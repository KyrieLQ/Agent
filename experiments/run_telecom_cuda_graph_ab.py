"""Compare 4B serving modes through Telecom's actual approval and MCP tools."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import wraps
import json
from math import ceil, isclose
from pathlib import Path
import re
from statistics import median
import sys
import tempfile
from threading import local
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, ToolMessage, UserMessage
from tau2.domains.telecom.environment import get_environment

from run_telecom_line_usage_ab import EnrichedAgent, read_json, write_json
from serve_telecom import APPROVAL_POLICY, TelecomService, dump_history
from telecom_mcp import READ_TOOLS, REFUEL_TOOL, connect_mcp
from telecom_refuel import TicketError, dump_environment, restore_environment
from telecom_state import TelecomAgentProxy
from telecom_workflow import TRANSFER_NOTICE


MODES = ("eager", "cuda-graph")
ROLES = ("agent", "simulator")
DEVICE_REPORT = (
    "You just received the result of a device operation. In this response, report the actual result "
    "to the support agent in plain language, including any error or measured value. If the tool reports "
    "failure, say it failed. Report only what happened. Do not propose another operation or write any "
    "tool-call markup. Wait for the support agent before trying anything else."
)
DEVICE_TOOL_MARKUP = re.compile(r"<\s*/?\s*(?:tool_call|function)(?=[\s=>/])", re.IGNORECASE)
EXPLICIT_APPROVAL = re.compile(r"^i (?:approve|confirm|agree|accept)(?:[.!]+(?=\s|$)|$)")
APPROVAL_CONFLICT = re.compile(
    r"\b(?:don't|do not|never) (?:add|refuel|buy|purchase|charge|proceed|go ahead)\b|"
    r"\b(?:please cancel|cancel (?:this|the) (?:purchase|quote|order))\b|"
    r"\bi (?:don't|do not) want (?:to (?:add|refuel|buy|purchase|proceed)|(?:this|the) (?:purchase|quote|charge))\b|"
    r"\bi (?:withdraw my (?:approval|consent)|changed my mind|(?:no longer|don't|do not) consent)\b"
)
APPROVAL_HOLD = re.compile(
    r"\b(?:only (?:if|after|once|when)|but only|provided that|as long as|subject to|on condition)\b|"
    r"(?:^|[.!?;\n])\s*(?:(?:but|and|however|actually)[\s,]+)*(?:if|unless|wait|hold off|not yet)\b|"
    r"\bplease (?:wait|hold off|pause)\b|"
    r"\bbefore (?:you )?(?:add|refuel|buy|purchase|charge|proceed|go ahead|do (?:it|that))\b|"
    r"\b(?:change|revise) (?:the|this|that) (?:quote|amount|price|line)\b|"
    r"\b(?:make it|add|refuel|buy|purchase)[^.!?;\n]*\binstead\b"
)


def approval_decision(text):
    """Map explicit simulator replies to the work order's confirmation action."""
    reply = (text or "").strip().casefold().replace("’", "'")
    if re.match(r"^no(?:\W|$)", reply) or re.search(r"\bi (?:decline|refuse|(?:don't|do not) (?:approve|agree))\b", reply):
        return False
    if APPROVAL_CONFLICT.search(reply):
        return False
    if APPROVAL_HOLD.search(reply):
        return None
    # A standalone decision remains valid when the user adds an unrelated remark or question.
    if EXPLICIT_APPROVAL.match(reply):
        return True
    if re.search(r"\bi (?:don't|do not) want\b", reply):
        return False
    if "?" in reply or re.search(r"\b(?:but|before|if|unless)\b", reply):
        return None
    if re.fullmatch(r"(?:yes|ok|okay|sure|go ahead)[.! ]*", reply) or re.search(
        r"\b(?:i (?:approve|confirm|agree|accept)|please (?:proceed|go ahead|add|refuel)|go ahead and (?:add|proceed|refuel))\b", reply
    ):
        return True
    return None


def transfer_succeeded(proposal, results):
    if not isinstance(proposal, AssistantMessage):
        return False
    transfer_ids = {call.id for call in proposal.tool_calls or () if call.name == "transfer_to_human_agents"}
    return any(
        isinstance(result, ToolMessage) and result.id in transfer_ids and result.requestor == "assistant"
        and not result.error and result.content == "Transfer successful"
        for result in results
    )


class DeviceReportingUser:
    """Return control to the agent after each simulated customer tool operation."""

    def __init__(self, user):
        self.user = user
        self.events = Counter()

    def __getattr__(self, name):
        return getattr(self.user, name)

    def generate_next_message(self, message, state):
        receipts = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
        if receipts and all(isinstance(item, ToolMessage) and item.requestor == "user" for item in receipts):
            return self.report_results(message, state, receipts)

        reply, state = self.user.generate_next_message(message, state)
        if len(reply.tool_calls or ()) > 1:
            self.events["extra_device_calls_dropped"] += len(reply.tool_calls) - 1
            reply = reply.model_copy(update={"tool_calls": reply.tool_calls[:1]})
            state.messages[-1] = reply
        return reply, state

    def report_results(self, message, state, receipts):
        instructions = state.system_messages[-1]
        temporary = instructions.model_copy(update={"content": f"{instructions.content}\n\n{DEVICE_REPORT}"})
        report_state = state.model_copy(update={"system_messages": [*state.system_messages[:-1], temporary]})
        original_args = self.user.llm_args
        self.user.llm_args = {**original_args, "tool_choice": "none"}
        try:
            reply, report_state = self.user.generate_next_message(message, report_state)
        finally:
            self.user.llm_args = original_args
        report_state.system_messages = state.system_messages

        self.events["device_reports"] += 1
        content = (reply.content or "").strip()
        markup = DEVICE_TOOL_MARKUP.search(content)
        if markup:
            self.events["literal_tool_reports_blocked"] += 1
        if reply.tool_calls or not content or markup:
            self.events["receipt_report_fallbacks"] += 1
            content = "\n".join(f"{'Device error' if item.error else 'Device result'}: {item.content}"
                                for item in receipts)
            reply = reply.model_copy(update={"content": content, "tool_calls": None})
            report_state.messages[-1] = reply
        return reply, report_state


@dataclass
class PendingQuote:
    ticket_id: str
    quote_id: str
    prompt: str
    proposal: AssistantMessage
    authorized: bool = False


class ApprovalAgent(TelecomAgentProxy):
    def __init__(self, agent, environment, service, task_id):
        super().__init__(agent)
        self.environment = environment
        self.service = service
        self.task_id = task_id
        self.native_response = environment.get_response
        environment.get_response = self.execute_tool
        self.pending = None
        self.transfer_notice = None
        self.rejected = {}
        self.started = False
        self.events = Counter()

    @staticmethod
    def error(call, message):
        return ToolMessage(id=call.id, role="tool", requestor="assistant", content=message, error=True)

    def execute_tool(self, call):
        if not self.started or call.requestor != "assistant":
            return self.native_response(call)
        if call.id in self.rejected:
            return self.error(call, self.rejected.pop(call.id))
        if call.name in READ_TOOLS:
            self.events["mcp_reads"] += 1
            return self.service.execute_tool(self.environment, call)
        if call.name != REFUEL_TOOL:
            return self.native_response(call)

        pending = self.pending
        if pending is None or not pending.authorized or call.id != pending.proposal.tool_calls[0].id:
            return self.error(call, "The customer must confirm the current quote")
        saved = self.service.store.load(pending.ticket_id)
        # Device changes made by the simulated customer while the quote was pending remain in the live user state.
        if json.loads(saved.snapshot)["agent"] != self.environment.tools.db.model_dump(mode="json"):
            self.service.workflow.confirm(pending.ticket_id, pending.quote_id, False)
            self.pending = None
            return self.error(call, "Account state changed. Create a new quote")
        self.service.workflow.confirm(pending.ticket_id, pending.quote_id, True)
        updated = self.service.store.load(pending.ticket_id)
        response = ToolMessage.model_validate(updated.history[-1])
        self.environment.tools.db = restore_environment(updated.snapshot).tools.db
        self.environment.sync_tools()
        self.events["mcp_refuels"] += 1
        self.pending = None
        return response

    def is_stop(self, message):
        return (self.transfer_notice is not None and message is self.transfer_notice) or self.agent.is_stop(message)

    def generate_next_message(self, message, state):
        self.started = True
        results = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
        if state.messages and transfer_succeeded(state.messages[-1], results):
            self.transfer_notice = AssistantMessage(role="assistant", content=TRANSFER_NOTICE)
            state.messages.extend((*results, self.transfer_notice))
            self.events["handoffs_closed"] += 1
            return self.transfer_notice, state

        pending = self.pending
        if pending is not None and isinstance(message, UserMessage):
            decision = approval_decision(message.content)
            if decision is True:
                pending.authorized = True
                response = AssistantMessage(role="assistant", tool_calls=pending.proposal.tool_calls)
            elif decision is False:
                self.service.workflow.confirm(pending.ticket_id, pending.quote_id, False)
                self.pending = None
                self.events["declined_quotes"] += 1
                response = AssistantMessage(role="assistant", content="Understood. I will continue troubleshooting without purchasing data.")
            else:
                self.events["confirmation_clarifications"] += 1
                response = AssistantMessage(role="assistant", content=pending.prompt)
            state.messages.extend((message, response))
            return response, state

        response, state = self.agent.generate_next_message(message, state)
        calls = response.tool_calls or []
        if not any(call.name == REFUEL_TOOL for call in calls):
            return response, state
        if len(calls) != 1:
            self.rejected.update((call.id, "Submit refuel_data as a single tool call") for call in calls)
            return response, state

        try:
            quote = self.service.propose(self.environment, calls[0])
        except TicketError as exc:
            self.rejected[calls[0].id] = str(exc)
            return response, state
        ticket_id = self.service.store.create(self.task_id, self.environment, dump_history(state.messages), quote)
        opened = self.service.status(ticket_id)
        assert opened["status"] == "awaiting_confirmation"
        prompt = opened["message"] + " Please reply ‘I approve’ or ‘I decline’."
        self.pending = PendingQuote(ticket_id, quote["id"], prompt, response)
        self.events["quotes"] += 1
        visible = response.model_copy(update={"content": prompt, "tool_calls": None})
        state.messages[-1] = visible
        return visible, state


def build_service(folder, playbook, mcp):
    folder.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(state_dir=folder, playbook=playbook, task_catalog=None, mcp_refuel=True)
    return TelecomService(args, mcp)


def attach_telecom_runtime(orchestrator, service, task_id, *, check_line_usage=False):
    approval = ApprovalAgent(orchestrator.agent, orchestrator.environment, service, task_id)
    reporting = DeviceReportingUser(orchestrator.user)
    usage = EnrichedAgent(approval, approval.execute_tool) if check_line_usage else None
    orchestrator.agent = usage or approval
    orchestrator.user = reporting
    return approval, reporting, usage


def smoke(folder, mcp):
    class ScriptedAgent:
        def generate_next_message(self, message, state):
            response = AssistantMessage(role="assistant", tool_calls=[ToolCall(
                id="smoke-refuel", name=REFUEL_TOOL, requestor="assistant",
                arguments={"customer_id": "C1001", "line_id": "L1002", "gb_amount": 1},
            )])
            state.messages.extend((message, response))
            return response, state

    service = build_service(folder / "bridge-smoke", folder / "telecom_playbook.txt", mcp)
    try:
        for can_approve in (True, False):
            environment = get_environment()
            history = SimpleNamespace(messages=[])
            bridge = ApprovalAgent(ScriptedAgent(), environment, service, "smoke")
            initial = dump_environment(environment)
            prompt, _ = bridge.generate_next_message(UserMessage(role="user", content="Please add one GB."), history)
            assert "Confirm adding" in prompt.content and prompt.tool_calls is None
            ticket = bridge.pending.ticket_id
            stray = bridge.pending.proposal.tool_calls[0]
            assert bridge.execute_tool(stray).error
            clarify, _ = bridge.generate_next_message(UserMessage(role="user", content="What does this cost?"), history)
            assert clarify.content == prompt.content and service.store.receipt(ticket) is None
            text = "I approve." if can_approve else "I decline."
            approved, _ = bridge.generate_next_message(UserMessage(role="user", content=text), history)
            if can_approve:
                assert dump_environment(environment) == initial
                assert not environment.get_response(approved.tool_calls[0]).error
                assert bridge.pending is None and service.store.receipt(ticket)["gb_added"] == 1
                check = ToolCall(id="smoke-usage", name="get_data_usage", requestor="assistant",
                                 arguments={"customer_id": "C1001", "line_id": "L1002"})
                usage = json.loads(environment.get_response(check).content)
                assert isclose(float(usage["data_refueling_gb"]), float(service.store.receipt(ticket)["data_refueling_gb"]))
                assert environment.get_response(stray).error
            else:
                assert service.store.receipt(ticket)["charge_usd"] == 0 and dump_environment(environment) == initial
        result = {"passed": True, "mcp_lookup_after_approval": True, "decline_without_charge": True,
                  "missing_approval_and_duplicate_writes_blocked": True}
        write_json(folder / "bridge-smoke.json", result)
        print(json.dumps(result), flush=True)
    finally:
        service.workflow.connection.close()


def specs(benchmark, args, protocol):
    agent = benchmark.make_llm_spec("openai/evol-llm-agent", args.agent_url, temperature=protocol["agent_temperature"],
                                    enable_thinking=protocol["agent_thinking"], max_tokens=protocol["agent_max_tokens"])
    agent.args["top_p"] = protocol["agent_top_p"]
    user = benchmark.make_llm_spec("openai/evol-llm-user", args.user_url, temperature=protocol["user_temperature"],
                                   enable_thinking=protocol["user_thinking"], max_tokens=protocol["user_max_tokens"])
    return agent, user


def run_arm(args, mcp):
    import tau2.agent.llm_agent as llm_agent
    import tau2.evaluator.evaluator as evaluator
    import tau2.runner
    import tau2.user.user_simulator as user_simulator
    from loguru import logger
    from tau2_evolve import benchmark

    logger.remove()
    logger.add(sys.stderr, level="CRITICAL")
    folder = args.folder
    manifest = read_json(folder / "manifest.json")
    protocol = manifest["protocol"]
    tasks = read_json(folder / "dev_tasks.json")
    ids = manifest["task_ids"]
    assert [task["id"] for task in tasks] == ids and len(ids) == len(set(ids)) == 12
    benchmark._TASKS_CACHE["telecom"] = tasks
    playbook = folder / "telecom_playbook.txt"
    manual = "\n\n".join((playbook.read_text(encoding="utf-8").strip(), APPROVAL_POLICY))
    agent, user = specs(benchmark, args, protocol)
    arm_dir = folder / args.mode
    arm_dir.mkdir(parents=True, exist_ok=True)
    reset = arm_dir / "simulator-cache-reset.status"
    if not reset.is_file() or reset.read_text().strip() != "200":
        namespace = arm_dir / "simulator-cache-salt.txt"
        if not namespace.is_file():
            namespace.write_text(uuid4().hex + "\n", encoding="utf-8")
        user.args.setdefault("extra_body", {})["cache_salt"] = namespace.read_text(encoding="utf-8").strip()
    benchmark.prime_nl_judge_routing(user)
    build_native = tau2.runner.build_text_orchestrator
    native_score = evaluator.evaluate_simulation
    active = local()
    state_root = folder / args.mode / "tickets"
    state_root.mkdir(parents=True, exist_ok=True)

    def build(*parameters, **kwargs):
        orchestrator = build_native(*parameters, **kwargs)
        active.wrapper, active.reporting, _ = attach_telecom_runtime(orchestrator, active.service, active.task_id)
        return orchestrator

    def timed(role, generate):
        @wraps(generate)
        def measure(*parameters, **kwargs):
            begin = perf_counter()
            reply = generate(*parameters, **kwargs)
            active.durations[role].append(perf_counter() - begin)
            return reply
        return measure

    def score(*parameters, **kwargs):
        try:
            return native_score(*parameters, **kwargs)
        except Exception as exc:
            active.scoring_error = exc
            raise

    def evaluate(task_id):
        run_dir = Path(tempfile.mkdtemp(prefix=f"task-{ids.index(task_id) + 1:02d}-", dir=state_root))
        active.service = build_service(run_dir, playbook, mcp)
        active.task_id, active.scoring_error = task_id, None
        active.durations = {role: [] for role in ROLES}
        try:
            begin = perf_counter()
            row = benchmark.run_task(domain="telecom", task_id=task_id, agent_spec=agent, user_spec=user,
                                     seed=manifest["seed"], max_steps=protocol["max_steps"],
                                     max_errors=protocol["max_errors"], skill_text=manual)
            duration = perf_counter() - begin
            if active.scoring_error is not None:
                raise RuntimeError(f"Official scoring failed for {task_id}") from active.scoring_error
            if not all(active.durations.values()):
                raise RuntimeError(f"Missing model timings for {task_id}")
            assert manual in row["system_prompt"]
            return {**row, "duration_seconds": duration, "model_requests_seconds": active.durations,
                    "approval_and_mcp_events": dict(active.wrapper.events), "state_dir": str(run_dir),
                    "device_reporting_events": dict(active.reporting.events), "arm": args.mode}
        finally:
            active.service.workflow.connection.close()

    output_path = folder / args.mode / "dialogues.jsonl"
    done = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as archive:
            for row in map(json.loads, archive):
                assert row["task_id"] in ids and row["task_id"] not in done and row["arm"] == args.mode
                done.add(row["task_id"])
    with patch.object(tau2.runner, "build_text_orchestrator", build), \
            patch.object(evaluator, "evaluate_simulation", score), \
            patch.object(llm_agent, "generate", timed("agent", llm_agent.generate)), \
            patch.object(user_simulator, "generate", timed("simulator", user_simulator.generate)), \
            output_path.open("a", encoding="utf-8") as output, \
            ThreadPoolExecutor(max_workers=protocol["workers_per_arm"]) as executor:
        futures = [executor.submit(evaluate, task_id) for task_id in ids if task_id not in done]
        for future in as_completed(futures):
            row = future.result()
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            done.add(row["task_id"])
            print(f"{args.mode} {len(done)}/{len(ids)} complete, reward={row['reward']}", flush=True)


def distribution(numbers):
    ordered = sorted(numbers)
    return {"p50": round(median(ordered), 3), "p95": round(ordered[ceil(.95 * len(ordered)) - 1], 3)}


def summarize(folder):
    manifest = read_json(folder / "manifest.json")
    tables, scores = {}, {}
    for mode in MODES:
        with (folder / mode / "dialogues.jsonl").open(encoding="utf-8") as src:
            table = {row["task_id"]: row for row in map(json.loads, src)}
        assert set(table) == set(manifest["task_ids"])
        tables[mode] = table
        rows = list(table.values())
        scores[mode] = {
            "passes": sum(row["reward"] == 1.0 for row in rows),
            "tasks": len(rows),
            "end_to_end_seconds": distribution([row["duration_seconds"] for row in rows]),
            "model_seconds_per_task": {role: distribution([sum(row["model_requests_seconds"][role]) for row in rows])
                                       for role in ROLES},
            "model_seconds_per_request": {role: distribution([d for row in rows for d in row["model_requests_seconds"][role]])
                                          for role in ROLES},
            "model_requests": {role: sum(len(row["model_requests_seconds"][role]) for row in rows) for role in ROLES},
            "approval_and_mcp_events": dict(sum((Counter(row["approval_and_mcp_events"]) for row in rows), Counter())),
            "step_limit": sum(row["termination_reason"].endswith("MAX_STEPS") for row in rows),
        }
        telemetry = folder / mode / "gpu-memory.csv"
        if telemetry.is_file():
            memory = [int(line.split(",")[1]) for line in telemetry.read_text().splitlines() if "," in line]
            if memory:
                scores[mode]["peak_gpu_mib"] = max(memory)
    pairs = [(tables["eager"][tid], tables["cuda-graph"][tid]) for tid in manifest["task_ids"]]
    assert all(a["system_prompt"] == b["system_prompt"] and a["tools"] == b["tools"] for a, b in pairs)
    scores["paired_pass_outcomes"] = dict(Counter(f"{int(a['reward'] == 1)}->{int(b['reward'] == 1)}" for a, b in pairs))
    write_json(folder / "summary.json", scores)
    print(json.dumps(scores, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--mode", required=True, choices=(*MODES, "smoke", "summary"))
    parser.add_argument("--mcp-url")
    parser.add_argument("--agent-url")
    parser.add_argument("--user-url")
    args = parser.parse_args()
    if args.mode == "summary":
        summarize(args.folder)
    else:
        if not args.mcp_url or (args.mode in MODES and not (args.agent_url and args.user_url)):
            parser.error("The requested mode requires MCP and its inference endpoints")
        with connect_mcp(args.mcp_url, refuel=True) as mcp:
            if args.mode == "smoke":
                smoke(args.folder, mcp)
            else:
                run_arm(args, mcp)


if __name__ == "__main__":
    main()
