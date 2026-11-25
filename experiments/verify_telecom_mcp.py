"""Replay native Telecom tool calls and exercise the MCP approval boundary on CPU."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
import json
from math import ceil, isclose
from pathlib import Path
import sqlite3
from statistics import median
import subprocess
import sys
from time import monotonic, perf_counter, sleep
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError

from tau2.agent import llm_agent
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from tau2.data_model.tasks import Task
from tau2.domains.telecom.environment import get_environment

from serve_telecom import DEFAULT_PLAYBOOK, HISTORY, TelecomService
from telecom_mcp import READ_TOOLS, connect_mcp
from telecom_refuel import TicketError, dump_environment
from verify_telecom_refuel import available_port, ledger, request


TOOLS_DIRECTORY = Path(__file__).resolve().parents[1] / "src"


@contextmanager
def boot(script, port, log, *args):
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen([sys.executable, "-B", str(TOOLS_DIRECTORY / script), "--port", str(port),
                                    *map(str, args)], stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = monotonic() + 42
            while process.poll() is None and monotonic() < deadline:
                try:
                    request(port, "/health")
                    break
                except URLError:
                    sleep(0.25)
            else:
                raise RuntimeError(f"Service failed to start. See {log}")
            yield
        finally:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def archive_rows(path):
    with path.open(encoding="utf-8") as archive:
        for row in (json.loads(line) for line in archive if line.strip()):
            if row["domain"] == "telecom":
                yield row


def canonical(message):
    try:
        content = json.loads(message.content)
    except (TypeError, ValueError):
        content = message.content
    return message.id, message.role, message.requestor, bool(message.error), content


def percentile(values):
    return {"p50_ms": round(median(values), 4), "p95_ms": round(sorted(values)[ceil(len(values) * .95) - 1], 4)}


def snapshot_after_a_write(folder, mcp):
    states = []
    for label, tools in (("direct", None), ("mcp", mcp)):
        state = folder / f"turn-{label}"
        state.mkdir()
        args = SimpleNamespace(state_dir=state, playbook=DEFAULT_PLAYBOOK, task_catalog=None, replay_traces=None,
                               agent_model="openai/evol-llm-agent", agent_url="http://127.0.0.1:1/v1", max_tool_rounds=5)
        service = TelecomService(args, tools)
        try:
            env = get_environment()
            line = next(obj for obj in env.tools.db.lines if obj.line_id == "L1002")
            line.roaming_enabled = False
            ticket = service.store.create("roaming-check", env, [])
            actions = [("enable_roaming", {"customer_id": "C1001", "line_id": line.line_id}),
                       ("get_details_by_id", {"id": line.line_id})]
            outputs = [AssistantMessage(role="assistant", tool_calls=[ToolCall(id=f"step-{number}", name=name,
                       arguments=arguments, requestor="assistant")]) for number, (name, arguments) in enumerate(actions)]
            outputs.append(AssistantMessage(role="assistant", content="I checked the updated roaming status."))
            with patch.object(llm_agent, "generate", side_effect=outputs):
                final = service.respond(ticket, "Please enable roaming on my line and confirm the current status.")
            assert final["status"] == "ready"
            history = service.store.load(ticket).history
            receipts = [canonical(ToolMessage.model_validate(message)) for message in history if message["role"] == "tool"]
            assert len(receipts) == 2 and receipts[1][-1]["roaming_enabled"] is True
            states.append((receipts, service.store.load(ticket).snapshot))
        finally:
            service.workflow.connection.close()
    assert states[0] == states[1]


def verify_reads(args):
    port = available_port()
    directory = args.output_dir
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    counts, timings, records, quota_states = Counter(), {"direct": [], "mcp": []}, [], {}
    with boot("serve_telecom_mcp.py", port, directory / "read-server.log", "--state-dir", directory), \
            connect_mcp(f"http://127.0.0.1:{port}/mcp/") as mcp:
        assert {tool.name for tool in mcp.portal.call(mcp.client.list_tools).tools} == READ_TOOLS
        for row in archive_rows(args.trace):
            task_id = row["task_id"]
            if task_id not in catalog:
                continue
            initial = Task.model_validate(catalog[task_id]).initial_state
            messages = HISTORY.validate_python(row["messages"])
            receipts = {message.id: message for message in messages if isinstance(message, ToolMessage)}
            for index, message in enumerate(messages):
                calls = (message.tool_calls or []) if isinstance(message, AssistantMessage) else []
                if not any(call.name in READ_TOOLS and call.id in receipts for call in calls):
                    continue
                env = get_environment()
                env.set_state(initial.initialization_data if initial else None,
                              initial.initialization_actions if initial else None, messages[:index])
                for call in calls:
                    if call.id not in receipts:
                        break
                    call = call.model_copy(update={"requestor": "assistant"})
                    start = perf_counter()
                    direct = env.get_response(call)
                    elapsed = (perf_counter() - start) * 1000
                    if call.name not in READ_TOOLS:
                        continue
                    start = perf_counter()
                    snapshot = dump_environment(env)
                    via_mcp = mcp.read(snapshot, call)
                    milliseconds = (perf_counter() - start) * 1000
                    assert canonical(direct) == canonical(via_mcp), (task_id, call.name, index)
                    counts[call.name] += 1
                    timings["direct"].append(elapsed)
                    timings["mcp"].append(milliseconds)
                    records.append({"task_id": task_id, "tool": call.name, "direct_ms": elapsed, "mcp_ms": milliseconds,
                                    "error": bool(direct.error)})
                    if call.name == "get_data_usage" and not direct.error:
                        fields = json.loads(direct.content)
                        overage = float(fields["data_used_gb"]) >= float(fields["data_limit_gb"]) + float(fields["data_refueling_gb"])
                        pair = tuple(call.arguments.get(key) for key in ("customer_id", "line_id"))
                        quota_states.setdefault(pair, {})[overage] = (snapshot, call, canonical(direct))

        healthy, exhausted = next((states[False], states[True]) for states in quota_states.values() if len(states) == 2)

        def verify_isolated(check):
            snapshot, call, expected = check
            assert canonical(mcp.read(snapshot, call)) == expected

        with ThreadPoolExecutor(max_workers=8) as workers:
            list(workers.map(verify_isolated, [healthy, exhausted] * 12))
        env = get_environment()
        for name, arguments in [("get_details_by_id", {"id": "L999999"}),
                                ("get_data_usage", {"customer_id": "C999999", "line_id": "L1002"})]:
            call = ToolCall(id=f"error-{name}", name=name, requestor="assistant", arguments=arguments)
            native = env.get_response(call)
            assert native.error and canonical(native) == canonical(mcp.read(dump_environment(env), call))
        snapshot_after_a_write(directory, mcp)
    report = {"passed": True, "matched_reads": sum(counts.values()), "calls_by_tool": dict(counts),
              "archived_tickets": len({row["task_id"] for row in records}), "isolated_concurrent_reads": 24,
              "matching_native_errors": 2, "same_turn_state_after_write": True,
              "latency": {key: percentile(values) for key, values in timings.items()}}
    save(directory / "read_details.json", records)
    return report


def refuel_cases(trace, catalog):
    for row in archive_rows(trace):
        if row["task_id"] not in catalog:
            continue
        messages = row["messages"]
        for index, message in enumerate(messages[:-1]):
            calls = message.get("tool_calls") or []
            if index < 2 or message["role"] != "assistant" or len(calls) != 1 or calls[0]["name"] != "refuel_data" or messages[index - 1]["role"] != "user":
                continue
            receipt = messages[index + 1]
            if receipt["role"] == "tool" and receipt.get("id") == calls[0]["id"] and not receipt.get("error"):
                yield row["task_id"], calls[0]["arguments"], float(json.loads(receipt["content"])["charge"])
            break


def operations_count(directory):
    with closing(sqlite3.connect(directory / "operations.sqlite3")) as db:
        return db.execute("SELECT count(*) FROM operations").fetchone()[0]


def compare_ledger(state, opened, arguments, expected, before, receipt):
    after = ledger(state, opened["ticket_id"], arguments["customer_id"], arguments["line_id"])
    assert after["confirmation_receipts"] == 1
    assert isclose(after["data_refueling_gb"], before["data_refueling_gb"] + arguments["gb_amount"])
    assert len(after["refuel_charges"]) == len(before["refuel_charges"]) + 1
    assert isclose(sum(after["refuel_charges"]) - sum(before["refuel_charges"]), expected)
    assert receipt["gb_added"] == arguments["gb_amount"] and receipt["charge_usd"] == expected


def verify_refuels(args):
    directory = args.output_dir
    cases = list(refuel_cases(args.trace, json.loads(args.catalog.read_text(encoding="utf-8"))))
    assert len(cases) > 1
    tool_state, app_state, interrupted = [directory / name for name in ("mcp-state", "agent-state", "interrupted")]
    for folder in (tool_state, app_state, interrupted):
        folder.mkdir()
    tool_port, app_port = available_port(), available_port()
    url = f"http://127.0.0.1:{tool_port}/mcp/"
    app_options = ("--state-dir", app_state, "--replay-traces", args.trace, "--task-catalog", args.catalog,
                   "--mcp-url", url, "--mcp-refuel")

    def read_ledger(state, opened, call):
        return ledger(state, opened["ticket_id"], call["customer_id"], call["line_id"])

    def confirm(opened):
        return request(app_port, f"/tickets/{opened['ticket_id']}/confirm",
                       {"quote_id": opened["quote"]["id"], "approved": True})

    with boot("serve_telecom_mcp.py", tool_port, directory / "mcp-before.log", "--state-dir", tool_state, "--with-refuel"):
        tickets = []
        with boot("serve_telecom.py", app_port, directory / "app-before.log", *app_options):
            for case_id, call, charge in cases:
                opened = request(app_port, "/tickets", {"task_id": case_id}, 201)
                assert opened["status"] == "awaiting_confirmation" and opened["quote"]["charge_usd"] == charge
                before = read_ledger(app_state, opened, call)
                assert before["confirmation_receipts"] == 0
                tickets.append((opened, call, charge, before))
        assert operations_count(tool_state) == 0
        with boot("serve_telecom.py", app_port, directory / "app-restart.log", *app_options):
            for opened, call, _, before in tickets:
                assert request(app_port, f"/tickets/{opened['ticket_id']}") == opened
                assert read_ledger(app_state, opened, call) == before
            wrong_ticket = f"/tickets/{tickets[0][0]['ticket_id']}/confirm"
            request(app_port, wrong_ticket, {"quote_id": tickets[1][0]["quote"]["id"], "approved": True}, 409)
            assert operations_count(tool_state) == 0
            with ThreadPoolExecutor(max_workers=8) as workers:
                duplicates = list(workers.map(confirm, [entry[0] for entry in tickets for _ in range(2)]))
            assert duplicates[::2] == duplicates[1::2]
            for (opened, call, charge, before), receipt in zip(tickets, duplicates[::2], strict=True):
                compare_ledger(app_state, opened, call, charge, before, receipt)
            rejected = request(app_port, "/tickets", {"task_id": cases[0][0]}, 201)
            before = read_ledger(app_state, rejected, cases[0][1])
            route = f"/tickets/{rejected['ticket_id']}/confirm"
            decline = request(app_port, route, {"quote_id": rejected["quote"]["id"], "approved": False})
            after = read_ledger(app_state, rejected, cases[0][1])
            assert decline["gb_added"] == decline["charge_usd"] == 0
            assert after == {**before, "confirmation_receipts": 1}
        assert operations_count(tool_state) == len(cases)

        case_id, arguments, charge = cases[0]
        service_args = SimpleNamespace(state_dir=interrupted, playbook=DEFAULT_PLAYBOOK, task_catalog=args.catalog,
                                       replay_traces=args.trace, mcp_refuel=True)
        with connect_mcp(url, refuel=True) as tools:
            service = TelecomService(service_args, tools)
            try:
                opened = service.create(case_id)
                before = read_ledger(interrupted, opened, arguments)
                ticket_id, quote_id = opened["ticket_id"], opened["quote"]["id"]

                def lose_response(*parameters):
                    tools.refuel(*parameters)
                    raise TicketError("Simulated response loss after tool commit", 503)

                with patch.object(service.store, "refuel_executor", lose_response):
                    try:
                        service.workflow.confirm(ticket_id, quote_id, True)
                    except TicketError as exc:
                        assert exc.status == 503
                    else:
                        raise AssertionError("Expected an interrupted confirmation")
                assert read_ledger(interrupted, opened, arguments) == before
                assert operations_count(tool_state) == len(cases) + 1
            finally:
                service.workflow.connection.close()

    with boot("serve_telecom_mcp.py", tool_port, directory / "mcp-restart.log", "--state-dir", tool_state, "--with-refuel"), \
            connect_mcp(url, refuel=True) as tools:
        service = TelecomService(service_args, tools)
        try:
            result = service.workflow.confirm(ticket_id, quote_id, True)
            compare_ledger(interrupted, opened, arguments, charge, before, result)
            assert service.workflow.confirm(ticket_id, quote_id, True) == result
        finally:
            service.workflow.connection.close()
    recorded = operations_count(tool_state)
    executions = sum(path.read_text(encoding="utf-8").count("Data refueled for line ")
                     for path in directory.glob("mcp-*.log"))
    assert executions == recorded == len(cases) + 1, (executions, recorded)
    return {"passed": True, "archived_approved_tickets": len(cases), "concurrent_confirmation_requests": len(duplicates),
            "declined_ticket_without_charge": True, "wrong_ticket_rejected": True,
            "recovered_after_tool_response_loss_and_both_restarts": True,
            "remote_refuel_executions_for_all_approved_tickets": executions}


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("read", "refuel"))
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = (verify_reads if args.mode == "read" else verify_refuels)(args)
    save(args.output_dir / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
