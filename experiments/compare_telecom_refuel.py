"""Compare direct Telecom refueling with the durable LangGraph path on fixed trajectories."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from math import isclose
from pathlib import Path
import sys
import tempfile
from time import perf_counter_ns

from loguru import logger
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.tasks import Task
from tau2.domains.telecom.environment import get_environment

from serve_telecom import HISTORY, dump_history
from telecom_refuel import RefuelWorkflow, TicketStore, dump_environment, make_quote, restore_environment
from verify_telecom_stability import distribution


@dataclass(frozen=True)
class Case:
    id: str
    task: Task
    source: str
    history: list
    proposal: AssistantMessage
    suffix: list
    environment_before: str
    archive_reward: float
    termination: str


def make_case(row, task, source):
    messages = HISTORY.validate_python(row["messages"])
    for index, message in enumerate(messages[:-1]):
        calls = message.tool_calls if isinstance(message, AssistantMessage) else None
        if (index >= 2 and calls and len(calls) == 1 and calls[0].name == "refuel_data"
                and isinstance(messages[index - 1], UserMessage)):
            result = messages[index + 1]
            if not isinstance(result, ToolMessage) or result.id != calls[0].id or result.error:
                break
            initial = task.initial_state
            environment = get_environment()
            environment.set_state(
                initial.initialization_data if initial else None,
                initial.initialization_actions if initial else None,
                messages[:index],
            )
            return Case(str(task.id), task, source, messages[:index], message, messages[index + 2:],
                        dump_environment(environment), float(row["reward"]), row["termination_reason"])
    raise ValueError(f"Missing verified refuel receipt for {task.id} in {source}")


def load_cases(reference, catalog):
    wanted = defaultdict(set)
    for entry in reference:
        wanted[entry["archive"]].add(entry["task_id"])
    cases = {}
    for source, ids in wanted.items():
        with Path(source).open(encoding="utf-8") as archive:
            for row in (json.loads(line) for line in archive if line.strip()):
                task_id = row["task_id"]
                if row["domain"] == "telecom" and task_id in ids:
                    cases[task_id] = make_case(row, Task.model_validate(catalog[task_id]), source)
    return [cases[entry["task_id"]] for entry in reference]


def tool_content(message):
    try:
        return json.loads(message.content)
    except (TypeError, ValueError):
        return message.content


def billing_outcome(environment, call):
    customer = call.arguments["customer_id"]
    line = call.arguments["line_id"]
    usage = environment.tools.get_data_usage(customer, line)
    charges = Counter(item.amount for bill in environment.tools.db.bills if bill.customer_id == customer
                      for item in bill.line_items if item.description.startswith("Data refueling:"))
    return usage["data_refueling_gb"], charges


def official_outcome(case, environment):
    # The archived tasks use only ENV_ASSERTION in their official reward basis.
    criteria = case.task.evaluation_criteria
    assert {item.value for item in criteria.reward_basis} == {"ENV_ASSERTION"}
    if case.termination.rsplit(".", 1)[-1].upper() not in {"USER_STOP", "AGENT_STOP"}:
        return 0.0

    # Follow the official environmental replay convention: replay completed writes.
    pending = {}
    for message in case.suffix:
        if isinstance(message, (AssistantMessage, UserMessage)):
            for call in message.tool_calls or []:
                pending[call.requestor, call.id] = call
        elif isinstance(message, ToolMessage):
            call = pending.pop((message.requestor, message.id))
            if environment._is_mutating_tool(call.name):
                result = environment.get_response(call)
                if result.error != message.error or tool_content(result) != tool_content(message):
                    raise ValueError(f"Suffix tool response for {case.id}/{call.name} diverged from the archive")
    checks = [environment.run_env_assertion(item, raise_assertion_error=False) for item in criteria.env_assertions]
    return float(all(checks))


def compare(case, workflow, *, old_first, score):
    store = workflow.store
    before = restore_environment(case.environment_before)
    after = restore_environment(case.environment_before)
    call = case.proposal.tool_calls[0]
    history = dump_history(case.history)
    with_proposal = [*history, *dump_history([case.proposal])]
    ticket = store.load(store.create(case.id, after, history))

    def direct():
        start = perf_counter_ns()
        tool = before.get_response(call)
        return tool, (perf_counter_ns() - start) / 1_000_000

    def protected():
        start = perf_counter_ns()
        quote = make_quote(after, call)
        store.save_turn(ticket, after, with_proposal, quote)
        waiting = workflow.prepare(store.load(ticket.id))
        paused = perf_counter_ns()
        receipt = workflow.confirm(ticket.id, quote["id"], True)
        completed = perf_counter_ns()
        assert any(task.interrupts for task in waiting.tasks), case.id
        settled = store.load(ticket.id)
        tool = ToolMessage.model_validate(settled.history[-1])
        return (tool, receipt, settled.snapshot, (paused - start) / 1_000_000,
                (completed - paused) / 1_000_000)

    if old_first:
        old_tool, old_ms = direct()
        new_tool, receipt, snapshot, pause_ms, confirm_ms = protected()
    else:
        new_tool, receipt, snapshot, pause_ms, confirm_ms = protected()
        old_tool, old_ms = direct()
    assert not old_tool.error and not new_tool.error
    old_content, new_content = tool_content(old_tool), tool_content(new_tool)
    assert all(isclose(float(old_content[key]), float(new_content[key]))
               for key in ("new_data_refueling_gb", "charge")), case.id
    assert isclose(float(old_content["charge"]), receipt["charge_usd"]), case.id
    applied = restore_environment(snapshot)
    assert billing_outcome(before, call) == billing_outcome(applied, call), case.id

    result = {
        "task_id": case.id, "archive": case.source, "old_ms": old_ms, "pause_ms": pause_ms,
        "confirm_ms": confirm_ms, "new_ms": pause_ms + confirm_ms,
        "delta_ms": pause_ms + confirm_ms - old_ms,
        "tool_response_exact_match": old_content == new_content,
    }
    if score:
        result.update(archive_score=case.archive_reward, old_score=official_outcome(case, before),
                      new_score=official_outcome(case, applied))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-catalog", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="compare-", dir=args.output_dir))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    catalog = json.loads(args.task_catalog.read_text(encoding="utf-8"))
    cases = load_cases(reference, catalog)
    print(f"Paired cases {len(cases)}, run directory {run}", flush=True)
    workflow = RefuelWorkflow(run, TicketStore(run / "tickets.sqlite3"))
    try:
        for index, case in enumerate(cases[:2]):
            compare(case, workflow, old_first=bool(index), score=False)
        outcomes = []
        for repeat in range(args.repeats):
            for index, case in enumerate(cases):
                outcome = compare(case, workflow, old_first=(index + repeat) % 2 == 0, score=(repeat == 0))
                outcome["repeat"] = repeat + 1
                outcomes.append(outcome)
            print(f"Completed paired round {repeat + 1}", flush=True)
    finally:
        workflow.connection.close()
    paired = [row for row in outcomes if row["repeat"] == 1]
    report = {
        "run_dir": str(run), "cases": len(cases), "repeats": args.repeats,
        "passed_tasks": {name: sum(row[f"{name}_score"] == 1 for row in paired) for name in ("archive", "old", "new")},
        "archive_reproduced": sum(row["archive_score"] == row["old_score"] for row in paired),
        "changed_scores": [row["task_id"] for row in paired if row["old_score"] != row["new_score"]],
        "equivalent_bill_and_quota": len(paired),
        "equivalent_tool_response_text": sum(row["tool_response_exact_match"] for row in paired),
        "latency": {key: distribution([row[key] for row in outcomes])
                    for key in ("old_ms", "pause_ms", "confirm_ms", "new_ms", "delta_ms")},
    }
    (run / "paired_results.json").write_text(json.dumps(outcomes, indent=2) + "\n", encoding="utf-8")
    body = json.dumps(report, indent=2) + "\n"
    (run / "summary.json").write_text(body, encoding="utf-8")
    print(body, end="")


if __name__ == "__main__":
    main()
