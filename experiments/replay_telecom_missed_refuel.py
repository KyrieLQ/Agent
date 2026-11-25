"""Replay unresolved Telecom tasks through the current approval-aware agent entry point."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from loguru import logger
import tau2.agent.llm_agent as llm_agent
from tau2.agent.base_agent import is_valid_agent_history_message
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.domains.telecom.environment import get_environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

from serve_telecom import DEFAULT_PLAYBOOK, HISTORY, TelecomService, dump_history


def archived_turns(messages):
    customer_indexes = [index for index, message in enumerate(messages) if isinstance(message, UserMessage)]
    for start, stop in zip(customer_indexes, [*customer_indexes[1:], len(messages)]):
        customer = messages[start]
        if customer.tool_calls or not customer.content:
            continue
        responses = [message for message in messages[start + 1:stop] if isinstance(message, AssistantMessage)]
        if responses and not responses[-1].tool_calls:
            yield start, stop, responses


def comparable(message):
    content = message.content
    if isinstance(message, ToolMessage):
        try:
            content = json.loads(content)
        except (TypeError, ValueError):
            pass
        return message.role, message.id, message.error, content
    return message.role, content, [call.model_dump() for call in message.tool_calls or []]


def score(task, messages, row):
    timestamps = [message.timestamp for message in messages if message.timestamp]
    simulation = SimulationRun(
        id=f"replay-{task.id}", task_id=task.id, start_time=timestamps[0], end_time=timestamps[-1],
        duration=0, termination_reason=TerminationReason[row["termination_reason"].rsplit(".", 1)[-1]],
        messages=messages,
    )
    return evaluate_simulation(simulation, task, EvaluationType.ALL, False, "telecom").reward


def replay(task, row, service):
    messages = HISTORY.validate_python(row["messages"])
    replayed = list(messages)
    initial = task.initial_state
    older_score = score(task, messages, row)
    assert older_score == row["reward"], task.id
    turns = 0
    decisions = 0
    for start, stop, responses in archived_turns(messages):
        environment = get_environment()
        environment.set_state(initial.initialization_data if initial else None,
                              initial.initialization_actions if initial else None, messages[:start])
        history = [message for message in messages[:start] if is_valid_agent_history_message(message)]
        ticket_id = service.store.create(task.id, environment, dump_history(history))
        recorded = iter(responses)
        with patch.object(llm_agent, "generate", side_effect=lambda **_: next(recorded).model_copy(deep=True)) as model:
            status = service.respond(ticket_id, messages[start].content)
        assert model.call_count == len(responses), (task.id, start, model.call_count, len(responses))
        assert status["status"] == "ready" and service.store.receipt(ticket_id) is None, (task.id, start, status)
        after = HISTORY.validate_python(service.store.load(ticket_id).history[len(history):])
        indexes = [pos for pos in range(start, stop) if is_valid_agent_history_message(messages[pos])]
        assert len(after) == len(indexes), (task.id, start, len(after), len(indexes))
        for position, observed in zip(indexes, after, strict=True):
            assert comparable(observed) == comparable(messages[position]), (task.id, position, observed.role)
            replayed[position] = observed
        turns += 1
        decisions += model.call_count
    newer_score = score(task, replayed, row)
    proposals = sum(call.name == "refuel_data" for message in replayed if isinstance(message, AssistantMessage)
                    for call in message.tool_calls or [])
    return {"task_id": task.id, "archive_score": older_score, "langgraph_score": newer_score,
            "termination_reason": row["termination_reason"], "replayed_customer_turns": turns,
            "replayed_model_decisions": decisions, "refuel_proposals": proposals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-catalog", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="replay-", dir=args.output_dir))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    wanted = {row["task_id"]: row for row in audit["cases"] if row["state"] == "no_refuel_called"}
    service = TelecomService(SimpleNamespace(
        state_dir=folder, task_catalog=args.task_catalog, playbook=DEFAULT_PLAYBOOK, replay_traces=None,
        agent_model="openai/evol-llm-agent", agent_url="http://127.0.0.1:1/v1", max_tool_rounds=60,
    ))
    results = []
    print(f"Run directory {folder}, expected cases {len(wanted)}", flush=True)
    try:
        with Path(audit["source"]).open(encoding="utf-8") as archive:
            for row in (json.loads(line) for line in archive if line.strip()):
                task_id = row["task_id"]
                if row["domain"] != "telecom" or task_id not in wanted:
                    continue
                result = replay(Task.model_validate(service.tasks[task_id]), row, service)
                result["overage_already_verified"] = bool(wanted[task_id]["overage_evidence"])
                results.append(result)
                if len(results) % 4 == 0:
                    print(f"Replayed {len(results)} cases", flush=True)
        checkpoint_exists = service.workflow.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'checkpoints'"
        ).fetchone()
        checkpoints = service.workflow.connection.execute("SELECT count(*) FROM checkpoints").fetchone()[0] if checkpoint_exists else 0
    finally:
        service.workflow.connection.close()
    assert {row["task_id"] for row in results} == set(wanted)
    report = {
        "run_dir": str(folder), "source": audit["source"], "tasks": len(results),
        "archive_passed": sum(row["archive_score"] == 1 for row in results),
        "langgraph_passed": sum(row["langgraph_score"] == 1 for row in results),
        "refuel_proposals": sum(row["refuel_proposals"] for row in results), "langgraph_approval_checkpoints": checkpoints,
        "replayed_customer_turns": sum(row["replayed_customer_turns"] for row in results),
        "replayed_model_decisions": sum(row["replayed_model_decisions"] for row in results),
        "verified_overage_cases": sum(row["overage_already_verified"] for row in results),
        "passed": True,
    }
    (folder / "per_task.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    text = json.dumps(report, indent=2) + "\n"
    (folder / "summary.json").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
