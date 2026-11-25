"""Verify line-detail enrichment and compare paired next decisions with the real 4B."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from statistics import median
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

from loguru import logger
from tau2.agent import llm_agent
from tau2.agent.base_agent import is_valid_agent_history_message
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolMessage
from tau2.data_model.tasks import Task
from tau2.domains.telecom.environment import get_environment
from tau2.utils.llm_utils import to_litellm_messages

from probe_telecom_refuel_decisions import request
from replay_telecom_missed_refuel import archived_turns
from serve_telecom import APPROVAL_POLICY, DEFAULT_PLAYBOOK, HISTORY, TelecomService, dump_history
from telecom_line_usage import USAGE_FIELD, enrich_line_usage, record
from telecom_refuel import dump_environment


MODES = ("approval_only", "with_line_usage")


def restore(task, history):
    initial = task.initial_state
    environment = get_environment()
    environment.set_state(initial.initialization_data if initial else None,
                          initial.initialization_actions if initial else None, history)
    return environment


def replay_receipts(row, task):
    messages = HISTORY.validate_python(row["messages"])
    history = []
    calls = {}
    reads = []
    additions = []
    for index, message in enumerate(messages):
        if isinstance(message, AssistantMessage):
            calls.update((call.id, call) for call in message.tool_calls or [])
        if isinstance(message, ToolMessage) and (call := calls.pop(message.id, None)) is not None:
            end = index + 1
            while end < len(messages) and isinstance(messages[end], ToolMessage):
                end += 1

            def read_usage(lookup):
                environment = restore(task, messages[:end])
                before = dump_environment(environment)
                start = perf_counter()
                receipt = environment.get_response(lookup)
                reads.append({"index": index, "line": lookup.arguments["line_id"],
                              "ms": (perf_counter() - start) * 1000, "error": bool(receipt.error)})
                assert dump_environment(environment) == before, task.id
                return receipt

            enriched = enrich_line_usage(history + messages[index:end], call, message, read_usage)
            if enriched is not message:
                line = record(message.content)
                facts = record(enriched.content).pop(USAGE_FIELD)
                assert record(enriched.content) == {**line, USAGE_FIELD: facts}, task.id
                additions.append((index, line["line_id"], facts))
            message = enriched
        history.append(message)
    return messages, history, additions, reads


def verify_service_turn(row, catalog, output_dir):
    task = Task.model_validate(catalog[row["task_id"]])
    original, _, additions, _ = replay_receipts(row, task)
    index = additions[0][0]
    start, _, outputs = next(turn for turn in archived_turns(original) if turn[0] < index < turn[1])
    state_dir = output_dir / "service_replay"
    state_dir.mkdir(exist_ok=True)
    prefix = [message for message in original[:start] if is_valid_agent_history_message(message)]
    args = SimpleNamespace(state_dir=state_dir, task_catalog=None, playbook=DEFAULT_PLAYBOOK,
                           replay_traces=None, agent_model="openai/evol-llm-agent", agent_url="http://127.0.0.1:1/v1",
                           max_tool_rounds=60, with_line_usage=True)
    service = TelecomService(args)
    try:
        ticket = service.store.create(task.id, restore(task, original[:start]), dump_history(prefix))
        outputs = iter(outputs)
        with patch.object(llm_agent, "generate", side_effect=lambda **_: next(outputs).model_copy(deep=True)) as model:
            response = service.respond(ticket, original[start].content)
        actual = service.store.load(ticket).history[len(prefix):]
        enriched = [record(item["content"]) for item in actual if item["role"] == "tool" and
                    USAGE_FIELD in record(item["content"])]
        assert len(enriched) == 1 and enriched[0][USAGE_FIELD] == additions[0][2], task.id
        assert response["status"] == "ready" and service.store.receipt(ticket) is None, task.id
        return {"verified": True, "model_calls": model.call_count}
    finally:
        service.workflow.connection.close()


def decision_input(row, original, enriched, addition, cohort):
    index, line_id, facts = addition
    decide = next(i for i in range(index + 1, len(original))
                  if isinstance(original[i], AssistantMessage) and original[i].usage)
    system = row["system_prompt"]
    head, marker, tail = system.rpartition("\n</policy>")
    if not marker:
        raise ValueError("Missing archived policy boundary")
    system = SystemMessage(role="system", content=f"{head}\n\n{APPROVAL_POLICY}{marker}{tail}")
    messages = {mode: to_litellm_messages([system, *(msg for msg in archive[:decide]
                                                   if is_valid_agent_history_message(msg))])
                for mode, archive in zip(MODES, (original, enriched), strict=True)}
    return {"id": row["task_id"], "cohort": cohort, "index": index, "line_id": line_id,
            "evidence": facts, "tools": row["tools"], "messages": messages}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def archive(args):
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    dev_catalog = {task["id"]: task for task in json.loads(args.dev_catalog.read_text(encoding="utf-8"))}
    omitted = {case["task_id"]: case["expected"][0]["line_id"] for case in audit["cases"]
               if case["state"] == "no_refuel_called" and not case["overage_evidence"]}
    reference = {case["task_id"] for case in audit["cases"]}
    old_rows = [row for row in read_jsonl(Path(audit["source"])) if row["domain"] == "telecom"]
    controls = [row for row in read_jsonl(args.dev) if "refuel_data" not in row["expected_tool_calls"]]
    assert {row["task_id"] for row in old_rows}.isdisjoint(dev_catalog)
    trials = []
    records = []
    for row in old_rows + controls:
        task_id = row["task_id"]
        is_dev = task_id in dev_catalog
        cohort = ("development_control" if is_dev else "missed_check" if task_id in omitted
                  else "other_refuel" if task_id in reference else "other_archive")
        task = Task.model_validate((dev_catalog if is_dev else catalog)[task_id])
        original, augmented, additions, reads = replay_receipts(row, task)
        records.append({"id": task_id, "cohort": cohort, "scenario": task_id.partition("]")[0][1:],
                        "reads": reads, "facts": additions})
        if task_id in omitted:
            assert len(additions) <= 1 and len(reads) == len(additions), task_id
            if additions:
                assert additions[0][1] == omitted[task_id] and additions[0][2]["data_usage_exceeded"], task_id
                trials.append(decision_input(row, original, augmented, additions[0], cohort))
        elif is_dev and additions and not additions[0][2]["data_usage_exceeded"]:
            trials.append(decision_input(row, original, augmented, additions[0], cohort))

    # Controls use independent archived development contexts with known healthy quotas.
    failures = [trial for trial in trials if trial["cohort"] == "missed_check"]
    healthy = [trial for trial in trials if trial["cohort"] == "development_control"]
    healthy = sorted(healthy, key=lambda trial: trial["id"])
    selected = [*failures, *healthy[:6]]
    assert failures and healthy
    assert len(selected) == len({item["id"] for item in selected})
    verification = verify_service_turn(next(row for row in old_rows if row["task_id"] == failures[0]["id"]),
                                       catalog, args.output_dir)
    groups = {}
    for cohort in sorted({row["cohort"] for row in records}):
        group = [row for row in records if row["cohort"] == cohort]
        timing = [read["ms"] for row in group for read in row["reads"]]
        groups[cohort] = {"cases": len(group), "cases_enriched": sum(bool(row["facts"]) for row in group),
                          "reads": len(timing), "read_p50_ms": median(timing) if timing else None,
                          "service_only_cases": sum(row["scenario"] == "service_issue" for row in group),
                          "service_only_enriched": sum(row["scenario"] == "service_issue" and bool(row["facts"])
                                                       for row in group)}
    save(args.output_dir / "archive_details.json", records)
    save(args.output_dir / "probe_inputs.json", selected)
    summary = {"groups": groups, "service_replay": verification, "probe_pairs": len(selected)}
    save(args.output_dir / "replay_summary.json", summary)
    print(json.dumps(summary), flush=True)


def reply_signals(reply):
    content = reply.get("content") or ""
    calls = [call["name"] for call in reply["calls"]]
    overage = re.search(r"\b(?:exceeded|exhausted|over (?:your|the) (?:monthly |plan )?(?:data )?(?:limit|allowance)|"
                        r"used up (?:all |your )?(?:monthly |mobile )?data|no (?:mobile )?data (?:left|remaining))\b",
                        content, re.I)
    return {"overage_mentioned": bool(overage), "read_usage": "get_data_usage" in calls,
            "proposed_refuel": "refuel_data" in calls, "other_tool": [x for x in calls if x != "get_data_usage"]}


def probe(args):
    items = json.loads((args.output_dir / "probe_inputs.json").read_text(encoding="utf-8"))
    url = args.endpoint.rstrip("/") + "/v1/chat/completions"

    def paired(number_item):
        number, item = number_item
        order = MODES if number % 2 == 0 else MODES[::-1]
        replies = {mode: request(url, item, mode) for mode in order}
        scores = {mode: reply_signals(reply) for mode, reply in replies.items()}
        return {"id": item["id"], "cohort": item["cohort"], "responses": replies, "signals": scores}

    path = args.output_dir / "probe_decisions.jsonl"
    rows = []
    with path.open("x", encoding="utf-8") as output, ThreadPoolExecutor(max_workers=args.workers) as workers:
        for case in workers.map(paired, enumerate(items)):
            rows.append(case)
            output.write(json.dumps(case, ensure_ascii=False) + "\n")
            output.flush()
            if len(rows) % 4 == 0 or len(rows) == len(items):
                print(f"Completed {len(rows)}/{len(items)} paired decisions", flush=True)
    scores = {}
    for cohort in sorted({row["cohort"] for row in rows}):
        subset = [row for row in rows if row["cohort"] == cohort]
        scores[cohort] = {"cases": len(subset), "arms": {mode: dict(sum((Counter(
            {key: value for key, value in row["signals"][mode].items() if isinstance(value, bool)})
            for row in subset), Counter())) for mode in MODES}}
    summary = {"paired_cases": len(rows), "scores": scores, "completed_full_dialogues": 0}
    save(args.output_dir / "probe_summary.json", summary)
    print(json.dumps(summary), flush=True)


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    modes = parser.add_subparsers(dest="mode", required=True)
    replay = modes.add_parser("archive")
    for option in ("audit", "catalog", "dev", "dev-catalog"):
        replay.add_argument(f"--{option}", type=Path, required=True)
    model = modes.add_parser("probe")
    model.add_argument("--endpoint", required=True)
    model.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="CRITICAL")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (archive if args.mode == "archive" else probe)(args)


if __name__ == "__main__":
    main()
