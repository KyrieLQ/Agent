"""Compare the next 4B decision at the same archived Telecom dialog states."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
from time import perf_counter
from urllib.request import Request, urlopen

from loguru import logger
from tau2.agent.base_agent import is_valid_agent_history_message
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolCall
from tau2.data_model.tasks import Task
from tau2.domains.telecom.environment import get_environment
from tau2.utils.llm_utils import to_litellm_messages

from serve_telecom import APPROVAL_POLICY, HISTORY, dump_history
from telecom_refuel import RefuelWorkflow, TicketStore, make_quote


LOOKUPS = {"get_customer_by_phone", "get_customer_by_name", "get_customer_by_id", "get_details_by_id"}
REFUEL_INTEREST = re.compile(r"\b(refuel|add (?:more |[12](?:\.0)?\s?)?(?:gb|data)|extra data)\b", re.I)


def decision_anchor(messages, evidence, line_id):
    if evidence:
        overage = evidence[0]["receipt_index"]
        decisions = [index for index in range(overage + 1, len(messages))
                     if messages[index]["role"] == "assistant" and messages[index].get("usage")]
        if not decisions:
            raise ValueError("Archived overage has no subsequent assistant decision")
        interested = [index for index in decisions if messages[index - 1]["role"] == "user"
                      and REFUEL_INTEREST.search(messages[index - 1].get("content") or "")]
        return (interested[0] if interested else decisions[0]), "known_overage"

    requested = {}
    for index, message in enumerate(messages[:-1]):
        if message["role"] == "assistant":
            requested.update((call["id"], call["name"]) for call in message.get("tool_calls") or [])
        if (message["role"] == "tool" and message.get("requestor") == "assistant" and not message.get("error")
                and requested.get(message["id"]) in LOOKUPS and line_id in (message.get("content") or "")
                and messages[index + 1]["role"] == "assistant" and messages[index + 1].get("usage")):
            return index + 1, "no_usage_check"
    raise ValueError(f"No recorded account lookup for {line_id}")


def assemble(audit, catalog):
    wanted = {row["task_id"]: row for row in audit["cases"] if row["state"] == "no_refuel_called"}
    items = []
    with Path(audit["source"]).open(encoding="utf-8") as archive:
        for row in (json.loads(line) for line in archive if line.strip()):
            tid = row["task_id"]
            if row["domain"] != "telecom" or tid not in wanted:
                continue
            case = wanted[tid]
            target = case["expected"][0]
            index, stage = decision_anchor(row["messages"], case["overage_evidence"], target["line_id"])
            prefix = HISTORY.validate_python(row["messages"][:index])
            history = [message for message in prefix if is_valid_agent_history_message(message)]
            original = row["system_prompt"]
            head, marker, tail = original.rpartition("\n</policy>")
            if not marker:
                raise ValueError(f"No policy boundary for {tid}")
            enriched = f"{head}\n\n{APPROVAL_POLICY}{marker}{tail}"
            items.append({"id": tid, "stage": stage, "index": index, "prefix": prefix,
                          "task": Task.model_validate(catalog[tid]), "target": target, "tools": row["tools"],
                          "messages": {mode: to_litellm_messages([SystemMessage(role="system", content=system), *history])
                                       for mode, system in (("without_approval_policy", original), ("with_approval_policy", enriched))}})
    assert {item["id"] for item in items} == set(wanted)
    return items


def request(url, item, mode):
    parameters = {
        "model": "evol-llm-agent", "messages": item["messages"][mode], "tools": item["tools"],
        "tool_choice": "auto", "temperature": 0.5, "top_p": 0.98, "max_tokens": 1024,
        "seed": 300, "chat_template_kwargs": {"enable_thinking": False},
    }
    query = Request(url, data=json.dumps(parameters).encode(), headers={"Content-Type": "application/json"})
    start = perf_counter()
    with urlopen(query, timeout=150) as response:
        payload = json.load(response)
    choice = payload["choices"][0]
    message = choice["message"]
    results = []
    for call in message.get("tool_calls") or []:
        function = call["function"]
        try:
            args = json.loads(function["arguments"])
        except (TypeError, ValueError):
            args = None
        results.append({"id": call["id"], "name": function["name"], "arguments": args})
    return {"seconds": perf_counter() - start, "finish": choice["finish_reason"],
            "content": message.get("content"), "calls": results}


def matches_target(call, target, name):
    args = call["arguments"]
    return call["name"] == name and isinstance(args, dict) and all(args.get(key) == target[key]
           for key in ("line_id", "customer_id"))


def validate_proposal(item, response, workflow):
    if len(response["calls"]) != 1:
        return False
    call = response["calls"][0]
    if not matches_target(call, item["target"], "refuel_data"):
        return False
    tool = ToolCall(**call, requestor="assistant")
    initial = item["task"].initial_state
    environment = get_environment()
    environment.set_state(initial.initialization_data if initial else None,
                          initial.initialization_actions if initial else None, item["prefix"])
    try:
        quote = make_quote(environment, tool)
    except ValueError:
        return False
    history = dump_history([*item["prefix"], AssistantMessage(role="assistant", tool_calls=[tool])])
    ticket_id = workflow.store.create(item["id"], environment, history, quote)
    checkpoint = workflow.prepare(workflow.store.load(ticket_id))
    assert workflow.store.receipt(ticket_id) is None
    return any(task.interrupts for task in checkpoint.tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--task-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    items = assemble(json.loads(args.audit.read_text()), json.loads(args.task_catalog.read_text()))
    modes = ("without_approval_policy", "with_approval_policy")
    url = args.endpoint.rstrip("/") + "/v1/chat/completions"
    workflow = RefuelWorkflow(args.output_dir, TicketStore(args.output_dir / "tickets.sqlite3"))
    print(f"Prepared paired inputs for {len(items)} cases", flush=True)

    def pair(number_and_item):
        number, item = number_and_item
        order = modes if number % 2 == 0 else modes[::-1]
        return {mode: request(url, item, mode) for mode in order}

    rows = []
    scores = {mode: Counter() for mode in modes}
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            for item, responses in zip(items, workers.map(pair, enumerate(items)), strict=True):
                stage = item["stage"]
                target_tool = "refuel_data" if stage == "known_overage" else "get_data_usage"
                for mode, result in responses.items():
                    scores[mode][stage + "_matching_tool"] += any(
                        matches_target(call, item["target"], target_tool) for call in result["calls"]
                    )
                    scores[mode]["refuel_proposals"] += any(call["name"] == "refuel_data" for call in result["calls"])
                    if mode == "with_approval_policy":
                        scores[mode]["valid_refuel_pending_confirmation"] += validate_proposal(item, result, workflow)
                rows.append({"task_id": item["id"], "stage": stage, "message_index": item["index"], "responses": responses})
                if len(rows) % 4 == 0:
                    print(f"Completed {len(rows)} paired decisions", flush=True)
    finally:
        workflow.connection.close()
    summary = {"cases": len(rows), "stage_counts": dict(Counter(row["stage"] for row in rows)), "scores": scores}
    (args.output_dir / "decisions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    text = json.dumps(summary, indent=2) + "\n"
    (args.output_dir / "summary.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
