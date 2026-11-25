"""Replay frozen agent requests against two vLLM serving configurations."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter
from urllib.request import Request, urlopen


def freeze(trace, output):
    from pydantic import TypeAdapter
    from tau2.agent.base_agent import is_valid_agent_history_message
    from tau2.data_model.message import Message, SystemMessage
    from tau2.utils.llm_utils import to_litellm_messages

    adapter = TypeAdapter(list[Message])
    samples = defaultdict(list)
    for line in trace.open():
        row = json.loads(line)
        if not row.get("error"):
            samples[row["domain"]].append(row)
    workload = []
    for domain, rows in sorted(samples.items()):
        order = sorted(rows, key=lambda r: hashlib.sha256(("serving-ab-v1:" + r["task_id"]).encode()).digest())
        for row in order[:4]:
            messages = row["messages"]
            decisions = [i for i, m in enumerate(messages) if m["role"] == "assistant" and m.get("usage")]
            if len(decisions) < 2:
                raise ValueError(f"Expected both an early and late decision in {row['task_id']}")
            for stage, i in (("early", decisions[0]), ("late", decisions[(len(decisions) * 3) // 4])):
                visible = [m for m in adapter.validate_python(messages[:i]) if is_valid_agent_history_message(m)]
                system = SystemMessage(role="system", content=row["system_prompt"])
                body = {
                    "model": "evol-llm-agent", "messages": to_litellm_messages([system, *visible]),
                    "tools": row["tools"], "tool_choice": "auto", "temperature": 0.0,
                    "max_tokens": 192, "seed": 300, "chat_template_kwargs": {"enable_thinking": False},
                    "stream": True, "stream_options": {"include_usage": True},
                }
                workload.append({"id": f"{domain}/{row['task_id']}/{stage}", "domain": domain,
                                 "stage": stage, "body": body})
    order = sorted(workload, key=lambda r: hashlib.sha256(r["id"].encode()).digest())
    output.write_text(json.dumps(order, ensure_ascii=False) + "\n")


def request(url, item):
    body = json.dumps(item["body"]).encode()
    query = Request(url, data=body, headers={"Content-Type": "application/json"})
    parts = []
    calls = defaultdict(lambda: {"name": "", "arguments": ""})
    first_output = finish = usage = None
    started = perf_counter()
    with urlopen(query, timeout=150) as stream:
        for line in stream:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            for choice in event.get("choices", []):
                delta = choice["delta"]
                if delta.get("content") or delta.get("tool_calls"):
                    first_output = first_output or perf_counter()
                if delta.get("content"):
                    parts.append(delta["content"])
                for item_delta in delta.get("tool_calls") or ():
                    function = item_delta.get("function") or {}
                    fields = calls[item_delta["index"]]
                    fields["name"] += function.get("name") or ""
                    fields["arguments"] += function.get("arguments") or ""
                finish = choice.get("finish_reason") or finish
            usage = event.get("usage") or usage
    if usage is None or finish is None:
        raise ValueError(f"Streaming response was incomplete for {item['id']}")
    elapsed = perf_counter() - started
    valid = True
    for call in calls.values():
        try:
            call["arguments"] = json.loads(call["arguments"])
        except ValueError:
            valid = False
    return {
        "id": item["id"], "domain": item["domain"], "stage": item["stage"],
        "seconds": elapsed, "first_visible_output_seconds": first_output - started if first_output else None,
        "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
        "finish_reason": finish, "content": "".join(parts), "tool_calls": [calls[k] for k in sorted(calls)],
        "valid_tool_arguments": valid,
    }


def quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return {"median": median(ordered), "p95": ordered[ceil(len(ordered) * 0.95) - 1]}


def measure(workload, output, endpoint):
    items = json.loads(workload.read_text())
    url = endpoint + "/v1/chat/completions"
    for domain in sorted({item["domain"] for item in items}):
        request(url, next(item for item in items if item["domain"] == domain and item["stage"] == "early"))
    runs = {}
    for workers in (1, 8):
        started = perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(lambda item: request(url, item), items))
        elapsed = perf_counter() - started
        runs[str(workers)] = {"requests": rows, "workload_seconds": elapsed,
                              "completion_tokens_per_second": sum(r["completion_tokens"] for r in rows) / elapsed}
        output.write_text(json.dumps(runs, ensure_ascii=False, indent=2) + "\n")


def compare(folder, candidate_name):
    data = {arm: json.loads((folder / f"{arm}_requests.json").read_text())
            for arm in ("baseline", candidate_name)}
    results = {}
    for workers, base in data["baseline"].items():
        candidate = data[candidate_name][workers]
        a, b = base["requests"], candidate["requests"]
        if [r["id"] for r in a] != [r["id"] for r in b] or any(x["prompt_tokens"] != y["prompt_tokens"] for x, y in zip(a, b)):
            raise ValueError("The two serving modes did not replay the same inputs")
        profile = {}
        for name, trial in (("baseline", base), (candidate_name, candidate)):
            rows = trial["requests"]
            profile[name] = {
                "requests": len(rows), "seconds": quantiles([r["seconds"] for r in rows]),
                "first_visible_output_seconds": quantiles([r["first_visible_output_seconds"] for r in rows
                                                           if r["first_visible_output_seconds"] is not None]),
                "completion_tokens_per_second": trial["completion_tokens_per_second"],
                "invalid_tool_argument_requests": sum(not r["valid_tool_arguments"] for r in rows),
                "tool_call_requests": sum(bool(r["tool_calls"]) for r in rows),
            }
        signature = lambda r: (r["content"], r["tool_calls"], r["finish_reason"])
        results[workers] = {**profile, "equal_responses": sum(signature(x) == signature(y) for x, y in zip(a, b))}
    (folder / "serving_ab_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("freeze")
    prepare.add_argument("trace", type=Path)
    prepare.add_argument("output", type=Path)
    run = sub.add_parser("measure")
    run.add_argument("workload", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--endpoint", default="http://127.0.0.1:28876")
    audit = sub.add_parser("compare")
    audit.add_argument("folder", type=Path)
    audit.add_argument("--candidate", choices=("ngram", "cuda_graph"), default="ngram")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.trace, args.output)
    elif args.command == "measure":
        measure(args.workload, args.output, args.endpoint)
    else:
        compare(args.folder, args.candidate)


if __name__ == "__main__":
    main()
