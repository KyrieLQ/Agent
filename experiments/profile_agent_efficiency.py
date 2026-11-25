"""Profile agent-side and simulator-side work from recorded benchmark traces."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from math import ceil
from pathlib import Path
from statistics import median


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return {"median": median(ordered), "p95": ordered[ceil(0.95 * len(ordered)) - 1]}


def span(messages):
    stamps = [datetime.fromisoformat(m["timestamp"]) for m in messages if m.get("timestamp")]
    return (max(stamps) - min(stamps)).total_seconds() if len(stamps) > 1 else None


def episode(trace):
    roles = defaultdict(list)
    calls = {}
    results = Counter()
    tool_output_chars = 0
    repeated_chars = 0
    messages = trace["messages"]
    for message in messages:
        role = message["role"]
        usage = message.get("usage")
        if role in {"assistant", "user"} and isinstance(usage, dict) and "prompt_tokens" in usage:
            roles[role].append(message)
        if role == "assistant":
            for call in message.get("tool_calls") or ():
                calls[call["id"]] = (call["name"], json.dumps(call["arguments"], sort_keys=True))
        if role == "tool" and message.get("requestor") == "assistant":
            text = message.get("content") or ""
            tool_output_chars += len(text)
            call = calls.pop(message.get("id"), None)
            if call is None:
                continue
            signature = (*call, text, bool(message.get("error")))
            repeated_chars += len(text) if results[signature] else 0
            results[signature] += 1

    stats = {}
    for role, requests in roles.items():
        prompt = [m["usage"]["prompt_tokens"] for m in requests]
        seconds = [m["generation_time_seconds"] for m in requests if m.get("generation_time_seconds") is not None]
        stats[role] = {
            "calls": len(requests), "input_tokens": sum(prompt),
            "output_tokens": sum(m["usage"].get("completion_tokens", 0) for m in requests),
            "first_input_tokens": prompt[0], "last_input_tokens": prompt[-1],
            "later_context_tokens": sum(max(0, p - prompt[0]) for p in prompt),
            "recorded_latency_seconds": seconds,
        }
    return {
        "key": (trace["domain"], trace["task_id"]), "passed": trace["reward"] == 1.0,
        "hit_step_limit": trace["termination_reason"].endswith("MAX_STEPS"),
        "trajectory_span_seconds": span(messages), "end_to_end_seconds": trace.get("duration_seconds"),
        "tool_output_chars": tool_output_chars, "repeated_tool_output_chars": repeated_chars,
        "assistant_tool_results": sum(results.values()),
        "repeated_assistant_tool_results": sum(n - 1 for n in results.values()), "roles": stats,
    }


def summarize(episodes):
    role_metrics = {}
    for role in ("assistant", "user"):
        data = [ep["roles"][role] for ep in episodes if role in ep["roles"]]
        input_tokens = sum(r["input_tokens"] for r in data)
        later_tokens = sum(r["later_context_tokens"] for r in data)
        timings = [seconds for r in data for seconds in r["recorded_latency_seconds"]]
        role_metrics[role] = {
            **{field: distribution([r[field] for r in data]) for field in (
                "calls", "input_tokens", "output_tokens", "first_input_tokens", "last_input_tokens"
            )},
            "total_calls": sum(r["calls"] for r in data), "episodes_with_usage": len(data),
            "input_tokens_above_first_request_share": later_tokens / input_tokens if input_tokens else None,
            "timed_request_count": len(timings), "request_latency_seconds": distribution(timings),
        }
    return {
        "tasks": len(episodes), "passed": sum(e["passed"] for e in episodes),
        "hit_step_limit": sum(e["hit_step_limit"] for e in episodes),
        "roles": role_metrics,
        "trajectory_span_seconds": distribution([x["trajectory_span_seconds"] for x in episodes
                                                 if x["trajectory_span_seconds"] is not None]),
        "end_to_end_seconds": distribution([x["end_to_end_seconds"] for x in episodes
                                            if x["end_to_end_seconds"] is not None]),
        "tool_output_chars": distribution([e["tool_output_chars"] for e in episodes]),
        "assistant_tool_results": sum(e["assistant_tool_results"] for e in episodes),
        "repeated_assistant_tool_results": sum(e["repeated_assistant_tool_results"] for e in episodes),
        "repeat_tool_output_character_share": (
            sum(e["repeated_tool_output_chars"] for e in episodes) / sum(e["tool_output_chars"] for e in episodes)
            if any(e["tool_output_chars"] for e in episodes) else None
        ),
    }


def profile(path, expected):
    traces = [json.loads(line) for line in path.open() if line.strip()]
    eligible = [r for r in traces if not r.get("error")]
    episodes = [episode(r) for r in eligible]
    keys = {tuple(e["key"]) for e in episodes}
    if expected is not None and (keys != expected or len(episodes) != len(expected)):
        raise ValueError(f"{path}: task identities differ from the requested baseline")
    grouped = defaultdict(list)
    for ep in episodes:
        grouped[ep["key"][0]].append(ep)
    return {
        "source": str(path), "unscored": len(traces) - len(eligible),
        "overall": summarize(episodes), "domains": {d: summarize(group) for d, group in sorted(grouped.items())},
        "termination_groups": {
            "step_limited": summarize([e for e in episodes if e["hit_step_limit"]]),
            "other": summarize([e for e in episodes if not e["hit_step_limit"]]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    expected = None if args.manifest is None else {tuple(t) for t in json.loads(args.manifest.read_text())["evaluation"]}
    report = profile(args.trace, expected)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
