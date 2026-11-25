"""Evaluate a fixed student, teacher, or upstream RoutedAgent on one frozen task set."""

import argparse
from collections import Counter
import json
from math import ceil
from pathlib import Path
import runpy
import sys
from threading import local
from time import perf_counter
from unittest.mock import patch


def collect(folder, arm, student_url, teacher_url, simulator_url, *, calibration=False):
    import tau2.evaluator.evaluator as evaluator
    from tau2.agent.llm_agent import LLMAgent
    from tau2_evolve import benchmark, collect_traces

    manifest = json.loads((folder / "manifest.json").read_text())
    protocol = manifest["protocol"]
    load = runpy.run_path(str(folder.parent / "balanced_v3_expanded" / "run_tasks.py"))
    tasks, _ = load["task_pool"]()
    if calibration:
        selected = load["pairs"](json.loads((folder / "calibration" / "tasks.json").read_text()))
        excluded = set(load["pairs"](manifest["evaluation"]))
    else:
        selected = load["pairs"](manifest["evaluation"])
        excluded = set(load["pairs"](manifest["training_partition"]))
    if len(selected) != len(set(selected)) or set(selected) & excluded:
        raise ValueError("The frozen task set includes duplicates or training tasks")
    for domain in tasks:
        benchmark._TASKS_CACHE[domain] = [
            tasks[domain][task_id] for label, task_id in selected if label == domain
        ]

    def task_ids(bucket, domain=None):
        return [item for item in selected if domain is None or item[0] == domain]

    context = local()
    make_llm_spec = benchmark.make_llm_spec
    run_task = benchmark.run_task
    native_score = evaluator.evaluate_simulation
    generate = LLMAgent.generate_next_message

    def sampling(model, api_base, *args, **kwargs):
        is_agent = model == "openai/evol-llm-agent"
        if is_agent:
            kwargs["temperature"] = protocol["agent_temperature"]
            kwargs["enable_thinking"] = protocol["agent_thinking"]
        else:
            kwargs["temperature"] = protocol["user_temperature"]
            kwargs["enable_thinking"] = protocol["user_thinking"]
        spec = make_llm_spec(model, api_base, *args, **kwargs)
        if is_agent:
            spec.args["top_p"] = protocol["agent_top_p"]
            if api_base == simulator_url:
                spec.args["extra_body"]["top_k"] = -1
        return spec

    def count_generation(agent, message, state):
        endpoint = agent.llm_args.get("api_base")
        if endpoint in {student_url, teacher_url}:
            context.calls["strong" if endpoint == teacher_url else "weak"] += 1
        return generate(agent, message, state)

    def check_score(*args, **kwargs):
        try:
            return native_score(*args, **kwargs)
        except Exception as exc:
            context.scorer_error = exc
            raise

    def timed_task(**kwargs):
        if bool(kwargs.get("router_model_path")) != (arm == "routed"):
            raise ValueError(f"Routing configuration did not match the requested arm {arm}")
        if kwargs.get("skill_text"):
            raise ValueError("This comparison evaluates the same official prompt in every arm")
        context.calls = Counter()
        context.scorer_error = None
        started = perf_counter()
        row = run_task(**kwargs)
        if context.scorer_error is not None:
            raise RuntimeError(f"Official scorer failed: {context.scorer_error}")
        row["duration_seconds"] = perf_counter() - started
        row["strong_calls"] = context.calls["strong"]
        row["weak_calls"] = context.calls["weak"]
        row["arm"] = arm
        return row

    url = teacher_url if arm == "27b" else student_url
    output = (folder / "calibration" if calibration else folder) / f"{arm}.jsonl"
    options = [
        "--bucket", "EVAL", "--seed", str(manifest["seed"]),
        "--workers", str(protocol["workers_per_arm"]),
        "--max-steps", str(protocol["max_steps"]), "--max-errors", str(protocol["max_errors"]),
        "--agent-model", "openai/evol-llm-agent", "--agent-base-url", url,
        "--user-model", "openai/evol-llm-user", "--user-base-url", simulator_url,
        "--agent-max-tokens", str(protocol["agent_max_tokens"]),
        "--user-max-tokens", str(protocol["user_max_tokens"]),
        "--tau2-log-level", "CRITICAL", "--resume", "--out", str(output),
    ]
    if arm == "routed":
        options.extend([
            "--router-model", str(folder / "router.json"), "--router-teacher-model", "openai/evol-llm-agent",
            "--router-teacher-base-url", teacher_url, "--router-teacher-max-tokens", str(protocol["agent_max_tokens"]),
        ])
    sys.argv = [sys.argv[0], *options]
    with patch.object(benchmark, "task_ids_for", task_ids), \
            patch.object(benchmark, "make_llm_spec", sampling), \
            patch.object(benchmark, "run_task", timed_task), \
            patch.object(evaluator, "evaluate_simulation", check_score), \
            patch.object(LLMAgent, "generate_next_message", count_generation):
        return collect_traces.main()


def read_arm(folder, arm, expected):
    rows = [json.loads(line) for line in (folder / f"{arm}.jsonl").open() if line.strip()]
    indexed = {(r["domain"], r["task_id"]): r for r in rows}
    if len(rows) != len(expected) or set(indexed) != expected:
        raise ValueError(f"{arm}: incomplete task coverage or duplicate keys")
    for key, row in indexed.items():
        if row.get("error") or row["arm"] != arm:
            raise ValueError(f"{arm}: unscored task or incorrect arm {key}")
        if arm == "4b" and row["strong_calls"] or arm == "27b" and row["weak_calls"]:
            raise ValueError(f"{arm}: request was sent to the wrong model")
        if arm == "routed" and bool(row["strong_calls"]) != bool(row["router_escalations"]):
            raise ValueError(f"{arm}: escalation and actual calls differ for {key}")
    return indexed


def statistics(rows):
    seconds = sorted(r["duration_seconds"] for r in rows)
    strong = sum(r["strong_calls"] for r in rows)
    total = strong + sum(r["weak_calls"] for r in rows)
    passed = sum(r["reward"] == 1.0 for r in rows)
    escalated = sum(r["strong_calls"] > 0 for r in rows)
    return {
        "passed": passed, "tasks": len(rows), "pass_rate": passed / len(rows),
        "strong_calls": strong, "customer_model_calls": total, "strong_call_share": strong / total,
        "strong_tasks": escalated, "strong_task_share": escalated / len(rows),
        "p95_seconds": seconds[ceil(0.95 * len(seconds)) - 1],
    }


def audit(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    expected = {tuple(x) for x in manifest["evaluation"]}
    arms = ("4b", "27b", "routed") if (folder / "routed.jsonl").exists() else ("4b", "27b")
    runs = {arm: read_arm(folder, arm, expected) for arm in arms}
    summary = {}
    for arm, run in runs.items():
        rows = list(run.values())
        domains = {d: statistics([r for r in rows if r["domain"] == d])
                   for d in sorted({d for d, _ in expected})}
        summary[arm] = {**statistics(rows), "domains": domains,
                        "macro_pass_rate": sum(v["pass_rate"] for v in domains.values()) / len(domains)}
    paired = {arm: {
        "wins": sum(r["reward"] == 1.0 and runs["4b"][k]["reward"] != 1.0 for k, r in runs[arm].items()),
        "losses": sum(r["reward"] != 1.0 and runs["4b"][k]["reward"] == 1.0 for k, r in runs[arm].items()),
    } for arm in arms if arm != "4b"}
    result = {"arms": summary, "paired_vs_4b": paired,
              "teacher_improved": summary["27b"]["passed"] > summary["4b"]["passed"]}
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("arm", choices=["4b", "27b", "routed", "audit"])
    parser.add_argument("--student-url")
    parser.add_argument("--teacher-url", default="http://127.0.0.1:28863/v1")
    parser.add_argument("--simulator-url", default="http://127.0.0.1:28862/v1")
    parser.add_argument("--calibration", action="store_true")
    args = parser.parse_args()
    if args.calibration and args.arm != "4b":
        parser.error("Calibration collection uses the fixed 4B student")
    if args.arm == "audit":
        audit(args.folder)
        return 0
    placement = args.folder / "student_endpoint.json"
    endpoint = json.loads(placement.read_text())["api_base"] if placement.exists() else "http://127.0.0.1:28864/v1"
    return collect(args.folder, args.arm, args.student_url or endpoint, args.teacher_url, args.simulator_url,
                   calibration=args.calibration)


if __name__ == "__main__":
    raise SystemExit(main())
