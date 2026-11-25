"""Calibrate the upstream task router on separate three-domain continuations."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def key(row):
    return row["domain"], str(row["task_id"])


def digest(seed, task):
    return hashlib.sha256((str(seed) + "\t" + "\t".join(task)).encode()).digest()


def fit_router(examples, seed):
    from tau2_evolve.router import RouterModel

    tasks = {label: {key(row) for row in examples if row["label"] == label} for label in (0, 1)}
    if not min(map(len, tasks.values())) or tasks[0] & tasks[1]:
        raise ValueError("Router calibration needs successful student and verified teacher-rescue tasks")
    if min(map(len, tasks.values())) < 4:
        return RouterModel.fit(examples), {"threshold_selection": "upstream default", "validation_tasks": 0}

    held = {task for group in tasks.values()
            for task in sorted(group, key=lambda item: digest(seed, item))[:max(1, len(group) // 4)]}
    train = [row for row in examples if key(row) not in held]
    validation = [row for row in examples if key(row) in held]
    router = RouterModel.fit(train)
    scored = [(router.probability(row["features"]), row["label"]) for row in validation]
    scores = sorted(set(score for score, _ in scored))
    thresholds = [scores[0] / 2, *[(a + b) / 2 for a, b in zip(scores, scores[1:])], (1 + scores[-1]) / 2]

    def metrics(threshold):
        result = [sum((score >= threshold) == bool(label) for score, y in scored if y == label)
                  / sum(y == label for _, y in scored) for label in (0, 1)]
        return sum(result) / 2, result[0]

    router.threshold = max(thresholds, key=metrics)
    accuracy, specificity = metrics(router.threshold)
    return router, {"threshold_selection": "task-disjoint balanced accuracy, then specificity",
                    "validation_tasks": len(held), "validation_balanced_accuracy": accuracy,
                    "validation_specificity": specificity}


def run(folder, teacher_url, simulator_url):
    from pydantic import TypeAdapter
    from tau2.agent.base_agent import is_valid_agent_history_message
    from tau2.data_model.message import Message
    from tau2_evolve import benchmark, collect_opd
    from tau2_evolve.router import router_features

    if not read(folder / "summary.json")["teacher_improved"]:
        raise ValueError("The fixed teacher did not improve over the fixed student")
    calibration = folder / "calibration"
    manifest = read(folder / "manifest.json")
    chosen = {tuple(task) for task in read(calibration / "tasks.json")}
    evaluation = {tuple(task) for task in manifest["evaluation"]}
    if not chosen or chosen & evaluation:
        raise ValueError("Calibration overlaps evaluation")
    traces = rows(calibration / "4b.jsonl")
    if len(traces) != len(chosen) or {key(row) for row in traces} != chosen or any(row.get("error") for row in traces):
        raise ValueError("Calibration traces have missing, duplicate, or unscored tasks")

    load = runpy.run_path(str(folder.parent / "balanced_v3_expanded" / "run_tasks.py"))
    catalog, _ = load["task_pool"]()
    for domain in catalog:
        benchmark._TASKS_CACHE[domain] = [
            catalog[domain][task_id] for label, task_id in sorted(chosen) if label == domain
        ]

    protocol = manifest["protocol"]
    make_spec = benchmark.make_llm_spec
    make_example = collect_opd.router_example
    parse_messages = TypeAdapter(list[Message])

    def sampling(model, api_base, *args, **kwargs):
        is_teacher = model == "openai/evol-llm-agent"
        kwargs["temperature"] = protocol["agent_temperature" if is_teacher else "user_temperature"]
        kwargs["max_tokens"] = protocol["agent_max_tokens" if is_teacher else "user_max_tokens"]
        kwargs["enable_thinking"] = False
        spec = make_spec(model, api_base, *args, **kwargs)
        if is_teacher:
            spec.args["top_p"] = protocol["agent_top_p"]
            if api_base == simulator_url:
                spec.args["extra_body"]["top_k"] = -1
        return spec

    def feature_example(row, prefix, label, source):
        visible = [message for message in parse_messages.validate_python(prefix)
                   if is_valid_agent_history_message(message)]
        example = make_example(row, visible, label, source)
        example["features"] = router_features(visible, max_steps=protocol["max_steps"])
        return example

    examples_path = calibration / "router_examples.jsonl"
    sys.argv = [
        sys.argv[0], "--traces", str(calibration / "4b.jsonl"),
        "--output", str(calibration / "teacher_rescues.jsonl"), "--router-data", str(examples_path),
        "--teacher-model", "openai/evol-llm-agent", "--teacher-base-url", teacher_url,
        "--user-base-url", simulator_url, "--max-steps", str(protocol["max_steps"]),
        "--workers", str(protocol["workers_per_arm"]), "--seed", str(manifest["seed"]),
        "--no-teacher-thinking",
    ]
    with patch.object(benchmark, "make_llm_spec", sampling), \
            patch.object(collect_opd, "router_example", feature_example):
        if collect_opd.main():
            raise RuntimeError("Upstream continuation collection failed")

    examples = rows(examples_path)
    if not {key(row) for row in examples} <= chosen:
        raise ValueError("The router examples extend beyond the frozen calibration tasks")
    router, stats = fit_router(examples, manifest["seed"])
    model = folder / "router.json"
    router.save(model.with_suffix(".pending"))
    model.with_suffix(".pending").replace(model)
    metrics = {
        "tasks": len(chosen), "teacher_rescue_examples": sum(r["label"] == 1 for r in examples),
        "student_continue_examples": sum(r["label"] == 0 for r in examples),
        "threshold": router.threshold, **stats,
    }
    (calibration / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--teacher-url", default="http://127.0.0.1:28863/v1")
    parser.add_argument("--simulator-url", default="http://127.0.0.1:28862/v1")
    args = parser.parse_args()
    run(args.folder, args.teacher_url, args.simulator_url)


if __name__ == "__main__":
    main()
