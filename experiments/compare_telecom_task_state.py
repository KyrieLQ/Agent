"""Run a paired Telecom experiment from a frozen development manifest."""

import argparse
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
from statistics import mean
import sys
from threading import local
from time import perf_counter
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from telecom_state import tool_results, with_telecom_task_state


def collect(folder, manifest, arm, options):
    import tau2.evaluator.evaluator as evaluator
    from tau2_evolve import benchmark, collect_traces

    tasks = json.loads((folder / "dev_tasks.json").read_text())
    selected = manifest["development"]
    if len(tasks) != len(selected) or {row["id"] for row in tasks} != set(selected):
        raise ValueError("The development catalog differs from the frozen task list")
    manual = (folder / "telecom_playbook.txt").read_text().strip()
    if not manual:
        raise ValueError("The Telecom playbook is empty")
    benchmark._TASKS_CACHE["telecom"] = tasks
    benchmark.task_ids_for = lambda bucket, domain=None: [("telecom", task) for task in selected]
    make_spec = benchmark.make_llm_spec
    run_task = benchmark.run_task
    native_score = evaluator.evaluate_simulation
    scoring = local()

    def sampling(model, api_base, *args, **kwargs):
        is_agent = model == "openai/evol-llm-agent"
        if is_agent:
            kwargs["temperature"] = manifest["protocol"]["agent_temperature"]
        spec = make_spec(model, api_base, *args, **kwargs)
        if is_agent:
            spec.args["top_p"] = manifest["protocol"]["agent_top_p"]
        return spec

    def score_simulation(*args, **kwargs):
        try:
            return native_score(*args, **kwargs)
        except Exception as exc:
            scoring.error = exc
            raise

    def call_task(**kwargs):
        if kwargs.get("router_model_path") or kwargs.get("skill_text"):
            raise ValueError("Use the frozen experiment playbook with routing disabled")
        kwargs["skill_text"] = manual
        scoring.error = None
        start = perf_counter()
        row = run_task(**kwargs)
        row["duration_seconds"] = perf_counter() - start
        if scoring.error:
            raise RuntimeError(f"Official scorer failed: {scoring.error}") from scoring.error
        if arm == "quota":
            row["quota_steps"] = tracker.agent.quota_steps
            if manifest.get("quota_context_neutral"):
                row["quota_reads"] = tracker.agent.quota_reads
        elif tracker is not None:
            row["structured_task_state_turns"] = tracker.agent.state_turns
            if arm == "workflow":
                row["workflow_steps"] = tracker.agent.workflow_steps
                row["workflow_corrections"] = tracker.agent.workflow_corrections
        # Validate the recorded context before treating this trial as scored.
        if manual not in row["system_prompt"]:
            raise RuntimeError("The frozen Telecom playbook was absent from the agent prompt")
        if (row.get("structured_task_state_turns", 0) > 0) != (arm in {"state", "workflow"}):
            raise RuntimeError("Telecom task-state injection did not match the requested arm")
        if arm == "workflow" and len(row["workflow_steps"]) != row["structured_task_state_turns"]:
            raise RuntimeError("Telecom workflow guidance was missing from a state turn")
        row["arm"] = arm
        return row

    sys.argv = [sys.argv[0], *options]
    state = (with_telecom_task_state(workflow=arm == "workflow", quota=arm == "quota",
                                     quota_probe=manifest.get("quota_probe", False),
                                     quota_auto_read=manifest.get("quota_auto_read", False),
                                     quota_context_neutral=manifest.get("quota_context_neutral", False),
                                     quota_post_mms=manifest.get("quota_post_mms", False),
                                     quota_mms_guide=manifest.get("quota_mms_guide", False),
                                     quota_auto_price=manifest.get("quota_auto_price", False),
                                     quota_data_only=manifest.get("quota_data_only", False),
                                     quota_early_mms=manifest.get("quota_early_mms", False))
             if arm in {"state", "workflow", "quota"} else nullcontext())
    with state as tracker, patch.object(benchmark, "make_llm_spec", sampling), \
            patch.object(benchmark, "run_task", call_task), \
            patch.object(evaluator, "evaluate_simulation", score_simulation):
        return collect_traces.main()


def statistics(row):
    successful = set()
    signatures = set()
    repeat_count = 0
    assistant_results = 0
    for event in tool_results(row["messages"]):
        if not event.failed:
            successful.add(event.name)
        if event.caller == "assistant":
            assistant_results += 1
            signature = (event.key, event.content, event.failed)
            repeat_count += signature in signatures
            signatures.add(signature)
    missing = sorted(set(row["expected_tool_calls"]) - successful)
    return {"passed": row["reward"] == 1.0, "missing": missing, "repeats": repeat_count,
            "assistant_results": assistant_results, "max_steps": str(row["termination_reason"]).endswith("MAX_STEPS")}


def read_trials(path, expected, arm, manual):
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows = {row["task_id"]: row for row in records}
    if len(records) != len(expected) or set(rows) != expected:
        raise ValueError(f"Missing or duplicate task IDs in {path}")
    for row in rows.values():
        if row.get("error"):
            continue
        if row["arm"] != arm or manual not in row["system_prompt"]:
            raise ValueError(f"Unexpected prompt for {row['task_id']}")
        if (row.get("structured_task_state_turns", 0) > 0) != (arm in {"state", "workflow"}):
            raise ValueError(f"Unexpected task-state injection for {row['task_id']}")
        if bool(row.get("workflow_steps")) != (arm == "workflow") or (
            arm == "workflow" and len(row["workflow_steps"]) != row["structured_task_state_turns"]
        ):
            raise ValueError(f"Unexpected workflow guidance for {row['task_id']}")
        if ("quota_steps" in row) != (arm == "quota"):
            raise ValueError(f"Unexpected quota instrumentation for {row['task_id']}")
    return rows


def totals(rows):
    return {
        "trials": len(rows), "passed": sum(r["passed"] for r in rows),
        "with_missing_steps": sum(bool(r["missing"]) for r in rows),
        "missing_action_types": sum(len(r["missing"]) for r in rows),
        "repeated_calls": sum(r["repeats"] for r in rows),
        "assistant_tool_results": sum(r["assistant_results"] for r in rows),
        "max_steps": sum(r["max_steps"] for r in rows),
    }


def compare(pairs):
    return {"static": totals([pair[0] for pair in pairs]), "state": totals([pair[1] for pair in pairs]),
            "wins": sum(right["passed"] and not left["passed"] for left, right in pairs),
            "losses": sum(left["passed"] and not right["passed"] for left, right in pairs)}


def audit(folder, manifest):
    if "quota" in manifest.get("arms", ()):
        return audit_quota(folder, manifest)
    if "workflow" in manifest.get("arms", ()):
        return audit_workflow(folder, manifest)
    keys = set(manifest["development"])
    manual = (folder / "telecom_playbook.txt").read_text().strip()
    pairs, by_seed, details, unscored = [], {}, [], []
    for seed in manifest["development_seeds"]:
        static, state = (read_trials(folder / f"{arm}_{seed}.jsonl", keys, arm, manual)
                         for arm in ("static", "state"))
        seed_pairs = []
        for key in manifest["development"]:
            if static[key].get("error") or state[key].get("error"):
                unscored.append({"seed": seed, "task_id": key})
                continue
            trial = statistics(static[key]), statistics(state[key])
            seed_pairs.append(trial)
            details.append({"seed": seed, "task_id": key, "static": trial[0], "state": trial[1]})
        pairs.extend(seed_pairs)
        by_seed[seed] = compare(seed_pairs)
    summary = {"by_seed": by_seed, "pooled": compare(pairs), "unscored": unscored,
               "definitions": {
                   "missing": "unique benchmark reference tool names without a successful result, across both roles",
                   "repeats": "assistant tool responses with identical call name, arguments, response and error status already seen in the episode",
               }}
    (folder / "paired_details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


def workflow_diagnostics(row):
    from telecom_workflow import CLOSE_CLAIM, decide_workflow, workflow_correction

    messages = row["messages"]
    seen = set()
    for i, message in enumerate(messages):
        if message.get("role") == "assistant":
            if not message.get("tool_calls") and not message.get("content"):
                continue
            # Only run the diagnostic at the decisions that can affect it.
            if any(call.get("name") == "transfer_to_human_agents" for call in (message.get("tool_calls") or ())) \
                    or CLOSE_CLAIM.search(message.get("content") or ""):
                correction = workflow_correction(decide_workflow(messages[:i]), message)
                if correction:
                    seen.add(correction[0])
    result = statistics(row)
    result.update({"premature_transfer": "premature_transfer" in seen,
                   "unverified_resolution": "unverified_resolution" in seen,
                   "repair_without_retest": decide_workflow(messages).stage == "verify",
                   "corrections": row.get("workflow_corrections", []),
                   "duration_seconds": row.get("duration_seconds")})
    return result


def audit_workflow(folder, manifest):
    tasks = manifest["development"]
    seeds = manifest["development_seeds"]
    arms = manifest["arms"]
    manual = (folder / "telecom_playbook.txt").read_text().strip()
    details = []
    for seed in seeds:
        runs = {arm: read_trials(folder / f"{arm}_{seed}.jsonl", set(tasks), arm, manual) for arm in arms}
        for task in tasks:
            if any(runs[arm][task].get("error") for arm in arms):
                raise ValueError(f"Unscored trial on seed {seed}: {task}")
            details.append({"seed": seed, "task_id": task, **{
                arm: workflow_diagnostics(runs[arm][task]) for arm in arms
            }})

    def summarize(group):
        results = {}
        for arm in arms:
            records = [row[arm] for row in group]
            timing = sorted(r["duration_seconds"] for r in records if r["duration_seconds"] is not None)
            counts = Counter(k for r in records for k in r["corrections"])
            results[arm] = {**totals(records), **{
                key: sum(r[key] for r in records)
                for key in ("premature_transfer", "unverified_resolution", "repair_without_retest")
            }, "mean_seconds": round(mean(timing), 2) if timing else None,
                "p95_seconds": round(timing[(95 * len(timing) + 99) // 100 - 1], 2) if timing else None,
                "workflow_corrections": dict(counts)}
        comparisons = {
            baseline: {
                "wins": sum(row["workflow"]["passed"] and not row[baseline]["passed"] for row in group),
                "losses": sum(row[baseline]["passed"] and not row["workflow"]["passed"] for row in group),
            } for baseline in ("static", "state")
        }
        return {"arms": results, "workflow_vs": comparisons}

    passed_by_task = {arm: Counter(row["task_id"] for row in details if row[arm]["passed"]) for arm in arms}
    categories = sorted({task.split("]")[0][1:] for task in tasks})
    summary = {"by_seed": {seed: summarize([row for row in details if row["seed"] == seed]) for seed in seeds},
               "pooled": summarize(details),
               "by_issue": {kind: summarize([row for row in details if row["task_id"].startswith(f"[{kind}]")])
                            for kind in categories},
               "pass_all_seeds": {arm: {"passed": sum(v == len(seeds) for v in scores.values()), "tasks": len(tasks)}
                                  for arm, scores in passed_by_task.items()},
               "definitions": {"premature_transfer": "episode attempted a transfer during a clear in-scope repair or retest",
                               "repair_without_retest": "episode ended with a recognized repair pending a symptom retest",
                               "unverified_resolution": "episode claimed resolution while a symptom retest was pending",
                               "reference_actions": "diagnostic tool-name coverage, not part of the official success score"}}
    (folder / "paired_details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


def audit_quota(folder, manifest):
    tasks = manifest["development"]
    seeds = manifest["development_seeds"]
    arms = manifest["arms"]
    manual = (folder / "telecom_playbook.txt").read_text().strip()
    details = []
    for seed in seeds:
        runs = {arm: read_trials(folder / f"{arm}_{seed}.jsonl", set(tasks), arm, manual) for arm in arms}
        for task in tasks:
            rows = {arm: runs[arm][task] for arm in arms}
            if any(row.get("error") for row in rows.values()):
                raise ValueError(f"Unscored trial for seed {seed}: {task}")
            required = {arm: "refuel_data" in row["expected_tool_calls"] for arm, row in rows.items()}
            if len(set(required.values())) != 1:
                raise ValueError(f"Different task definitions for {task}")
            record = {"seed": seed, "task_id": task, "reference_refuel": required["static"]}
            for arm, row in rows.items():
                calls = [e.name for e in tool_results(row["messages"]) if e.caller == "assistant" and not e.failed]
                inspections = row.get("quota_reads", ())
                audited_tools = {"get_data_usage"}
                if manifest.get("quota_auto_price"):
                    audited_tools.add("get_details_by_id")
                tools = [entry["name"] for entry in inspections]
                if len(tools) != len(set(tools)) or set(tools) - audited_tools:
                    raise ValueError(f"Invalid supplemental usage lookup for {task}")
                record[arm] = {
                    **statistics(row), "checked_usage": ("get_data_usage" in calls or
                                                        any(entry["name"] == "get_data_usage" and
                                                            not entry["failed"] for entry in inspections)),
                    "refueled": "refuel_data" in calls, "refuel_calls": calls.count("refuel_data"),
                    "duration_seconds": row.get("duration_seconds"),
                    "steps": row.get("quota_steps", []),
                }
            details.append(record)

    def summarize(group):
        results = {}
        for arm in arms:
            records = [row[arm] for row in group]
            timing = [r["duration_seconds"] for r in records if r["duration_seconds"] is not None]
            results[arm] = {**totals(records), **{
                key: sum(r[key] for r in records) for key in ("checked_usage", "refueled", "refuel_calls")
            }, "triggered": sum(bool(r["steps"]) for r in records),
                "step_counts": dict(Counter(s for r in records for s in r["steps"])),
                "mean_seconds": round(mean(timing), 2) if timing else None}
        return {
            "arms": results,
            "wins": sum(row["quota"]["passed"] and not row["static"]["passed"] for row in group),
            "losses": sum(row["static"]["passed"] and not row["quota"]["passed"] for row in group),
        }

    summary = {
        "cohort": manifest.get("cohort"),
        "pooled": summarize(details),
        "reference_refuel": summarize([r for r in details if r["reference_refuel"]]),
        "other": summarize([r for r in details if not r["reference_refuel"]]),
        "by_seed": {seed: summarize([r for r in details if r["seed"] == seed]) for seed in seeds},
        "pass_all_seeds": {arm: {
            "passed": sum(n == len(seeds) for n in Counter(r["task_id"] for r in details if r[arm]["passed"]).values()),
            "tasks": len(tasks),
        } for arm in arms},
    }
    (folder / "paired_details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("mode", choices=("static", "state", "workflow", "quota", "audit"))
    args, options = parser.parse_known_args()
    manifest = json.loads((args.folder / "manifest.json").read_text())
    if args.mode == "audit":
        return audit(args.folder, manifest)
    return collect(args.folder, manifest, args.mode, options)


if __name__ == "__main__":
    raise SystemExit(main())
