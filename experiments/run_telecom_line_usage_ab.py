"""Run complete Telecom dialogues with native simulation and one read-only treatment."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
from statistics import median
import sys
from threading import local
from time import perf_counter
from unittest.mock import patch

from telecom_line_usage import enrich_line_usage
from telecom_state import TelecomAgentProxy, tool_results


ARMS = ("baseline", "line_usage")
SALT = "telecom-line-usage-ab-20261124"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    stage = path.with_suffix(".json.new")
    stage.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stage.replace(path)


def strings(value):
    if isinstance(value, dict):
        yield from value
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


def needs_refuel(task):
    return any(action["name"] == "refuel_data" for action in task["evaluation_criteria"]["actions"])


def freeze(folder, project):
    results = project / "results"
    source = project / "upstream-tau2/data/tau2/domains/telecom/tasks_full.json"
    original = results / "telecom_langgraph_20261124/telecom_tasks.json"
    experiment = results / "telecom_quota_20261123/dev_v4b"
    config = read_json(experiment / "manifest.json")
    pool = {task["id"]: task for task in read_json(source)}
    exclusions = set(read_json(original))
    for pattern in ("telecom*/**/dev_tasks.json", "telecom*/**/train_tasks.json", "telecom*/**/teacher_schedule.json",
                    "telecom*/**/probe_tasks.json", "telecom*/**/heldout_tasks.json", "telecom*/grpo_task_counts.json"):
        for path in sorted(results.glob(pattern)):
            exclusions.update(pool.keys() & set(strings(read_json(path))))

    groups = (("mobile_data_issue", True, 3), ("mms_issue", True, 3),
              ("mobile_data_issue", False, 3), ("mms_issue", False, 3))

    def rank(task):
        return sha256(f"{SALT}:{task['id']}".encode()).digest()

    selected = []
    for scenario, refuel, count in groups:
        matches = [task for task in pool.values() if task["id"] not in exclusions
                   and task["id"].startswith(f"[{scenario}]") and needs_refuel(task) == refuel]
        assert len(matches) >= count, (scenario, refuel, len(matches))
        selected.extend(sorted(matches, key=rank)[:count])
    selected.sort(key=rank)
    task_ids = [task["id"] for task in selected]
    assert len(set(task_ids)) == 12 and not exclusions.intersection(task_ids)
    playbook = (experiment / "telecom_playbook.txt").read_bytes()
    manifest = {"source": str(source), "source_sha256": sha256(source.read_bytes()).hexdigest(),
                "selection_salt": SALT, "seed": config["development_seeds"][0], "protocol": config["protocol"],
                "models": {key: config[key] for key in ("agent", "simulator")}, "task_ids": task_ids,
                "playbook_sha256": sha256(playbook).hexdigest(),
                "excluded_matching_ids": len(pool.keys() & exclusions),
                "procedure": "Native tau2 customer, confirmation dialogue, tool execution, and official scoring in both arms"}
    (folder / "telecom_playbook.txt").write_bytes(playbook)
    write_json(folder / "dev_tasks.json", selected)
    write_json(folder / "manifest.json", manifest)
    print(f"Frozen {len(task_ids)} independent development tasks and seed {manifest['seed']}", flush=True)


class EnrichedAgent(TelecomAgentProxy):
    def __init__(self, agent, retrieve):
        super().__init__(agent)
        self.retrieve = retrieve
        self.quota_reads = []

    def read_usage(self, call):
        start = perf_counter()
        result = self.retrieve(call)
        self.quota_reads.append({"arguments": call.arguments, "ok": not result.error,
                                 "duration_seconds": perf_counter() - start})
        return result

    def generate_next_message(self, message, state):
        from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolMessage

        if not isinstance(message, (ToolMessage, MultiToolMessage)):
            return self.agent.generate_next_message(message, state)
        results = list(message.tool_messages) if isinstance(message, MultiToolMessage) else [message]
        calls = {call.id: call for past in state.messages if isinstance(past, AssistantMessage)
                 for call in past.tool_calls or ()}
        for index, result in enumerate(results):
            if result.requestor == "assistant" and (call := calls.get(result.id)) is not None:
                results[index] = enrich_line_usage(state.messages + results, call, result, self.read_usage)
        incoming = message.model_copy(update={"tool_messages": results}) if isinstance(message, MultiToolMessage) else results[0]
        return self.agent.generate_next_message(incoming, state)


def tally(row):
    tools = [event for event in tool_results(row["messages"]) if event.caller == "assistant"]
    return {"passed": row["reward"] == 1.0, "seconds": row["duration_seconds"],
            "steps": row["n_steps"], "automatic_usage_reads": len(row["quota_reads"]),
            "model_usage_reads": sum(event.name == "get_data_usage" for event in tools),
            "successful_refuels": sum(event.name == "refuel_data" and not event.failed for event in tools),
            "ran_out_of_steps": row["termination_reason"].endswith("MAX_STEPS")}


def summary(rows, tasks, manifest):
    groups = {"all": tasks, "refuel_required": [task for task in tasks if needs_refuel(task)],
              "without_refuel": [task for task in tasks if not needs_refuel(task)]}
    totals = {}
    for name, members in groups.items():
        ids = [task["id"] for task in members]
        paired = {task_id: {arm: tally(rows[(task_id, arm)]) for arm in ARMS} for task_id in ids}
        arms = {}
        for arm in ARMS:
            arm_rows = [row[arm] for row in paired.values()]
            durations = sorted(row["seconds"] for row in arm_rows)
            arms[arm] = {key: sum(row[key] for row in arm_rows) for key in (
                "passed", "automatic_usage_reads", "model_usage_reads", "successful_refuels", "ran_out_of_steps")}
            arms[arm].update({"median_seconds": round(median(durations), 2),
                              "p95_seconds": round(durations[ceil(0.95 * len(durations)) - 1], 2),
                              "median_steps": median(row["steps"] for row in arm_rows)})
        totals[name] = {"tasks": len(ids), "arms": arms,
                        "pair_outcomes": dict(Counter(f"{int(row['baseline']['passed'])}->{int(row['line_usage']['passed'])}"
                                                      for row in paired.values()))}
    return {"protocol": manifest["protocol"], "seed": manifest["seed"], "results": totals,
            "tasks": {task["id"]: {arm: tally(rows[(task["id"], arm)]) for arm in ARMS} for task in tasks}}


def run(folder, agent_url, user_url):
    import tau2.evaluator.evaluator as evaluator
    import tau2.runner
    from loguru import logger
    from tau2_evolve import benchmark

    logger.remove()
    logger.add(sys.stderr, level="CRITICAL")
    manifest = read_json(folder / "manifest.json")
    protocol = manifest["protocol"]
    tasks = read_json(folder / "dev_tasks.json")
    assert [task["id"] for task in tasks] == manifest["task_ids"]
    manual = (folder / "telecom_playbook.txt").read_bytes()
    assert sha256(manual).hexdigest() == manifest["playbook_sha256"]
    manual = manual.decode().strip()
    benchmark._TASKS_CACHE["telecom"] = tasks
    agent = benchmark.make_llm_spec("openai/evol-llm-agent", agent_url, temperature=protocol["agent_temperature"],
                                    enable_thinking=protocol["agent_thinking"], max_tokens=protocol["agent_max_tokens"])
    agent.args["top_p"] = protocol["agent_top_p"]
    user = benchmark.make_llm_spec("openai/evol-llm-user", user_url, temperature=protocol["user_temperature"],
                                   enable_thinking=protocol["user_thinking"], max_tokens=protocol["user_max_tokens"])
    build = tau2.runner.build_text_orchestrator
    official_score = evaluator.evaluate_simulation
    active = local()

    def wrapped_build(*args, **kwargs):
        orch = build(*args, **kwargs)
        if active.arm == "line_usage":
            orch.agent = EnrichedAgent(orch.agent, orch.environment.get_response)
            active.enriched = orch.agent
        return orch

    def strict_score(*args, **kwargs):
        try:
            return official_score(*args, **kwargs)
        except Exception as exc:
            active.scoring_error = exc
            raise

    def evaluate(task_id, arm):
        active.arm, active.enriched, active.scoring_error = arm, None, None
        start = perf_counter()
        result = benchmark.run_task(domain="telecom", task_id=task_id, agent_spec=agent, user_spec=user,
                                    seed=manifest["seed"], max_steps=protocol["max_steps"],
                                    max_errors=protocol["max_errors"], skill_text=manual)
        if active.scoring_error is not None:
            raise RuntimeError(f"Official scoring failed for {task_id}") from active.scoring_error
        assert (active.enriched is not None) == (arm == "line_usage")
        assert manual in result["system_prompt"]
        return {**result, "arm": arm, "quota_reads": active.enriched.quota_reads if active.enriched else [],
                "duration_seconds": perf_counter() - start}

    path = folder / "dialogues.jsonl"
    rows = {}
    if path.exists():
        with path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                key = row["task_id"], row["arm"]
                assert key not in rows and key[0] in manifest["task_ids"] and key[1] in ARMS
                rows[key] = row
    schedule = []
    for index, task_id in enumerate(manifest["task_ids"]):
        arms = ARMS if index % 2 == 0 else ARMS[::-1]
        schedule.extend((task_id, arm) for arm in arms if (task_id, arm) not in rows)
    with path.open("a", encoding="utf-8") as output, patch.object(tau2.runner, "build_text_orchestrator", wrapped_build), \
            patch.object(evaluator, "evaluate_simulation", strict_score), \
            ThreadPoolExecutor(max_workers=protocol["workers_per_arm"]) as workers:
        pending = {workers.submit(evaluate, *key): key for key in schedule}
        for future in as_completed(pending):
            key = pending[future]
            row = future.result()
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            rows[key] = row
            print(f"Completed {len(rows)}/{len(tasks) * 2} dialogues ({key[1]}, reward={row['reward']})", flush=True)

    assert all(rows[(tid, "baseline")]["system_prompt"] == rows[(tid, "line_usage")]["system_prompt"]
               and rows[(tid, "baseline")]["tools"] == rows[(tid, "line_usage")]["tools"]
               for tid in manifest["task_ids"])
    report = summary(rows, tasks, manifest)
    write_json(folder / "summary.json", report)
    print(json.dumps(report["results"], ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--agent-url")
    parser.add_argument("--user-url")
    args = parser.parse_args()
    if args.prepare:
        freeze(args.folder, args.prepare)
    else:
        if not (args.agent_url and args.user_url):
            parser.error("Both inference URLs are required")
        run(args.folder, args.agent_url, args.user_url)


if __name__ == "__main__":
    main()
