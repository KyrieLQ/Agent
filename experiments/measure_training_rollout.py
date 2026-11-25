"""Sample full tau2 trajectories using the training stack's native veRL workers."""

import argparse
import asyncio
from copy import deepcopy
import json
import math
from pathlib import Path
from statistics import median
import time

from hydra import compose, initialize_config_dir
import ray
import verl
from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.trainer.main_ppo import create_rl_dataset
from verl.utils import hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.workers.rollout.llm_server import LLMServerManager

from tau2_evolve.agent_loop import Tau2AgentLoop


class TimedTau2AgentLoop(Tau2AgentLoop):
    async def run(self, sampling_params, **kwargs):
        started = time.perf_counter()
        output = await super().run(sampling_params, **kwargs)
        output.extra_fields["trajectory_seconds"] = time.perf_counter() - started
        return output


def load_config(args):
    loop_config = args.out / "agent_loop.yaml"
    loop_config.write_text(
        "- name: tau2_timed_agent\n  _target_: measure_training_rollout.TimedTau2AgentLoop\n",
        encoding="utf-8",
    )
    config_dir = Path(verl.__file__).parent / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="ppo_trainer", overrides=[
            f"actor_rollout_ref.model.path={args.agent}",
            "actor_rollout_ref.model.trust_remote_code=True",
            "actor_rollout_ref.hybrid_engine=False",
            "actor_rollout_ref.rollout.name=vllm",
            "actor_rollout_ref.rollout.n_gpus_per_node=1",
            "actor_rollout_ref.rollout.nnodes=1",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "actor_rollout_ref.rollout.dtype=bfloat16",
            "actor_rollout_ref.rollout.load_format=safetensors",
            "actor_rollout_ref.rollout.enforce_eager=True",
            "actor_rollout_ref.rollout.free_cache_engine=False",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.70",
            "actor_rollout_ref.rollout.calculate_log_probs=True",
            "actor_rollout_ref.rollout.temperature=1.0",
            "actor_rollout_ref.rollout.top_p=0.98",
            "actor_rollout_ref.rollout.max_model_len=14336",
            "actor_rollout_ref.rollout.multi_turn.enable=True",
            "actor_rollout_ref.rollout.multi_turn.max_user_turns=24",
            "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=24",
            "actor_rollout_ref.rollout.multi_turn.format=hermes",
            "actor_rollout_ref.rollout.agent.num_workers=8",
            f"actor_rollout_ref.rollout.agent.agent_loop_config_path={loop_config}",
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton",
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True",
            "data.max_prompt_length=8192",
            "data.max_response_length=6144",
            "data.truncation=error",
            "data.return_raw_chat=True",
            "data.filter_overlong_prompts=True",
            "data.shuffle=False",
            "++data.apply_chat_template_kwargs.enable_thinking=False",
        ])


async def measure(args, config, rows):
    server = await LLMServerManager.create(config)
    manager = await AgentLoopManager.create(config, server.get_client())
    print("rollout workers ready", flush=True)
    batches = []
    with (args.out / "batches.jsonl").open("x", encoding="utf-8") as stream:
        for index in range(0, len(rows), 2):
            group = rows[index:index + 2]
            batch = DataProto.from_single_dict(collate_fn(group)).repeat(8, interleave=True)
            batch.meta_info["global_steps"] = index // 2
            started = time.perf_counter()
            output = await manager.generate_sequences(batch)
            seconds = time.perf_counter() - started
            scores = output.batch["rm_scores"].sum(-1).tolist()
            durations = output.non_tensor_batch["trajectory_seconds"].tolist()
            turns = output.non_tensor_batch["__num_turns__"].tolist()
            response_tokens = output.batch["response_mask"].sum(-1).tolist()
            if len(scores) != 16 or not set(scores) <= {0.0, 1.0} or min(response_tokens) < 1:
                raise ValueError("Expected sixteen complete, binary-scored trajectories")
            result = {
                "batch": index // 2 + 1,
                "tasks": [row["extra_info"]["interaction_kwargs"]["task_id"] for row in group],
                "seconds": seconds,
                "scores": [scores[:8], scores[8:]],
                "trajectory_seconds": durations,
                "turns": turns,
                "response_tokens": response_tokens,
            }
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            batches.append(result)
            print(f"completed batch {result['batch']}, 16 trajectories in {seconds:.1f}s", flush=True)

    durations = sorted(value for batch in batches for value in batch["trajectory_seconds"])
    groups = [group for batch in batches for group in batch["scores"]]
    elapsed = sum(batch["seconds"] for batch in batches)
    report = {
        "trajectories": len(durations),
        "elapsed_seconds": elapsed,
        "scored_trajectories_per_hour": len(durations) * 3600 / elapsed,
        "trajectory_p50_seconds": median(durations),
        "trajectory_p95_seconds": durations[math.ceil(len(durations) * 0.95) - 1],
        "passed": sum(sum(group) for group in groups),
        "mixed_groups": sum(len(set(group)) > 1 for group in groups),
        "groups": len(groups),
        "mean_turns": sum(sum(batch["turns"]) for batch in batches) / len(durations),
        "response_tokens": sum(sum(batch["response_tokens"]) for batch in batches),
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--user-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ray-dir", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=8)
    args = parser.parse_args()
    if args.groups < 2 or args.groups % 2:
        parser.error("groups must be a positive multiple of two")
    args.out.mkdir(parents=True, exist_ok=True)
    config = load_config(args)
    tokenizer = hf_tokenizer(str(args.agent), trust_remote_code=True)
    dataset = create_rl_dataset([str(args.dataset)], config.data, tokenizer, processor=None)
    rows = [deepcopy(dataset[index]) for index in range(args.groups)]
    for row in rows:
        row["agent_name"] = "tau2_timed_agent"
        row["extra_info"]["interaction_kwargs"]["user_base_url"] = args.user_url

    ray.init(address="local", include_dashboard=False, num_cpus=24, num_gpus=1, _temp_dir=str(args.ray_dir))
    try:
        asyncio.run(measure(args, config, rows))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
