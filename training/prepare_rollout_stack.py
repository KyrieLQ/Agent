"""Prepare the project's tau2 GRPO stack with optional rollout optimizations."""

import argparse
from pathlib import Path
import shutil

from launch_4b import replace_once


RECIPE = Path("experiments/tau-2/tau2_evolve")


def adapt_interaction(code):
    code = replace_once(
        code,
        "from tau2_evolve import benchmark\n",
        "from tau2_evolve import benchmark\nfrom tau2_evolve.rollout_environment import run_environment\n",
    )
    code = replace_once(
        code,
        "        # The rollout model owns assistant turns. The orchestrator owns all\n",
        "        return await run_environment(self._advance_environment, state)\n\n"
        "    def _advance_environment(self, state):\n"
        "        from tau2.data_model.message import ToolMessage\n"
        "        from tau2.orchestrator.orchestrator import Role\n\n"
        "        orch = state[\"orch\"]\n"
        "        # The rollout model owns assistant turns. The orchestrator owns all\n",
    )
    code = replace_once(
        code,
        "        from tau2.data_model.message import ToolMessage\n"
        "        from tau2.data_model.simulation import TerminationReason\n",
        "        from tau2.data_model.simulation import TerminationReason\n",
    )
    return replace_once(
        code,
        "    async def force_terminate(self, instance_id: str, reason: str = \"verl_limit\") -> float:\n"
        "        from tau2.data_model.simulation import TerminationReason\n"
        "        from tau2.orchestrator.orchestrator import Role\n\n"
        "        state = self._instances[instance_id]\n",
        "    async def force_terminate(self, instance_id: str, reason: str = \"verl_limit\") -> float:\n"
        "        return await run_environment(self._finish_environment, self._instances[instance_id], reason)\n\n"
        "    def _finish_environment(self, state, reason: str) -> float:\n"
        "        from tau2.data_model.simulation import TerminationReason\n"
        "        from tau2.orchestrator.orchestrator import Role\n\n",
    )


def adapt_budget_trainer(code):
    code = replace_once(code, "import asyncio\n", "import asyncio\nfrom time import perf_counter\n")
    code = replace_once(
        code,
        '        assert self.mode in ("plain", "dynamic")\n',
        '        assert self.mode in ("plain", "dynamic")\n'
        '        self.prefetch = os.getenv("CIRRUS_PREFETCH", "0") == "1"\n'
        '        self.prefetch_task = None\n'
        '        if self.prefetch:\n'
        '            rollout = self.config.actor_rollout_ref.rollout\n'
        '            correction = self.config.algorithm.get("rollout_correction") or {}\n'
        '            if self.mode != "plain" or not rollout.calculate_log_probs or not correction.get("bypass_mode"):\n'
        '                raise ValueError("Prefetch requires plain GRPO with rollout log probabilities")\n'
        '            per_batch = self.config.data.train_batch_size * rollout.n\n'
        '            if self.rollout_budget % per_batch:\n'
        '                raise ValueError("Prefetch requires a budget of whole training batches")\n',
    )
    code = replace_once(
        code,
        '        result = await super().fit()\n',
        '        try:\n'
        '            result = await super().fit()\n'
        '        finally:\n'
        '            if self.prefetch_task is not None:\n'
        '                self.prefetch_task.cancel()\n'
        '                await asyncio.gather(self.prefetch_task, return_exceptions=True)\n',
    )
    code = replace_once(
        code,
        '    def _record(self, event):\n',
        '    async def _async_gen_next_batch(self, continuous_iterator):\n'
        '        # veRL starts step one after synchronizing the initial policy.\n'
        '        policy_version = self.global_steps - 1\n'
        '        result = await super()._async_gen_next_batch(continuous_iterator)\n'
        '        if result is not None:\n'
        '            result[3].meta_info["cirrus_policy_version"] = policy_version\n'
        '        return result\n\n'
        '    def _record(self, event):\n',
    )
    code = replace_once(
        code,
        '            await super().fit_step(batch_data_future, continuous_iterator)\n',
        '            next_batch = await super().fit_step(batch_data_future, continuous_iterator)\n',
    )
    code = replace_once(
        code,
        '        self._fit_update_weights()\n        return asyncio.create_task(self._async_gen_next_batch(continuous_iterator))\n',
        '        if self.prefetch:\n'
        '            return next_batch\n'
        '        self._fit_update_weights()\n'
        '        return asyncio.create_task(self._async_gen_next_batch(continuous_iterator))\n',
    )
    code = replace_once(
        code,
        '            metrics, timing, epoch, batch, reward_future = await batch_data_future\n',
        '            wait_started = perf_counter()\n'
        '            metrics, timing, epoch, batch, reward_future = await batch_data_future\n'
        '            self.timing_raw["rollout_wait"] = self.timing_raw.get("rollout_wait", 0.0) + perf_counter() - wait_started\n'
        '            if batch_data_future is self.prefetch_task:\n'
        '                self.prefetch_task = None\n'
        '            rollout_version = batch.meta_info.pop("cirrus_policy_version")\n'
        '            policy_lag = self.global_steps - 1 - rollout_version\n'
        '            if not 0 <= policy_lag <= int(self.prefetch):\n'
        '                raise RuntimeError(f"Unexpected rollout policy lag: {policy_lag}")\n',
    )
    code = replace_once(
        code,
        '                                  raw_rollouts=len(indexes), rewards=rewards, mixed=mixed, selected=keep))\n',
        '                                  raw_rollouts=len(indexes), rewards=rewards, mixed=mixed, selected=keep,\n'
        '                                  rollout_policy_version=rollout_version, policy_lag=policy_lag))\n',
    )
    return replace_once(
        code,
        '        return batch, None\n',
        '        if self.prefetch and not self.is_last_step:\n'
        '            # Reuse veRL one-step-off ordering: sync the current policy, then start the\n'
        '            # next generation before updating the actor on the current batch.\n'
        '            self._fit_update_weights()\n'
        '            self.prefetch_task = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))\n'
        '        return batch, self.prefetch_task\n',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    destination = args.destination.resolve()
    if source == destination or source in destination.parents:
        parser.error("destination must be a separate checkout")
    interaction = RECIPE / "interaction.py"
    budget_trainer = RECIPE / "cirrus_budget.py"
    interaction_code = adapt_interaction((source / interaction).read_text(encoding="utf-8"))
    budget_code = adapt_budget_trainer((source / budget_trainer).read_text(encoding="utf-8"))
    catalog = (source.parent / "train_tasks.json").read_bytes()
    destination_catalog = destination.parent / "train_tasks.json"
    if destination_catalog.exists() and destination_catalog.read_bytes() != catalog:
        parser.error(f"a different training catalog already exists at {destination_catalog}")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    if not destination_catalog.exists():
        with destination_catalog.open("xb") as stream:
            stream.write(catalog)
    (destination / interaction).write_text(interaction_code, encoding="utf-8")
    (destination / budget_trainer).write_text(budget_code, encoding="utf-8")
    shutil.copyfile(
        Path(__file__).with_name("rollout_environment.py"),
        destination / RECIPE / "rollout_environment.py",
    )


if __name__ == "__main__":
    main()
