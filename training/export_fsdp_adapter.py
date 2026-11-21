"""Export a PEFT adapter from a two-rank VeRL checkpoint."""

import argparse
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def gather(checkpoint):
    import torch
    from torch.distributed.tensor import DTensor

    parts = []
    for rank in (0, 1):
        shard = checkpoint / f"model_world_size_2_rank_{rank}.pt"
        state = torch.load(shard, map_location="cpu", weights_only=False, mmap=True)
        parts.append({name: value for name, value in state.items() if ".lora_A." in name or ".lora_B." in name})
        del state
    left, right = parts
    if not left or left.keys() != right.keys():
        raise ValueError("The two ranks must contain the same nonempty LoRA key set")

    weights = {}
    for name, first in left.items():
        second = right[name]
        if isinstance(first, DTensor) and isinstance(second, DTensor):
            if (first.shape != second.shape or first.placements != second.placements
                    or len(first.placements) != 1 or first.device_mesh.size() != 2):
                raise ValueError(f"Unexpected FSDP placement for {name}")
            placement = first.placements[0]
            local = [first.to_local(), second.to_local()]
            if placement.is_shard():
                weight = torch.cat(local, dim=placement.dim)
            elif placement.is_replicate() and torch.equal(*local):
                weight = local[0]
            else:
                raise ValueError(f"Unsupported placement or inconsistent replica for {name}")
            if weight.shape != first.shape:
                raise ValueError(f"Unexpected gathered shape for {name}")
        elif not isinstance(first, DTensor) and not isinstance(second, DTensor) and torch.equal(first, second):
            weight = first
        else:
            raise ValueError(f"Inconsistent rank tensors for {name}")
        if not torch.isfinite(weight).all().item():
            raise ValueError(f"Non-finite weights in {name}")
        weights[name] = weight.contiguous().clone()
    return weights


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora-rank", required=True)
    parser.add_argument("--lora-alpha", required=True)
    parser.add_argument("--target-modules", required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="lora-", dir=args.output.parent) as directory:
        torch.save(gather(args.checkpoint_dir), Path(directory) / "model_world_size_1_rank_0.pt")
        subprocess.run([
            sys.executable, "-m", "verl_code_rl.extract_sft_lora_adapter",
            "--checkpoint-dir", directory, "--base-model", args.base_model,
            "--output", str(args.output), "--lora-rank", args.lora_rank,
            "--lora-alpha", args.lora_alpha, "--target-modules", args.target_modules,
        ], check=True)


if __name__ == "__main__":
    main()
