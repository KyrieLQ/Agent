"""Run the upstream VeRL launcher with two-rank adapter export."""

import os
from pathlib import Path
import shlex
import sys


def replace_once(code, original, replacement):
    if code.count(original) != 1:
        raise ValueError(f"Upstream launcher differs at {original!r}")
    return code.replace(original, replacement, 1)


def adapt(stage, code):
    exporter = shlex.quote(str(Path(__file__).with_name("export_fsdp_adapter.py").resolve()))
    module = "-m verl_code_rl.extract_sft_lora_adapter"
    code = replace_once(code, module, exporter)
    if stage == "grpo":
        code = replace_once(code, "${BASH_SOURCE[0]}", "$0")
        code = replace_once(code, "model_world_size_1_rank_0.pt", "model_world_size_2_rank_0.pt")
    return code


def main():
    stage, source, *options = sys.argv[1:]
    if stage not in {"sft", "grpo"}:
        raise ValueError(f"Unknown training stage: {stage}")
    launcher = Path(source).resolve(strict=True)
    code = adapt(stage, launcher.read_text(encoding="utf-8"))
    os.execvp("bash", ["bash", "-c", code, str(launcher), *options])


if __name__ == "__main__":
    main()
