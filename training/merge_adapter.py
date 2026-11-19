"""Merge a training adapter into the base checkpoint used to create it."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mera-root", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    base = args.base.resolve(strict=True)
    adapter = args.adapter.resolve(strict=True)
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    reference = Path(config["base_model_name_or_path"]).resolve()
    if reference != base:
        raise SystemExit(f"Adapter targets {reference}, expected {base}")
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    sys.path.insert(0, str(args.mera_root.resolve(strict=True)))
    from verl_code_rl.merge_lora import main as merge

    sys.argv[1:] = ["--adapter", str(adapter), "--output", str(args.output.resolve())]
    return merge()


if __name__ == "__main__":
    raise SystemExit(main())
