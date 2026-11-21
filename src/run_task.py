"""Run a user-specified τ²-bench task with the official evaluator."""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=("telecom", "retail", "airline"))
    parser.add_argument("task_id")
    parser.add_argument("--with-playbook", action="store_true")
    parser.add_argument("--agent-url", default="http://127.0.0.1:8260/v1")
    parser.add_argument("--user-url", default="http://127.0.0.1:8261/v1")
    parser.add_argument("--agent-model", default="openai/evol-llm-agent")
    parser.add_argument("--user-model", default="openai/evol-llm-user")
    parser.add_argument("--agent-temperature", type=float, default=0.0)
    parser.add_argument("--agent-top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def run(args, benchmark):
    agent = benchmark.make_llm_spec(
        args.agent_model, args.agent_url, temperature=args.agent_temperature,
        enable_thinking=False, max_tokens=1024,
    )
    agent.args.update(top_p=args.agent_top_p, seed=args.seed)
    user = benchmark.make_llm_spec(
        args.user_model, args.user_url, temperature=0.0,
        enable_thinking=False, max_tokens=768,
    )
    user.args["seed"] = args.seed
    manual = ""
    if args.with_playbook:
        manual = (ROOT / "playbooks" / f"{args.domain}.txt").read_text(encoding="utf-8").strip()
        if not manual:
            raise ValueError(f"Empty playbook for {args.domain}")
    result = benchmark.run_task(
        domain=args.domain, task_id=args.task_id, seed=args.seed,
        agent_spec=agent, user_spec=user, max_steps=args.max_steps,
        max_errors=10, skill_text=manual,
    )
    if manual and manual not in result["system_prompt"]:
        raise RuntimeError("The playbook did not reach the agent's system prompt")
    result["seed"] = args.seed
    return result


def main():
    args = build_parser().parse_args()
    if args.max_steps < 1 or args.agent_temperature < 0:
        raise SystemExit("max-steps must be positive and temperature non-negative")
    if not 0.0 < args.agent_top_p <= 1.0:
        raise SystemExit("agent-top-p must be in (0, 1]")
    try:
        from tau2_evolve import benchmark
    except ImportError as exc:
        raise SystemExit("Set up the runner environment and PYTHONPATH in deploy/README.md") from exc
    if args.out.exists():
        raise SystemExit(f"Output already exists: {args.out}")
    result = run(args, benchmark)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False)
        file.write("\n")
    print(f"Official reward: {result['reward']}. Wrote {args.out}.")


if __name__ == "__main__":
    main()
