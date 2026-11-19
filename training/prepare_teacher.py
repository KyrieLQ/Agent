"""Select verified training trajectories and project them onto the agent history."""
import argparse
import json
from pathlib import Path


def project(row, eligible, validate_messages, visible, teacher_suffix):
    if (row["domain"], str(row["task_id"])) not in eligible:
        return None
    if row.get("reward") != 1 or row.get("error") or row.get("scoring_exception"):
        return None
    raw = row["messages"]
    messages = [
        source for source, message in zip(raw, validate_messages(raw), strict=True)
        if visible(message)
    ]
    if not any(message["role"] == "assistant" and message.get("tool_calls") for message in messages):
        return None
    if any(
        (message["role"] == "tool" and message.get("requestor") != "assistant")
        or (message["role"] == "user" and message.get("tool_calls"))
        for message in messages
    ):
        raise ValueError(f"Simulator-only tool message leaked into {row['task_id']}")
    prompt = row["system_prompt"]
    if row.get("teacher_closing_instruction"):
        suffix = "\n\n" + teacher_suffix
        if not teacher_suffix or prompt.count(suffix) != 1:
            raise ValueError("Provide the exact teacher-only suffix for marked traces")
        prompt = prompt.replace(suffix, "")
    return {
        "domain": row["domain"], "task_id": str(row["task_id"]), "reward": 1.0,
        "passed": True, "messages": messages, "system_prompt": prompt, "tools": row["tools"],
        "used_skill": False if row.get("teacher_closing_instruction") else bool(row.get("used_skill")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True,
                        help="Private JSON array of [domain, task_id] training pairs")
    parser.add_argument("--teacher-suffix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from pydantic import TypeAdapter
    from tau2.agent.base_agent import is_valid_agent_history_message
    from tau2.data_model.message import Message

    eligible = {(domain, str(tid)) for domain, tid in json.loads(args.train_ids.read_text(encoding="utf-8"))}
    suffix = args.teacher_suffix.read_text(encoding="utf-8").strip() if args.teacher_suffix else ""
    validate = TypeAdapter(list[Message]).validate_python
    traces = []
    with args.traces.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = project(json.loads(line), eligible, validate, is_valid_agent_history_message, suffix)
                if row is not None:
                    traces.append(row)
    if not traces:
        raise SystemExit("No successful, tool-using training trajectories matched the allowlist")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        for row in traces:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(traces)} training trajectories to {args.output}")


if __name__ == "__main__":
    main()
