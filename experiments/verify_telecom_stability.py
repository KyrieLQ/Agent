"""Exercise Telecom approvals across archived tasks, concurrent requests and process exits."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import json
from math import ceil, isclose
from pathlib import Path
from statistics import median
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError

from verify_telecom_refuel import available_port, ledger, request


@dataclass(frozen=True)
class Case:
    task_id: str
    source: Path
    arguments: dict
    charge: float
    prefix: list[dict]


def archived_cases(paths, catalog):
    cases = {}
    for path in paths:
        with path.open(encoding="utf-8") as trace:
            for row in (json.loads(line) for line in trace if line.strip()):
                task_id = row.get("task_id")
                if row.get("domain") != "telecom" or task_id not in catalog or task_id in cases or row.get("error"):
                    continue
                messages = row["messages"]
                for index, message in enumerate(messages):
                    calls = message.get("tool_calls") or []
                    if (index < 2 or message["role"] != "assistant" or len(calls) != 1
                            or calls[0]["name"] != "refuel_data" or messages[index - 1]["role"] != "user"):
                        continue
                    if index + 1 < len(messages):
                        reply = messages[index + 1]
                        call = calls[0]
                        if reply.get("id") == call["id"] and not reply.get("error"):
                            charge = float(json.loads(reply["content"])["charge"])
                            prefix = [{key: value for key, value in item.items() if key != "raw_data"}
                                      for item in messages[:index + 2]]
                            cases[task_id] = Case(task_id, path, call["arguments"], charge, prefix)
                    break
    if not cases:
        raise ValueError("No successful archived refuel proposals found")
    return list(cases.values())


FAULT_SERVER = """
import os
import sys
import serve_telecom
from telecom_refuel import TicketStore

stage = sys.argv.pop(1)
apply_approval = TicketStore.apply_approval

def crash(self, *args):
    if stage == 'before_write':
        os._exit(71)
    apply_approval(self, *args)
    os._exit(72)

TicketStore.apply_approval = crash
serve_telecom.main()
"""


@contextmanager
def boot(args, folder, fixture, port, name, fault=None):
    command = [sys.executable, "-B"]
    command.extend(("-c", FAULT_SERVER, fault) if fault else (str(args.app),))
    command.extend(("--state-dir", str(folder), "--task-catalog", str(args.task_catalog),
                    "--replay-traces", str(fixture), "--port", str(port)))
    log_path = folder / (name + ".log")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    request(port, "/health")
                    break
                except URLError:
                    time.sleep(0.25)
            else:
                raise RuntimeError(f"Service not ready. See {log_path}")
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def approval(opened, approved=True):
    return {"quote_id": opened["quote"]["id"], "approved": approved}


def ticket_route(opened):
    return f"/tickets/{opened['ticket_id']}"


def audit(folder, opened, case):
    return ledger(folder, opened["ticket_id"], case.arguments["customer_id"], case.arguments["line_id"])


def propose(port, folder, case):
    opened = request(port, "/tickets", {"task_id": case.task_id}, 201)
    quote = opened["quote"]
    assert opened["status"] == "awaiting_confirmation", opened
    assert quote["line_id"] == case.arguments["line_id"]
    assert isclose(quote["gb_amount"], float(case.arguments["gb_amount"]))
    assert isclose(quote["charge_usd"], case.charge)
    state = audit(folder, opened, case)
    assert state["confirmation_receipts"] == 0
    assert state["data_refueling_gb"] == opened["data_refueling_gb"]
    return case, opened, state


def verify_ledger(folder, ticket, receipt, approved=True):
    case, opened, before = ticket
    after = audit(folder, opened, case)
    delta = float(case.arguments["gb_amount"]) if approved else 0.0
    additional_charges = [case.charge] if approved else []
    assert receipt["approved"] is approved
    assert receipt["quote_id"] == opened["quote"]["id"]
    assert isclose(receipt["gb_added"], delta)
    assert isclose(receipt["charge_usd"], sum(additional_charges))
    assert isclose(after["data_refueling_gb"], before["data_refueling_gb"] + delta)
    assert Counter(after["refuel_charges"]) == Counter(before["refuel_charges"] + additional_charges)
    assert after["confirmation_receipts"] == 1
    return after


def run_fault(args, folder, fixture, case, stage):
    from telecom_refuel import RefuelWorkflow, TicketStore

    folder.mkdir()
    port = available_port()
    with boot(args, folder, fixture, port, "fault", stage) as process:
        ticket = propose(port, folder, case)
        _, opened, before = ticket
        route = ticket_route(opened) + "/confirm"
        try:
            request(port, route, approval(opened))
        except (OSError, URLError):
            pass
        else:
            raise AssertionError("Expected the confirmation connection to close when the process exits")
        assert process.wait(12) == (71 if stage == "before_write" else 72)

    interrupted = audit(folder, opened, case)
    if stage == "before_write":
        assert interrupted == before
    else:
        assert interrupted["confirmation_receipts"] == 1
        assert isclose(interrupted["data_refueling_gb"], before["data_refueling_gb"] + float(case.arguments["gb_amount"]))

    with boot(args, folder, fixture, port, "recovered"):
        request(port, route, approval(opened, False), 409)
        receipt = request(port, route, approval(opened))
        assert request(port, route, approval(opened)) == receipt
        verify_ledger(folder, ticket, receipt)

    workflow = RefuelWorkflow(folder, TicketStore(folder / "tickets.sqlite3"))
    try:
        checkpoint = workflow.graph.get_state(workflow.config(opened["quote"]["id"]))
        assert not checkpoint.next and checkpoint.values.get("receipt") == receipt, {
            "stage": stage, "next": checkpoint.next, "receipt_committed": True,
            "task_id": case.task_id,
        }
    finally:
        workflow.connection.close()
    calls = sum(log.read_text(encoding="utf-8").count("Data refueled for line ") for log in folder.glob("*.log"))
    assert calls == 1, {"stage": stage, "refuel_tool_invocations": calls}
    return {"stage": stage, "recovered": True, "synchronous_checkpoint_completed": True, "tool_invocations": calls}


def distribution(items):
    items = sorted(items)
    return {"n": len(items), "p50_ms": round(median(items), 2),
            "p95_ms": round(items[ceil(0.95 * len(items)) - 1], 2), "max_ms": round(items[-1], 2)}


def run_batch(args, folder, fixture, cases):
    folder.mkdir()
    port = available_port()
    timings = {}

    def measure(bucket, function, *parameters):
        start = time.perf_counter()
        result = function(*parameters)
        timings.setdefault(bucket, []).append((time.perf_counter() - start) * 1000)
        return result

    def issue(ticket):
        _, opened, _ = ticket
        return request(port, ticket_route(opened) + "/confirm", approval(opened))

    with boot(args, folder, fixture, port, "pending") as process, ThreadPoolExecutor(max_workers=args.workers) as clients:
        tickets = list(clients.map(lambda case: measure("prepare_with_archive_replay", propose, port, folder, case), cases))
        for case, opened, old in tickets:
            assert audit(folder, opened, case) == old
        process.kill()
        process.wait(12)

    with boot(args, folder, fixture, port, "confirm"):
        for _, opened, _ in tickets:
            assert measure("read_quote_after_restart", request, port, ticket_route(opened)) == opened
        first, second = tickets[0][1], tickets[1][1]
        request(port, ticket_route(first) + "/confirm", approval(second), 409)
        request(port, ticket_route(first) + "/confirm", {"quote_id": first["quote"]["id"], "approved": "true"}, 422)
        for case, opened, old in tickets[:2]:
            assert audit(folder, opened, case) == old

        cutoff = min(12, len(tickets) // 2)
        serial_receipts = [measure("first_confirmation_serial", issue, item) for item in tickets[:cutoff]]
        paired = [item for item in tickets[cutoff:] for _ in range(2)]
        with ThreadPoolExecutor(max_workers=args.workers) as clients:
            concurrent = list(clients.map(lambda item: measure("confirmation_with_concurrent_retries", issue, item), paired))
        assert concurrent[::2] == concurrent[1::2]
        receipts = serial_receipts + concurrent[::2]
        for ticket, receipt in zip(tickets, receipts, strict=True):
            verify_ledger(folder, ticket, receipt)

        declines = cases[::max(1, len(cases) // 8)][:8]
        for case in declines:
            ticket = propose(port, folder, case)
            _, opened, _ = ticket
            route = ticket_route(opened) + "/confirm"
            receipt = request(port, route, approval(opened, False))
            verify_ledger(folder, ticket, receipt, False)
            request(port, route, approval(opened), 409)

    with boot(args, folder, fixture, port, "retry"), ThreadPoolExecutor(max_workers=args.workers) as clients:
        replayed = list(clients.map(lambda item: measure("already_committed_confirmation", issue, item), tickets))
        assert replayed == receipts
        for ticket, receipt in zip(tickets, receipts, strict=True):
            verify_ledger(folder, ticket, receipt)

    invocations = sum(log.read_text(encoding="utf-8").count("Data refueled for line ") for log in folder.glob("*.log"))
    assert invocations == len(cases), {"tool_invocations": invocations, "approved_tickets": len(cases)}
    return {
        "archived_tickets": len(tickets), "approved_tickets": len(tickets), "declined_tickets": len(declines),
        "wrong_ticket_quote_rejected": True, "non_boolean_approval_rejected": True,
        "concurrent_duplicate_requests": len(concurrent), "retried_after_restart": len(replayed),
        "committed_refuel_tool_invocations": invocations, "client_workers": args.workers,
        "latency": {name: distribution(values) for name, values in timings.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--task-catalog", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--faults-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="verify-", dir=args.output_dir))
    cases = archived_cases(args.trace, json.loads(args.task_catalog.read_text(encoding="utf-8")))
    if len(cases) < 2:
        raise ValueError("Stability verification requires at least two distinct archived tasks")
    fixture = run / "archived_proposals.jsonl"
    with fixture.open("w", encoding="utf-8") as output:
        for case in cases:
            output.write(json.dumps({"domain": "telecom", "task_id": case.task_id, "messages": case.prefix}) + "\n")
    (run / "cases.json").write_text(json.dumps([{
        "task_id": case.task_id, "archive": str(case.source), "gb": float(case.arguments["gb_amount"]),
        "charge_usd": case.charge,
    } for case in cases], indent=2) + "\n", encoding="utf-8")
    print(f"Run directory {run}, selected {len(cases)} archived tasks", flush=True)

    faults = [run_fault(args, run / stage, fixture, cases[0], stage) for stage in ("before_write", "after_commit")]
    print(f"Process-exit checks passed {len(faults)}", flush=True)
    report = {"passed": True, "run_dir": str(run), "faults": faults}
    if not args.faults_only:
        report["batch"] = run_batch(args, run / "batch", fixture, cases)
    body = json.dumps(report, indent=2) + "\n"
    (run / "summary.json").write_text(body, encoding="utf-8")
    print(body, end="")


if __name__ == "__main__":
    main()
