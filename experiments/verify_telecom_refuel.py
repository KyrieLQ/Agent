"""Verify pause and recovery against a recorded Telecom purchase without inference."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import tempfile
from threading import Thread
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def test_case(path):
    with path.open(encoding="utf-8") as trace:
        for row in (json.loads(line) for line in trace if line.strip()):
            if row.get("domain") != "telecom" or row.get("reward") != 1:
                continue
            messages = row["messages"]
            for index, message in enumerate(messages):
                calls = message.get("tool_calls") or []
                if len(calls) != 1 or calls[0]["name"] != "refuel_data" or index < 2:
                    continue
                reply = messages[index - 1]
                ack = (reply.get("content") or "").lower()
                if reply["role"] != "user" or not ack.startswith("yes, i confirm"):
                    break
                tool_message = messages[index + 1]
                if tool_message.get("id") != calls[0]["id"] or tool_message.get("error"):
                    break
                expected = json.loads(tool_message["content"])
                recorded_response = message.get("raw_data")
                if not recorded_response or not recorded_response.get("choices"):
                    break
                return row["task_id"], calls[0]["arguments"], float(expected["charge"]), reply["content"], recorded_response
    raise RuntimeError("No successful archived Telecom refuel with explicit customer approval")


def available_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def request(port, route, payload=None, expected_status=200):
    data = json.dumps(payload).encode() if payload is not None else None
    message = Request(f"http://127.0.0.1:{port}{route}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(message, timeout=15) as response:
            status, content = response.status, json.load(response)
    except HTTPError as error:
        with error:
            status, content = error.code, error.read().decode()
    assert status == expected_status, (route, status, content)
    return content


@contextmanager
def recorded_model(reply):
    payload = json.dumps(reply).encode()
    calls = []

    class ModelHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            tools = {tool["function"]["name"] for tool in body.get("tools", [])}
            if self.path != "/v1/chat/completions" or "refuel_data" not in tools:
                self.send_error(400)
                return
            calls.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    with ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", calls
        finally:
            server.shutdown()
            thread.join()


@contextmanager
def service(launcher, state, trace, port, log, agent_url=None):
    command = ["bash", str(launcher), "--state-dir", str(state), "--port", str(port)]
    if trace is not None:
        command.extend(("--replay-traces", str(trace)))
    if agent_url is not None:
        command.extend(("--agent-url", agent_url))
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    request(port, "/health")
                    break
                except URLError:
                    time.sleep(0.25)
            else:
                raise RuntimeError(f"Telecom service did not become ready. See {log}")
            yield process.pid
        finally:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def ledger(state, ticket, account, line):
    with closing(sqlite3.connect(state / "tickets.sqlite3")) as db:
        snapshot = db.execute("SELECT snapshot FROM tickets WHERE id = ?", (ticket,)).fetchone()
        agent = json.loads(snapshot[0])["agent"]
        receipts = db.execute("SELECT count(*) FROM receipts WHERE ticket_id = ?", (ticket,)).fetchone()[0]
    balance = next(row["data_refueling_gb"] for row in agent["lines"] if row["line_id"] == line)
    charges = [item["amount"] for bill in agent["bills"] if bill["customer_id"] == account
               for item in bill["line_items"] if item["description"].startswith("Data refueling:")]
    return {"data_refueling_gb": balance, "refuel_charges": charges, "confirmation_receipts": receipts}


def verify(args):
    task_id, tool_args, charge, historical_confirmation, model_reply = test_case(args.trace)
    state = Path(tempfile.mkdtemp(prefix="verify-", dir=args.output_dir))
    port = available_port()
    with service(args.serve, state, args.trace, port, state / "before-restart.log") as first_pid:
        opened = request(port, "/tickets", {"task_id": task_id}, 201)
        route = f"/tickets/{opened['ticket_id']}"
        quote = opened["quote"]
        assert opened["status"] == "awaiting_confirmation"
        assert quote["gb_amount"] == tool_args["gb_amount"] and quote["charge_usd"] == charge
        assert quote["line_id"] == tool_args["line_id"]
        before = ledger(state, opened["ticket_id"], tool_args["customer_id"], tool_args["line_id"])
        assert before["data_refueling_gb"] == opened["data_refueling_gb"]
        assert before["confirmation_receipts"] == 0
        request(port, route + "/messages", {"text": historical_confirmation}, 409)

    with service(args.serve, state, args.trace, port, state / "after-restart.log") as second_pid:
        recovered = request(port, route)
        assert recovered == opened
        request(port, route + "/confirm", {"quote_id": "wrong-quote", "approved": True}, 409)
        assert ledger(state, opened["ticket_id"], tool_args["customer_id"], tool_args["line_id"]) == before
        approval = {"quote_id": quote["id"], "approved": True}
        with ThreadPoolExecutor(max_workers=2) as clients:
            results = list(clients.map(lambda _: request(port, route + "/confirm", approval), range(2)))
        receipt = results[0]
        assert results[1] == receipt
        assert receipt["gb_added"] == tool_args["gb_amount"] and receipt["charge_usd"] == charge
        assert request(port, route)["last_receipt"] == receipt
        after = ledger(state, opened["ticket_id"], tool_args["customer_id"], tool_args["line_id"])
        assert after["data_refueling_gb"] == before["data_refueling_gb"] + tool_args["gb_amount"]
        assert len(after["refuel_charges"]) == len(before["refuel_charges"]) + 1
        assert sum(after["refuel_charges"]) == sum(before["refuel_charges"]) + charge
        assert after["confirmation_receipts"] == 1

        declined = request(port, "/tickets", {"task_id": task_id}, 201)
        decline_route = f"/tickets/{declined['ticket_id']}/confirm"
        decision = {"quote_id": declined["quote"]["id"], "approved": False}
        cancelled = request(port, decline_route, decision)
        assert cancelled["gb_added"] == 0 and cancelled["charge_usd"] == 0
        cancel_ledger = ledger(state, declined["ticket_id"], tool_args["customer_id"], tool_args["line_id"])
        assert cancel_ledger["data_refueling_gb"] == before["data_refueling_gb"]
        assert cancel_ledger["refuel_charges"] == before["refuel_charges"]
        request(port, decline_route, {**decision, "approved": True}, 409)

    with service(args.serve, state, args.trace, port, state / "after-commit-restart.log") as third_pid:
        assert request(port, route + "/confirm", approval) == receipt
        assert ledger(state, opened["ticket_id"], tool_args["customer_id"], tool_args["line_id"]) == after

    with recorded_model(model_reply) as (model_url, model_requests):
        with service(args.serve, state, None, port, state / "normal-entry.log", model_url) as app_pid:
            new_ticket = request(port, "/tickets", {"task_id": task_id}, 201)
            assert new_ticket["status"] == "ready"
            current = request(port, f"/tickets/{new_ticket['ticket_id']}/messages", {"text": historical_confirmation})
            assert model_requests == ["/v1/chat/completions"]
            assert current["status"] == "awaiting_confirmation" and current["quote"]["charge_usd"] == charge
            assert ledger(state, new_ticket["ticket_id"], tool_args["customer_id"], tool_args["line_id"]) == before

    return {
        "passed": True, "archive": str(args.trace), "task_id": task_id, "state_dir": str(state),
        "service_pids": [first_pid, second_pid, third_pid, app_pid], "quote": quote,
        "checks": {"paused_before_write": True, "resumed_same_quote": True, "wrong_quote_rejected": True,
                   "concurrent_confirmations_same_receipt": True, "repeated_after_restart": True,
                   "declined_without_charge": True, "ordinary_agent_request_reaches_approval": True},
        "ledger_before": before, "ledger_after": after, "receipt": receipt,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = verify(args)
    text = json.dumps(report, indent=2) + "\n"
    (Path(report["state_dir"]) / "summary.json").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
