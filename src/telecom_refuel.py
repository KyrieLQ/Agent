"""Checkpointed approval for the simulator's Telecom refuel_data tool."""

from contextlib import closing, contextmanager
from dataclasses import dataclass
import json
import math
import sqlite3
from threading import RLock
from typing import TypedDict
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from tau2.data_model.message import ToolCall, ToolMessage
from tau2.domains.telecom.data_model import TelecomDB
from tau2.domains.telecom.environment import get_environment
from tau2.domains.telecom.user_data_model import TelecomUserDB


class TicketError(Exception):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


def dump_environment(environment):
    return json.dumps({
        "agent": environment.tools.db.model_dump(mode="json"),
        "user": environment.user_tools.db.model_dump(mode="json"),
    })


def restore_environment(snapshot):
    data = json.loads(snapshot)
    return get_environment(
        db=TelecomDB.model_validate(data["agent"]),
        user_db=TelecomUserDB.model_validate(data["user"]),
    )


def refuel_terms(environment, call):
    if call.name != "refuel_data" or call.requestor != "assistant":
        raise ValueError("Expected an assistant refuel_data proposal")
    args = call.arguments
    if set(args) != {"customer_id", "line_id", "gb_amount"}:
        raise ValueError("refuel_data needs customer_id, line_id and gb_amount")
    customer_id, line_id = args["customer_id"], args["line_id"]
    if not isinstance(customer_id, str) or not isinstance(line_id, str):
        raise ValueError("Customer and line IDs must be strings")
    gb_amount = args["gb_amount"]
    if isinstance(gb_amount, bool):
        raise ValueError("Specify a data amount in GB")
    try:
        amount = float(gb_amount)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Specify a data amount in GB") from exc
    if not math.isfinite(amount) or not 0 < amount <= 2:
        raise ValueError("Data refueling must be greater than 0 and at most 2 GB")
    # The lookup also validates that the customer owns the requested line.
    usage = environment.tools.get_data_usage(customer_id, line_id)
    line = next(line for line in environment.tools.db.lines if line.line_id == line_id)
    plan = next(plan for plan in environment.tools.db.plans if plan.plan_id == line.plan_id)
    return {
        "customer_id": customer_id, "line_id": line_id,
        "gb_amount": gb_amount if isinstance(gb_amount, int) else amount,
        "charge_usd": round(amount * plan.data_refueling_price_per_gb, 2),
        "before_gb": usage["data_refueling_gb"],
    }


def make_quote(environment, call):
    terms = refuel_terms(environment, call)
    return {"id": uuid4().hex, "tool_call_id": call.id, **terms}


def public_quote(quote):
    return {key: quote[key] for key in ("id", "line_id", "gb_amount", "charge_usd")}


@dataclass(frozen=True)
class Ticket:
    id: str
    task_id: str
    snapshot: str
    history: list[dict]
    quote: dict | None
    revision: int


class TicketStore:
    def __init__(self, path, refuel_executor=None):
        self.path = path
        self.refuel_executor = refuel_executor
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
                    snapshot TEXT NOT NULL, history TEXT NOT NULL,
                    quote TEXT, revision INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS receipts (
                    quote_id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL,
                    receipt TEXT NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        with closing(sqlite3.connect(self.path, timeout=30)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db

    def create(self, task_id, environment, history, quote=None):
        ticket_id = uuid4().hex
        with self.connect() as db:
            db.execute(
                "INSERT INTO tickets (id, task_id, snapshot, history, quote) VALUES (?, ?, ?, ?, ?)",
                (ticket_id, task_id, dump_environment(environment), json.dumps(history),
                 json.dumps(quote) if quote else None),
            )
        return ticket_id

    def load(self, ticket_id, db=None):
        if db is None:
            with self.connect() as conn:
                return self.load(ticket_id, conn)
        row = db.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if row is None:
            raise TicketError("Unknown ticket", 404)
        return Ticket(row["id"], row["task_id"], row["snapshot"], json.loads(row["history"]),
                      json.loads(row["quote"]) if row["quote"] else None, row["revision"])

    def save_turn(self, ticket, environment, history, quote=None):
        with self.connect() as db:
            updated = db.execute(
                """UPDATE tickets SET snapshot = ?, history = ?, quote = ?, revision = revision + 1
                   WHERE id = ? AND revision = ? AND quote IS NULL""",
                (dump_environment(environment), json.dumps(history), json.dumps(quote) if quote else None,
                 ticket.id, ticket.revision),
            ).rowcount
            if updated != 1:
                raise TicketError("The ticket changed. Refresh its status before continuing")

    def receipt(self, ticket_id, quote_id=None, db=None):
        if db is None:
            with self.connect() as conn:
                return self.receipt(ticket_id, quote_id, conn)
        if quote_id is None:
            row = db.execute(
                "SELECT receipt FROM receipts WHERE ticket_id = ? ORDER BY rowid DESC LIMIT 1",
                (ticket_id,),
            ).fetchone()
        else:
            row = db.execute(
                "SELECT receipt FROM receipts WHERE ticket_id = ? AND quote_id = ?", (ticket_id, quote_id),
            ).fetchone()
        return json.loads(row["receipt"]) if row else None

    def apply_approval(self, ticket_id, quote_id, approved):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = self.receipt(ticket_id, quote_id, db)
            if existing is not None:
                return same_decision(existing, approved)
            ticket = self.load(ticket_id, db)
            quote = ticket.quote
            if quote is None or quote["id"] != quote_id:
                raise TicketError("This quote is no longer awaiting confirmation")

            snapshot = ticket.snapshot
            receipt = {"quote_id": quote_id, "approved": approved, "line_id": quote["line_id"],
                       "gb_added": 0, "charge_usd": 0, "data_refueling_gb": quote["before_gb"]}
            if approved:
                environment = restore_environment(snapshot)
                call = ToolCall(id=quote["tool_call_id"], name="refuel_data", requestor="assistant",
                                arguments={key: quote[key] for key in ("customer_id", "line_id", "gb_amount")})
                if any(quote[key] != value for key, value in refuel_terms(environment, call).items()):
                    raise TicketError("The account or quote changed before confirmation")
                if self.refuel_executor is None:
                    tool_message = environment.get_response(call)
                else:
                    tool_message, snapshot = self.refuel_executor(snapshot, call, quote_id)
                if tool_message.error:
                    raise TicketError(f"Data refueling failed: {tool_message.content}")
                if self.refuel_executor is not None:
                    environment = restore_environment(snapshot)
                usage = environment.tools.get_data_usage(quote["customer_id"], quote["line_id"])
                updated = usage["data_refueling_gb"]
                charge = round(float(json.loads(tool_message.content)["charge"]), 2)
                if not math.isclose(updated, quote["before_gb"] + quote["gb_amount"]) or charge != quote["charge_usd"]:
                    raise TicketError("The tool result differs from the confirmed quote")
                receipt.update(gb_added=quote["gb_amount"], charge_usd=charge, data_refueling_gb=updated)
                snapshot = dump_environment(environment)
            else:
                tool_message = ToolMessage(
                    id=quote["tool_call_id"], role="tool", requestor="assistant", error=True,
                    content="The customer declined this data refueling quote.",
                )

            history = [*ticket.history, tool_message.model_dump(mode="json")]
            db.execute("INSERT INTO receipts VALUES (?, ?, ?)", (quote_id, ticket_id, json.dumps(receipt)))
            db.execute(
                "UPDATE tickets SET snapshot = ?, history = ?, quote = NULL, revision = revision + 1 WHERE id = ?",
                (snapshot, json.dumps(history), ticket_id),
            )
            # The updated simulator database and receipt commit in the same transaction.
            return receipt


def same_decision(receipt, approved):
    if receipt["approved"] != approved:
        raise TicketError("This quote already has a different confirmation decision")
    return receipt


class ApprovalState(TypedDict, total=False):
    ticket_id: str
    quote: dict
    approved: bool
    receipt: dict


class RefuelWorkflow:
    def __init__(self, directory, store):
        self.store = store
        self.lock = RLock()
        self.connection = sqlite3.connect(directory / "checkpoints.sqlite3", timeout=30, check_same_thread=False)
        self.connection.execute("PRAGMA synchronous=FULL")
        builder = StateGraph(ApprovalState)
        builder.add_node("wait_for_confirmation", self.wait_for_confirmation)
        builder.add_node("apply_confirmation", self.apply_confirmation)
        builder.add_edge(START, "wait_for_confirmation")
        builder.add_edge("wait_for_confirmation", "apply_confirmation")
        builder.add_edge("apply_confirmation", END)
        self.graph = builder.compile(checkpointer=SqliteSaver(self.connection))

    @staticmethod
    def config(quote_id):
        return {"configurable": {"thread_id": quote_id}}

    @staticmethod
    def wait_for_confirmation(state):
        quote = state["quote"]
        decision = interrupt({"status": "awaiting_confirmation", "ticket_id": state["ticket_id"], **public_quote(quote)})
        if decision["quote_id"] != quote["id"] or type(decision["approved"]) is not bool:
            raise TicketError("Confirmation must identify this quote and provide a decision")
        return {"approved": decision["approved"]}

    def apply_confirmation(self, state):
        return {"receipt": self.store.apply_approval(state["ticket_id"], state["quote"]["id"], state["approved"])}

    def prepare(self, ticket):
        with self.lock:
            quote = ticket.quote
            config = self.config(quote["id"])
            checkpoint = self.graph.get_state(config)
            if not checkpoint.values:
                self.graph.invoke({"ticket_id": ticket.id, "quote": quote}, config, durability="sync")
            elif checkpoint.next == ("wait_for_confirmation",) and not any(task.interrupts for task in checkpoint.tasks):
                self.graph.invoke(None, config, durability="sync")
            return self.graph.get_state(config)

    def confirm(self, ticket_id, quote_id, approved):
        with self.lock:
            config = self.config(quote_id)
            receipt = self.store.receipt(ticket_id, quote_id)
            if receipt is not None:
                same_decision(receipt, approved)
                # A process can exit after the receipt commits but before the node checkpoints.
                if self.graph.get_state(config).next == ("apply_confirmation",):
                    self.graph.invoke(None, config, durability="sync")
                return receipt
            ticket = self.store.load(ticket_id)
            if ticket.quote is None or ticket.quote["id"] != quote_id:
                raise TicketError("This quote is no longer awaiting confirmation")
            state = self.prepare(ticket)
            if "approved" in state.values and state.values["approved"] != approved:
                raise TicketError("This quote already has a different confirmation decision")
            if any(task.interrupts for task in state.tasks):
                command = Command(resume={"quote_id": quote_id, "approved": approved})
            elif "approved" in state.values and state.next:
                command = None  # Continue an apply step interrupted by a service exit.
            else:
                raise TicketError("No confirmation checkpoint found for this quote")
            self.graph.invoke(command, config, durability="sync")
            return self.store.receipt(ticket_id, quote_id)
