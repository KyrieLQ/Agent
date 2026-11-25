"""Serve resumable Telecom tickets with the existing Qwen and tau2 tools."""

import argparse
from contextlib import ExitStack
from functools import partial
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictBool, TypeAdapter
from tau2.agent.base_agent import is_valid_agent_history_message
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, Message, MultiToolMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.tasks import Task
from tau2.domains.telecom.environment import get_environment
from tau2_evolve import benchmark
import uvicorn

from telecom_line_usage import enrich_line_usage
from telecom_mcp import READ_TOOLS, connect_mcp
from telecom_refuel import RefuelWorkflow, TicketError, TicketStore, dump_environment, make_quote, public_quote, restore_environment


HISTORY = TypeAdapter(list[Message])
DEFAULT_PLAYBOOK = Path(__file__).resolve().parents[1] / "playbooks" / "telecom.txt"
APPROVAL_POLICY = (
    "When the customer has requested a specific amount of data, submit refuel_data as the only tool call "
    "in its turn. The application will present the exact quote for confirmation before executing the tool. "
    "Report a successful purchase only after receiving a successful refuel_data result."
)


class CreateTicket(BaseModel, extra="forbid"):
    task_id: str = Field(min_length=1)


class CustomerMessage(BaseModel, extra="forbid"):
    text: str = Field(min_length=1)


class Confirmation(BaseModel, extra="forbid"):
    quote_id: str = Field(min_length=1)
    approved: StrictBool


def dump_history(messages):
    return [message.model_dump(mode="json", exclude={"raw_data"}) for message in messages]


def archived_proposal(path, task_id):
    """Replay the prefix immediately before the archive's customer approval."""
    with path.open(encoding="utf-8") as trace:
        for row in (json.loads(line) for line in trace if line.strip()):
            if row.get("domain") != "telecom" or row["task_id"] != task_id:
                continue
            messages = row["messages"]
            for index, message in enumerate(messages):
                calls = message.get("tool_calls") or []
                if (index >= 2 and message["role"] == "assistant" and len(calls) == 1
                        and calls[0]["name"] == "refuel_data" and messages[index - 1]["role"] == "user"):
                    return HISTORY.validate_python(messages[:index - 1]), ToolCall.model_validate(calls[0])
            break
    raise TicketError("No historical refueling proposal after a customer reply for this task", 404)


class TelecomService:
    def __init__(self, args, mcp=None):
        self.args = args
        self.mcp = mcp
        refuel = mcp.refuel if mcp is not None and getattr(args, "mcp_refuel", False) else None
        self.store = TicketStore(args.state_dir / "tickets.sqlite3", refuel_executor=refuel)
        self.workflow = RefuelWorkflow(args.state_dir, self.store)
        self.manual = args.playbook.read_text(encoding="utf-8").strip()
        self.tasks = json.loads(args.task_catalog.read_text(encoding="utf-8")) if args.task_catalog else None

    def create(self, task_id):
        try:
            entry = self.tasks[task_id] if self.tasks is not None else benchmark.load_task("telecom", task_id)
            task = Task.model_validate(entry)
        except (KeyError, ValueError, StopIteration) as exc:
            raise TicketError("Unknown Telecom task", 404) from exc
        initial = task.initial_state
        history = list(initial.message_history or []) if initial else []
        call = None
        if self.args.replay_traces is not None:
            history, call = archived_proposal(self.args.replay_traces, task_id)
        environment = get_environment()
        environment.set_state(
            initial.initialization_data if initial else None,
            initial.initialization_actions if initial else None,
            history,
        )
        quote = None
        if call is not None:
            quote = self.propose(environment, call)
            history.append(AssistantMessage(role="assistant", tool_calls=[call]))
        elif not history:
            history.append(AssistantMessage(role="assistant", content="How can I help you today?"))
        ticket_id = self.store.create(task_id, environment, dump_history(history), quote)
        return self.status(ticket_id)

    @staticmethod
    def propose(environment, call):
        try:
            return make_quote(environment, call)
        except ValueError as exc:
            raise TicketError(str(exc), 422) from exc

    def status(self, ticket_id):
        ticket = self.store.load(ticket_id)
        receipt = self.store.receipt(ticket_id)
        if ticket.quote:
            self.workflow.prepare(ticket)
            quote = public_quote(ticket.quote)
            quote_text = (
                f"Confirm adding {quote['gb_amount']:g} GB to line {quote['line_id']} "
                f"for ${quote['charge_usd']:.2f}."
            )
            return {"ticket_id": ticket_id, "task_id": ticket.task_id,
                    "status": "awaiting_confirmation", "quote": quote, "message": quote_text,
                    "data_refueling_gb": ticket.quote["before_gb"]}
        message = next((item["content"] for item in reversed(ticket.history)
                        if item["role"] == "assistant" and item.get("content") and not item.get("tool_calls")), None)
        return {"ticket_id": ticket_id, "task_id": ticket.task_id, "status": "ready",
                "message": message, "last_receipt": receipt}

    def execute_tool(self, environment, call):
        call = call.model_copy(update={"requestor": "assistant"})
        if self.mcp is not None and call.name in READ_TOOLS:
            return self.mcp.read(dump_environment(environment), call)
        return environment.get_response(call)

    def respond(self, ticket_id, text):
        ticket = self.store.load(ticket_id)
        if ticket.quote:
            raise TicketError("The customer must approve or decline the pending quote")
        if self.args.replay_traces is not None:
            raise TicketError("This service is replaying archival proposals")
        environment = restore_environment(ticket.snapshot)
        execute = partial(self.execute_tool, environment)
        policy = "\n\n".join((environment.get_policy(), self.manual, APPROVAL_POLICY))
        spec = benchmark.make_llm_spec(self.args.agent_model, self.args.agent_url,
                                      temperature=0.5, enable_thinking=False, max_tokens=1024)
        spec.args.update(top_p=0.98, seed=300)
        agent = LLMAgent(tools=environment.get_tools(), domain_policy=policy, llm=spec.model, llm_args=spec.args)
        history = [m for m in HISTORY.validate_python(ticket.history) if is_valid_agent_history_message(m)]
        state = agent.get_init_state(history)
        incoming = UserMessage(role="user", content=text)
        for _ in range(self.args.max_tool_rounds):
            response, state = agent.generate_next_message(incoming, state)
            calls = response.tool_calls or []
            if not calls:
                self.store.save_turn(ticket, environment, dump_history(state.messages))
                return self.status(ticket_id)

            if any(call.name == "refuel_data" for call in calls):
                try:
                    if len(calls) != 1:
                        raise TicketError("Submit refuel_data as a single tool call")
                    quote = self.propose(environment, calls[0])
                except TicketError as exc:
                    results = [ToolMessage(id=call.id, role="tool", content=str(exc), error=True) for call in calls]
                else:
                    self.store.save_turn(ticket, environment, dump_history(state.messages), quote)
                    return self.status(ticket_id)
            else:
                results = [execute(call) for call in calls]
                if getattr(self.args, "with_line_usage", False):
                    for index, (call, result) in enumerate(zip(calls, results, strict=True)):
                        results[index] = enrich_line_usage(state.messages + results, call, result, execute)
            incoming = results[0] if len(results) == 1 else MultiToolMessage(role="tool", tool_messages=results)
        raise TicketError("The agent reached its tool-round limit for this message", 422)


def create_app(service):
    app = FastAPI(title="Telecom customer-agent work orders")

    @app.exception_handler(TicketError)
    async def ticket_error(_request: Request, error: TicketError):
        return JSONResponse(status_code=error.status, content={"detail": str(error)})

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/tickets", status_code=201)
    def create(request: CreateTicket):
        return service.create(request.task_id)

    @app.get("/tickets/{ticket_id}")
    def status(ticket_id: str):
        return service.status(ticket_id)

    @app.post("/tickets/{ticket_id}/messages")
    def respond(ticket_id: str, request: CustomerMessage):
        return service.respond(ticket_id, request.text)

    @app.post("/tickets/{ticket_id}/confirm")
    def confirm(ticket_id: str, request: Confirmation):
        return service.workflow.confirm(ticket_id, request.quote_id, request.approved)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--agent-url", default="http://127.0.0.1:8260/v1")
    parser.add_argument("--agent-model", default="openai/evol-llm-agent")
    parser.add_argument("--playbook", type=Path, default=DEFAULT_PLAYBOOK)
    parser.add_argument("--task-catalog", type=Path)
    parser.add_argument("--max-tool-rounds", type=int, default=30)
    parser.add_argument("--replay-traces", type=Path)
    parser.add_argument("--with-line-usage", action="store_true", help="Include verified quota on the affected line")
    parser.add_argument("--mcp-url", help="Streamable HTTP URL for the Telecom tool service")
    parser.add_argument("--mcp-refuel", action="store_true", help="Send confirmed refuels to the MCP tool service")
    args = parser.parse_args()
    if args.max_tool_rounds <= 0:
        parser.error("--max-tool-rounds must be positive")
    if args.mcp_refuel and not args.mcp_url:
        parser.error("--mcp-refuel requires --mcp-url")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as resources:
        mcp = resources.enter_context(connect_mcp(args.mcp_url, args.mcp_refuel)) if args.mcp_url else None
        service = TelecomService(args, mcp)
        resources.callback(service.workflow.connection.close)
        uvicorn.run(create_app(service), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
