"""Route the existing Telecom tools through a persistent MCP client."""

from contextlib import contextmanager
from functools import partial

from pydantic import ValidationError
from tau2.data_model.message import ToolMessage
from tau2.domains.telecom.environment import get_environment

from telecom_refuel import TicketError


READ_TOOLS = frozenset({"get_details_by_id", "get_data_usage"})
REFUEL_TOOL = "refuel_data"
TOOL_CONTEXT = "tau2.customer-agent/context"


def native_specs():
    return {tool.name: tool.openai_schema["function"] for tool in get_environment().get_tools()
            if tool.name in READ_TOOLS or tool.name == REFUEL_TOOL}


class TelecomMCP:
    def __init__(self, portal, client, refuel):
        self.portal = portal
        self.client = client
        specs = native_specs()
        advertised = {tool.name: tool for tool in portal.call(client.list_tools).tools}
        selected = READ_TOOLS | ({REFUEL_TOOL} if refuel else set())
        for name in selected:
            tool = advertised.get(name)
            if tool is None or tool.input_schema != specs[name]["parameters"] or tool.description != specs[name]["description"]:
                raise TicketError(f"Telecom MCP tool {name} does not match its native contract", 502)

    def call(self, snapshot, call, quote_id=None):
        from mcp.types import RequestParamsMeta

        context = {"snapshot": snapshot, "tool_call_id": call.id}
        if quote_id is not None:
            context["quote_id"] = quote_id
        meta = RequestParamsMeta(**{TOOL_CONTEXT: context})
        try:
            response = self.portal.call(partial(self.client.call_tool, call.name, call.arguments, meta=meta))
        except Exception as exc:
            raise TicketError("Telecom MCP service could not complete the request", 502) from exc
        payload = response.structured_content
        if not isinstance(payload, dict) or "message" not in payload:
            raise TicketError("Telecom MCP service did not return a native tool receipt", 502)
        try:
            message = ToolMessage.model_validate(payload["message"])
        except ValidationError as exc:
            raise TicketError("Telecom MCP service returned an invalid tool receipt", 502) from exc
        if (message.id != call.id or message.requestor != "assistant" or
                bool(message.error) != bool(response.is_error)):
            raise TicketError("Telecom MCP tool receipt does not match the request", 502)
        return message, payload.get("snapshot")

    def read(self, snapshot, call):
        if call.name not in READ_TOOLS:
            raise ValueError("Expected a read-only Telecom tool")
        message, _ = self.call(snapshot, call)
        return message

    def refuel(self, snapshot, call, quote_id):
        if call.name != REFUEL_TOOL or not quote_id:
            raise ValueError("Expected a confirmed refuel operation")
        message, updated = self.call(snapshot, call, quote_id)
        if not message.error and not isinstance(updated, str):
            raise TicketError("Telecom MCP service returned no updated account state", 502)
        return message, updated


@contextmanager
def connect_mcp(url, refuel=False):
    from anyio.from_thread import start_blocking_portal
    from mcp import Client

    with start_blocking_portal() as portal:
        with portal.wrap_async_context_manager(Client(url, read_timeout_seconds=15)) as client:
            yield TelecomMCP(portal, client, refuel)
