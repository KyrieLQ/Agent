"""Serve native Telecom tools over MCP Streamable HTTP."""

import argparse
from collections.abc import Mapping
from contextlib import asynccontextmanager, closing, contextmanager
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from anyio import to_thread
from mcp import types
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from tau2.data_model.message import ToolCall
import uvicorn

from telecom_mcp import READ_TOOLS, REFUEL_TOOL, TOOL_CONTEXT, native_specs
from telecom_refuel import dump_environment, restore_environment


class RefuelOperations:
    def __init__(self, path):
        self.path = path
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS operations (
                quote_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, result TEXT NOT NULL
            )""")

    @contextmanager
    def connect(self):
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.execute("PRAGMA synchronous=FULL")
            yield db

    def apply(self, snapshot, call, quote_id):
        raw_request = json.dumps([snapshot, call.id, call.arguments], sort_keys=True)
        request_hash = sha256(raw_request.encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cached = db.execute("SELECT request_hash, result FROM operations WHERE quote_id = ?", (quote_id,)).fetchone()
            if cached is not None:
                if cached[0] != request_hash:
                    raise ValueError("Confirmation ID was already used with a different request")
                return json.loads(cached[1])
            environment = restore_environment(snapshot)
            message = environment.get_response(call)
            result = {"message": message.model_dump(mode="json")}
            if not message.error:
                result["snapshot"] = dump_environment(environment)
                db.execute("INSERT INTO operations VALUES (?, ?, ?)", (quote_id, request_hash, json.dumps(result)))
            return result


def error_result(text):
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=True)


def create_application(state_dir, with_refuel):
    specs = native_specs()
    enabled = READ_TOOLS | ({REFUEL_TOOL} if with_refuel else set())
    operations = RefuelOperations(state_dir / "operations.sqlite3") if with_refuel else None
    catalog = [types.Tool(name=name, description=specs[name]["description"], input_schema=specs[name]["parameters"],
                          annotations=types.ToolAnnotations(read_only_hint=name in READ_TOOLS))
               for name in sorted(enabled)]

    async def list_tools(_context, _params):
        return types.ListToolsResult(tools=catalog)

    async def call_tool(context, params):
        if params.name not in enabled:
            return error_result("Unknown Telecom MCP tool")
        meta = context.meta
        if meta is not None and not isinstance(meta, Mapping):
            meta = meta.model_extra
        payload = (meta or {}).get(TOOL_CONTEXT)
        if not isinstance(payload, dict) or not all(isinstance(payload.get(key), str) and payload[key]
                                                   for key in ("snapshot", "tool_call_id")):
            return error_result("This tool requires an active Telecom work order")
        tool = ToolCall(id=payload["tool_call_id"], name=params.name, requestor="assistant", arguments=params.arguments or {})
        snapshot = payload["snapshot"]
        try:
            if params.name == REFUEL_TOOL:
                quote_id = payload.get("quote_id")
                if not isinstance(quote_id, str) or not quote_id:
                    return error_result("Refueling requires a confirmed quote")
                result = await to_thread.run_sync(partial(operations.apply, snapshot, tool, quote_id))
            else:
                def lookup():
                    return restore_environment(snapshot).get_response(tool).model_dump(mode="json")

                result = {"message": await to_thread.run_sync(lookup)}
        except ValueError as exc:
            return error_result(str(exc))
        message = result["message"]
        return types.CallToolResult(content=[types.TextContent(type="text", text=message["content"] or "")],
                                    structured_content=result, is_error=bool(message["error"]))

    server = Server("Telecom tools", on_list_tools=list_tools, on_call_tool=call_tool)
    manager = StreamableHTTPSessionManager(app=server, stateless=True, json_response=True)

    @asynccontextmanager
    async def lifespan(_app):
        async with manager.run():
            yield

    async def health(_request):
        return JSONResponse({"tools": sorted(enabled)})

    return Starlette(routes=[Route("/health", health), Mount("/mcp", app=manager.handle_request)], lifespan=lifespan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--with-refuel", action="store_true")
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    app = create_application(args.state_dir, args.with_refuel)
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
