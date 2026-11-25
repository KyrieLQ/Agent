"""Derive a per-turn Telecom state from the messages the agent can see."""

from contextlib import contextmanager
from dataclasses import dataclass
import json
from threading import local
from unittest.mock import patch


LOOKUPS = {"get_customer_by_phone", "get_customer_by_id", "get_customer_by_name"}
WRITES = {"send_payment_request", "resume_line", "suspend_line", "enable_roaming",
          "disable_roaming", "refuel_data", "transfer_to_human_agents"}
FACT_KEYS = {"status", "account_status", "line_id", "bill_id", "total_due", "roaming_enabled",
             "data_used_gb", "data_limit_gb", "data_refueling_gb", "new_data_refueling_gb",
             "charge", "download_mbps", "upload_mbps", "success"}


def field(message, name, default=None):
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


@dataclass(frozen=True)
class ToolResult:
    index: int
    caller: str
    name: str
    arguments: dict
    content: str
    failed: bool

    @property
    def key(self):
        return self.name, json.dumps(self.arguments, sort_keys=True, separators=(",", ":"))


def tool_results(messages):
    pending = {}
    for index, message in enumerate(messages):
        role = field(message, "role")
        if role in {"assistant", "user"}:
            for call in field(message, "tool_calls") or ():
                caller = field(call, "requestor", role)
                if caller == role:
                    pending[caller, field(call, "id")] = call
        elif role == "tool":
            caller = field(message, "requestor")
            call = pending.pop((caller, field(message, "id")), None)
            if call is not None:
                content = field(message, "content") or ""
                failed = bool(field(message, "error")) or content.lstrip().lower().startswith("error:")
                yield ToolResult(index, caller, field(call, "name"), field(call, "arguments") or {}, content, failed)


def observations(content):
    def facts(record):
        return {key: value for key, value in record.items()
                if key in FACT_KEYS and isinstance(value, (int, float, bool, str))}

    try:
        result = json.loads(content)
    except (TypeError, ValueError):
        return content[:100]
    if isinstance(result, dict):
        fields = facts(result)
        if isinstance(result.get("line"), dict):
            fields.update(facts(result["line"]))
        return fields
    if isinstance(result, list):
        return {"records_returned": len(result), "recent_records": [facts(item) for item in result[-2:]
                                                                    if isinstance(item, dict)]}
    return str(result)[:100]


def entry(event):
    return {"tool": event.name, "arguments": event.arguments, "result": observations(event.content)}


def snapshot(messages):
    reads, changes = {}, {}
    lookup_done = False
    last_error = None
    opening = None
    last_user = None
    last_agent_text = None
    for event in tool_results(messages):
        if event.caller != "assistant":
            continue
        if event.failed:
            last_error = event
            continue
        if event.name in LOOKUPS:
            lookup_done = True
        destination = changes if event.name in WRITES else reads
        destination[event.key] = event

    for index, message in enumerate(messages):
        role, content = field(message, "role"), field(message, "content")
        if content and not field(message, "tool_calls"):
            if role == "user":
                last_user = (index, content[:180])
                if opening is None:
                    opening = last_user[1]
            elif role == "assistant":
                last_agent_text = (index, content[:180])

    latest_change = max(changes.values(), key=lambda result: result.index, default=None)
    review = "No service change recorded"
    if latest_change:
        if latest_change.name == "transfer_to_human_agents":
            review = "Transfer returned successfully"
        elif last_user and last_user[0] > latest_change.index:
            review = "Customer replied after the latest change. Check if the reply verifies the original issue."
        else:
            review = "No customer update since the latest change"

    return {
        "account_lookup": "completed" if lookup_done else "pending",
        "completed_checks": [entry(item) for item in sorted(reads.values(), key=lambda result: result.index)[-5:]],
        "completed_changes": [entry(item) for item in sorted(changes.values(), key=lambda result: result.index)[-4:]],
        "most_recent_tool_error": entry(last_error) if last_error else None,
        "opening_customer_message": opening,
        "last_agent_message": last_agent_text[1] if last_agent_text else None,
        "last_customer_update": last_user[1] if last_user else None,
        "resolution_check": review,
    }


class TelecomAgentProxy:
    def __init__(self, agent):
        self.agent = agent

    def __getattr__(self, name):
        return getattr(self.agent, name)

    @property
    def domain_policy(self):
        return self.agent.domain_policy

    @domain_policy.setter
    def domain_policy(self, value):
        self.agent.domain_policy = value


class TelecomStateAgent(TelecomAgentProxy):
    def __init__(self, agent, *, workflow=False):
        super().__init__(agent)
        self.state_turns = 0
        self.workflow = workflow
        self.workflow_steps = []
        self.workflow_corrections = []

    def generate_next_message(self, message, state):
        incoming = field(message, "tool_messages") or [message]
        history = [*state.messages, *incoming]
        recent = snapshot(history)
        extra = ""
        if self.workflow:
            from telecom_workflow import decide_workflow, workflow_correction

            step = decide_workflow(history)
            extra = "\n\nCurrent Telecom workflow step (apply the official policy; use one action at a time).\n" + step.prompt()
            self.workflow_steps.append(step.next_step)
        system = state.system_messages
        augmented = system[-1].model_copy(update={"content": (
            system[-1].content + "\n\nCurrent Telecom task state (observations from this conversation). "
            "Use it to track completed steps and check what remains under the official policy.\n"
            + json.dumps(recent, ensure_ascii=False, separators=(",", ":")) + extra
        )})
        state.system_messages = [*system[:-1], augmented]
        self.state_turns += 1
        try:
            response, next_state = self.agent.generate_next_message(message, state)
            if self.workflow and (correction := workflow_correction(step, response)):
                reason, guidance = correction
                if not next_state.messages or next_state.messages[-1] is not response:
                    raise RuntimeError("Cannot correct a proposal outside the agent's current message history")
                response = response.model_copy(update={"content": guidance, "tool_calls": None})
                next_state.messages[-1] = response
                self.workflow_corrections.append(reason)
            return response, next_state
        finally:
            state.system_messages = system


@contextmanager
def with_telecom_task_state(
    *, workflow=False, quota=False, quota_probe=False, quota_auto_read=False,
    quota_context_neutral=False, quota_post_mms=False, quota_mms_guide=False, quota_auto_price=False,
    quota_data_only=False, quota_early_mms=False
):
    """Install the state wrapper for the lifetime of one runner process or CLI call."""
    import tau2.runner

    build = tau2.runner.build_text_orchestrator
    current = local()

    def build_with_state(config, *args, **kwargs):
        orchestrator = build(config, *args, **kwargs)
        if config.domain == "telecom":
            if quota:
                from telecom_quota import TelecomQuotaAgent

                orchestrator.agent = TelecomQuotaAgent(
                    orchestrator.agent, probe=quota_probe, auto_read=quota_auto_read,
                    read_usage=orchestrator.environment.get_response if quota_context_neutral else None,
                    post_mms=quota_post_mms,
                    mms_guide=quota_mms_guide,
                    auto_price=quota_auto_price,
                    data_only=quota_data_only,
                    early_mms=quota_early_mms,
                )
            else:
                orchestrator.agent = TelecomStateAgent(orchestrator.agent, workflow=workflow)
            current.agent = orchestrator.agent
        return orchestrator

    with patch.object(tau2.runner, "build_text_orchestrator", build_with_state):
        yield current
