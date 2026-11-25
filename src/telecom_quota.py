"""Guide the Telecom agent when its own tools confirm exhausted data."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import re

from telecom_state import LOOKUPS, TelecomAgentProxy, ToolResult, field, tool_results
from telecom_mms import next_mms_step
from telecom_workflow import ISSUES, MMS_NEGATIVE, MMS_PERMISSION, MMS_POSITIVE, SIGNALS, SPEED, _user_statements


DECLINES = re.compile(
    r"\b(?:no thanks|no,? thank you|i (?:do not|don't) (?:want|agree)|"
    r"i (?:would|i'd) rather not|i (?:decline|refuse)|please (?:do not|don't) "
    r"(?:add|charge))\b", re.IGNORECASE
)
BUYING = re.compile(r"\b(?:please|go ahead and) (?:add|buy|refuel|purchase)\b", re.IGNORECASE)
PHONE = re.compile(r"(?<!\d)(?:\+1[- .]?)?\(?(\d{3})\)?[- .]?(\d{3})[- .]?(\d{4})(?!\d)")
WIFI_CALLING = re.compile(r"\bwi[- ]?fi calling (?:is |is currently |is now |now )?(on|off|enabled|disabled)\b")
CONNECTED = re.compile(r"\b(?:connected to|on|have) (?:an? |excellent |strong |good )?(?:3g|4g|5g|lte)\b")
MMSC_MISSING = re.compile(r"\bmmsc(?: url)? (?:is )?(?:missing|empty|not set|blank)\b")
MMSC_PRESENT = re.compile(r"\bmmsc(?: url)? (?:is )?(?:set|configured|present|https?://)")
HUMAN_REQUEST = re.compile(r"\b(?:transfer|connect) me\b|\b(?:talk|speak) to (?:a )?human\b")


def details(event):
    try:
        result = json.loads(event.content)
    except (TypeError, ValueError):
        return {}
    return result if isinstance(result, dict) else {}


def number(value):
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        return None
    return result if result.is_finite() and result >= 0 else None


def recent_users(messages, after=-1):
    return [field(m, "content") or "" for i, m in enumerate(messages)
            if i > after and field(m, "role") == "user" and not field(m, "tool_calls")]


@dataclass(frozen=True)
class QuotaStep:
    name: str
    guidance: str
    tool_arguments: dict | None = None
    reply: str | None = None
    tool_name: str = "get_data_usage"


def after_refuel_mms(messages, purchase, completed):
    """Resume the official MMS checks after a failed test with restored data connectivity."""
    latest = len(messages) - 1
    seen_steps = {step for step in completed if step.startswith("mms_post_")}
    if len(seen_steps) >= 2 or latest <= purchase.index:
        return None
    statements = list(_user_statements(messages))
    current = [line for index, line in statements if index == latest]
    if not any(MMS_NEGATIVE.search(line) for line in current):
        return None
    if any(HUMAN_REQUEST.search(line) or (MMS_POSITIVE.search(line) and not MMS_NEGATIVE.search(line))
           for line in current):
        return None
    network_ok = wifi = mmsc = None
    permissions = {}
    for index, line in statements:
        if index > purchase.index:
            speed = SPEED.search(line)
            if speed:
                network_ok = speed[1] != "no connection"
            elif CONNECTED.search(line):
                network_ok = True
        if match := WIFI_CALLING.search(line):
            wifi = match[1] in {"on", "enabled"}
        for permission in MMS_PERMISSION.finditer(line):
            permissions[permission[1]] = permission[2] in {"allowed", "granted"}
        if MMSC_MISSING.search(line):
            mmsc = False
        elif MMSC_PRESENT.search(line):
            mmsc = True
    if not network_ok:
        return None
    if wifi is not False:
        if wifi:
            step = "mms_post_disable_wifi_calling"
            reply = "Please turn off Wi-Fi Calling, then try sending a picture message in your default messaging app."
        else:
            step = "mms_post_check_wifi_calling"
            reply = "Could you check whether Wi-Fi Calling is currently on or off?"
    elif not all(permissions.get(p) for p in ("storage", "sms")):
        step = "mms_post_check_permissions"
        reply = "Please check whether your default messaging app has both Storage and SMS permissions."
    elif mmsc is not True:
        step = "mms_post_check_mmsc"
        reply = "Please check whether the MMSC URL is present in the APN settings."
    else:
        return None
    return None if step in seen_steps else QuotaStep(step, "", reply=reply)


def quota_probe(messages, issue, events, *, auto_read=False):
    """Request one read-only usage check for an unresolved symptom on an identified line."""
    users = recent_users(messages)
    if any(DECLINES.search(text) for text in users[-2:]):
        return None
    phones = {"".join(parts) for text in users for parts in PHONE.findall(text)}
    line_ids = set()
    for event in events:
        if event.name != "get_details_by_id":
            continue
        line = details(event)
        if isinstance(line.get("line"), dict):
            line = line["line"]
        line_id = event.arguments.get("id")
        if "data_used_gb" not in line or str(line.get("status", "")).lower() != "active":
            continue
        if not isinstance(line_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", line_id):
            continue
        phone = PHONE.search(str(line.get("phone_number", "")))
        if (phone and "".join(phone.groups()) in phones) or any(
            re.search(r"\b" + re.escape(line_id) + r"\b", text, re.I) for text in users
        ):
            line_ids.add(line_id)
    if len(line_ids) != 1:
        return None

    facts = {}
    unresolved = False
    for _, line in _user_statements(messages):
        for key in ("signal", "mobile_data"):
            pattern, values = SIGNALS[key]
            if match := re.search(pattern, line):
                facts[key] = values.get(match[1], "present")
        if issue == "mms":
            if MMS_NEGATIVE.search(line):
                unresolved = True
            elif MMS_POSITIVE.search(line):
                unresolved = False
        elif speed := SPEED.search(line):
            unresolved = speed[1] != "excellent"
    if not unresolved or facts.get("signal") != "present" or facts.get("mobile_data") != "on":
        return None
    line_id = next(iter(line_ids))
    if auto_read:
        customers = set()
        for event in events:
            if event.name not in LOOKUPS:
                continue
            account = details(event)
            customer_id = account.get("customer_id")
            owned_lines = account.get("line_ids")
            if (isinstance(customer_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", customer_id)
                    and isinstance(owned_lines, list) and line_id in owned_lines):
                customers.add(customer_id)
        if len(customers) != 1:
            return None
        return QuotaStep("read_usage", "", {
            "customer_id": next(iter(customers)), "line_id": line_id,
        })
    return QuotaStep("check_usage",
                     f"The reported symptom persists with cellular signal and mobile data enabled. "
                     f"Use get_data_usage for the identified line {line_id} and its customer ID "
                     "to check remaining allowance before continuing device troubleshooting.")


def quota_step(
    messages, *, allow_probe=False, auto_read=False, usage_evidence=None,
    post_mms_steps=None, mms_guide_steps=None, allow_price_read=False, price_evidence=None,
    data_only=False, early_mms_steps=None
):
    users = recent_users(messages)
    issue = next((kind for text in users for kind in ("mms", "data")
                  if re.search(ISSUES[kind][0], text, re.IGNORECASE)), None)
    if issue is None:
        return None
    if early_mms_steps is not None:
        if issue != "mms":
            return None
        decision = next_mms_step(messages, early_mms_steps, early_checks=True)
        return QuotaStep(decision[0], "", reply=decision[1]) if decision else None
    if data_only and (issue != "data" or not re.search(ISSUES["data"][0], users[0], re.IGNORECASE)):
        return None
    all_events = [e for e in tool_results(messages) if e.caller == "assistant"]
    if usage_evidence is not None:
        all_events.append(usage_evidence)
        all_events.sort(key=lambda event: event.index)
    events = [e for e in all_events if not e.failed]
    usage = next((e for e in reversed(events) if e.name == "get_data_usage"), None)
    if usage is None:
        if allow_probe and not any(e.name in {"get_data_usage", "refuel_data"} for e in all_events):
            return quota_probe(messages, issue, events, auto_read=auto_read)
        return None
    body = details(usage)
    line_id = usage.arguments.get("line_id")
    if not line_id or body.get("line_id") != line_id:
        return None
    used, limit, added = (number(body.get(k)) for k in (
        "data_used_gb", "data_limit_gb", "data_refueling_gb"
    ))
    if used is None or limit is None or added is None:
        return None
    if used <= limit + added:
        if issue == "mms" and mms_guide_steps is not None:
            decision = next_mms_step(messages, mms_guide_steps)
            if decision:
                return QuotaStep(decision[0], "", reply=decision[1])
        return None

    purchase = next((e for e in reversed(events) if e.name == "refuel_data" and e.arguments.get("line_id") == line_id), None)
    if purchase is not None:
        if issue == "mms" and mms_guide_steps is not None:
            decision = next_mms_step(messages, mms_guide_steps, after_device_state=purchase.index)
            if decision:
                return QuotaStep(decision[0], "", reply=decision[1])
        if purchase.index < usage.index:
            return None
        for text in recent_users(messages, purchase.index):
            if ((issue == "mms" and (MMS_POSITIVE.search(text) or MMS_NEGATIVE.search(text)))
                    or (issue == "data" and SPEED.search(text))):
                if issue == "mms" and post_mms_steps is not None:
                    return after_refuel_mms(messages, purchase, post_mms_steps)
                return None
        return QuotaStep("verify_after_refuel", ISSUES[issue][1])

    declined = False
    for text in recent_users(messages, usage.index):
        if DECLINES.search(text):
            declined = True
        elif BUYING.search(text):
            declined = False
        if issue == "data" and (result := SPEED.search(text)) and result[1].lower() == "excellent":
            return None
        if issue == "mms" and MMS_POSITIVE.search(text) and not MMS_NEGATIVE.search(text):
            return None
    if declined:
        return None

    reads = {e.arguments.get("id"): details(e) for e in events if e.name == "get_details_by_id"}
    line = reads.get(line_id, {})
    if isinstance(line.get("line"), dict):
        line = line["line"]
    if str(line.get("status", "")).lower() in {"closed", "suspended", "pending activation"}:
        return None
    plan_id = line.get("plan_id")
    plan = reads.get(plan_id, {})
    rate = number(plan.get("data_refueling_price_per_gb"))
    if price_evidence and price_evidence.arguments.get("id") == plan_id and not price_evidence.failed:
        receipt = details(price_evidence)
        if receipt.get("plan_id") == plan_id:
            rate = number(receipt.get("data_refueling_price_per_gb"))
    prefix = f"The affected line used {used} GB against an effective quota of {limit + added} GB. "
    if rate is None:
        if (allow_price_read and line.get("line_id") == line_id
                and str(line.get("status", "")).lower() == "active"
                and isinstance(plan_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", plan_id)):
            return QuotaStep("inspect_rate", "", {"id": plan_id}, tool_name="get_details_by_id")
        return QuotaStep("check_price", prefix + "Look up the affected line's plan and its refueling rate.")
    return QuotaStep("quote_and_refuel", prefix + f"The verified plan rate is {rate} USD per GB. "
                     "Confirm the requested amount, up to 2 GB, and its total cost with the customer. "
                     "If already agreed in the conversation, call refuel_data once for the same line and agreed amount. "
                     "If the customer declines, continue with the official policy.")


class TelecomQuotaAgent(TelecomAgentProxy):
    def __init__(
        self, agent, *, probe=False, auto_read=False, read_usage=None, post_mms=False,
        mms_guide=False, auto_price=False, data_only=False, early_mms=False
    ):
        super().__init__(agent)
        self.probe = probe
        self.auto_read = auto_read
        self.read_usage = read_usage
        self.post_mms = post_mms
        self.mms_guide = mms_guide
        self.auto_price = auto_price
        self.data_only = data_only
        self.early_mms = early_mms
        self.quota_steps = []
        self.quota_reads = []
        self.usage_evidence = None
        self.price_evidence = None

    def inspect_tool(self, call, index):
        response = self.read_usage(call)
        evidence = ToolResult(index, "assistant", call.name, call.arguments, response.content, response.error)
        self.quota_reads.append({
            "name": call.name, "arguments": call.arguments,
            "response": details(evidence) or response.content,
            "failed": response.error, "message_index": index,
        })
        return evidence

    def generate_next_message(self, message, state):
        incoming = field(message, "tool_messages") or [message]
        messages = [*state.messages, *incoming]
        probed = any(step in {"check_usage", "read_usage", "inspect_usage"} for step in self.quota_steps)
        options = {
            "post_mms_steps": self.quota_steps if self.post_mms else None,
            "mms_guide_steps": self.quota_steps if self.mms_guide else None,
            "data_only": self.data_only,
            "early_mms_steps": self.quota_steps if self.early_mms else None,
        }
        def decide(allow_probe=False):
            return quota_step(
                messages, allow_probe=allow_probe, auto_read=self.auto_read,
                usage_evidence=self.usage_evidence, price_evidence=self.price_evidence,
                allow_price_read=self.auto_price and self.read_usage is not None and self.price_evidence is None,
                **options,
            )

        step = decide(self.probe and not probed)
        while step is not None and step.tool_arguments is not None:
            if not any(tool.name == step.tool_name for tool in self.agent.tools):
                return self.agent.generate_next_message(message, state)
            from tau2.data_model.message import AssistantMessage, ToolCall
            suffix = "rate" if step.name == "inspect_rate" else "usage"
            call = ToolCall(id=f"call_telecom_quota_{suffix}", name=step.tool_name, arguments=step.tool_arguments)
            if self.read_usage is None:
                reply = AssistantMessage(role="assistant", tool_calls=[call])
                self.quota_steps.append(step.name)
                state.messages.extend([*incoming, reply])
                return reply, state
            self.quota_steps.append(f"inspect_{suffix}")
            evidence = self.inspect_tool(call, len(messages) - 1)
            if suffix == "rate":
                self.price_evidence = evidence
            else:
                self.usage_evidence = evidence
            step = decide()
        if step is None:
            return self.agent.generate_next_message(message, state)
        self.quota_steps.append(step.name)
        if step.reply is not None:
            from tau2.data_model.message import AssistantMessage
            reply = AssistantMessage.text(step.reply)
            state.messages.extend([*incoming, reply])
            return reply, state
        system = state.system_messages
        heading = "Telecom diagnostic check" if step.name == "check_usage" else "Confirmed Telecom quota workflow"
        state.system_messages = [*system[:-1], system[-1].model_copy(update={
            "content": system[-1].content + "\n\n" + heading + ": " + step.guidance
        })]
        try:
            return self.agent.generate_next_message(message, state)
        finally:
            state.system_messages = system
