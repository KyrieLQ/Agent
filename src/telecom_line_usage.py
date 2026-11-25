"""Attach verified quota to the existing line-detail tool response."""

import json
import re

from tau2.data_model.message import ToolCall

from telecom_quota import PHONE, number, recent_users
from telecom_state import LOOKUPS, tool_results
from telecom_workflow import ISSUES


LINE_ID = re.compile(r"\bL\d+\b", re.IGNORECASE)
SYMPTOMS = {name: re.compile(pattern, re.IGNORECASE) for name, (pattern, _) in ISSUES.items()}
USAGE_FIELD = "billing_cycle_usage"


def record(content):
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def affected_line(users, line):
    phone_match = PHONE.search(str(line.get("phone_number", "")))
    phone = "".join(phone_match.groups()) if phone_match else None
    line_id = line["line_id"].upper()
    for text in reversed(users):
        phones = {"".join(match.groups()) for match in PHONE.finditer(text)}
        lines = {match.group().upper() for match in LINE_ID.finditer(text)}
        if phones or lines:
            return (not phones or phones == {phone}) and (not lines or lines == {line_id})
    return False


def usage_facts(content, line_id):
    usage = record(content)
    if usage.get("line_id") != line_id:
        return None
    used = number(usage.get("data_used_gb"))
    limit = number(usage.get("data_limit_gb"))
    refueled = number(usage.get("data_refueling_gb"))
    if used is None or limit is None or refueled is None:
        return None
    allowance = limit + refueled
    return {
        "data_used_gb": str(used),
        "data_limit_gb": str(limit),
        "data_refueling_gb": str(refueled),
        "remaining_data_gb": str(max(allowance - used, 0)),
        "data_usage_exceeded": used >= allowance,
        "cycle_end_date": usage.get("cycle_end_date"),
    }


def enrich_line_usage(history, call, result, read_usage):
    """Use the same callable for the service and the benchmark agent's tool loop."""
    if call.name != "get_details_by_id" or result.error:
        return result
    line = record(result.content)
    line_id = line.get("line_id")
    if not isinstance(line_id, str) or line_id != call.arguments.get("id") or line.get("status") != "Active":
        return result
    users = recent_users(history)
    issue = next((name for text in users for name, pattern in SYMPTOMS.items() if pattern.search(text)), None)
    if issue not in {"data", "mms"} or not affected_line(users, line):
        return result

    customer_ids = set()
    has_usage = False
    for event in tool_results(history):
        if event.caller != "assistant" or event.failed:
            continue
        if event.name in LOOKUPS:
            account = record(event.content)
            customer_id, lines = account.get("customer_id"), account.get("line_ids")
            if isinstance(customer_id, str) and isinstance(lines, list) and line_id in lines:
                customer_ids.add(customer_id)
        elif event.name == "get_details_by_id" and event.arguments.get("id") == line_id:
            has_usage = has_usage or USAGE_FIELD in record(event.content)
        elif event.arguments.get("line_id") == line_id:
            if event.name == "get_data_usage" and usage_facts(event.content, line_id) is not None:
                has_usage = True
            elif event.name == "refuel_data":
                has_usage = False

    if len(customer_ids) != 1 or has_usage:
        return result
    lookup = ToolCall(id=f"{call.id}-usage", name="get_data_usage", requestor="assistant",
                      arguments={"customer_id": customer_ids.pop(), "line_id": line_id})
    usage = read_usage(lookup)
    facts = None if usage.error else usage_facts(usage.content, line_id)
    if facts is None:
        return result
    return result.model_copy(update={"content": json.dumps({**line, USAGE_FIELD: facts}, separators=(",", ":"))})
