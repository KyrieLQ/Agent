"""Policy-based Telecom branches using only evidence visible to the agent."""

from dataclasses import asdict, dataclass
import json
import re

from telecom_state import WRITES, field, tool_results


TRANSFER_NOTICE = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
REPAIR_TOOLS = WRITES - {"send_payment_request", "transfer_to_human_agents"}
ISSUES = {
    "mms": (r"\bmms\b|\bpicture messages?\b|\bgroup messages?\b|\bsend(?:ing)? (?:photos|pictures)\b",
            "Try sending a picture message and tell me whether it goes through."),
    "data": (r"\bmobile data\b|\bcellular data\b|\binternet\b|\bdata speeds?\b",
             "Please run a mobile-data speed test and tell me the result."),
    "service": (r"\bno (?:cellular )?(?:service|signal)\b|\b(?:lost|restore) (?:my )?(?:phone )?service\b|\bline (?:is )?suspended\b",
                "Please check the status bar and tell me whether your phone has cellular service now."),
}
SIGNALS = {
    "airplane": (r"\bairplane mode (?:is |is currently |is now |now )?(?:turned |switched )?(on|off)\b",
                 {"on": "on", "off": "off"}),
    "mobile_data": (r"\b(?:mobile |cellular )?data (?:is |is currently |is now |now )?(?:turned |switched )?(on|off|enabled|disabled)\b",
                    {"on": "on", "enabled": "on", "off": "off", "disabled": "off"}),
    "sim": (r"\bsim(?: card)?(?: status)?\s*(?:is\s+|shows\s+|:\s*)?(active|missing|locked)\b",
            {"active": "active", "missing": "missing", "locked": "locked"}),
    "vpn": (r"\bvpn(?: status)?\s*(?:is\s+|is still\s+|is now\s+|:\s*)?(connected|disconnected|off)\b",
            {"connected": "connected", "disconnected": "off", "off": "off"}),
    "data_saver": (r"\bdata saver(?: mode)? (?:is |is currently |is now |now )?(?:turned )?(on|off|enabled|disabled)\b",
                    {"on": "on", "enabled": "on", "off": "off", "disabled": "off"}),
    "signal": (r"\b(no service|no signal|[1-4] bars?|(?:poor|fair|good|excellent) signal)\b",
               {"no service": "absent", "no signal": "absent"}),
}
SPEED = re.compile(r"(?:speed test|download speed)[^\n!?]{0,155}?\b(no connection|very poor|poor|fair|good|excellent)\b", re.I)
MMS_NEGATIVE = re.compile(r"\b(?:cannot|can.t|couldn.t|unable to|failed to|still can.t) send (?:an? )?(?:mms|picture messages?)\b", re.I)
MMS_POSITIVE = re.compile(r"\b(?:can now send|can successfully send|successfully sent|able to send) (?:an? )?(?:mms|picture messages?)\b", re.I)
DEVICE_CHANGE = re.compile(r"\b(?:i (?:have |just )?(?:turned|switched|restarted|rebooted|reset|reseated|disconnected|granted)|i.ve (?:turned|switched|restarted|rebooted|reset|reseated|disconnected|granted)|(?:mobile data|airplane mode|vpn) (?:is )?now (?:on|off|enabled|disabled|disconnected)|(?:vpn (?:has been |is )?disconnected successfully))\b", re.I)
BLOCKER = re.compile(r"\b(?:i (?:can.t|cannot|won.t|couldn.t) (?:do|use|find|run|turn|switch|reset|disconnect|reboot|reseat)|please (?:transfer|connect) me|(?:need|want) (?:to speak|to talk) to (?:a )?human)\b", re.I)
CLOSE_CLAIM = re.compile(r"(?:^|[.!?]\s+)(?:great[,!]\s+|glad to hear )?(?:the |your )?(?:original |mms |mobile data |service )?(?:issue|problem) (?:is now|is|has been) (?:fixed|resolved)\b|\b(?:i.ve|i have) (?:resolved|fixed) (?:the|your)\b", re.I)
IN_SCOPE = (
    ("airplane", "on", "Please turn off Airplane Mode and tell me what the status bar shows.", "turn_off_airplane"),
    ("sim", "missing", "Please reseat the SIM card, then tell me whether its status is Active.", "reseat_sim"),
    ("mobile_data", "off", "Please turn on mobile data, then tell me when it is enabled.", "enable_mobile_data"),
    ("data_saver", "on", "Please turn off Data Saver, then tell me when it is off.", "disable_data_saver"),
)
MMS_PERMISSION = re.compile(r"\b(storage|sms)(?: permission)?\s*(?:is\s+|:\s*)?(not granted|not allowed|denied|granted|allowed)\b")


@dataclass
class WorkflowStep:
    issue: str
    stage: str
    next_step: str
    instruction: str = ""
    evidence: str = ""
    can_correct_action: bool = False

    def prompt(self):
        fields = asdict(self)
        fields.pop("can_correct_action")
        return json.dumps(fields, ensure_ascii=False, separators=(",", ":"))


def _user_statements(messages):
    for index, message in enumerate(messages):
        if field(message, "role") == "user" and not field(message, "tool_calls"):
            text = field(message, "content") or ""
            for line in re.split(r"[\n]+|(?<=[.!?])\s+", text):
                line = line.strip().lower().replace("’", "'")
                if line and not line.endswith("?") and not line.startswith(("if ", "maybe ", "i wonder ")):
                    yield index, line


def _scope_exception(events):
    for event in reversed(events):
        if event.name != "get_details_by_id":
            continue
        try:
            result = json.loads(event.content)
        except (ValueError, TypeError):
            continue
        if isinstance(result, dict):
            line = result.get("line", result)
            if isinstance(line, dict) and line.get("status"):
                return str(line["status"]).lower() in {"suspended", "closed"}
    return False


def decide_workflow(messages):
    """Find one policy step from observed facts. Unknown facts leave the decision open."""
    events = [e for e in tool_results(messages) if e.caller == "assistant" and not e.failed]
    transfers = [e for e in events if e.name == "transfer_to_human_agents"]
    if transfers:
        notice_sent = any(field(m, "role") == "assistant" and TRANSFER_NOTICE in (field(m, "content") or "")
                          for m in messages[transfers[-1].index + 1:])
        if not notice_sent:
            return WorkflowStep("transfer", "handoff", "send_transfer_notice", TRANSFER_NOTICE,
                                "Transfer tool succeeded", True)
        return WorkflowStep("transfer", "handoff", "await_transfer", "Acknowledge briefly if needed. Do not make a second transfer call.")

    statements = list(_user_statements(messages))
    issue = next((kind for _, line in statements for kind, (pattern, _) in ISSUES.items()
                  if re.search(pattern, line)), "unknown")
    if issue == "unknown":
        return WorkflowStep(issue, "qualify", "clarify_symptom")
    test = ISSUES[issue][1]
    facts = {}
    last_repair = max((e.index for e in events if e.name in REPAIR_TOOLS), default=-1)
    reboot_required = max((e.index for e in events if e.name == "resume_line"), default=-1)
    last_reboot = -1
    last_retest = -1
    verified = False
    vpn_poor = False
    last_blocker = -1
    mms_checks = set()
    permissions = {}
    for index, line in statements:
        for name, (pattern, values) in SIGNALS.items():
            match = re.search(pattern, line)
            if match:
                facts[name] = (values.get(match[1], "present"), index)
        if BLOCKER.search(line):
            last_blocker = index
        if DEVICE_CHANGE.search(line):
            last_repair = max(last_repair, index)
            if re.search(r"\breset\b.*\bapn\b|\bapn\b.*\breset\b", line):
                reboot_required = max(reboot_required, index)
            if re.search(r"\b(?:rebooted|restarted)\b", line):
                last_reboot = index
        if "vpn" in line and re.search(r"\bperformance(?: level)?\b[^.!?]{0,50}\b(?:very poor|poor)\b", line):
            vpn_poor = True
        if issue == "mms":
            if re.search(r"\bapn name\b|\bmmsc (?:url|and apn)\b", line):
                mms_checks.add("apn")
            if re.search(r"\bwi[- ]?fi calling\b[^.!?]{0,45}\b(?:on|off|enabled|disabled)\b", line):
                mms_checks.add("wifi_calling")
            for permission in MMS_PERMISSION.finditer(line):
                permissions[permission[1]] = (permission[2] in {"granted", "allowed"}, index)
        if issue == "mms" and (MMS_POSITIVE.search(line) or MMS_NEGATIVE.search(line)):
            last_retest, verified = index, bool(MMS_POSITIVE.search(line) and not MMS_NEGATIVE.search(line))
        elif issue == "data" and (speed := SPEED.search(line)):
            last_retest, verified = index, speed[1] == "excellent"
        elif issue == "service" and ("status bar" in line or "service" in line):
            if re.search(r"\b(?:service (?:is )?(?:back|restored)|(?:can now|able to) make calls)\b", line):
                last_retest, verified = index, True
            elif re.search(r"\b(?:no service|no signal|still can.t make calls)\b", line):
                last_retest, verified = index, False

    looked_up = any(e.name in {"get_customer_by_phone", "get_customer_by_id", "get_customer_by_name"} for e in events)
    if not looked_up:
        return WorkflowStep(issue, "identify", "identify_customer", "Identify the customer from information already provided.")
    if facts.get("sim", (None, -1))[0] == "locked" or _scope_exception(events):
        return WorkflowStep(issue, "scope_check", "check_policy_handoff", "Check the official SIM or suspension policy for an applicable transfer or recovery route.")
    if reboot_required > last_reboot and last_retest < reboot_required:
        return WorkflowStep(issue, "repair", "reboot_after_change", "Please restart your phone and tell me once it is back on.",
                            "APN was reset or the line was resumed; the device must restart before retesting.", last_blocker < reboot_required)
    if last_retest >= max(0, last_repair) and verified:
        return WorkflowStep(issue, "verified", "confirm_and_close", "Acknowledge the successful retest and close naturally.")
    if last_repair > last_retest:
        return WorkflowStep(issue, "verify", "retest_original_symptom", test, "A corrective action was reported; no later symptom test is recorded.", last_blocker < last_repair)

    for name, value, action, step in IN_SCOPE:
        match = facts.get(name)
        if match and match[0] == value and (name not in {"mobile_data", "data_saver"} or issue != "service"):
            return WorkflowStep(issue, "repair", step, action, f"Customer reported {name}={value}", last_blocker < match[1])
    if issue == "data" and last_retest >= 0:
        vpn_state, seen = facts.get("vpn", (None, -1))
        if vpn_state == "connected" and vpn_poor:
            return WorkflowStep(issue, "repair", "disconnect_poor_vpn", "Please disconnect the VPN, then tell me when it is disconnected.",
                                "The customer reported a connected VPN with poor VPN performance.", last_blocker < seen)
        if vpn_state != "off" and "signal" in facts and last_blocker < last_retest:
            return WorkflowStep(issue, "diagnose", "check_vpn", "If its performance is not known, ask for the VPN status and performance.",
                                "The data retest did not meet the customer's goal.")
    if issue == "mms" and last_retest >= 0:
        for permission, (granted, seen) in permissions.items():
            if not granted and last_blocker < seen:
                return WorkflowStep(issue, "repair", "grant_messaging_permission",
                                    f"Please grant the messaging app its {permission} permission.",
                                    f"The customer reported an ungranted {permission} permission.", True)
        if facts.get("signal", (None, -1))[0] == "present" and facts.get("mobile_data", (None, -1))[0] == "on":
            if "apn" not in mms_checks:
                return WorkflowStep(issue, "diagnose", "check_mms_apn", "Ask for the APN and MMSC settings if they have not been checked.")
            if "wifi_calling" not in mms_checks:
                return WorkflowStep(issue, "diagnose", "check_wifi_calling", "Ask whether Wi-Fi Calling is on if it has not been checked.")
            if len(permissions) < 2:
                return WorkflowStep(issue, "diagnose", "check_messaging_permissions", "Ask which of the messaging app's storage and SMS permissions are granted.")
    fallback = {"mms": "Use the reported cellular and mobile-data evidence. Then check one relevant APN, network mode, Wi-Fi Calling, or messaging permission.",
                "data": "Use the latest speed and network results to pick an unchecked data, roaming, plan, or device branch.",
                "service": "Use the latest signal, airplane, SIM, and line status to choose the next service check."}
    prompt = fallback[issue] if last_retest >= 0 else test
    return WorkflowStep(issue, "diagnose", "choose_relevant_check" if last_retest >= 0 else "test_original_symptom", prompt)


def workflow_correction(step, response):
    """Correct high-confidence proposals before dispatch to the environment."""
    calls = field(response, "tool_calls") or ()
    content = field(response, "content") or ""
    if step.next_step == "send_transfer_notice" and (calls or content.strip() != TRANSFER_NOTICE):
        return "transfer_notice", step.instruction
    if step.can_correct_action and step.stage in {"repair", "verify"} and any(
        field(call, "name") == "transfer_to_human_agents" for call in calls
    ):
        return "premature_transfer", step.instruction
    if step.can_correct_action and step.stage == "verify" and not calls and CLOSE_CLAIM.search(content):
        return "unverified_resolution", step.instruction
    return None
