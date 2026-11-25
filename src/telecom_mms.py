"""Select the next MMS diagnosis from the customer's observed device state."""

import re

from telecom_workflow import MMS_NEGATIVE, MMS_POSITIVE, SIGNALS, SPEED, _user_statements


MMS_NOT_WORKING = re.compile(
    r"\b(?:mms|picture messages?|picture messaging|the picture) (?:still )?"
    r"(?:isn't working|is not working|doesn't work|does not work|won't send|failed to send)\b"
)
MMS_WORKING = re.compile(r"\b(?:the (?:test )?picture (?:message )?went through|mms now works)\b")
WIFI_CALLING = re.compile(
    r"\bwi[- ]?fi calling\s*[:=-]?\s*(?:is |is currently |is now |now )?"
    r"(?:turned |switched )?(on|off|enabled|disabled)\b"
)
WIFI_CHANGE = re.compile(r"\b(?:turned|switched) (on|off) wi[- ]?fi calling\b")
NETWORK = re.compile(
    r"\b(?:network(?: type)?\s*[:=]\s*|connected to |on |have )"
    r"(?:an? |excellent |strong |good )?(2g|3g|4g|5g|lte)\b"
)
MMSC = re.compile(
    r"\bmmsc(?: url)?(?: \([^)]{1,40}\))?\s*[:=-]?\s*"
    r"(?:is |shows? (?:as )?)?(?:now |properly )?[\"']?"
    r"(not configured|not set|missing|empty|blank|none|null|configured|present|set|https?://)"
)
PERMISSION = re.compile(
    r"\b(storage|sms)(?: permission)?\s*[:=-]?\s*"
    r"(?:is |has been )?(not granted|not allowed|denied|granted|allowed)\b"
)
GRANTED = re.compile(r"\bgranted (?:the )?(storage|sms) permission\b")
HOLD_BACK = re.compile(
    r"\b(?:no thanks|don't want|do not want|transfer me|connect me|"
    r"(?:talk|speak) to (?:a )?human|already (?:did|checked|tried))\b"
)


def next_mms_step(messages, checked, after_device_state=-1, *, early_checks=False):
    """Return a single grounded customer action, or leave the static agent in control."""
    if sum(name.startswith("mms_guide_") for name in checked) >= 7:
        return None
    observations = list(_user_statements(messages))
    latest = len(messages) - 1
    current = [line for index, line in observations if index == latest]
    if not current or any(HOLD_BACK.search(line) for line in current):
        return None

    wifi = mmsc = network = None
    mobile_data = signal = None
    permissions = {}
    outcome = None
    diagnosis_seen = -1
    reset = reboot = test = -1
    repair_confirmed = -1
    for index, line in observations:
        if MMS_NEGATIVE.search(line) or MMS_NOT_WORKING.search(line):
            outcome, test = "failed", index
        elif MMS_POSITIVE.search(line) or (early_checks and MMS_WORKING.search(line)):
            outcome, test = "working", index
        if hit := re.search(SIGNALS["mobile_data"][0], line):
            mobile_data = SIGNALS["mobile_data"][1].get(hit[1])
        if hit := re.search(SIGNALS["signal"][0], line):
            signal = SIGNALS["signal"][1].get(hit[1], "present")
        if index >= after_device_state:
            if hit := SPEED.search(line):
                network = "available" if hit[1] != "no connection" else "unavailable"
            elif hit := NETWORK.search(line):
                network = "2g" if hit[1] == "2g" else "available"
        if match := WIFI_CALLING.search(line) or WIFI_CHANGE.search(line):
            wifi = match[1] in {"on", "enabled"}
            diagnosis_seen = index
            if wifi is False and "mms_guide_disable_wifi_calling" in checked:
                repair_confirmed = index
        for permission in PERMISSION.finditer(line):
            permissions[permission[1]] = permission[2] in {"granted", "allowed"}
            diagnosis_seen = index
            if permissions[permission[1]] and f"mms_guide_grant_{permission[1]}" in checked:
                repair_confirmed = index
        for permission in GRANTED.finditer(line):
            permissions[permission[1]] = True
            diagnosis_seen = index
            if f"mms_guide_grant_{permission[1]}" in checked:
                repair_confirmed = index
        if match := MMSC.search(line):
            mmsc = match[1] in {"set", "configured", "present", "http://", "https://"}
            diagnosis_seen = index
        if re.search(r"\b(?:i.ve|i have|i) reset (?:my |the )?apn settings\b", line):
            reset = index
        if re.search(r"\b(?:i.ve|i have|i) (?:rebooted|restarted)\b", line):
            reboot = index

    if network is None and signal == "present" and mobile_data == "on":
        network = "available"
    if outcome != "failed" or signal == "absent" or mobile_data == "off":
        return None
    if latest not in {test, diagnosis_seen, reset, reboot}:
        return None

    if early_checks and network == "available" and repair_confirmed > test:
        step = "retest"
        reply = "Please try sending a picture message in your default messaging app and tell me whether it goes through."
    elif network == "2g":
        step = "change_network"
        reply = "Please select a preferred network mode that includes 3G, 4G or 5G."
    elif network != "available":
        return None
    elif reset > reboot and reset >= test:
        step = "restart_after_apn"
        reply = "Please restart your phone so the default APN settings are applied."
    elif wifi is True:
        step = "disable_wifi_calling"
        reply = "Please turn off Wi-Fi Calling, then try sending a picture message in the default messaging app."
    elif missing := next((key for key in ("storage", "sms") if permissions.get(key) is False), None):
        step = f"grant_{missing}"
        reply = f"Please grant the default messaging app its {missing.upper()} permission, then try sending a picture message."
    elif mmsc is False and reset < 0:
        step = "reset_apn"
        reply = "Please reset the APN settings to the carrier defaults."
    elif wifi is None:
        step = "check_wifi_calling"
        reply = "Could you check whether Wi-Fi Calling is currently on or off?"
    elif len(permissions) < 2:
        step = "check_permissions"
        reply = "Please check whether your default messaging app has both Storage and SMS permissions."
    elif mmsc is None:
        step = "check_mmsc"
        reply = "Please check whether the MMSC URL is present in the APN settings."
    else:
        step = "retest"
        reply = "Please try sending a picture message in your default messaging app and tell me whether it goes through."

    name = "mms_guide_" + step
    if early_checks:
        if step == "disable_wifi_calling":
            reply = "Please turn Wi-Fi Calling off and tell me whether the setting now says Off."
        elif step.startswith("grant_"):
            reply = f"Please grant the default messaging app {missing.upper()} permission and tell me once it is granted."
        elif step not in {"check_wifi_calling", "check_permissions", "retest"}:
            return None
    return None if name in checked else (name, reply)
