"""
src/detection/alert.py
Alert scoring, dispatch, and automated response hooks.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Callable

logger = logging.getLogger("mediguard.alerts")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

AUTO_RESPONSE_MAP = {
    "ransomware":        ["isolate_host", "block_ip", "notify_soc"],
    "brute_force":       ["block_ip", "revoke_session"],
    "lateral_movement":  ["block_ip", "notify_soc"],
    "ehr_unauthorized":  ["revoke_session", "lock_account", "notify_compliance"],
    "data_exfiltration": ["block_ip", "isolate_host", "notify_soc"],
    "iomt_compromise":   ["isolate_device", "notify_soc"],
    "attack":            ["notify_soc"],
}


@dataclass
class Alert:
    timestamp:      str
    severity:       str
    attack_type:    str
    src_ip:         str
    dst_ip:         str
    combined_score: float
    message:        str
    actions_taken:  list = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class AlertEngine:
    """
    Dispatches alerts and triggers automated response actions.

    Actions are pluggable callables — in production these would call
    firewall APIs, SIEM webhooks, or ticketing systems.
    """

    def __init__(self, min_severity: str = "LOW"):
        assert min_severity in SEVERITY_LEVELS, f"Invalid severity: {min_severity}"
        self.min_severity   = min_severity
        self._min_level     = SEVERITY_LEVELS.index(min_severity)
        self._alert_log: List[Alert] = []
        self._action_hooks: dict[str, Callable] = {}
        self._register_default_hooks()

    def _register_default_hooks(self):
        """Register stub action handlers (replace with real integrations)."""
        for action in ["block_ip", "isolate_host", "revoke_session",
                       "lock_account", "notify_soc", "notify_compliance",
                       "isolate_device"]:
            self._action_hooks[action] = self._stub_action(action)

    def _stub_action(self, name: str) -> Callable:
        def action(alert: Alert):
            logger.info(f"[ACTION:{name}] src={alert.src_ip} severity={alert.severity}")
        return action

    def register_hook(self, action_name: str, fn: Callable):
        """Register a real action handler to override the stub."""
        self._action_hooks[action_name] = fn

    def dispatch(self, alert: Alert):
        """Process an alert: log it and trigger appropriate auto-responses."""
        level = SEVERITY_LEVELS.index(alert.severity) \
                if alert.severity in SEVERITY_LEVELS else 0

        if level < self._min_level:
            return

        self._alert_log.append(alert)
        logger.warning(alert.message)

        if level >= SEVERITY_LEVELS.index("HIGH"):
            self._auto_respond(alert)

    def _auto_respond(self, alert: Alert):
        actions = AUTO_RESPONSE_MAP.get(alert.attack_type, ["notify_soc"])
        taken = []
        for action in actions:
            fn = self._action_hooks.get(action)
            if fn:
                try:
                    fn(alert)
                    taken.append(action)
                except Exception as e:
                    logger.error(f"Action {action} failed: {e}")
        alert.actions_taken = taken

    def get_recent_alerts(self, n: int = 50) -> List[dict]:
        return [a.to_dict() for a in self._alert_log[-n:]]

    def get_stats(self) -> dict:
        if not self._alert_log:
            return {"total": 0}
        from collections import Counter
        severities  = Counter(a.severity    for a in self._alert_log)
        attack_types= Counter(a.attack_type for a in self._alert_log)
        return {
            "total":        len(self._alert_log),
            "severities":   dict(severities),
            "attack_types": dict(attack_types),
        }

    def clear(self):
        self._alert_log.clear()
