import re

path = "decision_engine.py"
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update DECISIONS map
code = re.sub(
    r'DECISIONS = \{.*?\n\}',
    '''DECISIONS = {
    "ESCALATE":    {"severity": "CRITICAL", "color": "#FF3B3B"},
    "SCALE":       {"severity": "HIGH",     "color": "#FF8C00"},
    "SCALE_READY": {"severity": "WARNING",  "color": "#FFD700"},
    "MONITOR":     {"severity": "WARNING",  "color": "#FFD700"},
    "STABLE":      {"severity": "INFO",     "color": "#00C851"},
}''',
    code,
    flags=re.DOTALL
)

# 2. Update _purge
code = re.sub(
    r'def _purge\(self, server_id: str, now: datetime\):.*?(?=\n\n|\n#)',
    '''def _purge(self, server_id: str, now: datetime):
        if server_id in self._events:
            threshold = now - timedelta(minutes=self._window_minutes)
            self._events[server_id] = [
                t for t in self._events[server_id]
                if t >= threshold
            ]''',
    code,
    flags=re.DOTALL
)

# 3. Update ESCALATION_ORDER
code = code.replace(
    'ESCALATION_ORDER = {"STABLE": 0, "MONITOR": 1, "RESTART": 2, "SCALE": 3, "ESCALATE": 4}',
    'ESCALATION_ORDER = {"STABLE": 0, "MONITOR": 1, "SCALE_READY": 2, "SCALE": 3, "ESCALATE": 4}'
)

# 4. Update init with cpu tracking
code = code.replace(
    '''        self._session_start        = time.time()
        # Evaluation metrics''',
    '''        self._session_start        = time.time()
        
        # CPU tracking for sustained time-based rules
        self._cpu_buffer: dict[str, list[tuple[float, float]]] = {}
        self._last_action_time: dict[str, float] = {}
        
        # Evaluation metrics'''
)

code = re.sub(
    r'        self._restart_exit_windows: dict\[str, int\] = \{\}',
    '''        self._scale_exit_windows: dict[str, int] = {}

    def _update_cpu_buffer(self, server_id: str, current_cpu: float):
        now = time.time()
        with self._lock:
            if server_id not in self._cpu_buffer:
                self._cpu_buffer[server_id] = []
            self._cpu_buffer[server_id].append((now, current_cpu))
            # Keep 120 seconds of history
            self._cpu_buffer[server_id] = [(t, c) for t, c in self._cpu_buffer[server_id] if now - t <= 120]

    def _sustained_above(self, server_id: str, threshold: float, duration: float) -> bool:
        now = time.time()
        with self._lock:
            buf = self._cpu_buffer.get(server_id, [])
            if not buf:
                return False
            oldest_above = None
            for t, c in reversed(buf):
                if c > threshold:
                    oldest_above = t
                else:
                    break
            if oldest_above is not None and (now - oldest_above) >= duration:
                return True
            return False''',
    code
)

# 5. build_explanation change RESTART
code = re.sub(
    r'    elif decision == "RESTART":.*?(?=\n    elif decision == "MONITOR":)',
    '''    elif decision == "SCALE_READY":
        reason = (
            f"Preemptive staging under {load_state} ({trend_str}). "
            f"Predicted CPU {corrected_pred:.1f}% exceeds 90% threshold. "
            f"Preparing to scale if trajectory continues."
        )
        action = "Staging autoscale resources — pending hard trigger"
''',
    code,
    flags=re.DOTALL
)

# 6. _decide logic replacement
code = re.sub(
    r'        # ── Hysteresis — read last decision ────────────────────.*?(?=\n        # ── Action confidence — how confident is the decision itself\? ──)',
    '''        # ── Update CPU buffer ──────────────────────────────────
        self._update_cpu_buffer(server_id, current_cpu)

        # ── 1. Prediction vs Action Consistency ────────────────
        force_scale = False
        if (corrected_pred > 90.0 or risk > 0.60) and adjusted_conf > 0.60:
            force_scale = True

        # ── 2. Decision Logic Core ─────────────────────────────
        if current_cpu > 90:
            # 8. EMERGENCY OVERRIDE (Hard Rule)
            decision = "ESCALATE"
        elif self._sustained_above(server_id, 85, 120):
            # 2. TIME-BASED ESCALATION
            decision = "ESCALATE"
        elif self._sustained_above(server_id, 80, 60):
            # 2. TIME-BASED ESCALATION
            decision = "SCALE"
        elif force_scale:
            decision = "SCALE"
        elif corrected_pred > 90.0 and trend in ("rising", "rapidly_rising"):
            # 4. PREEMPTIVE DECISION LAYER
            decision = "SCALE_READY"
        elif corrected_pred > MONITOR_CPU_THRESHOLD or risk > 0.55:
            decision = "MONITOR"
        else:
            decision = "STABLE"

        # ── 3. STATE MACHINE ENFORCEMENT ───────────────────────
        last_decision = self._last_decision_str.get(server_id, "STABLE")
        state_order = {"STABLE": 0, "MONITOR": 1, "SCALE_READY": 2, "SCALE": 3, "ESCALATE": 4}
        
        now = time.time()
        last_action_time = self._last_action_time.get(server_id, 0.0)
        
        target_order = state_order.get(decision, 0)
        last_order = state_order.get(last_decision, 0)

        # No skipping unless emergency override (CPU > 90%)
        if current_cpu <= 90:
            if target_order > last_order + 1:
                target_order = last_order + 1
        
        # Cooldown check: after SCALE or ESCALATE, do not downgrade for 60 seconds
        if last_decision in ("SCALE", "ESCALATE") and target_order < last_order:
            if now - last_action_time < 60:
                target_order = last_order  # prevent downgrade

        # Map back to decision string
        for k, v in state_order.items():
            if v == target_order:
                decision = k
                break

        # Update action tracking
        if decision in ("SCALE", "ESCALATE") and decision != last_decision:
            self._last_action_time[server_id] = now
            
        self._last_decision_str[server_id] = decision
''',
    code,
    flags=re.DOTALL
)

# 7. Action confidence update
code = code.replace(
    'decision == "MONITOR" and trend in ("rising", "elevated", "volatile")',
    'decision in ("MONITOR", "SCALE_READY") and trend in ("rising", "elevated", "volatile")'
)

# 8. Check total_interventions metric update
code = code.replace(
    'if decision in ("SCALE", "ESCALATE", "RESTART"):',
    'if decision in ("SCALE", "ESCALATE", "SCALE_READY"):'
)

with open(path, "w", encoding="utf-8") as f:
    f.write(code)

print("Updated decision_engine.py successfully.")
