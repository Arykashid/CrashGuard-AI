"""
server_simulator.py — CrashGuard AI  v3 (FINAL)
Simulates 5 servers with realistic CPU behavior patterns.
Provides a thread-safe in-memory data store for Flask consumption.

Fixes applied (v3):
  [1] DEMO_MODE env var — seed=42 only when DEMO_MODE=1
  [2] Burst cooldown randomint(8,25)
  [3] Thread error handling
  [4] Global _clip() helper
  [5] Inertia smoothing
  [6] Gradual server smooth cooldown
  [7] Timestamp milliseconds
  [8] Critical server micro dips
  [9] inject_reading() — public API for demo trigger
"""

import os
import time
import math
import random
import logging
import threading
from collections import deque
from datetime import datetime

if os.getenv("DEMO_MODE") == "1":
    random.seed(42)
    print("[simulator] DEMO_MODE=1 -> random.seed(42) applied.")
else:
    print("[simulator] Live mode -> truly random. Set DEMO_MODE=1 for stable demo.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("crashguard.simulator")

TICK_INTERVAL  = 2
HISTORY_LENGTH = 150
MIN_CPU = 20.0
MAX_CPU = 95.0

def _clip(cpu, low=MIN_CPU, high=MAX_CPU):
    return round(max(low, min(high, cpu)), 2)

def _apply_inertia(state, new_cpu, alpha=0.7):
    last = state.get("last_cpu", new_cpu)
    smooth = alpha * last + (1 - alpha) * new_cpu
    state["last_cpu"] = smooth
    return smooth

_lock = threading.Lock()
_histories = {}
_latest = {}

SERVERS = {
    "server_a": {"name": "Server A — Stable",        "short": "Server A", "behavior": "stable",      "state": {}},
    "server_b": {"name": "Server B — Gradual Spike",  "short": "Server B", "behavior": "gradual",     "state": {"tick": 0, "phase": "climb"}},
    "server_c": {"name": "Server C — Sudden Burst",   "short": "Server C", "behavior": "burst",       "state": {"burst_ticks_left": 0, "cooldown_ticks_left": 0}},
    "server_d": {"name": "Server D — Fluctuating",    "short": "Server D", "behavior": "fluctuating", "state": {"tick": 0}},
    "server_e": {"name": "Server E — Critical",       "short": "Server E", "behavior": "critical",    "state": {}},
}

def _noise(magnitude=1.0):
    return max(-magnitude*3, min(magnitude*3, random.gauss(0, magnitude)))

def _gen_stable(state):
    return _clip(_apply_inertia(state, 27.0 + _noise(3.5)))

def _gen_gradual(state):
    tick, phase = state["tick"], state["phase"]
    if phase == "climb":
        cpu = 40.0 + min(tick/120, 1.0)*40.0 + _noise(2.0)
        state["tick"] += 1
        if state["tick"] >= 120: state["phase"] = "hold"; state["tick"] = 0
    elif phase == "hold":
        cpu = 78.0 + _noise(2.5)
        state["tick"] += 1
        if state["tick"] >= 15: state["phase"] = "cooldown"; state["tick"] = 0
    elif phase == "cooldown":
        cpu = 80.0 - state["tick"]*2.0 + _noise(1.5)
        state["tick"] += 1
        if cpu <= 40.0: state["phase"] = "climb"; state["tick"] = 0
    else:
        cpu = 40.0
    return _clip(_apply_inertia(state, cpu))

def _gen_burst(state):
    if state["burst_ticks_left"] > 0:
        cpu = random.uniform(70.0, 92.0) + _noise(3.0)
        state["burst_ticks_left"] -= 1
        if state["burst_ticks_left"] == 0:
            state["cooldown_ticks_left"] = random.randint(8, 25)
    elif state["cooldown_ticks_left"] > 0:
        cpu = random.uniform(20.0, 38.0) + _noise(2.0)
        state["cooldown_ticks_left"] -= 1
    else:
        if random.random() < 0.1:
            state["burst_ticks_left"] = random.randint(3, 8)
            cpu = random.uniform(70.0, 92.0) + _noise(3.0)
            state["burst_ticks_left"] -= 1
        else:
            cpu = random.uniform(20.0, 38.0) + _noise(2.0)
            state["cooldown_ticks_left"] = random.randint(8, 25)
    return _clip(_apply_inertia(state, cpu))

def _gen_fluctuating(state):
    tick = state["tick"]
    cpu = 50.0 + 20.0*math.sin(2*math.pi*tick/100) + _noise(4.0)
    state["tick"] = (tick+1) % 1000
    return _clip(_apply_inertia(state, cpu))

def _gen_critical(state):
    cpu = random.uniform(60.0, 75.0) if random.random() < 0.1 else random.uniform(80.0, 95.0)
    cpu += _noise(2.0)
    return _clip(_apply_inertia(state, cpu), 60.0, 99.0)

_GENERATORS = {"stable": _gen_stable, "gradual": _gen_gradual, "burst": _gen_burst, "fluctuating": _gen_fluctuating, "critical": _gen_critical}

def _simulator_loop():
    with _lock:
        for sid in SERVERS:
            _histories[sid] = deque(maxlen=HISTORY_LENGTH)
    while True:
        ts = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        with _lock:
            for sid, meta in SERVERS.items():
                try:
                    cpu = _GENERATORS[meta["behavior"]](meta["state"])
                except Exception as exc:
                    logger.warning("Generator failed for %s: %s", sid, exc)
                    cpu = _latest.get(sid, {}).get("cpu", 50.0)
                record = {"timestamp": ts, "cpu": cpu, "server_id": sid, "server_name": meta["short"], "behavior": meta["behavior"]}
                _histories[sid].append(record)
                _latest[sid] = record
        time.sleep(TICK_INTERVAL)

_thread = threading.Thread(target=_simulator_loop, daemon=True, name="SimulatorThread")

# ── PUBLIC API ──────────────────────────────────────────────────

def start():
    if not _thread.is_alive():
        _thread.start()
        logger.info("Simulator thread started.")

def get_latest():
    with _lock:
        return {sid: dict(v) for sid, v in _latest.items()}

def get_history(server_id, n=None):
    with _lock:
        if server_id not in _histories:
            return []
        history = list(_histories[server_id])
    return history[-n:] if n else history

def get_all_history():
    with _lock:
        return {sid: list(dq) for sid, dq in _histories.items()}

def get_server_list():
    return [{"server_id": sid, "name": meta["name"], "behavior": meta["behavior"]} for sid, meta in SERVERS.items()]

def inject_reading(server_id: str, cpu: float) -> bool:
    """
    Public API — inject a custom CPU reading into a server's history.
    Used by /api/demo/trigger. No private state access needed from app.py.
    Returns True if server_id is valid, False otherwise.
    """
    with _lock:
        if server_id not in _histories:
            return False
        ts = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        record = {
            "timestamp":   ts,
            "cpu":         round(float(cpu), 2),
            "server_id":   server_id,
            "server_name": SERVERS.get(server_id, {}).get("short", server_id),
            "behavior":    "injected",
        }
        _histories[server_id].append(record)
        _latest[server_id] = record
        return True

# ── STANDALONE TEST ─────────────────────────────────────────────
if __name__ == "__main__":
    print("CrashGuard AI — Server Simulator v3 (FINAL)\n")
    start()
    import os as _os
    for _ in range(40):
        time.sleep(4)
        latest = get_latest()
        if not latest:
            print("Waiting for first tick...")
            continue
        _os.system("cls" if _os.name == "nt" else "clear")
        mode = "DEMO (seed=42)" if _os.getenv("DEMO_MODE") == "1" else "LIVE (random)"
        print(f"{'─'*68}")
        print(f"  CrashGuard AI — Simulator v3 [{mode}]   {datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]} UTC")
        print(f"{'─'*68}")
        for sid in SERVERS:
            rec = latest.get(sid)
            if rec:
                pct = rec["cpu"]
                bar = "█" * min(int(pct/5), 20) + "░" * (20 - min(int(pct/5), 20))
                print(f"  {rec['server_name']:<28} {pct:>5.1f}%  [{bar}]  {rec['behavior']}")
        print(f"{'─'*68}")
    print("\nTest complete.")