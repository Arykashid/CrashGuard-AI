"""
server_simulator.py — CrashGuard AI  v3 (FINAL)
Simulates 5 servers with realistic CPU behavior patterns.
Provides a thread-safe in-memory data store for Flask consumption.

Behaviors:
  Server A — Stable:      CPU stays 20–35%
  Server B — Gradual:     CPU slowly climbs 40→80%, smooth cooldown back to 40
  Server C — Burst:       CPU rare random bursts 20→90%
  Server D — Fluctuating: CPU oscillates 30–70%
  Server E — Critical:    CPU stays high 75–95% with micro dips

Fixes applied (v3):
  [1] DEMO_MODE env var — seed=42 only when DEMO_MODE=1, else truly random
  [2] Burst cooldown randomint(8,25) — breaks predictable burst pattern
  [3] Thread error handling — generator crash logs warning, never kills thread
  [4] Global _clip() helper — consistent CPU range enforcement
  [5] Inertia smoothing — exponential moving average per server
  [6] Gradual server: smooth cooldown phase instead of instant reset
  [7] Timestamp precision → milliseconds
  [8] Critical server: micro dips for realism

Run in demo mode (stable, repeatable):
  Windows:  set DEMO_MODE=1 && python server_simulator.py
  Linux:    DEMO_MODE=1 python server_simulator.py

Run in live mode (truly random):
  python server_simulator.py
"""

import os
import time
import math
import random
import logging
import threading
from collections import deque
from datetime import datetime

# ─────────────────────────────────────────────
# FIX 1 — DEMO_MODE ENV VAR SEED
# stable for demo, truly random otherwise
# ─────────────────────────────────────────────
if os.getenv("DEMO_MODE") == "1":
    random.seed(42)
    print("[simulator] DEMO_MODE=1 → random.seed(42) applied. Reproducible run.")
else:
    print("[simulator] Live mode → truly random. Set DEMO_MODE=1 for stable demo.")

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crashguard.simulator")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TICK_INTERVAL  = 2    # seconds between each CPU reading
HISTORY_LENGTH = 150  # readings to keep per server (5 min @ 2s)

# ─────────────────────────────────────────────
# GLOBAL CPU RANGE + _clip()
# ─────────────────────────────────────────────
MIN_CPU = 20.0
MAX_CPU = 95.0

def _clip(cpu: float, low: float = MIN_CPU, high: float = MAX_CPU) -> float:
    return round(max(low, min(high, cpu)), 2)

# ─────────────────────────────────────────────
# INERTIA HELPER
# alpha=0.7 → 70% last value, 30% new → smooth transitions
# ─────────────────────────────────────────────
def _apply_inertia(state: dict, new_cpu: float, alpha: float = 0.7) -> float:
    last   = state.get("last_cpu", new_cpu)
    smooth = alpha * last + (1 - alpha) * new_cpu
    state["last_cpu"] = smooth
    return smooth

# ─────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────
_lock      = threading.Lock()
_histories: dict[str, deque] = {}
_latest:    dict[str, dict]  = {}

# ─────────────────────────────────────────────
# SERVER DEFINITIONS
# ─────────────────────────────────────────────
SERVERS = {
    "server_a": {
        "name":     "Server A — Stable",
        "short":    "Server A",
        "behavior": "stable",
        "state":    {},
    },
    "server_b": {
        "name":     "Server B — Gradual Spike",
        "short":    "Server B",
        "behavior": "gradual",
        "state":    {"tick": 0, "phase": "climb"},
    },
    "server_c": {
        "name":     "Server C — Sudden Burst",
        "short":    "Server C",
        "behavior": "burst",
        "state":    {"burst_ticks_left": 0, "cooldown_ticks_left": 0},
    },
    "server_d": {
        "name":     "Server D — Fluctuating",
        "short":    "Server D",
        "behavior": "fluctuating",
        "state":    {"tick": 0},
    },
    "server_e": {
        "name":     "Server E — Critical",
        "short":    "Server E",
        "behavior": "critical",
        "state":    {},
    },
}

# ─────────────────────────────────────────────
# NOISE HELPER
# ─────────────────────────────────────────────
def _noise(magnitude: float = 1.0) -> float:
    return max(-magnitude * 3, min(magnitude * 3, random.gauss(0, magnitude)))

# ─────────────────────────────────────────────
# CPU GENERATORS
# ─────────────────────────────────────────────

def _gen_stable(state: dict) -> float:
    value = 27.0 + _noise(3.5)
    value = _apply_inertia(state, value)
    return _clip(value)


def _gen_gradual(state: dict) -> float:
    tick  = state["tick"]
    phase = state["phase"]
    CLIMB_TICKS = 120
    HOLD_TICKS  = 15

    if phase == "climb":
        progress = min(tick / CLIMB_TICKS, 1.0)
        cpu = 40.0 + progress * 40.0 + _noise(2.0)
        state["tick"] += 1
        if state["tick"] >= CLIMB_TICKS:
            state["phase"] = "hold"
            state["tick"]  = 0

    elif phase == "hold":
        cpu = 78.0 + _noise(2.5)
        state["tick"] += 1
        if state["tick"] >= HOLD_TICKS:
            state["phase"] = "cooldown"
            state["tick"]  = 0

    elif phase == "cooldown":
        cpu = 80.0 - state["tick"] * 2.0 + _noise(1.5)
        state["tick"] += 1
        if cpu <= 40.0:
            state["phase"] = "climb"
            state["tick"]  = 0
    else:
        cpu = 40.0

    cpu = _apply_inertia(state, cpu)
    return _clip(cpu)


def _gen_burst(state: dict) -> float:
    if state["burst_ticks_left"] > 0:
        cpu = random.uniform(70.0, 92.0) + _noise(3.0)
        state["burst_ticks_left"] -= 1
        if state["burst_ticks_left"] == 0:
            # FIX 2 — wider random cooldown breaks predictable pattern
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
            # FIX 2 — also randomise idle cooldown
            state["cooldown_ticks_left"] = random.randint(8, 25)

    cpu = _apply_inertia(state, cpu)
    return _clip(cpu)


def _gen_fluctuating(state: dict) -> float:
    tick   = state["tick"]
    PERIOD = 100
    cpu    = 50.0 + 20.0 * math.sin(2 * math.pi * tick / PERIOD) + _noise(4.0)
    state["tick"] = (tick + 1) % (PERIOD * 10)
    cpu = _apply_inertia(state, cpu)
    return _clip(cpu)


def _gen_critical(state: dict) -> float:
    if random.random() < 0.1:
        cpu = random.uniform(60.0, 75.0)
    else:
        cpu = random.uniform(80.0, 95.0)
    cpu += _noise(2.0)
    cpu  = _apply_inertia(state, cpu)
    return _clip(cpu, 60.0, 99.0)


_GENERATORS = {
    "stable":      _gen_stable,
    "gradual":     _gen_gradual,
    "burst":       _gen_burst,
    "fluctuating": _gen_fluctuating,
    "critical":    _gen_critical,
}

# ─────────────────────────────────────────────
# SIMULATOR THREAD
# FIX 3 — try/except per generator so one crash
#          never kills the whole thread
# ─────────────────────────────────────────────

def _simulator_loop():
    with _lock:
        for sid in SERVERS:
            _histories[sid] = deque(maxlen=HISTORY_LENGTH)

    while True:
        ts = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

        with _lock:
            for sid, meta in SERVERS.items():
                try:
                    gen_fn = _GENERATORS[meta["behavior"]]
                    cpu    = gen_fn(meta["state"])
                except Exception as exc:
                    # FIX 3 — log the error, fall back to last known CPU
                    logger.warning(
                        "Generator failed for %s (%s): %s — using last known CPU.",
                        sid, meta["behavior"], exc,
                    )
                    cpu = _latest.get(sid, {}).get("cpu", 50.0)

                record = {
                    "timestamp":   ts,
                    "cpu":         cpu,
                    "server_id":   sid,
                    "server_name": meta["short"],
                    "behavior":    meta["behavior"],
                }
                _histories[sid].append(record)
                _latest[sid] = record

        time.sleep(TICK_INTERVAL)


_thread = threading.Thread(target=_simulator_loop, daemon=True, name="SimulatorThread")

# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def start():
    """Start the background simulator thread. Call once at Flask startup."""
    if not _thread.is_alive():
        _thread.start()
        logger.info("Simulator thread started.")


def get_latest() -> dict:
    """Latest single reading per server. Thread-safe snapshot."""
    with _lock:
        return {sid: dict(v) for sid, v in _latest.items()}


def get_history(server_id: str, n: int | None = None) -> list[dict]:
    """Last `n` readings for one server (all if n is None). Thread-safe."""
    with _lock:
        if server_id not in _histories:
            return []
        history = list(_histories[server_id])
    return history[-n:] if n else history


def get_all_history() -> dict[str, list[dict]]:
    """Full history for all servers."""
    with _lock:
        return {sid: list(dq) for sid, dq in _histories.items()}


def get_server_list() -> list[dict]:
    """Metadata for all servers (id, name, behavior)."""
    return [
        {"server_id": sid, "name": meta["name"], "behavior": meta["behavior"]}
        for sid, meta in SERVERS.items()
    ]


# ─────────────────────────────────────────────
# STANDALONE TEST
# Normal:    python server_simulator.py
# Demo mode: set DEMO_MODE=1 && python server_simulator.py  (Windows)
#            DEMO_MODE=1 python server_simulator.py          (Linux)
# ─────────────────────────────────────────────
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
        print(f"  {'SERVER':<28} {'CPU%':>6}  {'BEHAVIOR'}")
        print(f"{'─'*68}")
        for sid in SERVERS:
            rec = latest.get(sid)
            if rec:
                pct     = rec["cpu"]
                bar_len = min(int(pct / 5), 20)
                bar     = "█" * bar_len + "░" * (20 - bar_len)
                print(f"  {rec['server_name']:<28} {pct:>5.1f}%  [{bar}]  {rec['behavior']}")
        print(f"{'─'*68}")

        hist_b = get_history("server_b", n=5)
        print(f"\n  Last 5 — Server B (Gradual Spike):")
        for r in hist_b:
            print(f"    {r['timestamp']}  CPU {r['cpu']}%")

    print("\nTest complete.")
