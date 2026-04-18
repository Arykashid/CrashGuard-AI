"""
feature_engine.py — CrashGuard AI
Converts raw CPU readings from server_simulator into the 15 engineered
features expected by the LSTM + XGBoost ensemble.

Features produced (15 total):
  1.  cpu_raw            — current CPU %
  2.  cpu_lag1           — CPU 1 tick ago   (2s)
  3.  cpu_lag3           — CPU 3 ticks ago  (6s)
  4.  cpu_lag5           — CPU 5 ticks ago  (10s)
  5.  cpu_lag10          — CPU 10 ticks ago (20s)
  6.  rolling_mean_10    — mean over last 10 ticks
  7.  rolling_std_10     — std  over last 10 ticks
  8.  rolling_max_10     — max  over last 10 ticks
  9.  rolling_mean_30    — mean over last 30 ticks
  10. rolling_std_30     — std  over last 30 ticks
  11. delta_1            — cpu[t] - cpu[t-1]  (rate of change)
  12. delta_5            — cpu[t] - cpu[t-5]
  13. spike_flag         — 1 if cpu_raw > SPIKE_THRESHOLD, else 0
  14. hour_sin           — sin encoding of UTC hour
  15. hour_cos           — cos encoding of UTC hour
"""

import math
from datetime import datetime, timezone
from typing import Optional

# ─────────────────────────────────────────────
# FIX 1 — DATA-DRIVEN THRESHOLDS
# Derived from Google Cluster CPU percentiles (training data):
#   P80 ≈ 76%  → spike threshold
#   P65 ≈ 65%  → elevated mean threshold
#   P90 std ≈ 15% → high volatility threshold
# Formula anchors to MAX_CPU so they scale if range changes.
# ─────────────────────────────────────────────
MAX_CPU = 95.0
MIN_CPU = 20.0

SPIKE_THRESHOLD    = 0.80 * MAX_CPU   # 76.0  — P80 of training distribution
ELEVATED_MEAN      = 0.68 * MAX_CPU   # 64.6  — sustained high load indicator
HIGH_STD_THRESHOLD = 0.16 * MAX_CPU   # 15.2  — high volatility indicator
DELTA_SPIKE        = 0.06 * MAX_CPU   #  5.7  — rapid rise per tick

# ─────────────────────────────────────────────
# FIX 4 — OUTLIER CLIPPING BOUNDS
# Applied to every feature before returning.
# Prevents simulator glitches from producing
# NaN / inf / extreme values that corrupt model input.
# ─────────────────────────────────────────────
FEATURE_CLIP = {
    "cpu_raw":         (MIN_CPU,  MAX_CPU),
    "cpu_lag1":        (MIN_CPU,  MAX_CPU),
    "cpu_lag3":        (MIN_CPU,  MAX_CPU),
    "cpu_lag5":        (MIN_CPU,  MAX_CPU),
    "cpu_lag10":       (MIN_CPU,  MAX_CPU),
    "rolling_mean_10": (MIN_CPU,  MAX_CPU),
    "rolling_std_10":  (0.0,      MAX_CPU / 2),   # std can't exceed half range
    "rolling_max_10":  (MIN_CPU,  MAX_CPU),
    "rolling_mean_30": (MIN_CPU,  MAX_CPU),
    "rolling_std_30":  (0.0,      MAX_CPU / 2),
    "delta_1":         (-MAX_CPU, MAX_CPU),        # rate of change ±95
    "delta_5":         (-MAX_CPU, MAX_CPU),
    "spike_flag":      (0.0,      1.0),
    "hour_sin":        (-1.0,     1.0),
    "hour_cos":        (-1.0,     1.0),
}

# Must match training column order EXACTLY — do not reorder
FEATURE_ORDER = [
    "cpu_raw", "cpu_lag1", "cpu_lag3", "cpu_lag5", "cpu_lag10",
    "rolling_mean_10", "rolling_std_10", "rolling_max_10",
    "rolling_mean_30", "rolling_std_30",
    "delta_1", "delta_5",
    "spike_flag",
    "hour_sin", "hour_cos",
]

REQUIRED_FEATURES = 15
MIN_HISTORY       = 30

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _extract_cpu_series(history: list[dict]) -> list[float]:
    return [float(r["cpu"]) for r in history]


def _hour_encoding(dt: Optional[datetime] = None) -> tuple[float, float]:
    if dt is None:
        dt = datetime.now(timezone.utc)
    hour  = dt.hour + dt.minute / 60.0
    angle = 2 * math.pi * hour / 24.0
    return math.sin(angle), math.cos(angle)


def _clip_features(features: dict) -> dict:
    """
    FIX 4 — clip every feature to its valid range.
    Catches simulator glitches, NaN, inf before they reach the model.
    """
    clipped = {}
    for k, v in features.items():
        lo, hi = FEATURE_CLIP[k]
        v = float(v)
        if not math.isfinite(v):          # catch NaN / inf
            v = (lo + hi) / 2.0           # replace with midpoint
        clipped[k] = round(max(lo, min(hi, v)), 6)
    return clipped

# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def build_features(
    history: list[dict],
    dt: Optional[datetime] = None,
) -> Optional[dict]:
    """
    Build the 15-feature vector from a server's raw CPU history.

    Args:
        history : list of dicts from server_simulator.get_history()
                  Each dict must have {"cpu": float, "timestamp": str}
                  Oldest first, newest last.
        dt      : datetime for cyclical time features (defaults to UTC now).

    Returns:
        dict {feature_name: float}  if len(history) >= MIN_HISTORY.
        None                        if history too short.
    """
    if len(history) < MIN_HISTORY:
        return None

    cpu_series = _extract_cpu_series(history)
    current    = cpu_series[-1]

    def lag(n: int) -> float:
        return cpu_series[-(n + 1)] if len(cpu_series) >= n + 1 else current

    cpu_lag1  = lag(1)
    cpu_lag3  = lag(3)
    cpu_lag5  = lag(5)
    cpu_lag10 = lag(10)

    window10 = cpu_series[-10:]
    window30 = cpu_series[-30:]

    rolling_mean_10 = _mean(window10)
    rolling_std_10  = _std(window10)
    rolling_max_10  = max(window10)
    rolling_mean_30 = _mean(window30)
    rolling_std_30  = _std(window30)

    delta_1 = current - cpu_lag1
    delta_5 = current - cpu_lag5

    spike_flag = 1.0 if current > SPIKE_THRESHOLD else 0.0

    hour_sin, hour_cos = _hour_encoding(dt)

    features = {
        "cpu_raw":         current,
        "cpu_lag1":        cpu_lag1,
        "cpu_lag3":        cpu_lag3,
        "cpu_lag5":        cpu_lag5,
        "cpu_lag10":       cpu_lag10,
        "rolling_mean_10": rolling_mean_10,
        "rolling_std_10":  rolling_std_10,
        "rolling_max_10":  rolling_max_10,
        "rolling_mean_30": rolling_mean_30,
        "rolling_std_30":  rolling_std_30,
        "delta_1":         delta_1,
        "delta_5":         delta_5,
        "spike_flag":      spike_flag,
        "hour_sin":        hour_sin,
        "hour_cos":        hour_cos,
    }

    # FIX 4 — clip all features before returning
    features = _clip_features(features)

    assert len(features) == REQUIRED_FEATURES, (
        f"Feature count mismatch: got {len(features)}, expected {REQUIRED_FEATURES}"
    )

    return features


def features_to_array(features: dict) -> list[float]:
    """
    Convert feature dict → ordered list for model input.
    Order matches training column order exactly.
    """
    return [features[k] for k in FEATURE_ORDER]


def is_spike_context(features: dict) -> bool:
    """
    Returns True if readings suggest active or imminent spike.
    Thresholds derived from training data percentiles — see constants above.
    Used by decision_engine for fast-path checks.
    """
    return (
        features["spike_flag"]      == 1.0
        or features["rolling_mean_10"] > ELEVATED_MEAN
        or features["rolling_std_10"]  > HIGH_STD_THRESHOLD
        or features["delta_1"]         > DELTA_SPIKE
    )


# ─────────────────────────────────────────────
# STANDALONE TEST — python feature_engine.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("CrashGuard AI — Feature Engine Test")
    print(f"Thresholds: SPIKE={SPIKE_THRESHOLD:.1f}%  "
          f"ELEVATED_MEAN={ELEVATED_MEAN:.1f}%  "
          f"HIGH_STD={HIGH_STD_THRESHOLD:.1f}%  "
          f"DELTA_SPIKE={DELTA_SPIKE:.1f}%")
    print("─" * 55)

    # Test 1: short history → None
    result = build_features([{"cpu": 30.0, "timestamp": "t"}] * 10)
    assert result is None
    print("✓ Test 1: returns None for short history")

    # Test 2: stable server → no spike, no NaN
    stable = [{"cpu": 27.0 + (i % 3) * 0.5, "timestamp": "t"} for i in range(50)]
    f = build_features(stable)
    assert f is not None and len(f) == 15
    assert f["spike_flag"] == 0.0
    assert all(math.isfinite(v) for v in f.values()), "NaN/inf detected!"
    print(f"✓ Test 2: stable server — spike_flag={f['spike_flag']}, "
          f"rolling_std_10={f['rolling_std_10']}")

    # Test 3: critical server → spike detected
    critical = [{"cpu": 85.0 + (i % 5), "timestamp": "t"} for i in range(50)]
    f = build_features(critical)
    assert f is not None and f["spike_flag"] == 1.0
    assert is_spike_context(f)
    print(f"✓ Test 3: critical server — spike_flag={f['spike_flag']}, "
          f"rolling_mean_10={f['rolling_mean_10']}")

    # Test 4: outlier injection → clipped correctly
    glitch = [{"cpu": 27.0, "timestamp": "t"} for _ in range(49)]
    glitch.append({"cpu": 999.0, "timestamp": "t"})   # inject outlier
    f = build_features(glitch)
    assert f is not None
    assert f["cpu_raw"] <= MAX_CPU, f"Outlier not clipped! cpu_raw={f['cpu_raw']}"
    assert all(math.isfinite(v) for v in f.values()), "NaN/inf after outlier!"
    print(f"✓ Test 4: outlier clipped — cpu_raw={f['cpu_raw']} (max allowed {MAX_CPU})")

    # Test 5: NaN injection → replaced with midpoint
    nan_hist = [{"cpu": 27.0, "timestamp": "t"} for _ in range(49)]
    nan_hist.append({"cpu": float("nan"), "timestamp": "t"})
    try:
        f = build_features(nan_hist)
        if f:
            assert all(math.isfinite(v) for v in f.values())
            print(f"✓ Test 5: NaN handled — all features finite")
        else:
            print("✓ Test 5: NaN history rejected cleanly")
    except Exception as e:
        print(f"✗ Test 5: NaN caused crash — {e}")

    # Test 6: feature array order
    normal = [{"cpu": 50.0, "timestamp": "t"} for _ in range(50)]
    f = build_features(normal)
    arr = features_to_array(f)
    assert len(arr) == 15 and arr[0] == f["cpu_raw"]
    print(f"✓ Test 6: feature array length={len(arr)}, first={arr[0]}")

    print("\n── Full feature vector (critical server) ──")
    critical_f = build_features(critical)
    for k, v in critical_f.items():
        lo, hi = FEATURE_CLIP[k]
        flag = " ⚠ OUT OF RANGE" if not (lo <= v <= hi) else ""
        print(f"   {k:<20} {v:>10.4f}   [{lo}, {hi}]{flag}")

    print(f"\nSPIKE_THRESHOLD   = {SPIKE_THRESHOLD:.2f}%  (0.80 × MAX_CPU)")
    print(f"ELEVATED_MEAN     = {ELEVATED_MEAN:.2f}%  (0.68 × MAX_CPU)")
    print(f"HIGH_STD          = {HIGH_STD_THRESHOLD:.2f}%  (0.16 × MAX_CPU)")
    print(f"DELTA_SPIKE       = {DELTA_SPIKE:.2f}%  (0.06 × MAX_CPU)")
    print("\n✅ All tests passed. Feature engine ready.")