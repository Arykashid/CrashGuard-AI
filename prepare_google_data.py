"""
prepare_google_data.py -- CrashGuard AI
Processes the Google Cluster Trace 2011 (task_usage) CSV into a format
compatible with preprocessing.py and the rest of the pipeline.

STRATEGY:
    The raw trace has 2.5M task measurements across 12K machines in ~83 minutes.
    A single-machine time series is too short for LSTM training.
    
    Solution: pick the top N busiest machines, build per-machine CPU time series,
    and stitch them end-to-end to simulate multi-day monitoring. This preserves
    REAL workload patterns (spikes, bursts, diurnal-like variance) while
    producing enough rows (60K+) for robust LSTM training.

Google Cluster Trace task_usage columns (no header, 0-indexed):
    0: start_time          (microseconds from trace start)
    1: end_time            (microseconds from trace start)
    2: job_id
    3: task_index
    4: machine_id
    5: mean_cpu_rate        <-- THIS IS THE CPU COLUMN
    6: canonical_memory
    7: assigned_memory
    8: unmapped_page_cache
    9: page_cache_memory
   10: max_memory_usage
   11: mean_disk_io_time
   12: mean_local_disk_used
   13: max_cpu_rate
   14: max_disk_io_time
   15: cycles_per_instruction
   16: memory_accesses_per_instruction
   17: sample_portion
   18: aggregation_type
   19: sampled_cpu_usage    (clusterdata-2011-2 only)

Output:
    data/google_cluster_processed.csv
    Columns: timestamp, cpu_usage
    Format: matches cpu_timeseries.csv (drop-in replacement)

Usage:
    python prepare_google_data.py
    python prepare_google_data.py --target-rows 80000
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime, timedelta


# =============================================
# CONFIGURATION
# =============================================
INPUT_PATH = "data/google_cluster_usage.csv"
OUTPUT_PATH = "data/google_cluster_processed.csv"
RESAMPLE_INTERVAL_SEC = 10       # 10-second intervals (match cpu_timeseries.csv)
TARGET_ROWS = 60000              # Target dataset size for LSTM training
SPIKE_THRESHOLD = 0.75           # CPU > 75% = spike
TRACE_BASE_DATE = datetime(2024, 1, 1, 0, 0, 0)

# Column indices (no header)
COL_START_TIME = 0
COL_END_TIME = 1
COL_MACHINE_ID = 4
COL_MEAN_CPU = 5
COL_MAX_CPU = 13


# =============================================
# STEP 1: LOAD RAW DATA
# =============================================
def load_raw(path):
    """Load only the columns we need from the headerless CSV."""
    print(f"[1/6] Loading {path} ...")

    use_cols = [COL_START_TIME, COL_END_TIME, COL_MACHINE_ID,
                COL_MEAN_CPU, COL_MAX_CPU]

    df = pd.read_csv(
        path,
        header=None,
        usecols=use_cols,
        dtype={
            COL_START_TIME: np.int64,
            COL_END_TIME: np.int64,
            COL_MACHINE_ID: np.int64,
            COL_MEAN_CPU: np.float64,
            COL_MAX_CPU: np.float64,
        },
        on_bad_lines="skip",
    )
    df.columns = ["start_us", "end_us", "machine_id", "mean_cpu", "max_cpu"]

    # Drop invalid CPU readings
    df = df.dropna(subset=["mean_cpu"])
    df = df[df["mean_cpu"] >= 0]

    print(f"       {len(df):,} valid rows | "
          f"{df['machine_id'].nunique():,} machines | "
          f"{df['start_us'].nunique():,} time buckets")
    return df


# =============================================
# STEP 2: BUILD PER-MACHINE TIME SERIES
# =============================================
def build_per_machine_series(df):
    """
    For each machine, aggregate all its task CPU measurements per time bucket
    to get a single machine-level CPU time series.
    """
    print("[2/6] Building per-machine CPU series ...")

    # Sum CPU across tasks on same machine at same time
    # (total CPU load on the machine, capped at 1.0)
    machine_ts = (
        df.groupby(["machine_id", "start_us"])
        .agg(cpu=("mean_cpu", "sum"), n_tasks=("mean_cpu", "count"))
        .reset_index()
    )
    machine_ts["cpu"] = machine_ts["cpu"].clip(0, 1.0)

    # Count time points per machine
    machine_counts = (
        machine_ts.groupby("machine_id")
        .size()
        .reset_index(name="n_points")
        .sort_values("n_points", ascending=False)
    )

    print(f"       {len(machine_counts):,} machines with data")
    print(f"       Top machine: {machine_counts.iloc[0]['n_points']} time points")
    print(f"       Median:      {machine_counts['n_points'].median():.0f} time points")

    return machine_ts, machine_counts


# =============================================
# STEP 3: SELECT & STITCH MACHINES
# =============================================
def stitch_machines(machine_ts, machine_counts, target_rows):
    """
    Select the busiest machines and stitch their time series end-to-end
    to build a long, continuous CPU signal for LSTM training.

    Each machine's segment is placed sequentially in time, starting from
    TRACE_BASE_DATE, with 10-second intervals.
    """
    print(f"[3/6] Stitching machine series to reach {target_rows:,} rows ...")

    # Sort machines by row count (busiest first)
    sorted_machines = machine_counts.sort_values("n_points", ascending=False)

    all_segments = []
    total_points = 0
    machines_used = 0
    current_time = TRACE_BASE_DATE

    for _, row in sorted_machines.iterrows():
        if total_points >= target_rows:
            break

        mid = row["machine_id"]
        seg = machine_ts[machine_ts["machine_id"] == mid].copy()
        seg = seg.sort_values("start_us")

        # Resample within this machine's segment to 10s intervals
        # Map the machine's relative timestamps to regular intervals
        n_points = len(seg)
        timestamps = [
            current_time + timedelta(seconds=i * RESAMPLE_INTERVAL_SEC)
            for i in range(n_points)
        ]
        seg = seg.reset_index(drop=True)
        seg["timestamp"] = timestamps
        seg["cpu_usage"] = seg["cpu"]
        all_segments.append(seg[["timestamp", "cpu_usage"]])

        current_time = timestamps[-1] + timedelta(seconds=RESAMPLE_INTERVAL_SEC)
        total_points += n_points
        machines_used += 1

    result = pd.concat(all_segments, ignore_index=True)

    # If we have more than target, trim
    if len(result) > target_rows:
        result = result.iloc[:target_rows]

    print(f"       Used {machines_used} machines")
    print(f"       Time span: {result['timestamp'].min()} to {result['timestamp'].max()}")
    print(f"       Duration:  {result['timestamp'].max() - result['timestamp'].min()}")
    print(f"       Rows:      {len(result):,}")

    return result


# =============================================
# STEP 4: ADD DIURNAL MODULATION
# =============================================
def add_diurnal_pattern(df):
    """
    The stitched signal has real workload variance but no day/night pattern
    (the original trace is only 83 minutes). We modulate the CPU signal with
    a realistic diurnal envelope so the LSTM can learn temporal patterns.

    This preserves the REAL spike shapes while adding the time-of-day signal
    that CrashGuard AI needs for forecasting.
    """
    print("[4/6] Adding diurnal modulation ...")

    hours = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    days = df["timestamp"].dt.dayofweek

    # Diurnal envelope: busy 8am-8pm, quiet overnight
    # Peak at ~2pm (hour 14), trough at ~4am (hour 4)
    diurnal = 0.3 * np.sin(2 * np.pi * (hours - 4) / 24) + 0.7
    diurnal = np.clip(diurnal, 0.4, 1.0)

    # Weekend dampening (20% less load)
    weekend = np.where(days >= 5, 0.8, 1.0)

    # Apply modulation: base_cpu * diurnal * weekend
    cpu = df["cpu_usage"].values
    modulated = cpu * diurnal.values * weekend
    modulated = np.clip(modulated, 0.0, 1.0)

    df["cpu_usage"] = modulated

    print(f"       Diurnal range: [{diurnal.min():.2f}, {diurnal.max():.2f}]")
    print(f"       Weekend factor: 0.80")
    return df


# =============================================
# STEP 5: SCALE TO REALISTIC RANGE
# =============================================
def scale_to_range(df):
    """
    Scale CPU values to a realistic monitoring range (0.05 - 0.95)
    using robust percentile-based normalization.
    """
    print("[5/6] Scaling to realistic CPU range ...")

    cpu = df["cpu_usage"].values
    p1, p99 = np.percentile(cpu, [1, 99])
    print(f"       Pre-scale  p1/p99: [{p1:.6f}, {p99:.6f}]")

    # Robust normalization
    if p99 - p1 > 1e-8:
        scaled = (cpu - p1) / (p99 - p1)
    else:
        scaled = cpu - cpu.min()

    # Map to 0.05 - 0.95
    scaled = scaled * 0.9 + 0.05
    scaled = np.clip(scaled, 0.0, 1.0)
    df["cpu_usage"] = np.round(scaled, 6)

    print(f"       Post-scale range:  [{df['cpu_usage'].min():.4f}, "
          f"{df['cpu_usage'].max():.4f}]")
    print(f"       Mean: {df['cpu_usage'].mean():.4f}, "
          f"Std:  {df['cpu_usage'].std():.4f}")
    return df


# =============================================
# STEP 6: FORMAT & SAVE
# =============================================
def format_and_save(df, output_path):
    """Save in the exact format preprocessing.py expects."""
    print(f"[6/6] Saving to {output_path} ...")

    df["timestamp"] = df["timestamp"].dt.strftime("%d-%m-%Y %H:%M:%S")
    df = df[["timestamp", "cpu_usage"]].reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"       Saved {len(df):,} rows")
    return df


# =============================================
# STATISTICS
# =============================================
def print_statistics(df):
    ts = pd.to_datetime(df["timestamp"], dayfirst=True)
    cpu = df["cpu_usage"]
    spikes = (cpu > SPIKE_THRESHOLD).sum()
    spike_pct = spikes / len(df) * 100

    print()
    print("=" * 60)
    print("  Google Cluster Trace -- Processing Complete")
    print("=" * 60)
    print(f"  Total rows:       {len(df):,}")
    print(f"  Date range:       {ts.min()} -> {ts.max()}")
    print(f"  Duration:         {ts.max() - ts.min()}")
    print(f"  Interval:         {RESAMPLE_INTERVAL_SEC}s")
    print(f"  CPU mean:         {cpu.mean():.4f}")
    print(f"  CPU std:          {cpu.std():.4f}")
    print(f"  CPU min/max:      [{cpu.min():.4f}, {cpu.max():.4f}]")
    print(f"  Spikes (>{SPIKE_THRESHOLD*100:.0f}%):    "
          f"{spikes:,} ({spike_pct:.2f}%)")
    print(f"  Output:           {OUTPUT_PATH}")
    print(f"  Columns:          {list(df.columns)}")
    print()
    print("  Drop-in replacement for cpu_timeseries.csv")
    print("  Usage: df = pd.read_csv('data/google_cluster_processed.csv')")
    print("=" * 60)


# =============================================
# MAIN
# =============================================
def main():
    t0 = time.time()

    # Parse args
    target = TARGET_ROWS
    if "--target-rows" in sys.argv:
        idx = sys.argv.index("--target-rows")
        target = int(sys.argv[idx + 1])

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found!")
        print(f"Place the Google Cluster Trace CSV at: {INPUT_PATH}")
        sys.exit(1)

    # Pipeline
    raw = load_raw(INPUT_PATH)
    machine_ts, machine_counts = build_per_machine_series(raw)
    del raw  # free memory

    df = stitch_machines(machine_ts, machine_counts, target)
    del machine_ts, machine_counts

    df = add_diurnal_pattern(df)
    df = scale_to_range(df)
    df = format_and_save(df, OUTPUT_PATH)

    print_statistics(df)
    print(f"  Processing time:  {time.time() - t0:.1f}s")
    print()


if __name__ == "__main__":
    main()
