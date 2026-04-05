import pandas as pd

print("Loading dataset...")

# load only needed rows
df = pd.read_csv(
    "data/google_cluster_usage.csv",
    header=None,
    nrows=200000
)

print("Dataset loaded")

# select useful columns
df = df[[0,5]]
df.columns = ["timestamp","cpu_usage"]

# convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us")

# sort by time
df = df.sort_values("timestamp")

# aggregate to make a single CPU time series
cpu_series = df.groupby("timestamp")["cpu_usage"].mean().reset_index()

cpu_series.to_csv("data/cpu_timeseries.csv", index=False)

print("CPU time series created")
print(cpu_series.head())