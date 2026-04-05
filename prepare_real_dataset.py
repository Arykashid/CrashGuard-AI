import pandas as pd

print("Loading dataset...")

df = pd.read_csv(
    "data/google_cluster_usage.csv",
    header=None,
    nrows=100000
)

print("Dataset loaded successfully")
print(df.head())