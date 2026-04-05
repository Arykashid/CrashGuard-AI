import pandas as pd
import os


RESULT_FILE = "experiment_results.csv"


def log_experiment(model_name, rmse, mae):

    new_row = {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae
    }

    if os.path.exists(RESULT_FILE):
        df = pd.read_csv(RESULT_FILE)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(RESULT_FILE, index=False)

    return df


def load_experiments():

    if os.path.exists(RESULT_FILE):
        return pd.read_csv(RESULT_FILE)

    return pd.DataFrame(columns=["Model", "RMSE", "MAE"])