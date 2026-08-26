# %% [markdown]
# # 03 — Baseline Models: FD001
#
# Continues from 02_feature_engineering.ipynb. Covers:
# - Linear Regression (with scaling)
# - Random Forest
# - XGBoost
# - Comparing all three on validation set using RMSE + NASA asymmetric score
#
# See DECISIONS3.md for reasoning behind each choice made here.
#
# NOTE ON GPU: scikit-learn (Linear Regression, Random Forest) and XGBoost's
# default CPU mode do not use GPU acceleration on any machine — this is a
# library limitation, not specific to Mac. GPU (via PyTorch MPS on this M2)
# becomes relevant starting in the next phase (sequence models / LSTM).

# %%
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import SENSOR_COLS
from src.data_loader import load_subset
from src.preprocessing import add_rul_to_train, add_rul_to_test
from src.evaluation import rmse, nasa_score, evaluate

pd.set_option("display.max_columns", 40)

# %% [markdown]
# ## Rebuild train_final / test_final from scratch
# (Self-contained, same as previous notebooks — doesn't rely on variables
# still in memory from 02_feature_engineering.ipynb.)

# %%
train, test, rul_truth = load_subset("FD001")
train = add_rul_to_train(train)
test = add_rul_to_test(test, rul_truth)

constant_sensors = ["sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
train_reduced = train.drop(columns=constant_sensors)
test_reduced = test.drop(columns=constant_sensors)
sensor_cols_remaining = [c for c in SENSOR_COLS if c not in constant_sensors]


def engineer_features(df, sensor_cols, windows=[5, 10]):
    """
    Applies baseline normalization, rolling mean, and rolling std
    to a CMAPSS dataframe (train OR test) — entirely per-engine,
    so it's safe to call independently on each. See DECISIONS2.md
    for reasoning behind each step.
    """
    df = df[["unit_number", "time_in_cycles", "RUL"] + sensor_cols].copy()

    for sensor in sensor_cols:
        baseline_avg = (
            df[df["time_in_cycles"] <= 10]
            .groupby("unit_number")[sensor]
            .mean()
        )
        baseline_mapped = df["unit_number"].map(baseline_avg)
        df[f"{sensor}_norm"] = df[sensor] - baseline_mapped

    for window in windows:
        for sensor in sensor_cols:
            for suffix in ["", "_norm"]:
                col = f"{sensor}{suffix}"
                new_col = f"{col}_rollmean_{window}"
                df[new_col] = (
                    df.groupby("unit_number")[col]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )

    for window in windows:
        for sensor in sensor_cols:
            new_col = f"{sensor}_rollstd_{window}"
            df[new_col] = (
                df.groupby("unit_number")[sensor]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
                .fillna(0)
            )

    return df


train_final = engineer_features(train_reduced, sensor_cols_remaining)
test_final = engineer_features(test_reduced, sensor_cols_remaining)

print("train_final:", train_final.shape)
print("test_final:", test_final.shape)
assert train_final.shape[1] == test_final.shape[1]

# %% [markdown]
# ## Engine-level train/validation split (see DECISIONS2.md entry 8 for why
# this must be split by engine, not by row)

# %%
np.random.seed(42)
all_engine_ids = train_final["unit_number"].unique()
np.random.shuffle(all_engine_ids)

n_val_engines = 20
val_engine_ids = all_engine_ids[:n_val_engines]
train_engine_ids = all_engine_ids[n_val_engines:]

train_split = train_final[train_final["unit_number"].isin(train_engine_ids)]
val_split = train_final[train_final["unit_number"].isin(val_engine_ids)]

print("Train engines:", len(train_engine_ids), "-> rows:", train_split.shape[0])
print("Val engines:", len(val_engine_ids), "-> rows:", val_split.shape[0])

# %% [markdown]
# ## Ready for modeling — to be built together from here