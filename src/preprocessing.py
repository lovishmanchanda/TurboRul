"""
Preprocessing: RUL labeling and normalization.
"""

import pandas as pd

from src.config import RUL_CLIP_VALUE


def add_rul_to_train(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train files contain full run-to-failure trajectories, so RUL at each
    row = (max cycle for that engine) - (current cycle).

    Applies the standard piecewise-linear clip: RUL is capped at
    RUL_CLIP_VALUE, since engines far from failure don't show a
    meaningful linear degradation signal and treating them as "at max
    health" is both more realistic and improves model performance.
    """
    df = train_df.copy()
    max_cycle_per_unit = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["RUL"] = max_cycle_per_unit - df["time_in_cycles"]
    df["RUL"] = df["RUL"].clip(upper=RUL_CLIP_VALUE)
    return df


def add_rul_to_test(test_df: pd.DataFrame, rul_truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Test files are truncated before failure. The RUL at the *last* observed
    cycle for each engine is given in RUL_FD00X.txt. To get RUL at every
    row (not just the last), we back-compute it the same way as train,
    then add the truth-file offset for the final cycle.
    """
    df = test_df.copy()
    max_cycle_per_unit = df.groupby("unit_number")["time_in_cycles"].transform("max")
    cycles_from_end = max_cycle_per_unit - df["time_in_cycles"]

    rul_at_truncation = df["unit_number"].map(
        rul_truth_df.set_index("unit_number")["RUL"]
    )
    df["RUL"] = cycles_from_end + rul_at_truncation
    df["RUL"] = df["RUL"].clip(upper=RUL_CLIP_VALUE)
    return df


def drop_constant_sensors(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """
    Drop sensor columns with (near) zero variance. Run this AFTER checking
    which sensors are actually constant for the specific subset you're
    working with (FD002/FD004 behave differently from FD001/FD003) —
    don't blindly reuse a hardcoded list across subsets.
    """
    to_drop = [c for c in sensor_cols if df[c].std() < 1e-6]
    return df.drop(columns=to_drop), to_drop
