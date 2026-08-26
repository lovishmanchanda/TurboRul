"""
Load raw C-MAPSS text files (train_FD00X.txt, test_FD00X.txt, RUL_FD00X.txt)
into pandas DataFrames with proper column names.

Usage:
    from src.data_loader import load_subset
    train, test, rul = load_subset("FD001")
"""

import pandas as pd

from src.config import ALL_COLS, RAW_DATA_DIR


def _read_cmapss_txt(path):
    """
    CMAPSS txt files are whitespace-separated with no header, and often
    have trailing whitespace that creates two extra empty columns if you
    naively split — this handles that.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")  # drop any fully-empty trailing columns
    df.columns = ALL_COLS
    return df


def load_train(subset: str) -> pd.DataFrame:
    path = RAW_DATA_DIR / f"train_{subset}.txt"
    return _read_cmapss_txt(path)


def load_test(subset: str) -> pd.DataFrame:
    path = RAW_DATA_DIR / f"test_{subset}.txt"
    return _read_cmapss_txt(path)


def load_rul_truth(subset: str) -> pd.DataFrame:
    """
    RUL_FD00X.txt has one value per line: the true RUL of each test engine
    at the point its trajectory was truncated. Row i corresponds to
    unit_number i+1 in the test set.
    """
    path = RAW_DATA_DIR / f"RUL_{subset}.txt"
    rul = pd.read_csv(path, sep=r"\s+", header=None, names=["RUL"])
    rul["unit_number"] = rul.index + 1
    return rul[["unit_number", "RUL"]]


def load_subset(subset: str):
    """Convenience loader: returns (train_df, test_df, rul_truth_df)."""
    train = load_train(subset)
    test = load_test(subset)
    rul = load_rul_truth(subset)
    return train, test, rul


if __name__ == "__main__":
    # Quick sanity check when run directly: python -m src.data_loader
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        try:
            train, test, rul = load_subset(subset)
            print(f"{subset}: train={train.shape}, test={test.shape}, rul={rul.shape}")
        except FileNotFoundError as e:
            print(f"{subset}: file not found — {e}")
