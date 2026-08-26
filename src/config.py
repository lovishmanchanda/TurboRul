"""
Central configuration: paths, column names, and constants shared across
the whole TurboRUL pipeline. Import from here instead of hardcoding
column names or magic numbers in notebooks/scripts.
"""

from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
EXTERNAL_DATA_DIR = ROOT_DIR / "data" / "external"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- CMAPSS column names ---
# Raw files have no header: 26 space-separated columns.
INDEX_COLS = ["unit_number", "time_in_cycles"]
SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]  # sensor_1 ... sensor_21

ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# --- Sensors known to be constant / near-constant in FD001 (single condition) ---
# Verify this yourself in EDA before dropping — it can differ for FD002/FD004
# since those have 6 operating conditions and some "constant" sensors vary there.
CANDIDATE_DROP_SENSORS_FD001 = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]

# --- RUL labeling ---
# Piecewise-linear RUL cap: engines far from failure are treated as having
# constant "full health" RUL rather than a linearly increasing value.
# 125 is the standard value used in CMAPSS literature.
RUL_CLIP_VALUE = 125

# --- Sequence modeling ---
SEQUENCE_LENGTH = 30  # sliding window length (cycles) fed into LSTM/CNN models

# --- Subsets ---
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

SUBSET_INFO = {
    "FD001": {"conditions": 1, "fault_modes": 1},
    "FD002": {"conditions": 6, "fault_modes": 1},
    "FD003": {"conditions": 1, "fault_modes": 2},
    "FD004": {"conditions": 6, "fault_modes": 2},
}
