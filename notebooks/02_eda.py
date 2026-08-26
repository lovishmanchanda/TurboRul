# %% [markdown]
# # 02 — Feature Engineering: FD001
#
# Continues from 01_eda.ipynb. Covers:
# - Per-engine baseline normalization
# - Rolling window features (mean, std)
#
# See DECISIONS2.md for reasoning behind each choice made here.

# %%
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

import matplotlib.pyplot as plt
import pandas as pd

from src.config import SENSOR_COLS
from src.data_loader import load_subset
from src.preprocessing import add_rul_to_train, add_rul_to_test

pd.set_option("display.max_columns", 40)

# %% [markdown]
# ## Reload FD001 and redo the steps from Phase 1
# (RUL labeling + dropping constant sensors) so this notebook is
# self-contained and doesn't depend on variables still sitting in memory
# from 01_eda.ipynb.

# %%
train, test, rul_truth = load_subset("FD001")
train = add_rul_to_train(train)
test = add_rul_to_test(test, rul_truth)

constant_sensors = ["sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
train_reduced = train.drop(columns=constant_sensors)
test_reduced = test.drop(columns=constant_sensors)

print("train_reduced shape:", train_reduced.shape)
train_reduced.head()

# %% [markdown]
# ## Step 1: Per-engine baseline normalization (sensor_14 first, as a test case)

# %%
# to be filled in together