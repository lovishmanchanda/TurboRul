# %% [markdown]
# # 01 — EDA: NASA C-MAPSS FD001
#
# This is a "percent" script — open it in VS Code with the Jupyter extension
# and it behaves like a notebook (# %% marks cell boundaries), or convert it
# with `jupytext` if you prefer a real .ipynb. Either way, run cell by cell.
# jupytext --to notebook script.py
# %%
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))  # so `from src...` imports work

import matplotlib.pyplot as plt
import pandas as pd

from src.config import SENSOR_COLS
from src.data_loader import load_subset
from src.preprocessing import add_rul_to_train, add_rul_to_test, drop_constant_sensors

pd.set_option("display.max_columns", 40)

# %% [markdown]
# ## Load FD001

# %%
train, test, rul_truth = load_subset("FD001")
print("train:", train.shape)
print("test:", test.shape)
print("rul_truth:", rul_truth.shape)
train.head()

# %% [markdown]
# ## Basic shape checks
# - How many engines? How many cycles does each run for before failure?

# %%
n_engines = train["unit_number"].nunique()
cycles_per_engine = train.groupby("unit_number")["time_in_cycles"].max()
print(f"{n_engines} engines in train")
cycles_per_engine.describe()

# %%
cycles_per_engine.hist(bins=30)
plt.xlabel("Cycles to failure")
plt.ylabel("Number of engines")
plt.title("Distribution of engine lifetimes (FD001 train)")
plt.show()

# %% [markdown]
# ## Add RUL labels

# %%
train = add_rul_to_train(train)
test = add_rul_to_test(test, rul_truth)
train[["unit_number", "time_in_cycles", "RUL"]].head()

# %% [markdown]
# ## Which sensors are constant? (candidates to drop)
# Don't trust the hardcoded list in config.py blindly — verify per subset.

# %%
train_reduced, dropped = drop_constant_sensors(train, SENSOR_COLS)
print("Dropped (near-)constant sensors:", dropped)

# %% [markdown]
# ## Sensor trends vs RUL for a single engine
# This is where you build intuition for which sensors actually degrade.

# %%
engine_id = 1
engine_data = train[train["unit_number"] == engine_id]

remaining_sensors = [c for c in SENSOR_COLS if c not in dropped]
fig, axes = plt.subplots(len(remaining_sensors), 1, figsize=(8, 2.2 * len(remaining_sensors)), sharex=True)
for ax, sensor in zip(axes, remaining_sensors):
    ax.plot(engine_data["time_in_cycles"], engine_data[sensor])
    ax.set_ylabel(sensor, rotation=0, labelpad=30, fontsize=8)
axes[-1].set_xlabel("Cycle")
fig.suptitle(f"Sensor trends over lifetime — engine {engine_id}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Correlation of each sensor with RUL
# Sensors with strong (negative, usually) correlation with RUL are your
# strongest degradation indicators — prioritize these in feature engineering.

# %%
corrs = train[remaining_sensors + ["RUL"]].corr()["RUL"].drop("RUL").sort_values()
corrs.plot(kind="barh", figsize=(6, 6))
plt.title("Sensor correlation with RUL")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Next steps
# - Repeat this for FD002/FD003/FD004 — note how sensor behavior differs
#   with multiple operating conditions (FD002/FD004).
# - Move to `src/features.py` (to be built) for rolling-window feature
#   engineering once you've picked your key sensors here.
