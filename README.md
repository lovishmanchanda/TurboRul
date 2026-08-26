# TurboRUL

Predictive maintenance system for turbofan jet engines — predicts Remaining Useful
Life (RUL) from multivariate sensor time-series, using NASA's C-MAPSS dataset.

Currently built and validated on **FD001** only (1 operating condition, 1 fault
mode). FD002-FD004 and the 2021 N-CMAPSS dataset are planned as later
generalization extensions (see Phase 9) — not part of the core pipeline yet.

Every modeling decision in this project — and why it was made, including
approaches that were tried and rejected — is logged in detail:
- [`DECISIONS.md`](DECISIONS.md) — Phase 1: EDA
- [`DECISIONS2.md`](DECISIONS2.md) — Phase 2: Feature engineering
- [`DECISIONS3.md`](DECISIONS3.md) — Phase 3: Baseline models

## Project Structure

```
turborul/
├── data/
│   ├── raw/             # Original NASA txt files (train_FD00X.txt, test_FD00X.txt, RUL_FD00X.txt) — not committed, see Dataset section
│   ├── processed/       # Feature-engineered train/test CSVs (committed — see DECISIONS2.md for how these are built)
│   └── external/        # N-CMAPSS (2021) data, for Phase 9 (not yet used)
├── notebooks/
│   ├── 01_eda.ipynb                  # RUL labeling, constant-sensor detection, correlation analysis
│   ├── 02_feature_engineering.ipynb  # Baseline normalization, rolling mean/std
│   └── 03_baseline_models.ipynb      # Linear Regression, Ridge, Random Forest, XGBoost
├── src/
│   ├── config.py         # Paths, column names, constants
│   ├── data_loader.py    # Load raw CMAPSS text files into DataFrames
│   ├── preprocessing.py  # RUL labeling (piecewise-linear), sensor filtering
│   ├── evaluation.py     # RMSE + official NASA asymmetric scoring function
│   ├── device.py         # PyTorch device auto-detection (MPS/CUDA/CPU), for Phase 4
│   └── models/           # (Phase 4+) sequence model definitions go here
├── api/                  # FastAPI serving endpoint — Phase 8
├── dashboard/             # Streamlit monitoring dashboard — Phase 8
├── models/                # Saved trained model artifacts
├── reports/figures/       # Generated plots
├── tests/                 # Unit tests for src/
├── requirements.txt
├── DECISIONS.md           # Phase 1 decision log
├── DECISIONS2.md          # Phase 2 decision log
├── DECISIONS3.md          # Phase 3 decision log
└── README.md
```

## Roadmap

- [x] Phase 0 — Setup, data understanding
- [x] Phase 1 — RUL labeling (piecewise-linear), EDA
- [x] Phase 2 — Feature engineering (per-engine baseline normalization, rolling mean/std)
- [x] Phase 3 — Baseline models (Linear Regression, Ridge, Random Forest, XGBoost — cross-validated comparison)
- [ ] Phase 4 — Sequence models (LSTM, CNN-LSTM, Transformer) — GPU (MPS) training
- [ ] Phase 5 — Official asymmetric scoring evaluation on held-out test set
- [ ] Phase 6 — Uncertainty quantification (MC-Dropout / quantile regression)
- [ ] Phase 7 — Explainability (SHAP)
- [ ] Phase 8 — Productize: FastAPI + Streamlit dashboard + Docker
- [ ] Phase 9 (extension) — Generalize to FD002-FD004, then N-CMAPSS (2021 real-flight-condition data)

## Baseline model results (Phase 3)

5-fold cross-validated, engine-level splits (never splitting a single
engine's cycles across train/validation). Full reasoning and search process
in [`DECISIONS3.md`](DECISIONS3.md).

| Model | Mean RMSE (CV) | Notes |
|---|---|---|
| Linear Regression | ~16.4 (single split) | Baseline reference |
| Ridge (alpha=35000) | ~15.6 (single split) | Regularization confirmed useful (multicollinear features) |
| Random Forest (top-30 features, depth=10) | 12.80 | Single-split score (11.48) was optimistic — corrected via CV |
| **XGBoost (depth=4, lr=0.01, n_estimators=500)** | **12.43** | **Current best baseline** |

## Dataset

NASA C-MAPSS Turbofan Engine Degradation Simulation (2008), subset FD001.
Download from the NASA Prognostics Data Repository and place the raw `.txt`
files (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) in `data/raw/`.
Not committed to this repo — see `.gitignore`.

## Setup

Use any Python environment manager you prefer (conda, venv, etc.) with
Python 3.10+, then:

```bash
pip install -r requirements.txt
```

If training on Apple Silicon, verify GPU (MPS) is available for the
upcoming Phase 4:
```bash
python -m src.device
```