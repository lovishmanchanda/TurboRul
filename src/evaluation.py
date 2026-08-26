"""
Evaluation metrics for RUL prediction.

Includes the official NASA/PHM08 asymmetric scoring function, which is the
metric actually used in the CMAPSS literature to compare models — RMSE
alone understates how costly a late prediction (predicted RUL > true RUL)
is compared to an early one, since predicting an engine has more life left
than it actually does risks an in-service failure.
"""

import numpy as np


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nasa_score(y_true, y_pred):
    """
    Official PHM08/CMAPSS scoring function.

    d = y_pred - y_true
      d < 0  -> early prediction (predicted RUL lower than actual): mild penalty
      d >= 0 -> late prediction (predicted RUL higher than actual): steep penalty

    score = sum(exp(-d/13) - 1)          for d < 0
          + sum(exp( d/10) - 1)          for d >= 0

    Lower is better. Note this is NOT symmetric — a model that tends to
    under-predict RUL (conservative, recommends maintenance early) scores
    much better than one that over-predicts by the same margin.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    early = d[d < 0]
    late = d[d >= 0]
    score = np.sum(np.exp(-early / 13) - 1) + np.sum(np.exp(late / 10) - 1)
    return float(score)


def evaluate(y_true, y_pred):
    """Returns both metrics as a dict for easy logging (e.g. to MLflow)."""
    return {
        "rmse": rmse(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
    }
