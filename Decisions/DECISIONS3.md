# TurboRUL — Decision Log: Phase 3 (Baseline Models)

Continues from DECISIONS.md (Phase 1 — EDA) and DECISIONS2.md (Phase 2 —
Feature Engineering). This file covers model choices, hyperparameter
decisions, and evaluation results for baseline (non-deep-learning) models.

---

## 1. Hyperparameters — approach for this phase
**Decision:** Do not use default/arbitrary hyperparameter values without
justification. For each model, key hyperparameters will be introduced with
an explanation of what each controls and its tradeoff, then tested with a
few concrete values against the validation set (val_split), with the choice
made based on results — not assumed upfront.

---

## 2. Scaling — needed for Linear Regression only
- **Reasoning:** Linear Regression is sensitive to features being on very
different numeric scales. Random Forest and XGBoost are tree-based and
split on per-feature thresholds independently of scale, so they don't
require it.
- **Leakage-safety plan:** Any scaler used will be fit strictly on
train_split only, then the same fitted parameters applied to val_split and
test_final without refitting — consistent with the leakage principle
established in DECISIONS2.md entry 7.

---

## 3. No categorical encoding needed
**Reasoning:** All features in this dataset are already numeric. Encoding is
only needed for categorical/text data, which does not exist in this
dataset.

---

## 4. Model 1: Linear Regression — why chosen as the first baseline
- **What it does:** Predicts RUL as a weighted sum of all input features,
learning the weights that minimize prediction error across training rows.
- **Why start here:** Simplest model capable of learning a real pattern —
establishes an honest reference point for every future model. Fast, fully
interpretable, standard practice across the ML field.
- **Known limitation:** Assumes a straight-line relationship between each
feature and RUL — engine degradation is plausibly non-linear.

---

## 5. Feature/target preparation for modeling
- **Excluded from features:** `unit_number` (arbitrary ID) and
`time_in_cycles` (excluded because test engines are truncated at somewhat
arbitrary points — including cycle number risks the model learning "test
engines get cut off around cycle X" instead of genuine sensor-based
degradation patterns).
- **Result:** 112 usable features.

---

## 6. Scaling — implemented with StandardScaler, fit on train only
- **Method:** `scaler.fit_transform(X_train)` learns mean/std from train;
`scaler.transform(X_val)` (not fit_transform) reuses those values on
validation — direct implementation of the leakage-safety rule.
- **Verified:** X_train_scaled mean ≈ 0 (confirmed via np.isclose(), small
non-zero value was floating-point rounding noise) and std ≈ 1.

---

## 7. Linear Regression — baseline results (OFFICIAL BASELINE NUMBER)

| | RMSE | NASA score (per row) |
|---|---|---|
| Train | 15.79 | 3.93 |
| Validation | 16.42 | 4.81 |

- **Metric-calculation catch:** Initially compared raw (summed) NASA scores
between train (16,561 rows) and val (4,070 rows) and saw a misleadingly
large gap — nasa_score sums over all rows rather than averaging, so a
smaller dataset produces a smaller sum regardless of model quality. Fixed by
dividing by row count before comparing train vs. val.
- **Interpretation:** Train/val performance close on both metrics — healthy,
no serious overfitting. NASA/row grew proportionally more than RMSE,
consistent with its asymmetric design (penalizes late/optimistic
predictions much more heavily).
- **Status:** RMSE ≈ 16.42, NASA/row ≈ 4.81 (val) — official baseline for
every future model.

---

## 8. Model 2: Ridge Regression — hyperparameter search for alpha
- **What it adds:** A penalty shrinking feature weights toward zero — chosen
because the 112 features have meaningful multicollinearity (raw/normalized/
rolling variants of the same sensors).
- **Search — three rounds, progressively narrowed:**
1. Coarse sweep (0.001-1000): improved steadily across the whole range, no
   turning point yet — signal the true optimum was beyond the tested range.
2. Extended sweep (1000-1,000,000): found the turning point. Best RMSE at
   alpha=10000 (15.62); best NASA/row at alpha=50000 (3.824). Beyond
   alpha=100000, both metrics collapsed sharply (underfitting — at very
   high alpha, the weight penalty dominates and predictions collapse toward
   guessing near the average RUL, ignoring the actual data).
3. Narrowed sweep (10000-50000, step 5000): best RMSE at alpha=25000
   (15.578); best NASA/row at alpha=35000 (3.812).
- **Final choice: alpha=35000** — chosen over alpha=25000 (best RMSE) since
NASA score is the more operationally meaningful metric, and the RMSE
difference between the two was negligible.

---

## 9. Ridge (alpha=35000) vs Linear Regression baseline

| Model | RMSE | NASA score (per row) |
|---|---|---|
| Linear Regression (baseline) | 16.42 | 4.81 |
| Ridge (alpha=35000) | 15.60 | 3.81 |

**Conclusion:** Real, evidence-backed improvement over plain Linear
Regression — confirms the multicollinearity hypothesis. New benchmark going
into tree-based models.

---

## 10. Model 3: Random Forest — what it is and why tried next
- **What it does:** Trains many decision trees (each on a random subset of
rows and a random subset of features per split), then averages their
predictions. Individual trees overfit in different, semi-random directions;
averaging cancels out most of that individual noise.
- **Why tried after Linear/Ridge:** Directly addresses their known weakness —
no straight-line assumption, can capture non-linear degradation patterns.
No scaling required (tree splits are scale-independent).

---

## 11. Random Forest — initial grid search (n_estimators × max_depth)
- **Method:** Full grid, 5 values each of n_estimators [50,100,200,300,500]
and max_depth [5,10,15,20,None], on raw (unscaled) features.
- **Result:** Massive improvement over Ridge — best result (n_estimators=500,
max_depth=None): val RMSE 11.37 vs Ridge's 15.60, val NASA/row 2.44 vs
Ridge's 3.81. Confirmed the non-linearity hypothesis.
- **Red flag noticed before accepting at face value:** max_depth=None
(unlimited) and n_estimators=500 (largest tested value) both won — same
"best result sits at the edge of the tested range" pattern seen with
Ridge's alpha, meaning the true optimum may not have been found yet.

---

## 12. Overfitting check — max_depth=None vs finite depths
- **Method:** Compared train vs. validation RMSE (not just validation alone)
across depths, to check whether max_depth=None's edge-of-range win was real
signal or an overfitting artifact hidden by Random Forest's
ensemble-averaging effect.

| max_depth | Train RMSE | Val RMSE | Gap |
|---|---|---|---|
| 5 | 11.24 | 12.81 | 1.56 |
| 10 | 5.57 | 11.50 | 5.93 |
| 15 | 2.92 | 11.38 | 8.46 |
| 20 | 2.23 | 11.35 | 9.12 |
| None | 1.98 | 11.37 | ~9.4 |

- **Interpretation:** Going from depth 10 to 20 (or unlimited) buys almost
nothing on validation (~1%) but costs a massively larger train-val gap —
far more reliance on memorized training specifics for negligible benefit.
Depth 5 underfits (worse on both train and val). **Decision: max_depth=10**
— prioritizes a robust, generalizable model over squeezing out the last ~1%
of validation score.

---

## 13. n_estimators — extended range check
- **Method:** Tested n_estimators [500,700,1000,1500] at max_depth=20 to
check if 500 (edge of the original grid) was truly optimal.
- **Result:** val RMSE barely changed (11.377 → 11.364 → 11.353 → 11.360) —
essentially flat, confirming performance plateaus well before 1500.
- **Decision:** n_estimators=1000 (marginal best, negligible difference vs
500-1500).

---

## 14. Evaluating an external suggestion — tested, not adopted on faith
Two suggestions received: (a) check feature_importances_ and drop
low-importance features; (b) keep max_depth strictly between 4-8 and
increase min_samples_leaf to 10-20.

**(a) Feature importance — tested and CONFIRMED:**

| Top N features | Cumulative importance |
|---|---|
| 15 | 93.3% |
| 20 | 95.5% |
| 30 | 97.2% |

Top feature: `sensor_11_norm_rollmean_10` alone = 59.6% of total importance
— notable that sensor_11 was also the single strongest raw correlation with
RUL found back in Phase 1 EDA (-0.775), independently confirmed here by the
model. The "top ~20 features capture ~95%" claim was accurate.

**(b) max_depth 4-8 — REJECTED, contradicted by our own evidence.** Entry
12 already showed max_depth=5 underfits (val RMSE 12.81, clearly worse than
depth 10's 11.50). Kept max_depth=10 instead.

**(b) min_samples_leaf — tested independently, PARTIALLY CONFIRMED:**

| min_samples_leaf | Train RMSE | Val RMSE | Gap | Val NASA/row |
|---|---|---|---|---|
| 1 | 5.57 | 11.50 | 5.93 | 2.620 |
| 5 | 5.74 | 11.50 | 5.75 | 2.622 |
| 10 | 6.10 | 11.48 | 5.38 | 2.605 |
| 20 | 6.89 | 11.49 | 4.60 | 2.593 |
| 50 | 8.60 | 11.58 | 2.97 | 2.561 |

Gap shrinks steadily as min_samples_leaf increases, val RMSE stays
essentially flat through leaf=20, then degrades slightly at leaf=50 (mild
underfitting). **Decision: min_samples_leaf=20** — meaningfully smaller gap
than leaf=1, no real val cost.

**Takeaway:** Testing external suggestions against this project's own
evidence caught one bad suggestion (max_depth 4-8) and confirmed two good
ones — reinforces the project's evidence-first approach over accepting
advice at face value.

---

## 15. Feature trimming — top 30 vs all 112 features
**Method:** Retrained final config (max_depth=10, min_samples_leaf=20,
n_estimators=1000) using only the top 30 features by importance, compared
against the full 112-feature version.

| | Val RMSE | Val NASA/row |
|---|---|---|
| Full 112 features | 11.486 | 2.593 |
| Top 30 features | 11.476 | 2.678 |

**Interpretation:** RMSE essentially identical (top-30 even marginally
better); NASA/row slightly worse with top-30, likely noise at this scale.
Confirms most of the 112 features add negligible predictive value on top of
the top ~30. **Decision: use top-30-feature model** — same accuracy, less
than a third of the features, faster training, smaller overfitting surface.

---

## 16. FINAL Random Forest model and updated scoreboard
**Final config:** top-30 features (by importance), max_depth=10,
min_samples_leaf=20, n_estimators=1000.

| Model | Val RMSE | Val NASA/row |
|---|---|---|
| Linear Regression | 16.42 | 4.81 |
| Ridge (alpha=35000) | 15.60 | 3.81 |
| **Random Forest (final)** | **11.48** | **2.68** |

Random Forest is now the benchmark to beat going into XGBoost.

---
## 17. Model 4: XGBoost — what it is and why different from Random Forest
- **What it does:** Builds decision trees sequentially, not independently —
each new tree is trained specifically to predict the error (residual) of
all previous trees combined, and its correction is added on top
(gradient boosting), rather than averaging many independent trees.
- **Key new hyperparameter vs Random Forest — learning_rate:** controls how
much weight each new tree's correction gets. Low learning_rate = many small,
careful corrections (needs more trees, usually generalizes better); high
learning_rate = fewer, bigger corrections (faster but riskier).
- **Expected tradeoff going in:** Because each tree directly targets training
error, XGBoost was expected to overfit more aggressively than Random Forest
if unchecked — a hypothesis to verify with evidence, not assumed as fact.

---

## 18. XGBoost — initial grid search (max_depth × learning_rate × n_estimators)
- **Method:** Grid of max_depth [3,4,5,6] (kept shallow — standard practice
for boosting trees, unlike Random Forest's much wider range, since each
tree here is meant to be a small incremental corrector not a strong
standalone model), learning_rate [0.01,0.05,0.1,0.3], n_estimators
[100,300,500]. Gap (val_rmse - train_rmse) computed from the start this
time, not as an afterthought.
- **Result:** Best val_rmse (11.554) at max_depth=5, learning_rate=0.01,
n_estimators=500 — but gap for that config was 4.01. Scanning the gap
column across all top results showed gap growing sharply with depth/
learning_rate/n_estimators (from ~2.7 up to ~10.6 at the most aggressive
settings) — confirmed the overfitting hypothesis: XGBoost's gap grew far
more steeply than Random Forest's did across a comparable hyperparameter
range.
- **Notable candidate:** max_depth=4, learning_rate=0.01, n_estimators=500 —
val_rmse 11.69 (only 0.14 worse than the top result) but gap only 2.72
(smallest in the top 15) — flagged as the more conservative choice.

---

## 19. Critical review of the modeling approach (external input, evaluated
## before acting on it)
- **Concern raised:** Selecting a model purely by lowest val_rmse OR purely by
smallest gap is both flawed — a small gap alone can indicate underfitting
rather than good generalization (as already seen with Random Forest's
max_depth=5, which had a small gap but was genuinely worse on both train
and val). Also flagged that only 3 of XGBoost's many hyperparameters had
been tested — concluding "XGBoost inherently overfits" from an incomplete
search would repeat the same mistake made earlier with Ridge's alpha (see
DECISIONS3.md entry 8) and Random Forest's initial edge-of-range results.
- **Response/decision:** Agreed with the critique. Rather than running a full
5-hyperparameter regularization search with cross-validation on everything
(judged too computationally heavy for a baseline-models phase), agreed on a
scoped plan: test 2 of the most relevant regularization hyperparameters
(subsample, min_child_weight) on the already-promising depth=4/lr=0.01/
n_estimators=500 region using the existing val split (cheap), then run
5-fold cross-validation only on the top 2-3 resulting candidates (expensive,
reserved for final confirmation) — sequencing suggested directly.
- **New hyperparameters introduced:** subsample (row sampling fraction per
tree, similar to Random Forest's bagging), min_child_weight (minimum data
required to allow a split — higher is more conservative). Also introduced:
colsample_bytree, reg_alpha, reg_lambda (not tested this round, flagged as
available for future tuning if needed).

---

## 20. XGBoost — regularization grid (subsample × min_child_weight)
- **Method:** subsample [0.6,0.8,1.0] x min_child_weight [1,5,10], fixed at
max_depth=4, learning_rate=0.01, n_estimators=500 (the flagged conservative
region from entry 18).
- **Result:** All 9 combinations landed in a tight band (val_rmse 11.62-11.69,
gap 2.68-2.72) — far less spread than the depth/learning_rate/n_estimators
sweep, indicating these two hyperparameters have only a marginal effect on
this dataset compared to the earlier three. Top 3 candidates (by val_rmse,
differences <0.02, likely noise): subsample=0.8/min_child_weight=5;
subsample=0.6/min_child_weight=10; subsample=0.8/min_child_weight=10.

---

## 21. 5-fold cross-validation — top 3 XGBoost candidates
- **Method:** KFold(n_splits=5, shuffle=True, random_state=42) applied to the
full pool of 100 engine IDs (train_final, not just the 80-engine
train_split) — folds built on engine IDs, never splitting a single engine's
cycles across train/val within a fold, consistent with the engine-level
split principle (DECISIONS2.md entry 8).
- **Result:**
| subsample | min_child_weight | mean_rmse | std_rmse |
|---|---|---|---|
| 0.8 | 5 | 12.431 | 0.564 |
| 0.6 | 10 | 12.426 | 0.577 |
| 0.8 | 10 | 12.426 | 0.557 |

- **Key finding:** All 3 candidates are statistically indistinguishable —
differences between them (~0.005 RMSE) are far smaller than the
fold-to-fold variation within any single candidate (std ~0.56-0.58). The
apparent ranking from the single-val-split grid (entry 20) was mostly noise,
not a real difference between these hyperparameter values.
- **Final choice: subsample=0.8, min_child_weight=10** — statistically tied
with the others, chosen for having the (marginally) lowest std_rmse.

---

## 22. Fair comparison — cross-validated Random Forest vs cross-validated
## XGBoost (the real, honest result)
- **Motivation:** Since CV revealed the single-split Random Forest score
(11.48) might also be optimistic/lucky, re-ran the SAME 5-fold CV (identical
folds, same random_state=42) on the final Random Forest config (top-30
features, max_depth=10, min_samples_leaf=20, n_estimators=1000) for a fair,
apples-to-apples comparison.
- **Result:**
| Model | Mean RMSE (5-fold CV) | Std RMSE |
|---|---|---|
| Random Forest (final config) | 12.798 | 0.845 |
| XGBoost (subsample=0.8, min_child_weight=10) | 12.426 | 0.557 |

- **Critical finding:** XGBoost actually wins under fair cross-validated
comparison — the OPPOSITE of what the single-split numbers suggested
(Random Forest's single-split val_rmse of 11.48 looked better than
XGBoost's ~11.6-11.9, but that Random Forest number turned out to be its
single luckiest fold — other folds ranged 12.66-13.98). XGBoost also showed
meaningfully lower variance across folds (std 0.557 vs 0.845), indicating
more stable, consistent performance regardless of which engines land in
validation.
- **Lesson:** A single 80/20 val split can meaningfully mislead model
selection — this was caught only because cross-validation was applied
consistently to both candidate models rather than trusting the original
single-split leaderboard. Also implies the eventual test_final RMSE may
land closer to Random Forest's cross-validated ~12.8 or XGBoost's ~12.4,
rather than the more optimistic single-split numbers seen earlier in this
phase (Ridge, Linear Regression included) — those may carry similar
optimism, not yet re-verified with CV.

---

## 23. FINAL baseline model selection
- **Winner: XGBoost** — max_depth=4, learning_rate=0.01, n_estimators=500,
subsample=0.8, min_child_weight=10.
- **Cross-validated performance:** mean RMSE 12.426, std 0.557 (5-fold,
engine-level splits).
- **Status:** This is the baseline model to beat once the deep learning
(LSTM/sequence model) phase begins.

---
