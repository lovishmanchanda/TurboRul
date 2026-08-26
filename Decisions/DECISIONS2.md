# TurboRUL — Decision Log: Phase 2 (Feature Engineering)

Continues from DECISIONS.md (Phase 1 — EDA). This file covers every choice
made while turning raw sensor readings into model-ready features.

---

## 1. Per-engine baseline normalization — tested on sensor_14
**Motivation:** Phase 1 (DECISIONS.md, entry 5) found that sensor_14's pooled
correlation with RUL (-0.370) looked weak despite a clear visual downward
trend within a single engine — traced to engines having different starting
baselines for this sensor. Hypothesis: subtracting each engine's own baseline
from its readings should isolate the real degradation signal and strengthen
the correlation.

**Attempt 1 — baseline = engine's cycle-1 reading:**
```
sensor_14_normalized = sensor_14 - (engine's value at cycle 1)
```
Result: correlation improved from -0.370 to -0.397. Direction confirms the
hypothesis, but the improvement is small.

**Attempt 2 — baseline = average of engine's first 10 cycles:**
Reasoning: a single cycle-1 reading can itself be noisy/unrepresentative: any
error in that one reading gets baked into every row for that engine, since
the same number is subtracted throughout. Averaging over the first 10 cycles
smooths out one-off noise before using it as the baseline.
```
sensor_14_normalized_v2 = sensor_14 - (mean of engine's first 10 cycles)
```
Result: correlation improved further to -0.408.

**Conclusion:** Both attempts moved the correlation in the predicted
direction, confirming per-engine baseline offset is a real contributor to
sensor_14's weak correlation. However, the total improvement (-0.370 to
-0.408) is modest, not dramatic — baseline offset is a real but partial
explanation, not the full story. Two likely reasons more is going on:
  1. Correlation only measures linear relationships. If sensor_14's true
     relationship with RUL is non-linear (e.g. flat for most of life, then
     drops sharply near failure — a realistic degradation shape), a linear
     correlation coefficient will always understate the true relationship,
     no matter how well the baseline is normalized.
  2. sensor_14 may simply be a weaker/noisier signal than sensors like
     sensor_11 or sensor_4 — not every sensor needs to be equally predictive.
**Decision:** Accept baseline normalization as a worthwhile step (small,
real improvement, cheap to compute, no downside) and apply it to all 14
remaining sensors, not just sensor_14 — rather than over-fitting further
investigation to this one sensor. Move forward using average-of-first-10-cycles
as the baseline method, since it beat the single-cycle-1 version.

---

## 2. Applied baseline normalization to all 14 sensors
**What we did:** Generalized the sensor_14 experiment (Attempt 2: baseline =
mean of engine's first 10 cycles) to all 14 remaining sensors, using a loop.
**Bug caught along the way:** First attempt at gathering "all sensor columns"
used `[c for c in train_reduced.columns if c.startswith("sensor_")]`. This
silently picked up leftover intermediate columns from earlier one-off
experiments (`sensor_14_normalized`, `sensor_14_baseline_avg`, etc.), which
aren't real sensors — polluted the correlation table with garbage/NaN rows.
Fixed by explicitly using `[c for c in SENSOR_COLS if c not in constant_sensors]`
instead of inferring the list from column name patterns. Lesson: don't trust
name-based filters in a notebook with accumulated experimental columns —
always sanity check what a filter actually captured.
**Result:** 12 of 14 sensors improved (higher |correlation| with RUL) after
normalization; 2 (sensor_3, sensor_17) were essentially unchanged (both
within 0.003 of their raw value — not a real regression). Confirms the
sensor_14 finding generalizes: per-engine baseline normalization is a real,
broadly beneficial transformation for this dataset, not a one-sensor fluke.

---

## 3. Decision: keep both raw AND normalized versions as features
**Question:** Given normalized outperforms raw for most sensors, should we
drop raw entirely, keep only normalized, or keep both (doubling sensor
feature count from 14 to 28)?
**Reasoning:** Raw absolute readings still carry information normalization
deliberately discards (e.g. an engine's typical/absolute operating level,
not just its drift from that level) — didn't want to throw that away without
checking if it's actually redundant.
**Checked for redundancy:** Computed correlation between each sensor's raw
and normalized version. Range: 0.766 (sensor_13) to 0.943 (sensor_9) — all
sensors meaningfully correlated (expected, since normalized is derived from
raw) but well below the ~0.95+ threshold that would signal near-total
redundancy. Notably, sensors with the lowest raw-vs-normalized correlation
(sensor_13, sensor_8, sensor_12) were also the sensors that improved most
from normalization — consistent, since a bigger baseline-offset problem
means normalization changes the signal more.
**Decision:** Keep both raw and normalized versions for all 14 sensors
(28 sensor-derived features total). Not redundant enough to justify dropping
either. Tree-based models (Random Forest, XGBoost) and neural nets (LSTM)
both tolerate correlated inputs reasonably well — will revisit trimming
feature count only if a specific model shows overfitting or training becomes
too slow, not preemptively.

---
## 4. Rolling window features — tested window sizes 5, 10, 15, 20 on raw sensors
**Motivation:** A single cycle's reading can be noisy (seen earlier with
sensor_9). A rolling mean smooths short-term noise so the underlying trend
is clearer — but risks blurring real short, sudden changes if the window is
too large (raised as a concern before running any numbers: "small change in
sensor data can make a lot of change" — worth checking, not assuming).
**Method:** Computed rolling mean per engine (grouped by unit_number, so
windows never cross engine boundaries) for all 14 sensors, at window sizes
5/10/15/20, and compared each window's correlation with RUL against the
others.
**Result:**
  - 8/14 sensors peaked (strongest correlation) at window=5, 6/14 peaked at
    window=10. Zero sensors peaked at window=15 or 20 — every sensor got
    WORSE past window=10.
  - Confirms the original concern: past a certain window size, smoothing
    starts removing real signal, not just noise. window=15/20 provide no
    benefit for this dataset and were dropped from further use.
  - Even the worst-performing window (20) still beat no smoothing at all
    for most sensors — rolling mean is net positive, the question is how
    much smoothing, not whether to smooth.
**Decision:** Keep window sizes 5 and 10 only (roughly split which one wins
per sensor — no single global winner, so kept both rather than forcing one
choice).

---

## 5. Rolling window features on NORMALIZED sensors — combining both techniques
**Question:** Does applying rolling mean to the already-baseline-normalized
sensors (rather than raw) perform better, and does stacking normalization +
rolling together beat either technique alone?
**Method:** Repeated the window 5/10 rolling-mean correlation check, this
time on the `_norm` (baseline-normalized) sensor columns instead of raw.
**Result:**
  - Every one of the 14 sensors preferred window=10 over window=5 on the
    normalized versions — a much more consistent pattern than raw rolling,
    where the window preference was split roughly 8-vs-6. Hypothesis (not
    fully confirmed, worth treating as a working theory): once the baseline
    offset noise is removed, more of what remains is real degradation
    trend, so a larger smoothing window has more genuine signal to preserve
    and less to accidentally blur.
  - Normalized+rolling(10) beat raw+rolling for every single sensor,
    sometimes substantially. Example: sensor_11 raw correlation -0.775 →
    raw+rolling(5) -0.814 → normalized-only -0.819 → normalized+rolling(10)
    -0.883. Each individual technique (normalization, rolling) gave a modest
    improvement alone; combined, the improvement was much larger than either
    alone — evidence the two techniques address different problems
    (baseline offset vs. noise) rather than overlapping/redundant fixes.
  - Even sensor_9, the weakest performer throughout this whole
    investigation, improved meaningfully once both techniques were stacked
    (-0.462 raw → -0.507 normalized+rolling(10)).
**Decision:** Use normalized + rolling(window=10) as the primary engineered
sensor features going forward, in addition to (not instead of) the raw and
plain-normalized versions already kept — same reasoning as before: no strong
evidence of harmful redundancy, and different processing steps capture
different aspects of the signal (absolute level, baseline-relative drift,
smoothed trend).

---
## 6. Rolling standard deviation — added, then found and removed a redundancy
**Motivation:** Rolling mean captures the smoothed level of a sensor, but not
how erratic/volatile it is. Hypothesis: rolling std (variation within the
window) could catch instability that appears before the mean itself shifts —
a potentially independent degradation signal.
**Method:** Computed rolling std (window 5 and 10) for both raw and
normalized versions of all 14 sensors, same groupby-per-engine approach as
rolling mean. Cycle-1 rows produced NaN (std of a single value is
undefined) — exactly 100 NaNs per column, one per engine, confirming this
was purely the "no history yet" case rather than a bug. Filled these with 0,
reasoned to be the mathematically correct answer here (no preceding cycles
existed to vary against), not an arbitrary guess.
**Finding — redundancy caught before it wasted downstream work:** Correlation
values for `sensor_X_rollstd` and `sensor_X_norm_rollstd` came out digit-for-
digit identical for every sensor. Root cause: normalization subtracts a
constant (per engine) from every reading — shifting a set of numbers by a
constant changes their mean but leaves their spread (standard deviation)
completely unchanged. So rolling std computed on raw vs. normalized sensor
values will always be mathematically identical; normalization only ever
affects rolling mean, never rolling std.
**Action:** Dropped the 28 redundant `_norm_rollstd_` columns entirely,
keeping only the 28 real `_rollstd_` (raw-based) columns. No information was
lost — the values were exact duplicates, only the computation was wasted.
**Secondary finding — rolling std is a much weaker signal than rolling mean:**
Correlation of rollstd with RUL ranged roughly -0.02 to -0.30 (strongest:
sensor_14 and sensor_9 at ~-0.30 with window=10), compared to rolling mean's
-0.6 to -0.88 range. As a standalone linear predictor, rolling std carries
far less signal than rolling mean here. Kept it anyway — a weak standalone
correlation doesn't rule out it being useful in combination with other
features for a non-linear model (tree-based / neural net), and the feature
is cheap to keep now that the duplication is resolved.
**Takeaway:** Worth thinking through the math of a transformation (what does
subtracting a constant do to spread?) before running it broadly — would have
caught this without needing the correlation table to reveal it after the
fact.

---
## 7. Clarified: what "don't touch the test set" actually means, before processing test features
**Question raised:** Since we're about to apply the same normalization and
rolling-window transformations to the test set, does that violate the rule
of keeping train and test separate for accurate evaluation?
**Clarified principle:** The rule isn't "never transform test data" — a
model can't even read raw, untransformed test data if it doesn't match the
feature format it was trained on. The actual rule is: **no information from
the test set (or from other engines) may influence how a computation is
defined**, especially anything computed globally across the whole dataset.
**Two categories of feature computation, with different rules:**
  - **Per-row / per-engine computations (safe to redo independently on
    test):** Our baseline normalization ("this engine's own first-10-cycle
    average") and rolling mean/std ("this engine's own trailing window")
    only ever look at that one engine's own history. Recomputing them fresh
    on the test set, using test's own values, does not leak anything from
    train into test, or between engines — each engine's calculation is
    self-contained.
  - **Dataset-wide/global computations (leakage risk — must reuse train's
    numbers, never recompute from test):** Not used yet in this project, but
    the classic example is a scaler (e.g. StandardScaler) fit using the
    overall mean/std across all train engines. If such a scaler is used
    later (e.g. before feeding a neural net), it must be fit ONCE on train
    only, and the same fitted values applied to test — recomputing mean/std
    from test data itself would let the test set "see its own distribution,"
    producing misleadingly optimistic evaluation results.
**Conclusion:** Safe and correct to apply baseline normalization and rolling
window features to the test set independently, using each test engine's own
values — this is not a leakage violation. Will need to revisit this
distinction carefully if/when a global scaler or other dataset-wide fitted
transformation is introduced later in the project.

---