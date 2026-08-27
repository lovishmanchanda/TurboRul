# TurboRUL — Decision Log

Every meaningful choice made in this project, with reasoning, alternatives
considered, and how it was verified. Written as we go, not after the fact.

---

## 1. Dataset: NASA C-MAPSS (2008),for Phase 1
- **Decision:** Build the core pipeline on the original 2008 CMAPSS (FD001–FD004).
- **Why:** It's the standard benchmark used across the CMAPSS research literature,
so results are directly comparable to published work. N-CMAPSS (2021) is larger,
heavier (HDF5, GBs per file), and has far less tutorial support — better suited
as a later "generalization" extension once the core pipeline works.
- **Alternative considered:** Start directly with N-CMAPSS for a more impressive-sounding
dataset. Rejected — too much data-engineering overhead before any modeling starts.

---

## 2. RUL labeling: piecewise-linear clip at 125 cycles
- **Decision:** RUL = (max cycle for engine) − (current cycle), capped at 125.
- **Why:** Early in an engine's life, sensors show no meaningful degradation
signal — the engine is just healthy. Asking a model to predict a precise
"280 cycles left" during this flat/healthy period is unrealistic and wastes
model capacity on a signal that isn't really there. Capping tells the model
"treat this as healthy" until real degradation begins.
- **Verified by:** Plotting raw RUL vs. clipped RUL for a single engine (Engine 1) —
confirmed clipped line is flat at 125 early on, then slopes down like the raw
line once it crosses the threshold.
- **Alternative considered:** Uncapped linear RUL (used in some papers). Rejected
in favor of the more common piecewise-linear convention, which generally performs
better in reported CMAPSS results.

---

## 3. Dropped constant/near-constant sensors
- **Decision:** Dropped sensor_1, sensor_5, sensor_6, sensor_10, sensor_16,
sensor_18, sensor_19 (7 of 21 sensors) from FD001.
- **Why:** A sensor that doesn't change over an engine's lifetime carries zero
predictive information — it can't help distinguish "healthy" from "about to fail."
Keeping it just adds noise and unnecessary input dimensions.
- **Verified by:**
  - Checked `.std()` for all 21 sensors — found a natural split: 6 sensors
    with std of exactly 0 or ~1e-15/1e-18 (floating-point rounding noise, not
    real variation).
  - sensor_6 was borderline (std ≈ 0.0014) — checked `.describe()` to compare
    against its own min/max range, and found min ≈ max (real range is
    negligible) — confirmed visually with a per-engine plot (flat line with
    only pixel-level jitter). Classified as constant too.
  - Did NOT blindly use a hardcoded threshold — inspected the actual scale of
    each borderline sensor before deciding.
- **Important caveat:** This list is FD001-specific (1 operating condition).
Must re-check for FD002/FD004, since sensors that look constant under a single
operating condition may show real variation once operating conditions change —
they might be reacting to the operating setting, not degradation, or a
combination of both.

---

## 4. Correlation of remaining sensors with RUL
- **What we did:** Computed Pearson correlation between each of the 14 remaining
sensors and RUL, across all rows (all 100 engines pooled together).
- **Result:** 8 sensors negatively correlated (2, 3, 4, 8, 11, 13, 15, 17),
6 positively correlated (7, 9, 12, 14, 20, 21). Strongest: sensor_11 (-0.775),
sensor_4 (-0.757). Weakest: sensor_9 (-0.462), sensor_14 (-0.370).
- **Cross-checked against visual trend plots** (single engine, sensor value vs.
cycle) before trusting the numbers. Predicted sign from the visual trend
(up-trending sensor → should be negatively correlated with RUL, since RUL
decreases as the engine ages; down-trending sensor → should be positively
correlated) matched for 12 of 14 sensors. sensor_9 and sensor_14 disagreed —
investigated further below rather than accepted at face value.

---

## 5. Diagnosed a between-engine baseline effect masking sensor_14's real trend
- **Problem found:** sensor_14 visually showed a clear, consistent downward
trend within a single engine's lifetime (Engine 1) — which should produce a
*positive* correlation with RUL (sensor drops as RUL drops). Instead, the
pooled correlation across all 100 engines came out *negative* (-0.370),
contradicting the single-engine visual read.
- **Investigation:** Plotted sensor_14 for multiple engines (1-4) on the same
chart to compare not just their trend direction but their starting/baseline
levels. Found engines do NOT all start from the same sensor_14 baseline —
Engine 1 trended down from a lower starting point, while Engines 3 and 4 sat
at a noticeably higher level throughout, even early in life.
- **Conclusion:** The pooled correlation was tangled between two different
effects: (1) the real within-engine degradation trend as each engine ages,
and (2) between-engine baseline differences unrelated to wear (e.g.
manufacturing variation, sensor calibration, or possibly incomplete control
of operating conditions). These two effects partially cancelled out in the
single pooled number, making a real degradation signal look weak/misleading.
**Why this matters for the project:** A raw pooled correlation number can
hide or distort real per-engine signal when engines don't share a common
baseline. Confirms this with actual data inspection rather than trusting the
summary statistic blindly.
- **Implication for feature engineering (next phase):** Consider normalizing
sensors relative to each engine's own early-life/starting value (e.g.
subtracting each engine's first-cycle reading, or its rolling baseline)
rather than using raw sensor values directly — this should isolate the true
within-engine degradation signal from between-engine baseline noise. To be
implemented and tested in Phase 2 (feature engineering).
