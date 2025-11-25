
## Overview
Behavioral neuroscience metrics.
Author: Wiktoria Zaniewska 🐁

## Mathematical Formulations

### 1. Classic Preference Index (PI)
**Formula:** `PI = (T_in - T_out) / T_total × 100`

Where:
- T_in = Time spent inside ROI
- T_out = Time spent outside ROI  
- T_total = Total experiment time

**Interpretation:**
- PI > 0: Preference for ROI
- PI < 0: Avoidance of ROI
- PI = 0: No preference


---

### 2. Discrimination Index (DI)
**Formula:** `DI = (T_in - T_out) / (T_in + T_out)`

**Range:** -1 to +1
- DI = +1: Exclusive preference for ROI
- DI = -1: Complete avoidance of ROI
- DI = 0: Equal time distribution




---

### 3. Exploration Ratio (ER)
**Formula:** `ER = T_in / T_total`

**Range:** 0 to 1
- ER > expected_proportion: Preference
- ER < expected_proportion: Avoidance
- ER = expected_proportion: No preference


---

### 4. Entry Preference Score (EPS)
**Formula:** `EPS = (N_entries / D_total) × 1000`

Where:
- N_entries = Number of entries to ROI
- D_total = Total distance traveled (pixels)


**Interpretation:** Normalizes entry frequency by locomotor activity

---

### 5. Cohen's d Effect Size
**Formula:** `d = (P_observed - P_expected) / σ_expected`

Where:
- P_observed = Observed proportion of time in ROI
- P_expected = Expected proportion based on ROI area
- σ_expected = Standard deviation of expected proportion

**Interpretation:**
- |d| < 0.2: Negligible effect
- |d| = 0.2-0.5: Small effect
- |d| = 0.5-0.8: Medium effect
- |d| > 0.8: Large effect



---

## Statistical Tests

### 1. Binomial Test
Tests if time spent in ROI differs from chance expectation based on ROI area.

Null Hypothesis: Time in ROI = Area proportion × Total time

Alternative: Two-tailed (preference or avoidance)

---

### 2. Bout Duration t-test
Tests if visit durations differ from expected random distribution.

Null Hypothesis: Mean bout duration = Expected duration

---

### 3. Mann-Whitney U Test
Non-parametric test for velocity differences inside vs outside ROI.

Null Hypothesis: No difference in movement speed

---

## Behavioral Metrics

### Time-Based Metrics
- **Total time in ROI** (seconds)
- **Percent time in ROI** (%)
- **Expected time** (based on area proportion)

### Entry Metrics
- **Number of entries** (count)
- **Entry frequency** (entries/minute)
- **First entry latency** (seconds)

### Bout Analysis
- **Number of bouts** (continuous periods in ROI)
- **Mean bout duration** (seconds)
- **Median bout duration** (seconds)
- **Maximum bout duration** (seconds)
- **Bout variability** (standard deviation)

### Activity Metrics
- **Mean velocity in ROI** (pixels/frame)
- **Mean velocity outside** (pixels/frame)
- **Percent active in ROI** (% frames with movement > threshold)

---

## Interpretation 

### Strong Preference
- Binomial test p < 0.05
- PI > +20%
- DI > +0.2
- Cohen's d > 0.8

### Moderate Preference
- Binomial test p < 0.05
- PI = +10 to +20%
- DI = +0.1 to +0.2
- Cohen's d = 0.5-0.8

### No Preference
- Binomial test p > 0.05
- PI = -10 to +10%
- DI = -0.1 to +0.1
- Cohen's d < 0.5

### Avoidance
- Binomial test p < 0.05
- PI < -10%
- DI < -0.1
- Cohen's d < -0.5


## Output Files

1. **CSV file** with all data.
2. **Visualization** contains 9 panels:
   - ROI occupancy timeline
   - Cumulative time plot
   - Bout duration histogram
   - Entry frequency over time
   - Velocity comparison
   - Preference indices
   - Statistical tests summary
   - Key metrics summary
   - Trajectory with ROI highlighted

Enjoy!!!
Wiktoria 🐁