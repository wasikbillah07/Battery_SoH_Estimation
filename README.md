# Battery SoH Estimation
Benchmarking ensemble ML methods (XGBoost, Random Forest, Gradient Boosting, Linear Regression) for lithium-ion battery State-of-Health estimation using the NASA Battery Dataset. Built as part of self-directed study in battery systems research.

> **Note:** The notebook cannot be rendered directly on GitHub due to a known compatibility issue. To view the notebook, open it in [Google Colab](https://colab.research.google.com).

> **Versions:** `Battery_SoH_Estimation.ipynb` is the original 5-feature version. `Battery_SoH_Estimation_v2_new_features.ipynb` adds two additional engineered features (equal-voltage-drop-time and impedance-derived resistance) — see [v2 Update](#v2-update-additional-features) below.

---

## Overview
This project benchmarks four machine learning models for estimating the **State-of-Health (SoH)** of lithium-ion batteries using cycling data from the NASA Battery Dataset. 

---

## Dataset
**NASA Battery Dataset** — Kaggle cleaned version  
Source: [NASA PCOE Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)  
Kaggle: [patrickfleith/nasa-battery-dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)

The dataset contains 34 lithium-ion batteries cycled through charge, discharge, and impedance profiles under varying conditions:

| Batteries | Temperature | 
|---|---|
| B0005, B0006, B0007, B0018 | 24°C | 
| B0025–B0028 | 24°C | 
| B0029–B0032 | 43°C | 
| B0033–B0040 | 24°C / 44°C | 
| B0041–B0056 | 4°C | 

Each battery was cycled until end-of-life (30% capacity fade from 2.0 Ah to 1.4 Ah).

---

## Features
Five features are extracted from each discharge cycle in the original version, extended to eight in v2. Each feature is connected to a specific physical degradation mechanism:

| Feature | Physical Meaning |
|---|---|
| `discharge_time` | Total discharge duration. Decreases with aging as deliverable capacity reduces — directly reflects SEI growth and active material loss. |
| `max_temperature` | Peak temperature during discharge. Increases with aging due to higher ohmic heating as internal resistance grows from SEI thickening. |
| `min_voltage` | Lowest voltage reached during discharge. Reflects the end-of-discharge knee point, which shifts as lithium inventory decreases. |
| `mean_voltage` | Average terminal voltage. Decreases with aging due to increased overpotentials from resistance growth and reduced active surface area. |
| `voltage_drop` | Start voltage minus end voltage. Captures total polarisation across the discharge cycle — increases as internal resistance grows. |
| `time_4p0_to_3p8` *(v2)* | Equal-voltage-drop-time (EVDT): time to discharge from 4.0V to 3.8V. Shortens with aging as rising internal resistance causes voltage to sag faster under load — more sensitive to resistance growth than total discharge time. |
| `Re` *(v2)* | Electrolyte resistance, from the nearest impedance (EIS) measurement. A direct physical measurement of resistance growth, not a derived proxy. |
| `Rct` *(v2)* | Charge-transfer resistance, from the nearest impedance (EIS) measurement. Reflects degradation at the electrode-electrolyte interface. |

---

## Models
Four models are trained and compared:
- **Linear Regression** — baseline; captures monotonic degradation trend directly
- **Random Forest** — ensemble of decision trees; handles nonlinear relationships
- **Gradient Boosting** — sequential ensemble; strong on structured tabular data
- **XGBoost** — regularised gradient boosting; generally strong on small datasets

---

## Train / Test Split Methodology
The dataset is split **per battery by cycle number** — the first 70% of each battery's cycles go to training, the last 30% go to testing. This simulates a realistic deployment scenario: a model trained on a battery's early degradation history predicts its future SoH. Random splitting was deliberately avoided to prevent data leakage, where future cycle information would leak into the training set.

---

## Key Findings
> **Note:** The three experiments below were run with modified versions of the feature-extraction and battery-selection code (different `selected_batteries` lists and SoH filtering logic), not the single notebook in this repo as-is. The notebook here reflects the final configuration used in Finding 3 (and its v2 extension). Findings 1 and 2 are included to document the exploration process and the reasoning that led to the final filtering approach.

### Finding 1 — Single-condition data (4 batteries, room temperature, no filter)
*Code variant: `selected_batteries` restricted to B0005/B0006/B0007/B0018, no SoH filter*

636 discharge cycles | 443 training | 193 test

| Model | RMSE |
|---|---|
| **Linear Regression** | **0.0256 ← Best** |
| XGBoost | 0.0346 |
| Gradient Boosting | 0.0355 |
| Random Forest | 0.0404 |

**Linear Regression outperforms all ensemble methods.** SoH degradation in controlled, single-condition data follows a largely monotonic downward trend across cycles. A linear model captures this directly. Ensemble methods overfit to cycle-to-cycle measurement noise rather than the underlying degradation trajectory.

---

### Finding 2 — Cross-condition data (all 34 batteries, no filter)
*Code variant: all 34 batteries included, no SoH filter*

2,755 discharge cycles | 1,911 training | 844 test

| Model | RMSE |
|---|---|
| Random Forest | 2.6701 |
| XGBoost | 2.9514 |
| Gradient Boosting | 3.1410 |
| Linear Regression | 4.7126 |

**All RMSE values exceed the physically valid SoH range of 0–1, indicating a data quality problem.** The SoH is computed as measured capacity divided by the first cycle's capacity. Several low-temperature batteries (B0049–B0056) contain anomalous capacity readings in early cycles — as noted in the original dataset README files. When the first recorded capacity is anomalously low (e.g. 0.2 Ah instead of the expected ~2.0 Ah), all subsequent cycles produce SoH values far above 1.0, corrupting the training and evaluation.

This highlights a real challenge in battery ML: **cross-condition generalisation requires robust data preprocessing before model training.** Random Forest shows the most resilience to these outliers due to its averaging mechanism across many trees, which partially suppresses the effect of extreme SoH values.

---

### Finding 3 — Cross-condition data (all 34 batteries, SoH filter applied)
*Code variant: all 34 batteries, `0.5 < SoH ≤ 1.1` filter applied — this is the configuration in the notebook in this repo*

Filter: `0.5 < SoH ≤ 1.1` — removes physically invalid readings  
1,516 discharge cycles retained (1,239 anomalous cycles removed) | 1,043 training | 470 test

| Model | RMSE |
|---|---|
| **Random Forest** | **0.0896 ← Best** |
| Gradient Boosting | 0.1026 |
| XGBoost | 0.1118 |
| Linear Regression | 0.1380 |

**After filtering, all RMSE values fall within the valid 0–1 range. Random Forest is now the best model — the opposite of Finding 1.**

This reversal has a clear physical explanation. Across 34 batteries tested at 4°C, 24°C, and 44°C with different discharge currents and cutoff voltages, degradation patterns become nonlinear and condition-dependent. Temperature alone significantly affects all five features — discharge time, voltage profiles, and thermal behaviour differ substantially between a 4°C and a 44°C battery. Random Forest captures this nonlinear, multi-condition complexity while Linear Regression, which assumes a single global trend, cannot.

**The SoH filter removed 1,239 out of 2,755 cycles — nearly 45% of the full dataset.** This is not random noise but a systematic issue concentrated in the low-temperature battery groups, consistent with the warnings in the original dataset documentation.

---

## v2 Update: Additional Features
`Battery_SoH_Estimation_v2_new_features.ipynb` extends the cross-condition, SoH-filtered experiment (Finding 3) with two additional features: `time_4p0_to_3p8` (EVDT) and impedance-derived `Re`/`Rct`, joined from the nearest impedance test per battery.

1,513 discharge cycles retained | 1,043 training | 470 test | 8 features total

| Model | RMSE (v1, 5 features) | RMSE (v2, 8 features) |
|---|---|---|
| **Random Forest** | 0.0896 | **0.0861 ← Best** |
| Gradient Boosting | 0.1026 | 0.0955 |
| XGBoost | 0.1118 | 0.0926 |
| Linear Regression | 0.1380 | 0.1387 |

Random Forest remains the best model, with a modest ~4% RMSE improvement. XGBoost sees the largest gain, moving from worst to second-best — likely because Re/Rct and EVDT give it more structured signal to split on. The relatively small overall improvement is attributed to impedance tests being run periodically rather than every cycle: Re/Rct values are forward/back-filled within each battery, so they behave more like a coarse per-battery signal than a smooth per-cycle degradation trend. Interpolating Re/Rct between the two nearest impedance readings (rather than step-filling) is a likely next step to unlock more of their predictive value.

---

## Summary of All Findings
| Experiment | Batteries | Filter | Best Model | Best RMSE |
|---|---|---|---|---|
| Single-condition | 4 (room temp) | None | Linear Regression | 0.0256 |
| Cross-condition (raw) | 34 (all) | None | Random Forest | 2.6701 ⚠️ Invalid |
| Cross-condition (clean) | 34 (all) | SoH 0.5–1.1 | Random Forest | 0.0896 |
| Cross-condition (clean, v2 features) | 34 (all) | SoH 0.5–1.1 | Random Forest | **0.0861** |

---

## Real-Life Implications
Accurate SoH estimation enables:
- Informed battery replacement and second-life grading decisions
- Adaptive cell balancing strategies based on degradation state
- Degradation-aware charge and discharge control in BMS
- Predictive maintenance for EV and grid-scale energy storage

Physics-based models such as the Doyle-Fuller-Newman (DFN) model provide accurate SoH estimates but are computationally expensive for real-time BMS applications. Data-driven ML approaches can approximate these models at much lower computational cost, provided the features are chosen to reflect the underlying physical degradation processes — as demonstrated in this project.

---

## How to Run
This project runs in **Google Colab** with direct Google Drive integration.

**Step 1 — Download the dataset**  
Download from [Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset) and upload to Google Drive at:
```
MyDrive/Research/Kaggle_Battery_Dataset/archive/cleaned_dataset/
```

**Step 2 — Open in Colab**  
Upload the notebook to Google Colab or open directly from GitHub.

**Step 3 — Mount Drive and run**  
Run Cell 2 to mount Google Drive, then run all cells in order.

---

## Limitations and Future Work
- **Cross-battery generalisation:** The current split trains and tests on the same batteries at different cycle points. A stricter evaluation would train on some batteries entirely and test on completely unseen batteries — a more realistic deployment scenario that this project has not yet attempted.
- **Impedance feature resolution:** Re/Rct are currently forward/back-filled per battery since impedance tests run less frequently than discharge cycles. Interpolating between readings instead of step-filling would give a smoother, more informative degradation signal.
- **Physics-informed methods:** Benchmarking against Physics-Informed Neural Networks (PINNs).
- **Data cleaning pipeline:** A systematic outlier detection method (rather than a fixed SoH threshold) would be more robust for production use.
- **Reproducibility:** Findings 1 and 2 above were produced with earlier code variants not preserved in this repo. A future revision should parameterise battery selection and filtering so all three findings can be reproduced from a single notebook.

---

## Author
**Wasik Billah Ibn Rashid**  
B.Sc. in Electrical and Electronic Engineering, Islamic University of Technology (IUT), Dhaka  
wasikbillah07@gmail.com

---

## References
1. B. Saha and K. Goebel (2007). *Battery Data Set*, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
2. L. Mbagaya, K. Reddy, and A. Botes (2025). "Machine Learning Techniques for Battery State of Health Prediction: A Comparative Review," *World Electric Vehicle Journal*, 16(11), 594. https://doi.org/10.3390/wevj16110594
