# Battery_SoH_Estimation
Benchmarking ensemble ML methods (XGBoost, Random Forest, Gradient Boosting, Linear Regression) for lithium-ion battery State-of-Health estimation using the NASA Battery Dataset. Built as part of self-directed study in battery systems research.

---

## Overview

This project benchmarks four machine learning models for estimating the **State-of-Health (SoH)** of lithium-ion batteries using cycling data from the NASA Battery Dataset. 

---

## Research Questions

1. Which ensemble ML method best estimates battery SoH from non-invasive discharge measurements?
2. Does a linear model capture degradation trends as well as more complex ensemble methods?
3. How do data quality issues (anomalous capacity readings) affect model performance when training across batteries tested under different conditions?

---

## Dataset

**NASA Battery Dataset** — Kaggle cleaned version  
Source: [NASA PCOE Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)  
Kaggle: [patrickfleith/nasa-battery-dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)

The dataset contains lithium-ion batteries cycled through charge, discharge, and impedance profiles under varying conditions:

Batteries | Temperature |
|---|---|
| B0005, B0006, B0007, B0018 | Room temperature (24°C) |
| B0025–B0028 | 24°C |
| B0029–B0032 | 43°C |
| B0033–B0040 | 24°C / 44°C |
| B0041–B0056 | 4°C |

---

## Features

Five features are extracted from each discharge cycle. Each feature is connected to a specific physical degradation mechanism:

| Feature | Physical Meaning |
|---|---|
| `discharge_time` | Total discharge duration. Decreases with aging as deliverable capacity reduces — directly reflects SEI growth and active material loss. |
| `max_temperature` | Peak temperature during discharge. Increases with aging due to higher ohmic heating as internal resistance grows from SEI thickening. |
| `min_voltage` | Lowest voltage reached during discharge. Reflects the end-of-discharge knee point, which shifts as lithium inventory decreases. |
| `mean_voltage` | Average terminal voltage. Decreases with aging due to increased overpotentials from resistance growth and reduced active surface area. |
| `voltage_drop` | Start voltage minus end voltage. Captures total polarisation across the discharge cycle — increases as internal resistance grows. |

---

## Models

Four models are trained and compared:

- **Linear Regression** — baseline; captures monotonic degradation trend directly
- **Random Forest** — ensemble of decision trees; handles nonlinear relationships
- **Gradient Boosting** — sequential ensemble; strong on structured tabular data
- **XGBoost** — regularised gradient boosting; generally strong on small datasets

---

## Key Findings

### Finding 1 — Single-condition data (4 batteries, room temperature)

| Model | RMSE |
|---|---|
| **Linear Regression** | **0.0256** ← Best |
| XGBoost | 0.0346 |
| Gradient Boosting | 0.0355 |
| Random Forest | 0.0404 |

**Linear Regression outperforms all ensemble methods.** The physical explanation: SoH degradation in controlled conditions follows a largely monotonic downward trend across cycles. A linear model captures this trend directly, while ensemble methods overfit to cycle-to-cycle measurement noise rather than the underlying degradation trajectory.

---

### Finding 2 — Cross-condition data (all 34 batteries)

| Model | RMSE |
|---|---|
| **Random Forest** | **2.6701** ← Best |
| XGBoost | 2.9514 |
| Gradient Boosting | 3.1410 |
| Linear Regression | 4.7126 |

The RMSE values far exceed the valid SoH range of 0–1, indicating a data quality problem. Several low-temperature batteries (B0049–B0056) contain anomalous capacity readings in early cycles. 

**This highlights a real challenge in battery ML:** cross-condition generalisation requires robust data preprocessing and outlier filtering before model training. Random Forest is most resilient to these outliers due to its averaging mechanism, which partially suppresses the effect of extreme values.

---

## How to Run

This project runs in **Google Colab** with direct Google Drive integration.

### Step 1 — Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset) and upload to Google Drive at:
```
MyDrive/Research/Kaggle_Battery_Dataset/archive/cleaned_dataset/
```

### Step 2 — Open in Colab
Upload the notebook to Google Colab.

### Step 3 — Mount Drive and run
Run Cell 2 to mount Google Drive, then run all cells in order.

## Project Structure

```
battery-soh-estimation/
│
├── Battery_SoH_Estimation.ipynb   # Main notebook
├── README.md                      # This file
│
└── outputs/
    ├── soh_predictions.png        # Predicted vs actual SoH for all 4 models
    ├── feature_importance.png     # XGBoost feature importance plot
    └── degradation_curve.png      # SoH degradation curve for one battery
```

---

## Background and Motivation

Battery State-of-Health estimation is a core function of the Battery Management System (BMS) in electric vehicles and grid-scale energy storage systems. Accurate SoH estimation enables:

- Informed battery replacement decisions
- Adaptive cell balancing strategies
- Degradation-aware charge/discharge control
- Second-life battery grading for repurposing

Physics-based models (such as the Doyle-Fuller-Newman model) provide accurate SoH estimates but are computationally expensive for real-time BMS applications. Data-driven ML approaches can approximate these models at much lower cost, provided the features are chosen to reflect the underlying physical degradation processes.

This project investigates whether simple ensemble ML models, when given physically meaningful features, can produce reliable SoH estimates — and where their limitations lie when operating conditions vary across batteries.

---

## Limitations and Future Work

- **Data cleaning:** Outlier SoH values from anomalous capacity readings in low-temperature batteries must be filtered before cross-condition training
- **Feature extension:** Adding impedance-based features (Re, Rct from EIS measurements) available in the dataset could improve accuracy
- **Physics-informed methods:** Benchmarking against Physics-Informed Neural Networks (PINNs) would quantify what is gained by embedding physical constraints into the model
- **Cross-condition generalisation:** Train on room-temperature batteries, test on low-temperature batteries — a more realistic deployment scenario

---

## Author

**Wasik Billah Ibn Rashid**  
B.Sc. in Electrical and Electronic Engineering, Islamic University of Technology (IUT), Dhaka  
wasikbillah07@gmail.com

---

## References

1. B. Saha and K. Goebel (2007). *Battery Data Set*, NASA Prognostics Data Repository, NASA Ames Research Center.
2. B. Bole, C. Kulkarni, and M. Daigle (2014). *Adaptation of an Electrochemistry-based Li-Ion Battery Model to Account for Deterioration Observed Under Randomized Use*, Annual Conference of the PHM Society.
3. Plett, G.L. (2015). *Battery Management Systems, Vol. 1: Battery Modeling*. Artech House.
