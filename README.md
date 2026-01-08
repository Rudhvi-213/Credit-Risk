# Credit Risk Modeling – End-to-End Machine Learning Project

## 📌 Project Overview

This project demonstrates an **end-to-end credit risk modeling workflow** focused on **model development, validation, and analysis**, following practices commonly used in **real-world credit risk modeling**.

The objective is to estimate the **probability of default (PD)** for loan applicants. The project currently supports **offline scoring**: given a prepared input dataset, the model applies the same preprocessing steps and produces a risk score.

---

## 🎯 Key Objectives

* Build credit risk models using **XGBoost, LightGBM, and CatBoost**
* Explicitly test **class imbalance handling** and make data-driven model choices
* Evaluate models using **KS as the primary performance metric**
* Perform disciplined **DEV–VAL–TEST** based model selection
* Validate **rank ordering** using decile / gains analysis
* Analyze **feature importance and Information Value (IV)** for interpretability
* Persist final models and scored datasets for reproducibility

---

## 🧱 Project Structure

```
credit-risk/
│
├── Data/
│   ├── credit_risk_scored_dataset.csv
│   ├── model_dataset_with_sampling.csv      # DEV / VAL / TEST flagged dataset
│   ├── model_Results.xlsx                   # Model results & comparisons
│
├── Notebooks/
│   ├── pre_processing.ipynb                 # Data cleaning, encoding, feature engineering
│   ├── training.ipynb                       # Model training & hyperparameter tuning
│   ├── model_results.ipynb                  # Model comparison, validation & analysis
│   ├── utils.py                             # KS, IV, scoring utilities
│
├── Models/
│   ├── catboost_model.cbm                   # Final selected model
│   ├── lightgbm_model.pkl                   # Challenger model
│   ├── xgboost_model.pkl                    # Challenger model
│
├── credit_risk_env.yml                      # Conda environment
├── requirements.txt
├── README.md
```

---

## 📊 Dataset & Target

* **Target Variable:** `target_default` (1 = default, 0 = non-default)
* **Sampling Strategy:** Explicit split into

  * `DEV` – model development & tuning
  * `VAL` – model validation
  * `TEST` – final unbiased evaluation

---

## ⚙️ Modeling Approach

### Models Trained

* XGBoost
* LightGBM
* CatBoost

Class imbalance was explicitly tested using model-specific weighting strategies (e.g., `scale_pos_weight`, `class_weight`, `auto_class_weights`). However, **unweighted models consistently demonstrated better validation KS and stability**. Therefore, unweighted models were selected based on empirical performance.

### Hyperparameter Tuning Strategy

* Manual grid search
* Models trained **only on DEV**
* Model selection based on:

  * **Validation KS**
  * **KS Gap (DEV − VAL)**
* No re-tuning after observing TEST performance

This approach mirrors practical credit risk development workflows.

---

## 🏆 Model Selection Results

### Final Performance (Frozen Models)

| Model                         | DEV_KS    | VAL_KS    | TEST_KS   |
| ----------------------------- | --------- | --------- | --------- |
| XGBoost                       | 38.45     | 33.21     | 31.31     |
| LightGBM                      | 40.85     | 33.00     | 31.35     |
| **CatBoost (Selected Model)** | **36.99** | **33.36** | **31.80** |

**CatBoost** was selected as the final model due to:

* Highest validation and test KS
* Lowest overfitting (stable KS gap)
* Consistent generalization across DEV, VAL, and TEST

Detailed model results are documented in the **model results Excel file**.

---

## 📈 Rank Ordering Validation (Decile / Gains Analysis)

* Deciles created on **DEV, VAL, TEST datasets** using predicted PD scores
* Strong monotonic increase in bad rates from lowest to highest risk deciles


This confirms the model is **cut-off ready** for risk-based decisioning.

---

## 🔍 Stability Considerations

Basic stability checks were performed across DEV, VAL, and TEST samples. Given that datasets were generated using controlled sampling with identical target rates, no material population shift was observed during development.

---

## 🧠 Feature Analysis

### Feature Importance

* Model-based feature importance was extracted from CatBoost
* Importance reflects **multivariate contribution** and interaction effects

### Information Value (IV)

* IV was computed on the DEV dataset
* IV was used to assess **univariate predictive strength**
* No features exhibited abnormally high IV values indicative of leakage

> Note: Full **model explainability** using SHAP or local explanation techniques has not yet been implemented and is planned as a future extension.

---

## 💾 Model Persistence & Reproducibility

* Final models are saved for **offline scoring and analysis**
* Scored datasets are exported to CSV for downstream evaluation
* Conda environment is captured via `credit_risk_env.yml`

This ensures reproducibility of results within the scope of this project.

---

## 🛠️ Utilities

Key custom utilities implemented in `utils.py`:

* KS calculation
* KS scorer for model selection
* Information Value (IV) computation

---

## 📌 Key Learnings

* Validation and stability matter more than raw performance
* ML-based credit models rely heavily on **feature interactions**
* Proper DEV–VAL–TEST discipline is critical to avoid leakage
* Explainability and governance are essential for real-world deployment

---

## 📄 Disclaimer

This project is for **educational and portfolio purposes only** and does not represent a production credit decisioning system.

---

## 👤 Author

**Rudhvi Uggirala**
Data Scientist – Risk & Credit Analytics