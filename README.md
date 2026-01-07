# Credit Risk Probability of Default (PD) Modeling

## 1. Project Overview

This project focuses on building, validating, and scoring a **Probability of Default (PD) model** for credit risk assessment. The objective is to accurately rank-order customers by default risk while ensuring strong generalization, stability across development (DEV), validation (VAL), and test (TEST) samples.

The final model demonstrates:

* Improved **KS** for
* Stable and monotonic **rank ordering across all tiers** in DEV, VAL, TEST
* Minimal and acceptable KS degradation on TEST, indicating reduced overfitting

This repository contains the full modeling workflow—from data preparation to scoring and performance evaluation—designed to be **production-ready** and **model governance friendly**.

---

## 2. Key Objectives

* Build a robust PD model with strong discriminatory power
* Ensure monotonic risk ranking across score bands
* Minimize overfitting while maintaining performance
* Provide explainability via feature importance
* Enable reproducible scoring for DEV / VAL / TEST datasets

---

## 3. Data Description

The modeling dataset consists of:

* **Encoded feature set** (`df_dataset_encoded`)
* Binary target variable: `target_default`
* Sampling flag: `Sampling` (DEV / VAL / TEST)

### Sampling Strategy

* **DEV**: Model training and tuning
* **VAL**: Model selection, rank-order validation
* **TEST**: Final unbiased performance assessment

---

## 4. Feature Engineering & Selection

* Input features were preprocessed and encoded prior to modeling

The final feature list is stored as:

```
selected_features
```

---

## 5. Model Development

* **Algorithm**: CatBoost Classifier
* **Reason for selection**:

  * Handles non-linear relationships effectively
  * Robust to feature interactions
  * Strong out-of-the-box performance for tabular credit data

### Training

* Model trained on DEV sample
* Hyperparameters tuned to balance bias–variance tradeoff
* Retrained using finalized feature set

---

## 6. Model Scoring

PD scores are generated using the retrained model:

```python
pd_score = model.predict_proba(X)[:, 1]
```

Scores are appended to the dataset for further analysis and monitoring.

---

## 7. Model Performance Summary

### 7.1 KS Statistics

* **DEV**: Improved
* **VAL**: Improved and fully monotonic
* **TEST**: Marginal decrease (~0.1), within acceptable tolerance

📌 Interpretation:

> The slight TEST KS reduction indicates reduced overfitting and better generalization.

---

### 7.2 Rank Ordering

* Perfect rank ordering achieved across **all tiers in VAL**
* No inversions observed in bad rates or PD bands

This confirms the model’s suitability for:

* Risk segmentation
* Policy rules
* Cutoff-based decisioning

---

## 8. Feature Importance & Explainability

Feature importance is computed using:

* CatBoost native feature importance
* SHAP values (for global and local explainability)

This ensures:

* Regulatory interpretability
* Transparent risk drivers
* Model governance readiness

---

## 9. Stability & Validation Checks

The following validation checks are part of the workflow:

* KS by sample (DEV / VAL / TEST)
* Bad rate monotonicity by decile
* Population Stability Index (PSI)
* Feature consistency across datasets

These checks ensure the model is stable and production-safe.

---

## 10. Project Structure

```
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
│   ├── feature_engineering.ipynb
│   ├── model_training.ipynb
│   ├── validation_analysis.ipynb
├── src/
│   ├── scoring.py
│   ├── metrics.py
│   ├── feature_importance.py
├── artifacts/
│   ├── trained_model.cbm
│   ├── feature_importance.csv
├── README.md
```

---

## 11. How to Run

1. Create conda environment from `environment.yml`
2. Train model on DEV sample
3. Validate using VAL sample
4. Score TEST sample
5. Generate performance and explainability reports

---

## 12. Business Impact

* Improved risk discrimination
* Stable decision thresholds
* Transparent risk drivers
* Ready for deployment and monitoring

This model supports **risk-based pricing**, **credit policy**, and **portfolio monitoring** use cases.

---

## 13. Final Notes

This project follows best practices in:

* Credit risk modeling
* Model validation
* Explainability
* Production readiness

The final model balances **performance, stability, and interpretability**, making it suitable for real-world deployment.

---

✅ **Status**: Model finalized and validated

📌 **Next steps**:

* Calibration (if required)
* Production deployment
* Ongoing monitoring (PSI / KS drift)