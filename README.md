# E-Commerce Risk: Predictive ML Classification

Part of a multi-platform e-commerce governance analytics portfolio. This repo covers the predictive machine learning layer -- supervised classification for fraud and return risk, trained and evaluated against the same governance rule engine documented in the SQL notebook. The dashboards and SQL/LLM co-pilot live in the companion repos linked below.

## Contents

| File | What it is |
|---|---|
| `04_Fraud_Return_Risk_ML_Classification.ipynb` | EDA, feature engineering, a from-scratch scored baseline of the governance rule engine, two supervised classifiers (fraud and returns), evaluation against imbalance-appropriate metrics, a head-to-head comparison against the rule engine, a validation-split methodology check, and a full closing summary |
| `synthetic_ecommerce_order_risk_dataset.csv` | The source 12,000-row, 23-column dataset -- the same one used across all three repos in this portfolio |

## What this notebook actually does

Two things get predicted: whether an order is fraudulent (`is_fraud`) and whether it gets returned (`is_returned`). Both are built from one shared, reusable pipeline -- same feature engineering, same train/test setup, same evaluation approach -- but scored and discussed separately, since Section 2's EDA proves they're statistically independent problems with very different underlying signal strength.

Before training anything, the SQL notebook's governance rule engine gets rebuilt in pandas and scored like a classifier -- not just described. That surfaced the most important finding in the notebook: the rule engine's apparent near-perfect fraud recall turned out to be almost entirely circular, because its first condition reads a label (`risk_label`) that already encodes the fraud outcome with zero exceptions. Once that shortcut was removed, the honest rule-engine baseline dropped from an apparent F1 of 0.83 to a real 0.11 -- which is the actual number the trained models are compared against.

## Headline results

| | Rule engine (honest) | Logistic regression (tuned) |
|---|---|---|
| **Fraud** F1 | 0.106 | 0.222 |
| **Returns** F1 | not measurable -- zero non-circular signal | 0.722 |

## Key findings beyond the headline numbers

- **Random Forest results are not reproducible across machines; logistic regression's are.** The same code, same `random_state`, run on three different machines, produced three different Random Forest outcomes. Logistic regression matched to three decimal places every time. This is why the notebook treats the logistic regression's numbers as the ones worth stating with confidence, and documents the Random Forest's instability directly rather than picking whichever run looked best.
- **Model-only alerts mean different things for fraud versus returns.** When the fraud model flags something the rule engine doesn't, it was wrong 100% of the time in testing. When the returns model does the same thing, it's catching real signal the rule engine structurally cannot see -- 115 of 160 correct return catches came with zero rule agreement at all.
- **A test-set threshold-tuning caveat was checked, not just flagged.** Tuning a classification threshold on the same test set used to report the final score is a mild version of the same leakage problem caught in the rule engine. A proper three-way train/validation/test split was run to check whether this mattered -- it moved the reported F1 by less than a hundredth.
- **Three separate imbalance-handling techniques (class weighting, manual oversampling, SMOTE) all hit the same wall** trying to fix the Random Forest's inability to cross the default 0.5 classification threshold for fraud -- pointing at something structural in how the model aggregates probabilities across trees on a rare positive class, not at any one resampling method being insufficient.

## Running the notebook

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
jupyter-lab
```

`imbalanced-learn` is only required for the SMOTE comparison cell; everything else runs on the core scientific Python stack. The notebook loads directly from the CSV in this repo -- no database setup required.

## Honest limitations

- The fraud test set is small (89 positive cases) -- an unavoidable consequence of fraud being genuinely rare in this data, not a flaw in the method.
- Random Forest's specific figures throughout this notebook should be read as illustrative of a pattern (threshold tuning recovers real signal from a model that looks broken at the default cutoff), not as precise, citable results -- see the reproducibility finding above.
- No SHAP -- feature interpretation here explains what each model cares about on average, not why any single prediction was made.
- No gradient boosting (XGBoost/LightGBM) -- only linear and bagged-tree model families were compared.

## Portfolio context

This is one of three linked pieces:
1. **SQL fraud/risk auditing + Power BI / Tableau / Looker Studio dashboards** -- see [multi-platform-ecommerce-governance-analytics](https://github.com/bernard-omondi/multi-platform-ecommerce-governance-analytics)
2. **LLM text-to-SQL co-pilot** -- see [ecommerce-risk-sql-llm-copilot](https://github.com/bernard-omondi/ecommerce-risk-sql-llm-copilot)
3. **This repo (predictive ML classification)** -- supervised fraud/return probability scoring, extending the same dataset's `is_fraud` / `is_returned` labels, checked directly against the governance rule engine from repo #1
