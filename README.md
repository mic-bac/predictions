# Practical Machine Learning Predictions

Educational examples of common prediction tasks in machine learning and data science,
built for students and beginners: each script is a readable, runnable walkthrough of one
method family, with the reasoning written into the code rather than hidden behind it.

## 🎯 Learning Objectives

- Understand different types of prediction problems in machine learning
- Learn how to work with real-world datasets
- Gain practical experience with popular ML libraries and frameworks
- Master visualization techniques for data analysis
- Apply best practices in machine learning workflows

## 🗂️ Project Structure

```
predictions/
├── src/churn_data.py   shared loading/cleaning for the two churn lessons
├── propensity.py       lesson 1 — will they churn?      (classification)
├── survival.py         lesson 2 — when will they churn? (time-to-event)
├── timeseries.py       lesson 3 — sales forecasting     (separate topic)
└── data/               Kaggle downloads (git-ignored, see below)
```

### 1. Propensity Models — Look-alike Modeling
`propensity.py`

Estimates, per customer, the **probability** of churning, and turns those scores into
targetable segments. Covers the look-alike time-window design (observation → buffer →
outcome), three model families (logistic regression, XGBoost, neural network),
cross-validation, `GridSearchCV`/`RandomizedSearchCV`, the metrics that matter for
imbalanced data (precision/recall/F1, ROC-AUC, PR-AUC, confusion matrix), and
learning/validation curves for diagnosing under- and overfitting.

### 2. Survival Analysis — Time to Churn
`survival.py`

Estimates **when** the event happens, and handles the customers who have not churned yet
(right-censored data) instead of discarding them. Covers Kaplan-Meier survival curves,
Cox proportional hazards, Random Survival Forest and Survival SVM, plus the
survival-specific metrics: concordance index, time-dependent AUC(t), and the integrated
Brier score.

### 3. Sales Forecasting (Time Series)
`timeseries.py`

Retail sales forecasting with Prophet and XGBoost, seasonal decomposition, and time-series
feature engineering. A separate topic from the two churn lessons above.

## 🛠️ Prerequisites

- Python 3.13 (see `.python-version`)
- Basic understanding of Python programming
- Familiarity with data analysis concepts
- Basic statistics knowledge

## 📦 Installation

Clone the repository and sync the environment with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/mic-bac/predictions.git
cd predictions
uv sync
```

A conda alternative is kept in `conda_env.yaml`, but `uv` is the supported path — it is
what `pyproject.toml` and `uv.lock` describe.

## 📊 Datasets

The CSVs are **not** in git (they are ~40 MB); download them from Kaggle into `data/`.

### Customer Churn Dataset → `data/churn/`
- `customer_churn_dataset-training-master.csv`
- `customer_churn_dataset-testing-master.csv`

Source: [Kaggle Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)

Both files are two halves of one customer base, not a modelling split — `src/churn_data.py`
concatenates them and each lesson makes its own split.

### Walmart Sales Dataset → `data/sales/`
- `train.csv`, `test.csv`, `stores.csv`, `features.csv`

Source: [Kaggle Walmart Sales Dataset](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)

## 📚 Getting Started

Each script runs top to bottom, prints its results, and opens interactive Plotly figures.
They can also be stepped through cell by cell — the `# %%` markers make them notebooks in
VS Code or Jupyter.

```bash
uv run python propensity.py
uv run python survival.py
uv run python timeseries.py
```

### Sample size

`propensity.py` and `survival.py` each define a `SAMPLE_SIZE` constant near the top. The
full dataset is ~505,000 customers, where a hyperparameter search takes hours; the defaults
(20,000 and 5,000) keep every result under a minute without changing the conclusions. Set
`SAMPLE_SIZE = None` to use everything.

## 📋 Dependencies

Main libraries used:
- pandas & numpy: Data manipulation
- scikit-learn: Machine learning algorithms
- scikit-survival: Survival analysis
- prophet: Time series forecasting
- xgboost: Gradient boosting
- plotly: Interactive visualizations
- statsmodels: Statistical models and tests

## 🎓 Learning Path

1. `propensity.py` — the ML workflow end to end, and classification metrics
2. `survival.py` — the same business question, now with time and censoring
3. `timeseries.py` — forecasting a continuous series into the future

Each file includes:
- Detailed comments explaining concepts
- Step-by-step implementation
- Visualization of results
- Model evaluation metrics
- Business insights interpretation

## 🤝 Contributing

Contributions to improve the educational content or add new examples are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a detailed description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
