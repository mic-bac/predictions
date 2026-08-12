# Practical Machine Learning Predictions

Four runnable lessons on predicting things that matter to a business: who will leave,
when they will leave, and what next quarter's sales will be. Each script is a readable
walkthrough of one method family, with the reasoning written into the code rather than
hidden behind it — and each ends by naming what its method *cannot* do, because that is
usually the part nobody tells you.

The forecasting lessons take that further. They report several results where the method
on the slides loses to a baseline with no parameters at all, and explain why. That is
deliberate: a student who has only seen models succeed has not been taught to evaluate one.

## 🎯 The four lessons

| Script | Question it answers | Method | Data |
|---|---|---|---|
| `propensity.py` | **Will** this customer churn? | Classification — logistic regression, XGBoost, neural net | Customer churn |
| `survival.py` | **When** will they churn? | Time-to-event — Kaplan-Meier, Cox, RSF, Survival SVM | Customer churn |
| `timeseries.py` | Can this series be forecast at all, and what do classical models make of it? | Decomposition, ADF, ACF/PACF, exponential smoothing, SARIMA | Walmart sales |
| `forecasting.py` | Are Prophet and XGBoost worth it here? | Prophet, XGBoost, Random Forest, detrending hybrid | Walmart sales |

```
predictions/
├── propensity.py       lesson 1 — will they churn?      (classification)
├── survival.py         lesson 2 — when will they churn? (time-to-event)
├── timeseries.py       lesson 3 — analysing a series    (classical forecasting)
├── forecasting.py      lesson 4 — Prophet & XGBoost     (modern forecasting)
├── src/
│   ├── churn_data.py       loading + encoding for the two churn lessons
│   ├── sales_data.py       loading, cleaning and the three sales grains
│   ├── ts_core.py          splits, metrics, decomposition, baselines, features
│   ├── forecast_models.py  ETS, SARIMA, Prophet, XGBoost, RF, hybrid wrappers
│   └── control_series.py   one extra series, used to show a method working
└── data/                   Kaggle downloads (git-ignored, see below)
```

Lessons 1–2 and lessons 3–4 are two independent pairs: they share no data and no code, so
either pair can be taught on its own.

## 🚀 Getting started

Requires Python 3.13 (see `.python-version`) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync

uv run python propensity.py
uv run python survival.py
uv run python timeseries.py
uv run python forecasting.py

SHOW_FIGURES=0 uv run python timeseries.py    # numbers only, no browser tabs
```

Every script runs top to bottom, prints its results and opens interactive Plotly figures.
They can also be stepped through cell by cell — the `# %%` markers make them notebooks in
VS Code or Jupyter. All paths are anchored to the script, so they run from any directory.

Approximate runtimes: `timeseries.py` ~40 s (SARIMA with 52 seasonal lags is most of it),
`forecasting.py` ~20 s, `propensity.py` a few minutes with its hyper-parameter search,
`survival.py` under a minute.

### Sample size

`propensity.py` and `survival.py` each define a `SAMPLE_SIZE` constant near the top. The
full dataset is ~505,000 customers, where a hyper-parameter search takes hours; the
defaults (20,000 and 5,000) keep every result under a minute without changing the
conclusions. Set `SAMPLE_SIZE = None` to use everything.

## 📊 The datasets, and why

The CSVs are **not** in git (~40 MB together). Download them from Kaggle into `data/`.
The forecasting lessons fall back to generated data if they are missing, so a fresh clone
still runs; the churn lessons need the real files.

### Customer churn → `data/churn/`

`customer_churn_dataset-training-master.csv`, `customer_churn_dataset-testing-master.csv` —
[Kaggle](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)

Both files are two halves of one customer base, **not** a modelling split, so
`src/churn_data.py` concatenates them and each lesson makes its own. The column that makes
this dataset usable for both lessons is `Tenure`: propensity predicts *whether* `Churn`
happens, survival predicts *when*, using tenure as the observed time.

### Walmart weekly sales → `data/sales/`

`train.csv`, `stores.csv`, `features.csv` —
[Kaggle](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)

421,570 store-department-weeks from 45 stores, 2010-02-05 to 2012-10-26 — 143 consecutive
Fridays with no gaps. `test.csv` also ships with the download and is **never used**: it has
no `Weekly_Sales` column, because on Kaggle that is the hidden answer, and you cannot
measure an error against values you do not have.

Two quirks drive whole sections of the lessons. `MarkDown1-5` (promotion spend) is entirely
missing before 2011-11-11 — a structural break, not missing data. And `CPI` rises 5.3%
across the window, which is why a chain total that looks flat in euros may be shrinking in
goods.

### The control series

`src/control_series.py` loads weekly atmospheric CO₂ from `statsmodels` — real data, no
download, no extra dependency. It appears twice, for one reason: the Walmart chain total has
no trend at all (r² = 0.001), so the lesson on why tree models cannot extrapolate a trend
has nothing to demonstrate on it. The CO₂ record (r² = 0.948) does. It is not sales data and
does not pretend to be; it is a measuring stick.

## 📘 Key concepts

### Propensity (`propensity.py`)

Estimates, per customer, the **probability** of churning, and turns those scores into
targetable segments. Covers the look-alike time-window design (observation → buffer →
outcome), three model families, cross-validation, `GridSearchCV`/`RandomizedSearchCV`, the
metrics that matter for imbalanced data (precision/recall/F1, ROC-AUC, PR-AUC, confusion
matrix), and learning/validation curves for diagnosing under- and overfitting.

### Survival (`survival.py`)

Estimates **when** the event happens, and handles the customers who have not churned yet
(right-censored) instead of discarding them. Covers Kaplan-Meier curves, Cox proportional
hazards, Random Survival Forest and Survival SVM, plus the survival-specific metrics:
concordance index, time-dependent AUC(t) and the integrated Brier score.

### Analysing a time series (`timeseries.py`)

Starts with the question people skip — *is this series forecastable?* — and answers it with
a report covering grid regularity, available seasonal cycles, trend significance, seasonal
strength and stationarity. Then decomposition, ADF, ACF/PACF, the correct way to split a
time series, the four error measures (MAE, MSE, RMSE, MAPE) and when each is the right one,
naive and seasonal-naive baselines, exponential smoothing written out by hand before
Holt-Winters, SARIMA, and rolling-origin cross-validation.

**What this dataset can and cannot show.** 143 weeks is 2.75 seasonal cycles. A 52-week
seasonal model needs two complete cycles before it can be fitted at all — so by the time
enough history exists, every Christmas is already inside it. *The hardest weeks of the retail
year cannot be evaluated with this data.* Where the baseline can be scored on a Christmas
window it degrades from 2.0% to 3.3% MAPE. Every headline number in these lessons comes from
a quiet August–October window and is flattered by it, and the lessons say so.

### Prophet and XGBoost (`forecasting.py`)

Prophet's additive components, changepoints and the `changepoint_prior_scale` knob that
causes overshooting; holiday and event effects; feature engineering for models that cannot
see time; early stopping with a time-ordered validation split; the bias-variance trade-off
made visible through tree depth.

The through-line is that **the grain decides the winner**. The same XGBoost model, the same
features, the same code:

| Grain | Series | Result |
|---|---|---|
| Chain total | 1 | loses to a no-parameter baseline — only 78 usable training rows |
| Per store | 45 | beats the baseline; one model in 0.2 s replaces 45 fits taking 4.6 s |
| Per department | ~3,000 | MAPE reports 169% for a model that is not broken |

Which gives three lessons the slides do not: a booster needs rows before it needs tuning;
its real advantage is operational (*forecasting at scale*), not accuracy; and MAPE breaks on
series that pass near zero. The final section takes the standard "detrend before boosting"
fix and shows it improving the error 3.6× in one regime and making it worse in another —
because a lag feature already carries the level, so the fix is only right when the diagnosis
is right.

## 📝 License

MIT — see [LICENSE](LICENSE).
