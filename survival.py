"""
==============================================================================
SURVIVAL ANALYSIS — HOW LONG DO OUR CUSTOMERS STAY?
==============================================================================
Course: Big Data and Machine Learning
Topic: Time-to-event modeling with Python

Classification answers *whether* a customer churns. Survival analysis answers
*when* — and that is what campaign timing needs.

Three ideas make it its own discipline
--------------------------------------
1. **Censoring.** Most customers have not churned *yet*. They are not "negative
   examples"; they are observations that are still running (right-censored).
   Throwing them away biases everything downwards; treating them as "stayed
   forever" biases everything upwards. Survival models use them correctly.
2. **The survival function** S(t) = P(T > t) — the probability of still being a
   customer after t months.
3. **Different metrics.** Censoring makes MSE/RMSE inapplicable. We use the
   concordance index, time-dependent AUC, and the Brier score instead.

Learning objectives
-------------------
1. Turn a customer table into (event, time) survival data
2. Read a Kaplan-Meier curve
3. Compare Cox PH, Random Survival Forest and Survival SVM
4. Evaluate with c-index, AUC(t) and the integrated Brier score
5. Predict individual survival curves and stratify customers by risk

Run it
------
    uv sync
    uv run python survival.py

Dataset: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset
==============================================================================
"""

# %% 1. SETUP
# ==============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.svm import FastSurvivalSVM

# Make `src/` importable no matter where this script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.churn_data import load_churn  # noqa: E402

# Survival models scale worse than classifiers: Cox does a full likelihood
# optimisation and the SVM solves a ranking problem over pairs. 5,000 customers
# keeps every fit in this script to a few seconds. Set to None for all ~505,000.
SAMPLE_SIZE = 5_000
RANDOM_STATE = 42

df = load_churn(sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE)

print("=" * 78)
print("SURVIVAL ANALYSIS — CUSTOMER CHURN")
print("=" * 78)
print(f"\nCustomers: {len(df):,}   Columns: {df.shape[1]}")

# %% 2. FROM CUSTOMER TABLE TO SURVIVAL DATA
# ==============================================================================
# Survival data needs exactly two target columns:
#
#   event  did the event happen during our observation?  (1 churned, 0 censored)
#   time   how long we watched this customer             (months of tenure)
#
# Note what "0" means: NOT "this customer will never churn", but "as of the
# cut-off date they were still here". A customer with time=8, event=0 tells us
# something real — they survived at least 8 months — and Kaplan-Meier uses
# exactly that partial information.
#
# In the raw data every customer is measured from their own signup date, so the
# calendar dates differ but the clock always starts at t0 = signup:
#
#   calendar time                     time from signup
#   |--#####x                         |#####x        churned at 6
#   |   |#########?                   |#########?    still active (censored)
#   |     |###x                       |###x          churned at 4
#                ^ cut-off

df["event"] = df["Churn"].astype(bool)
df["time"] = df["Tenure"].astype(float)

print("\n" + "-" * 78)
print("EVENT / TIME")
print("-" * 78)
print(f"  Churned (event=1):   {df['event'].sum():,} ({df['event'].mean() * 100:.1f}%)")
print(f"  Censored (event=0):  {(~df['event']).sum():,} ({(~df['event']).mean() * 100:.1f}%)")
print(f"  Observed time: {df['time'].min():.0f}-{df['time'].max():.0f} months (median {df['time'].median():.0f})")

# %% 3. KAPLAN-MEIER: THE SURVIVAL FUNCTION
# ==============================================================================
# Kaplan-Meier estimates S(t) without assuming any distribution. At every time
# where an event happens it multiplies in the fraction that survived it:
#
#     S(t) = PROD over t_i <= t of (1 - d_i / n_i)
#
#     d_i = churned at t_i        n_i = still "at risk" just before t_i
#
# Censored customers stay in n_i until they leave the study — which is precisely
# how their partial information gets used.

km_time, km_survival, km_conf = kaplan_meier_estimator(df["event"], df["time"], conf_type="log-log")

fig_km = go.Figure()
fig_km.add_trace(
    go.Scatter(
        x=np.concatenate([km_time, km_time[::-1]]),
        y=np.concatenate([km_conf[1], km_conf[0][::-1]]),
        fill="toself",
        fillcolor="rgba(0,100,255,0.15)",
        line=dict(width=0),
        name="95% CI",
        hoverinfo="skip",
    )
)
fig_km.add_trace(
    go.Scatter(x=km_time, y=km_survival, mode="lines", line=dict(color="#1e88e5", width=3, shape="hv"), name="S(t)")
)
for t in (6, 12, 24, 36):
    if t <= km_time.max():
        fig_km.add_annotation(
            x=t, y=km_survival[np.searchsorted(km_time, t)],
            text=f"{km_survival[np.searchsorted(km_time, t)] * 100:.0f}% at {t}m",
            showarrow=True, arrowhead=2,
        )
fig_km.update_layout(
    title="<b>Kaplan-Meier: share of customers still with us after t months</b>",
    xaxis_title="Months since signup",
    yaxis_title="S(t) — survival probability",
    yaxis_range=[0, 1],
    height=520,
)
fig_km.show()

# Curves split by a segment show *where* the difference is, not just that the
# averages differ. A gap that opens early calls for onboarding work; one that
# opens late calls for loyalty work.
fig_seg = go.Figure()
for contract in sorted(df["Contract Length"].unique()):
    mask = df["Contract Length"] == contract
    t_g, s_g = kaplan_meier_estimator(df.loc[mask, "event"], df.loc[mask, "time"])
    fig_seg.add_trace(go.Scatter(x=t_g, y=s_g, mode="lines", name=str(contract), line=dict(width=3, shape="hv")))
fig_seg.update_layout(
    title="<b>Survival by contract length</b>",
    xaxis_title="Months since signup",
    yaxis_title="S(t)",
    yaxis_range=[0, 1],
    height=500,
)
fig_seg.show()

print("\n" + "-" * 78)
print("READING THE CURVE")
print("-" * 78)
for t in (6, 12, 24):
    if t <= km_time.max():
        print(f"  S({t:>2}) = {km_survival[np.searchsorted(km_time, t)] * 100:5.1f}% still customers")

# %% 4. FEATURES
# ==============================================================================
# `Tenure` must NOT be a feature: it *is* the time we are modelling. Feeding it
# in is textbook target leakage.

exclude = ["CustomerID", "Churn", "event", "time", "Tenure"]
feature_cols = [c for c in df.columns if c not in exclude]

X = pd.get_dummies(df[feature_cols], drop_first=True)
numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
X[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])
X = X.astype(float)  # scikit-survival wants a plain float matrix

# scikit-survival expects the target as a structured array of (event, time).
y = np.array(list(zip(df["event"], df["time"])), dtype=[("event", bool), ("time", float)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=df["event"]
)

print("\n" + "-" * 78)
print("FEATURES")
print("-" * 78)
print(f"  {X.shape[1]} features: {', '.join(X.columns)}")
print(f"  Train {len(X_train):,} / Test {len(X_test):,}")

# %% 5. MODELS
# ==============================================================================
# | Model                  | Idea                             | Trade-off              |
# |------------------------|----------------------------------|------------------------|
# | Cox PH                 | semi-parametric; covariates shift | interpretable hazard   |
# |                        | the hazard proportionally        | ratios, but the        |
# |                        |                                  | proportionality        |
# |                        |                                  | assumption must hold   |
# | Random Survival Forest | ensemble of survival trees       | non-linear effects,    |
# |                        |                                  | few assumptions,       |
# |                        |                                  | harder to interpret    |
# | Survival SVM           | SVM idea as a ranking problem    | flexible via kernels,  |
# |                        |                                  | outputs only a SCORE   |
# |                        |                                  | — no S(t)              |

models = {}

print("\n" + "=" * 78)
print("TRAINING")
print("=" * 78)

models["Cox PH"] = CoxPHSurvivalAnalysis(alpha=0.1).fit(X_train, y_train)
print("  ✓ Cox Proportional Hazards")

models["Random Survival Forest"] = RandomSurvivalForest(
    n_estimators=50, max_depth=5, min_samples_leaf=15, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1
).fit(X_train, y_train)
print("  ✓ Random Survival Forest")

models["Survival SVM"] = FastSurvivalSVM(
    rank_ratio=1.0, alpha=0.01, max_iter=50, random_state=RANDOM_STATE
).fit(X_train, y_train)
print("  ✓ Survival SVM")

# %% 6. EVALUATION — CONCORDANCE INDEX
# ==============================================================================
# The c-index asks: for every pair of customers whose order we actually know,
# did the model rank them correctly? (A churns at 3 months, B at 10 -> the model
# should give A the higher risk.) 0.5 is coin-flipping, 1.0 is perfect ordering.
# It is the survival analogue of ROC-AUC and works for *every* model here,
# because it only needs a risk score.
#
# One sign convention to remember: Cox and the SVM return a **risk** score
# (higher = churns sooner). RandomSurvivalForest.predict returns a cumulative
# hazard, which is also "higher = worse", so all three go in unflipped.

cindex_rows = []
for name, model in models.items():
    train_c = concordance_index_censored(y_train["event"], y_train["time"], model.predict(X_train))[0]
    test_c = concordance_index_censored(y_test["event"], y_test["time"], model.predict(X_test))[0]
    cindex_rows.append({"Model": name, "Train c-index": train_c, "Test c-index": test_c, "Gap": train_c - test_c})

cindex_df = pd.DataFrame(cindex_rows)
print("\n" + "-" * 78)
print("CONCORDANCE INDEX (0.5 = random, >0.7 usable, >0.8 good)")
print("-" * 78)
print(cindex_df.round(4).to_string(index=False))

fig_c = go.Figure()
fig_c.add_trace(go.Bar(x=cindex_df["Model"], y=cindex_df["Train c-index"], name="Train", marker_color="#90caf9"))
fig_c.add_trace(go.Bar(x=cindex_df["Model"], y=cindex_df["Test c-index"], name="Test", marker_color="#1565c0"))
fig_c.update_layout(
    title="<b>Model comparison: concordance index</b>", yaxis_range=[0.5, 1.0], barmode="group", height=500
)
fig_c.show()

# %% 7. EVALUATION — TIME-DEPENDENT AUC AND BRIER SCORE
# ==============================================================================
# The c-index is one number for the whole horizon. Two models with the same
# c-index can behave completely differently at month 6 versus month 36.
#
#   AUC(t)   how well does the model separate "churns by t" from "does not"?
#            One ROC-AUC per time point.
#   Brier(t) mean squared error between predicted S(t) and what happened,
#            corrected for censoring. It grades calibration AND discrimination.
#            Integrated over t -> a single number, lower is better.
#
# Censoring bites in the EVALUATION too. Both metrics reweight observations by
# the inverse probability of still being uncensored, G(t) — and G drops to zero
# at the very end of the follow-up, where nobody is left to observe. So we grade
# the models strictly *inside* the observed window and say so out loud.
horizon = min(y_train["time"].max(), y_test["time"].max())
in_window = y_test["time"] < horizon
X_window, y_window = X_test[in_window], y_test[in_window]
eval_times = np.unique(np.percentile(y_window["time"][y_window["event"]], np.linspace(10, 90, 9)))

print("\n" + "-" * 78)
print("EVALUATION WINDOW")
print("-" * 78)
print(f"  Horizon: {horizon:.0f} months — {in_window.sum():,} of {len(y_test):,} test customers are inside it")
print(f"  Scored at t = {', '.join(f'{t:.0f}' for t in eval_times)} months")

fig_auc = go.Figure()
brier_rows = []
for name, model in models.items():
    auc_t, auc_mean = cumulative_dynamic_auc(y_train, y_window, model.predict(X_window), eval_times)
    fig_auc.add_trace(
        go.Scatter(x=eval_times, y=auc_t, mode="lines+markers", name=f"{name} (mean {auc_mean:.3f})")
    )

    # Only models that produce a survival FUNCTION can be scored with Brier.
    if hasattr(model, "predict_survival_function"):
        surv_funcs = model.predict_survival_function(X_window)
        preds = np.vstack([[fn(t) for t in eval_times] for fn in surv_funcs])
        brier_rows.append(
            {"Model": name, "Integrated Brier score": integrated_brier_score(y_train, y_window, preds, eval_times)}
        )
    else:
        # This is a real limitation, not a gap to paper over: FastSurvivalSVM
        # learns a ranking, so it has no S(t) to compare against reality. The
        # honest answer is "not applicable — judge this model by the c-index".
        brier_rows.append({"Model": name, "Integrated Brier score": np.nan})

fig_auc.add_hline(y=0.5, line_dash="dot", annotation_text="Chance")
fig_auc.update_layout(
    title="<b>Time-dependent AUC — discrimination is not constant over time</b>",
    xaxis_title="Months since signup",
    yaxis_title="AUC(t)",
    height=500,
)
fig_auc.show()

print("\n" + "-" * 78)
print("INTEGRATED BRIER SCORE (lower is better; 0.25 = uninformative)")
print("-" * 78)
print(pd.DataFrame(brier_rows).round(4).to_string(index=False, na_rep="n/a — score-only model, no S(t)"))

# %% 8. INDIVIDUAL SURVIVAL CURVES
# ==============================================================================
# This is what a classifier cannot give you: for one specific customer, the
# whole curve. "80% chance of still being here in 6 months, 45% in 24" turns
# into a concrete question — when is the intervention worth its cost?

rng = np.random.default_rng(RANDOM_STATE)  # seeded: the same three customers every run
sample_idx = rng.choice(len(X_test), size=3, replace=False)
curve_models = {n: m for n, m in models.items() if hasattr(m, "predict_survival_function")}

fig_ind = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[f"Customer {i + 1}" for i in range(3)],
    x_title="Months since signup",
    y_title="S(t)",
)
for col, idx in enumerate(sample_idx, start=1):
    for name, model in curve_models.items():
        fn = model.predict_survival_function(X_test.iloc[[idx]])[0]
        fig_ind.add_trace(
            go.Scatter(x=fn.x, y=fn.y, mode="lines", name=name, line=dict(width=2, shape="hv"), showlegend=(col == 1)),
            row=1,
            col=col,
        )
    fig_ind.add_vline(
        x=y_test[idx]["time"],
        line_dash="dash",
        line_color="gray",
        annotation_text="churned" if y_test[idx]["event"] else "still active",
        row=1,
        col=col,
    )
fig_ind.update_yaxes(range=[0, 1])
fig_ind.update_layout(title_text="<b>Predicted survival curves for three test customers</b>", height=450)
fig_ind.show()

# %% 9. WHAT DRIVES THE HAZARD?
# ==============================================================================
# Cox coefficients are log hazard ratios: exp(coef) is the multiplicative effect
# on the churn hazard of a one-standard-deviation increase in that feature.
#   coef > 0  -> churns sooner      coef < 0  -> stays longer

cox_coefs = (
    pd.DataFrame({"Feature": X_train.columns, "Coefficient": models["Cox PH"].coef_})
    .assign(**{"Hazard ratio": lambda d: np.exp(d["Coefficient"])})
    .sort_values("Coefficient")
)

fig_coef = go.Figure(
    go.Bar(
        x=cox_coefs["Coefficient"],
        y=cox_coefs["Feature"],
        orientation="h",
        marker_color=["#2e7d32" if c < 0 else "#c62828" for c in cox_coefs["Coefficient"]],
    )
)
fig_coef.update_layout(
    title="<b>Cox coefficients — red speeds churn up, green slows it down</b>",
    xaxis_title="Coefficient (log hazard ratio)",
    height=500,
)
fig_coef.show()

print("\n" + "-" * 78)
print("COX COEFFICIENTS")
print("-" * 78)
print(cox_coefs.round(4).to_string(index=False))

# %% 10. RISK STRATIFICATION
# ==============================================================================
# Split the test customers into risk quartiles by predicted risk, then draw the
# *actual* Kaplan-Meier curve of each group. If the model works, the four curves
# fan out — and that fan is what a campaign plan is built on.

best_name = cindex_df.loc[cindex_df["Test c-index"].idxmax(), "Model"]
risk_scores = models[best_name].predict(X_test)
risk_levels = ["Low", "Medium", "High", "Very high"]
risk_groups = np.asarray(pd.qcut(risk_scores, q=4, labels=risk_levels))

fig_strata = go.Figure()
for level in risk_levels:
    mask = risk_groups == level
    t_r, s_r = kaplan_meier_estimator(y_test["event"][mask], y_test["time"][mask])
    fig_strata.add_trace(go.Scatter(x=t_r, y=s_r, mode="lines", name=f"{level} risk", line=dict(width=3, shape="hv")))
fig_strata.update_layout(
    title=f"<b>Observed survival by predicted risk quartile ({best_name})</b>",
    xaxis_title="Months since signup",
    yaxis_title="S(t)",
    yaxis_range=[0, 1],
    height=500,
)
fig_strata.show()

strata_summary = pd.DataFrame(
    {
        "Risk group": risk_levels,
        "Customers": [int((risk_groups == g).sum()) for g in risk_levels],
        "Churn rate": [y_test["event"][risk_groups == g].mean() for g in risk_levels],
        "Mean tenure": [y_test["time"][risk_groups == g].mean() for g in risk_levels],
    }
)

print("\n" + "=" * 78)
print(f"RISK STRATIFICATION — {best_name}")
print("=" * 78)
print(strata_summary.round(3).to_string(index=False))

print("""
WHAT SURVIVAL ANALYSIS ADDS
  * censored customers are used instead of discarded or mislabelled
  * the answer is a curve, not a yes/no — so you can pick the MOMENT to act
  * risk groups can be timed: contact the "very high" group before their curve
    drops, not after
  * the same machinery powers churn-time forecasting and lifetime-value models

CAVEATS
  * Cox assumes proportional hazards — check it before trusting the ratios
  * score-only models (the SVM here) can be ranked but not calibrated
  * compare several models by cross-validation and more than one metric

Previous lesson: propensity.py — WHETHER a customer churns.
""")
