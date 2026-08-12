"""
==============================================================================
PROPENSITY MODELS — LOOK-ALIKE MODELING FOR CHURN
==============================================================================
Course: Big Data and Machine Learning
Topic: Propensity / look-alike modeling with Python

Business question
-----------------
"Acquiring and retaining customers costs money — who should we talk to?"

A **propensity model** answers that by estimating, for every single customer,
the probability that they perform a behaviour we care about: buying, accepting
an offer, or — here — churning. Because we have *labelled history* (we know who
churned), this is **supervised classification**.

Look-alike modeling
-------------------
We train on profiles that already showed the behaviour change, then look for
present-day customers with similar features ("look-alikes") who are likely to
show it next.

Learning objectives
-------------------
1. Frame churn as a look-alike problem, including the time-window design
2. Build and compare three model families (linear / tree / neural)
3. Generalise with cross-validation and tune with Grid/RandomizedSearchCV
4. Judge a classifier with the right metrics (not just accuracy)
5. Diagnose under-/overfitting with learning and validation curves
6. Turn propensity scores into targetable risk segments

Run it
------
    uv sync
    uv run python propensity.py

Dataset: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset
==============================================================================
"""

# %% 1. SETUP
# ==============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Make `src/` importable no matter where this script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.churn_data import encode_features, load_churn  # noqa: E402

# How many customers to model. The full dataset is ~505,000 rows; a
# GridSearchCV over that takes hours. 20,000 keeps every result in this script
# under a minute while the conclusions stay the same. Set to None for all data.
SAMPLE_SIZE = 20_000
RANDOM_STATE = 42

df = load_churn(sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE)

print("=" * 78)
print("PROPENSITY MODELS — LOOK-ALIKE MODELING FOR CHURN")
print("=" * 78)
print(f"\nCustomers: {len(df):,}   Columns: {df.shape[1]}")
print(f"Churn rate: {df['Churn'].mean() * 100:.1f}%")
print(f"\n{df.head()}")

# %% 2. THE LOOK-ALIKE SETUP (why time matters)
# ==============================================================================
# The single most common way a propensity model fails is not a bad algorithm —
# it is a leaky time design. A profile is built from an OBSERVATION period, and
# the label comes from a later OUTCOME period:
#
#     |---- observation ----|-- buffer --|-- outcome --|
#      features are built        gap        label is read
#            here             (lead time)      here
#
# * Observation period: everything the model is allowed to see (usage, support
#   calls, spend). Only data from *before* the cut-off.
# * Buffer: the lead time your business actually needs. If the retention team
#   needs two weeks to act, a model that predicts churn one day ahead is
#   worthless. The buffer also protects against label leakage — the last days
#   before a cancellation are full of give-away signals (cancelled auto-renew,
#   a "how do I close my account" ticket).
# * Outcome period: the window in which the event counts as a "yes".
#
# The same profile is then scored for a *different* customer in the *future* —
# that is the extra complexity a plain classification exercise does not have.
#
# This Kaggle table is already aggregated to one row per customer, so the
# windows are baked in rather than something we can slide. We name them anyway,
# because on real data defining them is the first task, not an afterthought.

print("\n" + "-" * 78)
print("LOOK-ALIKE WINDOWS (illustrative)")
print("-" * 78)
for name, weeks, role in [
    ("Observation", 12, "build features from behaviour in this window"),
    ("Buffer", 2, "lead time to act + protection against label leakage"),
    ("Outcome", 4, "did the customer churn in this window? -> label"),
]:
    print(f"  {name:<12} {weeks:>3} weeks   {role}")

# %% 3. DATA PREPARATION
# ==============================================================================
# Feature quality decides more than algorithm choice. Features can be
# socio-demographic (Age, Gender), behavioural (Usage Frequency, Support Calls),
# or commercial (Total Spend, Contract Length) — this dataset has all three.

X, y, encoders = encode_features(df)

print("\n" + "-" * 78)
print("FEATURES")
print("-" * 78)
print(f"  {X.shape[1]} features, {X.shape[0]:,} rows")
print(f"  Label-encoded categoricals: {list(encoders)}")

# Stratified split keeps the churn rate identical in train and test, so the
# test score measures the model and not an accident of sampling.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Logistic regression and neural nets are distance/gradient based and need
# comparable scales; trees split on thresholds and do not care.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Train: {len(X_train):,} ({y_train.mean() * 100:.1f}% churn)")
print(f"  Test:  {len(X_test):,} ({y_test.mean() * 100:.1f}% churn)")

# %% 4. EXPLORATORY DATA ANALYSIS
# ==============================================================================

fig_churn = px.pie(
    df,
    names="Churn",
    title="<b>Churn distribution</b>",
    color="Churn",
    color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
    hole=0.35,
)
fig_churn.update_traces(textinfo="percent+label")
fig_churn.show()

key_features = ["Support Calls", "Payment Delay", "Total Spend", "Last Interaction"]
fig_dist = make_subplots(rows=2, cols=2, subplot_titles=[f"<b>{f}</b>" for f in key_features])
for i, feat in enumerate(key_features):
    row, col = (i // 2) + 1, (i % 2) + 1
    for churned, label, color in [(0, "Stayed", "#2ecc71"), (1, "Churned", "#e74c3c")]:
        fig_dist.add_trace(
            go.Histogram(
                x=df.loc[df["Churn"] == churned, feat],
                name=label,
                marker_color=color,
                opacity=0.7,
                legendgroup=label,
                showlegend=(i == 0),
            ),
            row=row,
            col=col,
        )
fig_dist.update_layout(title_text="<b>Feature distributions by churn status</b>", barmode="overlay", height=600)
fig_dist.show()

churn_corr = X.corrwith(y).sort_values(ascending=False)
print("\n" + "-" * 78)
print("CORRELATION WITH CHURN (a first, linear-only hint at what matters)")
print("-" * 78)
for feat, corr in pd.concat([churn_corr.head(4), churn_corr.tail(4)]).items():
    print(f"  {feat:<22} {corr:+.4f}")

# %% 5. BASELINE MODELS
# ==============================================================================
# Three families, three trade-offs (slide: "Auswahl des besten Classification-
# Algorithmus"). All three expose predict_proba — essential here, because a
# look-alike model needs the *probability*, not just the predicted class.
#
#   Logistic Regression  linear, interpretable coefficients, fast
#   XGBoost              trees + boosting, captures interactions, strong default
#   Neural Network       flexible, needs the most data and tuning, opaque


def evaluate(name, model, X_fit, y_fit, X_eval, y_eval):
    """Fit a classifier and return one row of results.

    One helper for every model keeps the comparison honest: identical data,
    identical metrics, identical thresholds.
    """
    model.fit(X_fit, y_fit)
    probabilities = model.predict_proba(X_eval)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "name": name,
        "model": model,
        "probabilities": probabilities,
        "predictions": predictions,
        "metrics": {
            "Accuracy": accuracy_score(y_eval, predictions),
            "Precision": precision_score(y_eval, predictions, zero_division=0),
            "Recall": recall_score(y_eval, predictions, zero_division=0),
            "F1-Score": f1_score(y_eval, predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_eval, probabilities),
            "PR-AUC": average_precision_score(y_eval, probabilities),
        },
    }


results = {}

results["LR_Default"] = evaluate(
    "Logistic Regression",
    LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
)
results["XGB_Default"] = evaluate(
    "XGBoost",
    xgb.XGBClassifier(
        random_state=RANDOM_STATE, eval_metric="logloss", max_depth=2, learning_rate=0.01, n_estimators=100
    ),
    X_train,  # trees need no scaling
    y_train,
    X_test,
    y_test,
)
results["NN_Default"] = evaluate(
    "Neural Network",
    MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        alpha=0.1,
        random_state=RANDOM_STATE,
        max_iter=200,
        early_stopping=True,
    ),
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
)

print("\n" + "=" * 78)
print("BASELINE COMPARISON (untuned)")
print("=" * 78)
print(pd.DataFrame({k: v["metrics"] for k, v in results.items()}).round(4))

# %% 6. GENERALISATION: CROSS-VALIDATION
# ==============================================================================
# A single train/test split is one draw of a random variable. k-fold CV trains
# and tests k times and averages, which is a far more stable estimate. Stratified
# folds preserve the class ratio in every fold — mandatory for imbalanced data.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(
    LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    X_train_scaled,
    y_train,
    cv=cv,
    scoring="roc_auc",
)

print("\n" + "-" * 78)
print("5-FOLD STRATIFIED CV — Logistic Regression")
print("-" * 78)
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: ROC-AUC {score:.4f}")
print(f"  Mean {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  <- report this, not one lucky split")

# %% 7. HYPERPARAMETER TUNING
# ==============================================================================
# Hyperparameters (tree depth, regularisation strength, ...) steer *how* a model
# learns; they are not learned from the data. GridSearchCV tries every
# combination; RandomizedSearchCV samples n_iter of them, which wins as soon as
# the grid gets large. Both score each candidate by cross-validation.

lr_grid = GridSearchCV(
    LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"], "solver": ["liblinear"]},
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
)
lr_grid.fit(X_train_scaled, y_train)
print("\n" + "-" * 78)
print("GRIDSEARCHCV — Logistic Regression")
print("-" * 78)
print(f"  Candidates: {len(lr_grid.cv_results_['params'])} x 3 folds = {len(lr_grid.cv_results_['params']) * 3} fits")
print(f"  Best params: {lr_grid.best_params_}")
print(f"  Best CV ROC-AUC: {lr_grid.best_score_:.4f}")

xgb_search = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
    {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.3],
        "n_estimators": [100, 200],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
    n_iter=10,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
xgb_search.fit(X_train, y_train)
print("\n" + "-" * 78)
print("RANDOMIZEDSEARCHCV — XGBoost")
print("-" * 78)
print(f"  Sampled 10 of 72 combinations x 3 folds = 30 fits")
print(f"  Best params: {xgb_search.best_params_}")
print(f"  Best CV ROC-AUC: {xgb_search.best_score_:.4f}")

results["LR_Tuned"] = evaluate(
    "Logistic Regression (tuned)", lr_grid.best_estimator_, X_train_scaled, y_train, X_test_scaled, y_test
)
results["XGB_Tuned"] = evaluate("XGBoost (tuned)", xgb_search.best_estimator_, X_train, y_train, X_test, y_test)

comparison = pd.DataFrame({k: v["metrics"] for k, v in results.items()}).round(4)
best_name = comparison.loc["ROC-AUC"].idxmax()
print("\n" + "=" * 78)
print("ALL MODELS")
print("=" * 78)
print(comparison)
print(f"\nBest by ROC-AUC: {best_name} ({comparison.loc['ROC-AUC', best_name]:.4f})")

# %% 8. EVALUATION METRICS
# ==============================================================================
# Accuracy alone is a trap: with 57% churners, "predict churn for everyone"
# already scores 57%. What each metric answers:
#
#   Precision   of those we flagged, how many really churn? (cost of wasted offers)
#   Recall      of those who churn, how many did we catch?  (cost of missed saves)
#   F1          harmonic mean of the two
#   ROC-AUC     separability across ALL thresholds
#   PR-AUC      same idea, but focused on the positive class — the honest choice
#               when the positives are the rare and expensive ones

best = results[best_name]

cm = confusion_matrix(y_test, best["predictions"])
tn, fp, fn, tp = cm.ravel()
print("\n" + "-" * 78)
print(f"CONFUSION MATRIX — {best['name']}")
print("-" * 78)
print("                  predicted stay   predicted churn")
print(f"  actual stay        {tn:>8,}         {fp:>8,}   <- false alarms, wasted offers")
print(f"  actual churn       {fn:>8,}         {tp:>8,}   <- missed churners (top-left of the two costs)")
print()
print(classification_report(y_test, best["predictions"], target_names=["Stay", "Churn"], digits=4))

fig_roc = go.Figure()
for key, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result["probabilities"])
    fig_roc.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{key} (AUC={result['metrics']['ROC-AUC']:.3f})")
    )
fig_roc.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance (0.500)", line=dict(dash="dot", color="gray"))
)
fig_roc.update_layout(
    title="<b>ROC curves</b>", xaxis_title="False positive rate", yaxis_title="True positive rate", height=550
)
fig_roc.show()

precision, recall, _ = precision_recall_curve(y_test, best["probabilities"])
fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=best["name"], line=dict(width=3)))
fig_pr.add_hline(
    y=y_test.mean(),
    line_dash="dot",
    annotation_text=f"Chance level (AP = {y_test.mean():.2f})",
)
fig_pr.update_layout(
    title=f"<b>Precision-Recall curve — AP = {best['metrics']['PR-AUC']:.3f}</b>",
    xaxis_title="Recall",
    yaxis_title="Precision",
    height=500,
)
fig_pr.show()

# %% 9. LEARNING AND VALIDATION CURVES
# ==============================================================================
# Two diagnostics that answer different questions.
#
# Learning curve — "would more data help?"
#   both curves low and together .... underfitting, the model is too simple
#   train high, validation low ...... overfitting, the model memorises
#   curves converging high ........... healthy
#
# Validation curve — "how does ONE hyperparameter change things?"
#   Sweep it and watch where validation performance peaks and starts to fall.

sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    X_train_scaled,
    y_train,
    train_sizes=np.linspace(0.1, 1.0, 6),
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
)

fig_lc = go.Figure()
for scores, label, color in [(train_scores, "Training", "#3498db"), (val_scores, "Validation", "#e67e22")]:
    mean, std = scores.mean(axis=1), scores.std(axis=1)
    fig_lc.add_trace(go.Scatter(x=sizes, y=mean, mode="lines+markers", name=label, line=dict(color=color, width=3)))
    fig_lc.add_trace(
        go.Scatter(
            x=np.concatenate([sizes, sizes[::-1]]),
            y=np.concatenate([mean + std, (mean - std)[::-1]]),
            fill="toself",
            fillcolor=color,
            opacity=0.15,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
fig_lc.update_layout(
    title="<b>Learning curve — Logistic Regression</b>",
    xaxis_title="Training samples",
    yaxis_title="ROC-AUC",
    height=500,
)
fig_lc.show()

c_values = [0.001, 0.01, 0.1, 1, 10, 100]
train_scores_vc, val_scores_vc = validation_curve(
    LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver="liblinear"),
    X_train_scaled,
    y_train,
    param_name="C",
    param_range=c_values,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
)

fig_vc = go.Figure()
fig_vc.add_trace(
    go.Scatter(x=c_values, y=train_scores_vc.mean(axis=1), mode="lines+markers", name="Training")
)
fig_vc.add_trace(go.Scatter(x=c_values, y=val_scores_vc.mean(axis=1), mode="lines+markers", name="Validation"))
fig_vc.update_layout(
    title="<b>Validation curve — regularisation strength C</b>",
    xaxis_title="C (low = strong regularisation)",
    xaxis_type="log",
    yaxis_title="ROC-AUC",
    height=500,
)
fig_vc.show()

print("\n" + "-" * 78)
print("CURVE READING")
print("-" * 78)
print(f"  Learning curve  — train {train_scores.mean(axis=1)[-1]:.4f} vs validation {val_scores.mean(axis=1)[-1]:.4f}")
print(f"  Gap: {train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]:+.4f} (a large positive gap = overfitting)")
print(f"  Validation curve — best C = {c_values[int(np.argmax(val_scores_vc.mean(axis=1)))]}")

# %% 10. FEATURE IMPORTANCE
# ==============================================================================
# Two different questions: the linear model reports *direction and size* of an
# effect, the tree model reports *how useful a feature was for splitting*.

lr_importance = (
    pd.DataFrame({"Feature": X_train.columns, "Weight": np.abs(results["LR_Tuned"]["model"].coef_[0])})
    .sort_values("Weight", ascending=False)
    .head(10)
)
xgb_importance = (
    pd.DataFrame({"Feature": X_train.columns, "Weight": results["XGB_Tuned"]["model"].feature_importances_})
    .sort_values("Weight", ascending=False)
    .head(10)
)

fig_imp = make_subplots(rows=1, cols=2, subplot_titles=("<b>Logistic Regression |coef|</b>", "<b>XGBoost gain</b>"))
fig_imp.add_trace(
    go.Bar(x=lr_importance["Weight"], y=lr_importance["Feature"], orientation="h", marker_color="#3498db"),
    row=1,
    col=1,
)
fig_imp.add_trace(
    go.Bar(x=xgb_importance["Weight"], y=xgb_importance["Feature"], orientation="h", marker_color="#2ecc71"),
    row=1,
    col=2,
)
fig_imp.update_yaxes(autorange="reversed")
fig_imp.update_layout(title_text="<b>What drives the prediction?</b>", height=500, showlegend=False)
fig_imp.show()

# %% 11. PROPENSITY SCORES -> RISK SEGMENTS
# ==============================================================================
# This is the deliverable. Not "churn: yes/no", but a score per customer that
# marketing can sort, cut, and act on. The cut-offs are a *business* decision:
# they trade wasted retention budget (false positives) against lost customers
# (false negatives).

scores = pd.DataFrame(
    {
        "CustomerID": X_test.index,
        "Propensity": best["probabilities"],
        "Actual_Churn": y_test.to_numpy(),
    }
)
scores["Segment"] = pd.cut(
    scores["Propensity"], bins=[0, 0.4, 0.7, 1.0], labels=["Low risk", "Medium risk", "High risk"]
)

segment_summary = (
    scores.groupby("Segment", observed=False)
    .agg(Customers=("Propensity", "size"), Actual_churn_rate=("Actual_Churn", "mean"))
    .assign(Share=lambda d: d["Customers"] / d["Customers"].sum())
)

print("\n" + "=" * 78)
print(f"RISK SEGMENTS — scored with {best['name']}")
print("=" * 78)
print(segment_summary.round(3))

fig_scores = px.histogram(
    scores,
    x="Propensity",
    color="Actual_Churn",
    nbins=40,
    barmode="overlay",
    opacity=0.7,
    color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
    title="<b>Propensity score distribution — a good model separates the two colours</b>",
)
fig_scores.add_vline(x=0.4, line_dash="dash", annotation_text="Low | Medium")
fig_scores.add_vline(x=0.7, line_dash="dash", annotation_text="Medium | High")
fig_scores.update_layout(height=500)
fig_scores.show()

print("""
ACTIONS PER SEGMENT
  High risk    personal outreach, retention offer, escalate high-value accounts
  Medium risk  automated engagement, satisfaction survey, usage tips
  Low risk     standard communication, upsell and loyalty programmes

The same machinery answers the acquisition question from the lecture: swap the
label from "churned" to "bought a membership" and the model finds the
look-alikes worth approaching.

Next lesson: survival.py — not just WHETHER a customer churns, but WHEN.
""")
