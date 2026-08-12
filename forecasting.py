"""
==============================================================================
SALES FORECASTING — PROPHET AND XGBOOST, AND WHERE EACH ONE FAILS
==============================================================================
Course: Big Data and Machine Learning
Topic: Modern forecasting models on retail sales data

Business question
-----------------
"Prophet and XGBoost are supposed to be state of the art. Should we be using
them for the sales forecast?"

The answer this lesson arrives at is "it depends on the grain", and it gets there
by measuring rather than asserting. Both models are fitted on the same series
``timeseries.py`` used, then on the same sales aggregated two other ways, and the
winner changes each time.

The two models, in one line each
--------------------------------
**Prophet** decomposes a series into additive components — trend, yearly season,
holiday effects — and fits them with a piecewise-linear trend. It is built for
business data with events and outliers, and its selling point is that every
component can be pulled out and shown to somebody who does not model.

**XGBoost** is not a time-series model at all. It is a gradient-boosted tree
ensemble on a feature table, so the time axis has to be handed to it as columns:
week number, lags, rolling means. That is a weakness and a strength — it cannot
see time unless you engineer it, but it can use any other column you have.

What this lesson will find
--------------------------
1. On a single 143-point series, XGBoost **loses badly** to the classical models
   from the previous lesson, for a reason that has nothing to do with tuning.
2. On 45 parallel store series it beats the seasonal baseline, and its real
   advantage turns out to be **scale**, not accuracy.
3. At department grain MAPE stops being a usable measure entirely.
4. Adding external features (temperature, fuel price, CPI) buys nothing here.
5. Detrending before boosting — the standard fix for the fact that trees cannot
   extrapolate — helps enormously in one regime and hurts in another, and the
   difference is diagnosable in advance.

Learning objectives
-------------------
1. Fit Prophet and read its components, changepoints and holiday effects
2. Engineer calendar, Fourier and lag features for a tree model
3. Use early stopping correctly, with a time-ordered validation split
4. See how the grain of aggregation decides which model wins
5. Recognise when MAPE stops being a usable metric
6. Diagnose when a tree needs detrending, and when detrending hurts
7. Say what a model family cannot do, before someone else finds out

Run it
------
    uv sync
    uv run python forecasting.py

    SHOW_FIGURES=0 uv run python forecasting.py   # numbers only, no browser tabs

Previous lesson: timeseries.py — is this series forecastable, and classical models.

Dataset: https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast
==============================================================================
"""

# %% 1. SETUP
# ==============================================================================

import inspect
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Make `src/` importable no matter where this script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import forecast_models as fm  # noqa: E402
from src import ts_core as tc  # noqa: E402
from src.control_series import load_control_series  # noqa: E402
from src.sales_data import EXTERNAL_FEATURES, WALMART_HOLIDAYS, prepare  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

SHOW_FIGURES = os.environ.get("SHOW_FIGURES", "1") != "0"

BLUE, ORANGE, GREEN, GREY, RED = "#1e88e5", "#e67e22", "#2ecc71", "#95a5a6", "#e74c3c"


def show(fig) -> None:
    """Display a figure unless SHOW_FIGURES=0."""
    if SHOW_FIGURES:
        fig.show()


def show_source(fn) -> None:
    """Print the real source of a function, so the lesson can't drift from it."""
    print(f"\n--- {fn.__module__}.{fn.__name__} " + "-" * (48 - len(fn.__name__)))
    print(inspect.getsource(fn))


print("=" * 78)
print("SALES FORECASTING — PROPHET AND XGBOOST")
print("=" * 78)

# %% 2. THE SAME DATA, THE SAME SPLIT
# ==============================================================================
# Both lessons import their split from src/ts_core.py, so every number printed
# here can be laid directly beside the ones timeseries.py printed. Re-deriving
# the split in each script is how two lessons about the same data end up
# disagreeing for reasons nobody can find.

print("\nLoading and cleaning")
print("-" * 78)
chain, store_panel, dept_panel, SOURCE = prepare()

sales = tc.as_series(chain)
train, test = tc.split_series(sales, test_weeks=tc.TEST_WEEKS)
HORIZON = len(test)

baseline = tc.seasonal_naive_forecast(train, HORIZON)
baseline_metrics = tc.forecast_metrics(test, baseline)

print(f"\n  train {len(train)} weeks, test {HORIZON} weeks "
      f"({test.index[0].date()} to {test.index[-1].date()})")
print(f"  seasonal-naive baseline to beat: MAPE {baseline_metrics['MAPE']:.2f}%")
print("  (timeseries.py: Holt-Winters 1.59%, SARIMA 1.57% on this same window)")

chain_results = {"seasonal naive": baseline}

# %% 3. PROPHET — AN ADDITIVE MODEL YOU CAN EXPLAIN
# ==============================================================================
# Prophet models the series as a sum of interpretable pieces:
#
#     y(t) = trend(t) + seasonality(t) + holidays(t) + noise
#
# That is the whole design, and it is why Prophet is popular in businesses rather
# than in forecasting competitions: each term can be extracted and shown to
# somebody, so "why is November up" has an answer that is not "the model said so".
#
# Two settings here are corrections to what the original version of this script
# did. Weekly and daily seasonality are switched OFF: there is one observation per
# week, so there is no within-week pattern to estimate and asking for one fits
# components to noise.

show_source(fm.fit_prophet)

timer = time.perf_counter()
prophet_model, prophet_forecast = fm.fit_prophet(
    train, HORIZON, holidays=WALMART_HOLIDAYS, country_holidays="US"
)
prophet_seconds = time.perf_counter() - timer

predicted = fm.prophet_forecast_only(prophet_forecast, test.index)
chain_results["Prophet"] = predicted["yhat"].to_numpy()

coverage = np.mean(
    (test.to_numpy() >= predicted["yhat_lower"].to_numpy())
    & (test.to_numpy() <= predicted["yhat_upper"].to_numpy())
)

print(f"\n  Prophet   MAPE {tc.forecast_metrics(test, chain_results['Prophet'])['MAPE']:.2f}%"
      f"   fitted in {prophet_seconds:.1f}s")
print(f"  95% interval contains the actual value {coverage:.0%} of the time")
print("""
  A 95% interval that captures everything is not a triumph — it means the
  interval is wider than it needs to be, which on 130 observations is honest
  rather than impressive. An interval is a claim about uncertainty, and it should
  be checked like any other claim.""")

figure = go.Figure()
figure.add_trace(go.Scatter(
    x=predicted["ds"], y=predicted["yhat_upper"], mode="lines",
    line=dict(width=0), showlegend=False, hoverinfo="skip"))
figure.add_trace(go.Scatter(
    x=predicted["ds"], y=predicted["yhat_lower"], mode="lines", fill="tonexty",
    fillcolor="rgba(30,136,229,0.18)", line=dict(width=0), name="95% interval"))
figure.add_trace(go.Scatter(x=train.index[-40:], y=train.values[-40:], mode="lines",
                            name="history", line=dict(color=GREY, width=1.5)))
figure.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines+markers",
                            name="actual", line=dict(color=BLUE, width=2.5)))
figure.add_trace(go.Scatter(x=predicted["ds"], y=predicted["yhat"], mode="lines",
                            name="Prophet", line=dict(color=ORANGE, width=2, dash="dot")))
figure.update_layout(height=460, title="<b>Prophet forecast with its uncertainty interval</b>",
                     yaxis_title="weekly sales")
show(figure)

# %% 4. CHANGEPOINTS — WHERE PROPHET LETS THE TREND BEND
# ==============================================================================
# Prophet's trend is piecewise linear. It scatters candidate changepoints across
# the history and lets the slope shift at each one, with `changepoint_prior_scale`
# controlling how freely.
#
# This is the knob that causes the "overshooting" the slides warn about. A high
# value bends the trend to follow every wobble, which produces a beautiful fit to
# the past and then extrapolates whatever slope it happened to end on.

components = fm.prophet_components(prophet_model, prophet_forecast)
changepoints = components["changepoints"]

print("\nCHANGEPOINTS")
print("-" * 78)
print(f"  candidate changepoints placed: {len(changepoints)}")
print(f"  largest slope change: {np.abs(changepoints['delta']).max():.4f}")

print("\n  effect of changepoint_prior_scale on the forecast")
print(f"  {'prior scale':>12}{'test MAPE':>12}{'trend over horizon':>22}")
for scale in (0.01, 0.05, 0.5):
    tuned_model, tuned_forecast = fm.fit_prophet(
        train, HORIZON, holidays=WALMART_HOLIDAYS, changepoint_prior_scale=scale
    )
    tuned = fm.prophet_forecast_only(tuned_forecast, test.index)
    trend_part = tuned_forecast[tuned_forecast["ds"].isin(test.index)]["trend"]
    drift = 100 * (trend_part.iloc[-1] / trend_part.iloc[0] - 1)
    mape = tc.forecast_metrics(test, tuned["yhat"])["MAPE"]
    print(f"  {scale:>12}{mape:>11.2f}%{drift:>21.2f}%")

print("""
  Read the last column, not the middle one. As the prior scale rises the trend
  drifts further over the 13-week horizon — the model has been given permission to
  believe recent wobbles are real changes of direction. On a 13-week forecast the
  damage is modest. Extend the horizon to a year and this is the parameter that
  decides whether the forecast is a plan or a fantasy.

  Since timeseries.py established this series has no trend at all (p = 0.70), the
  right setting here is the low one. That is a decision made from the data, not
  from a default.""")

# %% 5. HOLIDAYS AND EVENTS
# ==============================================================================
# The component that most justifies Prophet on retail data. Holidays are supplied
# as a frame of dates with a window around each — because a Thanksgiving effect is
# not confined to Thanksgiving Thursday; shoppers move spend into the days before
# and the weekend after.

print("\nHOLIDAY EFFECTS")
print("-" * 78)
print(WALMART_HOLIDAYS.groupby("holiday").agg(
    dates=("ds", "count"), first=("ds", "min"), last=("ds", "max")
).to_string())

if "holidays" in components:
    holiday_effect = components["holidays"].set_index("ds")["holidays"]
    largest = holiday_effect.abs().nlargest(5)
    print("\n  weeks where the holiday component moved the forecast most:")
    for date, _ in largest.items():
        print(f"    {date.date()}   {holiday_effect.loc[date]:>+14,.0f}")

print("""
  `add_country_holidays("US")` was also switched on, which loads the public
  holiday calendar. Note what this buys: the model no longer has to *infer* that
  late November matters from two observations of it. External knowledge, supplied
  directly, is worth more than any amount of tuning when history is short.""")

# %% 6. THE COMPONENTS, PLOTTED
# ==============================================================================
# Prophet ships a matplotlib `plot_components`. These lessons are Plotly, and more
# usefully the components are just data — so they are extracted and drawn here,
# with the changepoints overlaid on the trend.

figure = make_subplots(rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.09,
                       subplot_titles=("Trend (with changepoints)", "Yearly seasonality",
                                       "Holiday effects"))
trend = components["trend"]
figure.add_trace(go.Scatter(x=trend["ds"], y=trend["trend"], mode="lines",
                            line=dict(color=BLUE, width=2), name="trend"), row=1, col=1)
for date in changepoints.loc[changepoints["delta"].abs() > 1e-4, "ds"]:
    figure.add_vline(x=date, line=dict(color=RED, dash="dot", width=1), row=1, col=1)

if "yearly" in components:
    yearly = components["yearly"].drop_duplicates("ds").sort_values("ds")
    figure.add_trace(go.Scatter(x=yearly["ds"], y=yearly["yearly"], mode="lines",
                                line=dict(color=GREEN, width=2), name="yearly"), row=2, col=1)
if "holidays" in components:
    figure.add_trace(go.Bar(x=components["holidays"]["ds"], y=components["holidays"]["holidays"],
                            marker_color=ORANGE, name="holidays"), row=3, col=1)
figure.update_layout(height=760, showlegend=False,
                     title="<b>Prophet's additive components</b>")
show(figure)

# %% 7. FEATURES FOR A MODEL THAT CANNOT SEE TIME
# ==============================================================================
# XGBoost has no idea what a date is. Every temporal fact has to become a column.


def build_features(
    frame: pd.DataFrame, group: str | list[str] | None = None
) -> pd.DataFrame:
    """Turn a sales frame into the feature table a tree model needs.

    Three kinds of column, each solving a different problem:

    * **calendar** — week, month, quarter, plus the sine/cosine pair that stops
      week 52 and week 1 looking 51 apart;
    * **lags and rolling means** — the target, shifted. This is what carries the
      *level* of the series; without it a tree has no idea what scale it is
      predicting on;
    * **``t``, a simple counter** — the only column expressing "later than".

    ``lag_52`` costs a full year of every series: 52 rows that cannot have it.
    On 143 weekly points that is more than a third of the data, and section 8 is
    entirely about what that costs.
    """
    out = frame.copy()
    calendar = tc.calendar_features(out["Date"]).reset_index(drop=True)
    out = pd.concat(
        [out.reset_index(drop=True), calendar[["week", "month", "quarter", "week_sin", "week_cos"]]],
        axis=1,
    )
    if "Type" in out.columns:
        out["Type"] = out["Type"].map({"A": 0, "B": 1, "C": 2}).astype(float)
    keys = [group] if isinstance(group, str) else list(group or [])
    out = out.sort_values(keys + ["Date"])
    out["t"] = out.groupby(keys).cumcount() if keys else np.arange(len(out))
    out = tc.lag_features(out, by=keys or None, lags=(1, 52), windows=(4,))
    return out.dropna(subset=["lag_1", "lag_52", "roll_4"])


show_source(build_features)

CALENDAR = ["week", "month", "quarter", "week_sin", "week_cos", "t"]
LAGS = ["lag_1", "lag_52", "roll_4"]

# %% 8. XGBOOST ON ONE SERIES — AND WHY IT LOSES
# ==============================================================================

chain_features = build_features(chain)
chain_train = chain_features[chain_features["Date"] <= train.index[-1]]
chain_test = chain_features[chain_features["Date"] > train.index[-1]]

X_fit, X_val, y_fit, y_val = fm.time_ordered_validation_split(
    chain_train[CALENDAR + LAGS], chain_train["Weekly_Sales"]
)
chain_model = fm.fit_xgboost(X_fit, y_fit, X_val, y_val)
chain_results["XGBoost"] = chain_model.predict(chain_test[CALENDAR + LAGS])

rf_model = fm.fit_random_forest(chain_train[CALENDAR + LAGS], chain_train["Weekly_Sales"])
chain_results["Random Forest"] = rf_model.predict(chain_test[CALENDAR + LAGS])

print("\nONE SERIES, ALL MODELS")
print("-" * 78)
chain_table = tc.metrics_table(
    {name: tc.forecast_metrics(test, prediction) for name, prediction in chain_results.items()}
)
print(chain_table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

print(f"""
  XGBoost is beaten by a forecast with no parameters. The reason is arithmetic,
  not tuning:

    weeks in the series                {len(sales)}
    minus the year that lag_52 eats     -52
    minus the held-out test window      -{HORIZON}
    = rows the booster trains on        {len(chain_train)}

  {len(chain_train)} rows, and roughly {len(CALENDAR + LAGS)} features. Gradient boosting builds hundreds of
  trees, each splitting the data further; there is nothing here to split. This is
  not a model that needs better hyper-parameters, it is a model that needs more
  rows — which is exactly what the next section gives it.""")

# %% 9. GIVE IT ROWS — THE STORE PANEL
# ==============================================================================
# The same sales, aggregated one level finer: 45 stores, each with its own weekly
# series. The chain total is one of those series summed; the panel keeps them
# apart. Two things change at once — there are 45x more rows, and Size and Type
# stop being constants and start being features.

panel = build_features(store_panel, group="Store")
panel_train, panel_test = tc.split_panel(panel, test_weeks=tc.TEST_WEEKS)
PANEL_FEATURES = ["Store", "Size", "Type"] + CALENDAR + LAGS

X_fit, X_val, y_fit, y_val = fm.time_ordered_validation_split(
    panel_train[PANEL_FEATURES], panel_train["Weekly_Sales"]
)
timer = time.perf_counter()
panel_model = fm.fit_xgboost(X_fit, y_fit, X_val, y_val)
xgb_seconds = time.perf_counter() - timer
panel_prediction = panel_model.predict(panel_test[PANEL_FEATURES])

panel_scores = {
    "seasonal naive": tc.forecast_metrics(panel_test["Weekly_Sales"], panel_test["lag_52"]),
    "XGBoost": tc.forecast_metrics(panel_test["Weekly_Sales"], panel_prediction),
}

# The honest comparison: one booster against one classical model per store.
timer = time.perf_counter()
ets_errors = []
for store, group in store_panel.groupby("Store"):
    series = tc.as_series(group)
    store_train, store_test = tc.split_series(series, test_weeks=tc.TEST_WEEKS)
    try:
        _, store_forecast = fm.fit_ets(store_train, len(store_test))
        ets_errors.append(np.abs(store_test.to_numpy() - store_forecast.to_numpy()) / store_test.to_numpy())
    except Exception:  # noqa: BLE001 - a store with too little history simply opts out
        continue
ets_seconds = time.perf_counter() - timer
ets_mape = 100 * float(np.mean(np.concatenate(ets_errors)))

print("\n45 SERIES INSTEAD OF ONE")
print("-" * 78)
print(f"  training rows      {len(panel_train):,}  (was {len(chain_train)})")
print(f"  {'model':<28}{'MAPE':>8}{'fit time':>12}{'models fitted':>16}")
print(f"  {'seasonal naive':<28}{panel_scores['seasonal naive']['MAPE']:>7.2f}%{'0.0s':>12}{'0':>16}")
print(f"  {'XGBoost (one model)':<28}{panel_scores['XGBoost']['MAPE']:>7.2f}%{xgb_seconds:>11.1f}s{'1':>16}")
print(f"  {'Holt-Winters (per store)':<28}{ets_mape:>7.2f}%{ets_seconds:>11.1f}s{len(ets_errors):>16}")

print(f"""
  This is the honest case for gradient boosting on forecasting problems, and it
  is not the one the slides make. XGBoost does beat the seasonal baseline here
  ({panel_scores['XGBoost']['MAPE']:.2f}% against {panel_scores['seasonal naive']['MAPE']:.2f}%). It does not beat fitting a classical model to
  every store separately ({ets_mape:.2f}%).

  What it does is get close with **one model in {xgb_seconds:.1f} seconds** instead of {len(ets_errors)}
  models in {ets_seconds:.1f} seconds. Scale that to 4,000 stores and the classical approach
  is 4,000 objects to fit, store, monitor and re-fit; the booster is still one.
  That is what "forecasting at scale" means, and it is an operations argument
  rather than an accuracy one.""")

importance = pd.Series(panel_model.feature_importances_, index=PANEL_FEATURES).sort_values()
figure = go.Figure(go.Bar(x=importance.values, y=importance.index, orientation="h",
                          marker_color=BLUE))
figure.update_layout(height=520, title="<b>What the booster actually used</b>",
                     xaxis_title="feature importance")
show(figure)

print("\n  feature importance")
for name, value in importance.sort_values(ascending=False).head(5).items():
    print(f"    {name:<12} {value:.3f}")
print(f"""
  ``{importance.idxmax()}`` alone accounts for {importance.max():.0%} of the model, and the two lag
  features together for {importance[[c for c in LAGS if c in importance.index]].sum():.0%}. Read that plainly: the booster has largely
  learned to reproduce the seasonal-naive baseline and adjust it. That is a perfectly respectable thing for it to do — and it is also why the
  margin over that baseline is modest.""")

# %% 10. EARLY STOPPING AND THE BIAS-VARIANCE TRADE-OFF
# ==============================================================================
# Boosting adds trees until told to stop. Each tree fits what the previous trees
# got wrong, so training error falls forever — and validation error eventually
# turns back up. Early stopping watches the validation curve and keeps the best
# iteration.
#
# Two ways to get this wrong, both common: supplying `eval_set` without
# `early_stopping_rounds` (nothing stops), and building the validation set with a
# random split (it measures interpolation, so it stops too late).

show_source(fm.time_ordered_validation_split)

print(f"\n  trees requested        {panel_model.n_estimators}")
print(f"  best iteration         {panel_model.best_iteration}")
print(f"  trees actually used    {panel_model.best_iteration + 1}"
      f"  ({(panel_model.best_iteration + 1) / panel_model.n_estimators:.0%} of the budget)")

print("\n  depth controls the same trade-off structurally")
print(f"  {'max_depth':>10}{'train MAPE':>13}{'test MAPE':>12}{'gap':>9}")
for depth in (2, 4, 6, 10):
    tuned = fm.fit_xgboost(X_fit, y_fit, X_val, y_val, max_depth=depth)
    in_sample = tc.forecast_metrics(y_fit, tuned.predict(X_fit))["MAPE"]
    out_sample = tc.forecast_metrics(panel_test["Weekly_Sales"],
                                     tuned.predict(panel_test[PANEL_FEATURES]))["MAPE"]
    print(f"  {depth:>10}{in_sample:>12.2f}%{out_sample:>11.2f}%{out_sample - in_sample:>8.2f}")

print("""
  Deeper trees cut the training error and widen the gap to the test error. That
  gap *is* overfitting, made visible: the model is describing this particular
  training set rather than the process that generated it. The goal is not the
  lowest training error, it is the lowest test error — which usually arrives well
  before the deepest tree.""")

# %% 11. DO EXTERNAL FEATURES HELP?
# ==============================================================================
# The claim on the slides is that XGBoost shines when the series is enriched with
# external drivers — weather being the standard example. This dataset ships four:
# temperature, fuel price, CPI and unemployment. So test it.
#
# One fold would not settle this, because the differences are small enough to be
# noise. Three rolling folds, same features, same everything else.

dates = np.sort(panel["Date"].unique())


def panel_folds(n_folds: int = 3, horizon: int = tc.TEST_WEEKS):
    """Expanding-window folds over the panel, cut on the date so all stores align."""
    for i in range(n_folds):
        end = len(dates) - (n_folds - 1 - i) * horizon
        cutoff, stop = dates[end - horizon], dates[end - 1]
        yield (panel[panel["Date"] < cutoff],
               panel[(panel["Date"] >= cutoff) & (panel["Date"] <= stop)])


def score_panel(feature_columns: list[str], hybrid: bool = False) -> list[float]:
    """MAPE on each fold for one feature set, optionally via the detrending hybrid."""
    scores = []
    for fold_train, fold_test in panel_folds():
        if hybrid:
            _, prediction = fm.fit_hybrid(fold_train, fold_test, feature_columns, group="Store")
        else:
            a, b, c, d = fm.time_ordered_validation_split(
                fold_train[feature_columns], fold_train["Weekly_Sales"]
            )
            prediction = fm.fit_xgboost(a, c, b, d).predict(fold_test[feature_columns])
        scores.append(tc.forecast_metrics(fold_test["Weekly_Sales"], prediction)["MAPE"])
    return scores


print("\nEXTERNAL FEATURES, THREE FOLDS")
print("-" * 78)
comparison = {
    "calendar + lags": score_panel(PANEL_FEATURES),
    "+ temperature, fuel, CPI, unemployment": score_panel(PANEL_FEATURES + EXTERNAL_FEATURES),
}
print(f"  {'feature set':<40}{'fold0':>8}{'fold1':>8}{'fold2':>8}{'mean':>9}")
for name, scores in comparison.items():
    print(f"  {name:<40}" + "".join(f"{s:>7.2f}%" for s in scores) + f"{np.mean(scores):>8.2f}%")

base_scores = comparison["calendar + lags"]
external_scores = comparison["+ temperature, fuel, CPI, unemployment"]
difference = np.mean(external_scores) - np.mean(base_scores)
fold_spread = max(base_scores) - min(base_scores)
per_fold = np.array(external_scores) - np.array(base_scores)

print(f"""
  Adding four external features moved the mean error by {difference:+.2f} points.
  Before calling that an improvement, measure it against the noise:

    effect of the features        {abs(difference):.2f} points
    spread between folds          {fold_spread:.2f} points
    per-fold change               {', '.join(f'{d:+.2f}' for d in per_fold)}

  The effect is roughly {fold_spread / max(abs(difference), 1e-9):.0f}x smaller than the disagreement between folds, and
  it does not even keep the same sign across them. That is not evidence of an
  improvement; it is evidence that three folds cannot resolve an effect this
  small. The honest conclusion is "no measurable benefit here", not "helps" or
  "hurts" — and the way to get a real answer would be more folds, not a bolder
  claim from these three.

  Why would the effect be small? These are *macro* variables: they move slowly
  and hit all 45 stores at once, so they say almost nothing about which store
  will deviate from its own seasonal pattern next week. The slides' example is
  weather, which for an ice-cream maker would genuinely matter. The lesson is not
  "external features are useless" but "they have to be causally close to what you
  are predicting, and these are not".""")

# %% 12. DEPARTMENT GRAIN — WHERE MAPE STOPS WORKING
# ==============================================================================
# One level finer again: store x department, ~3,300 series. The same model, the
# same features, the same code. Watch the metric, not the model.

dept_features = build_features(dept_panel, group=["Store", "Dept"])
dept_train, dept_test = tc.split_panel(dept_features, test_weeks=tc.TEST_WEEKS)
DEPT_FEATURES = ["Store", "Dept", "Size", "Type"] + CALENDAR + LAGS

timer = time.perf_counter()
dept_model = fm.fit_xgboost(dept_train[DEPT_FEATURES], dept_train["Weekly_Sales"])
dept_seconds = time.perf_counter() - timer
dept_prediction = dept_model.predict(dept_test[DEPT_FEATURES])

dept_metrics = tc.forecast_metrics(dept_test["Weekly_Sales"], dept_prediction)
dept_baseline = tc.forecast_metrics(dept_test["Weekly_Sales"], dept_test["lag_52"])

# The same predictions, summed back up to the chain total.
summed = dept_test.assign(prediction=dept_prediction).groupby("Date")[
    ["Weekly_Sales", "prediction"]
].sum()
summed_metrics = tc.forecast_metrics(summed["Weekly_Sales"], summed["prediction"])

print("\nTHE SAME MODEL AT THREE GRAINS")
print("-" * 78)
print(f"  {'grain':<22}{'series':>9}{'MAPE':>10}{'MAE':>16}")
print(f"  {'chain total':<22}{1:>9}"
      f"{tc.forecast_metrics(test, chain_results['XGBoost'])['MAPE']:>9.2f}%"
      f"{tc.forecast_metrics(test, chain_results['XGBoost'])['MAE']:>16,.0f}")
print(f"  {'per store':<22}{store_panel['Store'].nunique():>9}"
      f"{panel_scores['XGBoost']['MAPE']:>9.2f}%{panel_scores['XGBoost']['MAE']:>16,.0f}")
print(f"  {'per department':<22}{dept_features.groupby(['Store', 'Dept']).ngroups:>9}"
      f"{dept_metrics['MAPE']:>9.2f}%{dept_metrics['MAE']:>16,.0f}")
print(f"\n  department predictions summed back to the chain: MAPE {summed_metrics['MAPE']:.2f}%")

small = dept_test[dept_test["Weekly_Sales"] < 100]
print(f"""
  The MAPE at department grain is {dept_metrics['MAPE']:.0f}%. The model has not become
  {dept_metrics['MAPE'] / max(panel_scores['XGBoost']['MAPE'], 1e-9):.0f} times worse — the *metric* has broken.

  MAPE divides by the actual value. {len(small):,} of the {len(dept_test):,} test rows are departments
  that sold under 100 in a week; being 200 out on a week that sold 40 is a 500%
  error and a rounding difference in euros. The same predictions summed back to
  the chain total score {summed_metrics['MAPE']:.2f}%.

  Notice the baseline suffers identically ({dept_baseline['MAPE']:.0f}%), which is the tell: when
  every method scores absurdly, suspect the ruler rather than the methods. At
  this grain use MAE, or a weighted measure that lets big series count for more.
  This is the caveat that belongs beside "MAE, RMSE and MAPE are indispensable".""")

# %% 13. DETRENDING — THE RIGHT FIX FOR THE WRONG DIAGNOSIS
# ==============================================================================
# A tree predicts by averaging training observations, so it can never output a
# value above the highest level it has seen. On a growing series every forecast is
# therefore capped, and the standard fix is to fit the trend separately — with a
# linear regression — and let the booster model only the residual.
#
# It is a good technique. Whether it helps here depends on a diagnosis, and this
# section runs the experiment both ways.

show_source(fm.fit_hybrid)

NO_LAGS = ["Store", "Size", "Type"] + CALENDAR

print("\nDETRENDING, THREE FOLDS, TWO REGIMES")
print("-" * 78)
regimes = {
    "with lags   | plain XGBoost": score_panel(PANEL_FEATURES),
    "with lags   | trend + XGBoost": score_panel(PANEL_FEATURES, hybrid=True),
    "no lags     | plain XGBoost": score_panel(NO_LAGS),
    "no lags     | trend + XGBoost": score_panel(NO_LAGS, hybrid=True),
}
print(f"  {'regime':<34}{'fold0':>8}{'fold1':>8}{'fold2':>8}{'mean':>9}")
for name, scores in regimes.items():
    print(f"  {name:<34}" + "".join(f"{s:>7.2f}%" for s in scores) + f"{np.mean(scores):>8.2f}%")

with_lag_plain = np.mean(regimes["with lags   | plain XGBoost"])
with_lag_hybrid = np.mean(regimes["with lags   | trend + XGBoost"])
no_lag_plain = np.mean(regimes["no lags     | plain XGBoost"])
no_lag_hybrid = np.mean(regimes["no lags     | trend + XGBoost"])

print(f"""
  Two opposite results from the same technique, and the difference is the whole
  lesson:

  WITHOUT LAG FEATURES the tree genuinely has to extrapolate the level, and it
  cannot. Plain XGBoost scores {no_lag_plain:.2f}% — worse than every baseline in either
  lesson. Bolt a linear trend on and it becomes {no_lag_hybrid:.2f}%, a {no_lag_plain / no_lag_hybrid:.1f}x improvement.
  The fix works exactly as advertised.

  WITH LAG FEATURES it backfires: {with_lag_hybrid:.2f}% against {with_lag_plain:.2f}% for the plain model.
  ``lag_52`` already hands the tree last year's level, so the level was never
  missing — and the fitted straight line now adds an extrapolation of its own,
  which on stores whose decline flattens out marches confidently past them.

  THE RULE: detrend when the model has no way to know the level. If a lag of the
  target is already in the feature set, it is carrying the level and a trend term
  is redundant at best. "Best practice" applied without the diagnosis is how you
  make a model worse while believing you improved it.""")

# Confirmation on a series with an unmistakable trend, so that the "it works"
# half of the argument does not rest on Walmart's smallest stores.
control = load_control_series()
control_frame = pd.DataFrame({"Date": control.index, "Weekly_Sales": control.to_numpy()})
control_features = build_features(control_frame)
control_train, control_test = tc.split_panel(control_features, test_weeks=52)
CONTROL_FEATURES = ["week", "week_sin", "week_cos", "t"]

plain = fm.fit_xgboost(control_train[CONTROL_FEATURES], control_train["Weekly_Sales"])
plain_mape = tc.forecast_metrics(
    control_test["Weekly_Sales"], plain.predict(control_test[CONTROL_FEATURES])
)["MAPE"]
_, hybrid_prediction = fm.fit_hybrid(control_train, control_test, CONTROL_FEATURES)
hybrid_mape = tc.forecast_metrics(control_test["Weekly_Sales"], hybrid_prediction)["MAPE"]

control_trend = tc.trend_test(control)
print(f"""
  Confirmation, on the trending control series from timeseries.py section 3
  (weekly CO2, trend r2 = {control_trend['r_squared']:.3f} against this chain's 0.001), 52-week horizon:

    plain XGBoost        {plain_mape:.3f}%
    trend + XGBoost      {hybrid_mape:.3f}%     {plain_mape / hybrid_mape:.1f}x better

  Same code, same features, opposite conclusion — decided entirely by whether the
  series has a trend the model cannot otherwise see.""")

# %% 14. WHAT WE ARE NOT DOING: LSTM
# ==============================================================================
# The slides list LSTM alongside these models. It is not implemented here, and
# the reason is a lesson rather than an omission.

print("\nWHY NO LSTM")
print("-" * 78)
print(f"""
  {'model':<22}{'training rows':>15}{'params':>12}{'fits in':>10}
  {'seasonal naive':<22}{'0':>15}{'0':>12}{'0s':>10}
  {'Holt-Winters':<22}{len(train):>15}{'~55':>12}{'0.1s':>10}
  {'Prophet':<22}{len(train):>15}{'~25':>12}{prophet_seconds:>9.1f}s
  {'XGBoost (panel)':<22}{len(panel_train):>15,}{'~1000s':>12}{xgb_seconds:>9.1f}s
  {'LSTM':<22}{'10,000+ wanted':>15}{'10,000s':>12}{'minutes':>10}

  An LSTM learns long-range dependencies through recurrent state, which is
  genuinely powerful on long, high-frequency series — energy demand at hourly
  resolution, say, where a year is 8,760 observations.

  This chain total has {len(sales)}. A network with more parameters than training
  observations will fit the training data perfectly and forecast noise, and no
  amount of regularisation converts 143 points into enough information to
  estimate a recurrent network. Knowing that a model is inapplicable is worth
  more than a demonstration of it failing.""")

# %% 15. WHAT THE SLIDES CLAIM, AND WHAT THIS DATA SHOWS
# ==============================================================================

print("\nSLIDES VS DATA")
print("-" * 78)
print(f"""
  "Prophet: additive modelling, fast fitting, good explainability"
      Holds, completely. {prophet_seconds:.1f}s to fit, and the components in section 6 are
      the reason to choose it — a trend, a season and a holiday effect that can
      be shown to somebody who does not model.

  "XGBoost ... partly better than classical models such as ARIMA"
      Depends entirely on the grain, which the slides do not mention.
        one series      XGBoost {tc.forecast_metrics(test, chain_results['XGBoost'])['MAPE']:.2f}%  vs  SARIMA 1.57%   loses badly
        45 stores       XGBoost {panel_scores['XGBoost']['MAPE']:.2f}%  vs  seasonal naive {panel_scores['seasonal naive']['MAPE']:.2f}%   wins
      A booster needs rows. Below a few thousand it is the wrong tool, and no
      hyper-parameter search fixes that.

  "Feature engineering important: also integrate external factors (e.g. weather)"
      Feature engineering: strongly supported — ``{importance.idxmax()}`` alone carries {importance.max():.0%}
      of the panel model. External factors: not supported here. Temperature, fuel
      price, CPI and unemployment moved the mean error by {difference:+.2f} points, which
      is {fold_spread / max(abs(difference), 1e-9):.0f}x smaller than the spread between folds. The features that
      mattered were lags of the target itself.

  "Scaling forecasts in companies / forecasting at scale"
      This is the real case for boosting, and it is an operations argument:
      one model in {xgb_seconds:.1f}s covering {store_panel['Store'].nunique()} stores against {len(ets_errors)} separate fits
      taking {ets_seconds:.1f}s — and {len(ets_errors)} objects to store, monitor and re-fit.

  "MAE, RMSE, MAPE are indispensable"
      With a caveat the slides omit: at department grain MAPE reports {dept_metrics['MAPE']:.0f}%
      for a model that is not broken. Pick the metric to fit the data.

  The transferable habit: state the grain, the baseline and the metric before
  the model. A forecasting result without all three is not yet a result.""")

print("\n" + "=" * 78)
print(f"Done. Compared {len(chain_results)} models on one series and {store_panel['Store'].nunique()} stores; "
      f"best chain MAPE {chain_table.iloc[0]['MAPE']:.2f}% ({chain_table.iloc[0]['model']})  (data: {SOURCE})")
print("=" * 78)
print("\nPrevious lesson: timeseries.py — is this series forecastable, and classical models.")
