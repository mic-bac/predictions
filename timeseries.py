"""
==============================================================================
TIME SERIES ANALYSIS — CAN THIS SERIES BE FORECAST AT ALL?
==============================================================================
Course: Big Data and Machine Learning
Topic: Analysing a time series, and the classical models that forecast it

Business question
-----------------
"After all our campaigns, offers and promotions, revenue has to grow. We need to
know how it will change so that we can plan."

That question hides a harder one. Before asking *what will sales be*, somebody
has to ask *what can this data actually tell us* — and that question has a real
answer, computable in about twenty lines, which is where this lesson starts.

A time series is a series of observations in time order, where time is the
independent variable. What makes it its own discipline is that the observations
are **not independent**: this week resembles last week, and resembles this week
last year, and a model that ignores those relationships throws away most of the
signal. Three ideas name the structure:

    Autocorrelation  how much an observation resembles earlier ones
    Seasonality      patterns that repeat on a fixed cycle
    Stationarity     whether the mean and variance stay put over time

Decomposition then splits the series into the parts that produced it::

    observed  =  trend  +  seasonality  +  residual

What this lesson will find
--------------------------
That this particular series is dominated by its yearly season (strength 0.98),
has **no** usable trend (p = 0.70), and offers only 2.75 seasonal cycles of
history. Every model result later on follows from those three numbers, and one
of them makes a whole category of evaluation impossible. Finding that out first
is the difference between forecasting and guessing.

Learning objectives
-------------------
1. Test whether a series is suitable for forecasting before modelling it
2. Decompose a series into trend, seasonality and residual
3. Read stationarity (ADF) and autocorrelation (ACF/PACF) plots
4. Split a time series correctly — and know why shuffling is fatal
5. Compute and choose between MAE, MSE, RMSE and MAPE
6. Beat, or fail to beat, a seasonal-naive baseline
7. Fit exponential smoothing and SARIMA, and judge what they cost
8. Cross-validate a forecast without leaking the future

Run it
------
    uv sync
    uv run python timeseries.py

    SHOW_FIGURES=0 uv run python timeseries.py    # numbers only, no browser tabs

Next lesson: forecasting.py — Prophet and XGBoost on the same data.

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
from src.sales_data import deflate, prepare  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# Figures open in a browser tab. Set SHOW_FIGURES=0 to run the numbers only —
# handy when you just want the printed output, or when presenting from a terminal.
SHOW_FIGURES = os.environ.get("SHOW_FIGURES", "1") != "0"

BLUE, ORANGE, GREY, RED = "#1e88e5", "#e67e22", "#95a5a6", "#e74c3c"


def show(fig) -> None:
    """Display a figure unless SHOW_FIGURES=0."""
    if SHOW_FIGURES:
        fig.show()


def show_source(fn) -> None:
    """Print the real source of a function, so the lesson can't drift from it.

    The primitives live in ``src/ts_core.py`` (``forecasting.py`` needs the same
    splits and metrics, and importing this script would re-run the whole lesson).
    Printing them here means the code you read is the code that just ran.
    """
    print(f"\n--- {fn.__module__}.{fn.__name__} " + "-" * (48 - len(fn.__name__)))
    print(inspect.getsource(fn))


print("=" * 78)
print("TIME SERIES ANALYSIS — WALMART WEEKLY SALES")
print("=" * 78)

# %% 2. THE DATA
# ==============================================================================
# One row per store, department and week. The lesson forecasts the chain total —
# every store summed into a single weekly number — because that is the series a
# board actually plans against, and because a single series is the right place to
# meet decomposition, ARIMA and the error measures for the first time.

print("\nLoading and cleaning")
print("-" * 78)
chain, store_panel, dept_panel, SOURCE = prepare()

sales = tc.as_series(chain)

print(f"\n  chain total: {len(sales)} weekly observations")
print(f"  {sales.index[0].date()} to {sales.index[-1].date()}")
print(f"  mean {sales.mean():,.0f}   min {sales.min():,.0f}   max {sales.max():,.0f}")

# %% 3. IS THIS SERIES FORECASTABLE?
# ==============================================================================
# The question to answer before choosing a model, not after being disappointed by
# one. Five properties decide what is possible:
#
#   regular grid       are the observations evenly spaced, with no gaps?
#   cycles             how many complete seasons of history are there?
#   trend              is there a real long-run direction, or just noise?
#   seasonal strength  how much of the variation is the repeating pattern?
#   stationarity       do the mean and variance stay put?
#
# `cycles` is the one people skip, and it is the one that binds hardest here. A
# yearly seasonal model estimates 52 separate seasonal terms. Two cycles is the
# bare minimum for it to be fitted at all; below three, each of those 52 terms
# rests on two or three observations.

show_source(tc.suitability_report)

report = tc.suitability_report(sales)

print("\nSUITABILITY REPORT")
print("-" * 78)
print(f"  observations        {report['n']}  ({report['start'].date()} to {report['end'].date()})")
print(f"  regular grid        {report['regular_grid']}  (every {report['step_days']} days)")
print(f"  seasonal cycles     {report['cycles']:.2f}      <- 52-week season")
print(f"  trend               {report['trend_pct_per_year']:+.2f}%/year   "
      f"p={report['trend_p_value']:.2f}  r2={report['trend_r_squared']:.3f}")
print(f"  seasonal strength   {report['seasonal_strength']:.2f}   trend strength {report['trend_strength']:.2f}")
print(f"  ADF stationarity    p={report['adf_p_value']:.2e}  -> stationary={report['stationary']}")
print(f"  strongest lag > 2   {report['strongest_lag']} weeks  (correlation {report['acf_at_period']:.2f})")

# A second series, purely for contrast: atmospheric CO2, weekly since 1980. It is
# not sales data and is not pretending to be — it is a measuring stick, so that
# the numbers above mean something. This is what a comfortable series looks like.
control = load_control_series()
control_report = tc.suitability_report(control)

print("\n  the same report on a well-conditioned series (weekly CO2, for scale)")
print(f"    cycles {control_report['cycles']:.1f} vs {report['cycles']:.2f}     "
      f"trend r2 {control_report['trend_r_squared']:.3f} vs {report['trend_r_squared']:.3f}")

print(f"""
  VERDICT
    + the time axis is clean: {report['n']} consecutive weeks, no gaps to repair
    + seasonality is overwhelming ({report['seasonal_strength']:.2f}) and lands exactly at lag 52,
      so "the same week last year" will be hard to beat
    - there is no trend worth modelling: {report['trend_pct_per_year']:+.2f}%/year at p={report['trend_p_value']:.2f} is a
      straight line through noise, and a model that extrapolates it will drift
    - {report['cycles']:.2f} cycles is thin. Seasonal models will fit, but every seasonal
      term is estimated from three observations, and one whole class of
      evaluation becomes impossible (section 12).""")

# %% 4. DECOMPOSITION
# ==============================================================================
# Splitting the series into the parts that made it. Additive here, meaning the
# three components sum back to the original:
#
#     observed = trend + seasonality + residual
#
# The alternative is multiplicative, where the seasonal swing scales with the
# level — the right choice for a business that doubles in size and whose
# Christmas peak doubles with it. This chain's level is flat, so the two forms
# barely differ.
#
# Note the trend panel is empty at both ends. A 52-week centred moving average
# cannot be computed for the first and last 26 weeks: there is no window. That is
# not a bug, it is the price of the method, and it is worth seeing.

parts = tc.decompose(sales, period=52, model="additive")

figure = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
    subplot_titles=("Observed", "Trend (52-week moving average)", "Seasonality", "Residual"),
)
for row, (column, colour) in enumerate(
    [("observed", BLUE), ("trend", ORANGE), ("seasonal", "#2ecc71"), ("residual", GREY)], start=1
):
    figure.add_trace(
        go.Scatter(x=parts["Date"], y=parts[column], mode="lines",
                   line=dict(color=colour, width=1.5), name=column),
        row=row, col=1,
    )
figure.update_layout(height=800, showlegend=False,
                     title="<b>Decomposition of weekly chain sales</b> (additive)")
show(figure)

seasonal_swing = parts["seasonal"].max() - parts["seasonal"].min()
trend_swing = parts["trend"].max() - parts["trend"].min()
print("\nDECOMPOSITION")
print("-" * 78)
print(f"  seasonal swing (peak to trough) {seasonal_swing:>14,.0f}")
print(f"  trend swing over 2.7 years      {trend_swing:>14,.0f}")
print(f"  residual std deviation          {parts['residual'].std():>14,.0f}")
print(f"\n  The season moves the business {seasonal_swing / trend_swing:.1f}x more than the trend does.")

# %% 5. THE TREND THAT ISN'T — AND WHERE IT WENT
# ==============================================================================
# Section 3 reported no trend, which is a strange thing for a retail chain over
# three years. Two pieces of domain knowledge explain it, and both are testable
# with columns already in this dataset.
#
# FIRST: money shrinks. Sales are counted in euros, and euros lose value. The CPI
# column rises across this window, so a flat *nominal* series can be a declining
# *real* one. Deflating rebases every week onto the purchasing power of the first
# week, which is the only way to ask whether the chain sold more goods rather
# than merely more euros.

cpi = tc.as_series(chain, value="CPI")
real_sales = deflate(sales, cpi)

nominal_trend = tc.trend_test(sales)
real_trend = tc.trend_test(real_sales)

print("\nNOMINAL VS REAL")
print("-" * 78)
print(f"  CPI over the window   {cpi.iloc[0]:.1f} -> {cpi.iloc[-1]:.1f}   "
      f"({100 * (cpi.iloc[-1] / cpi.iloc[0] - 1):+.1f}%)")
print(f"  nominal sales trend   {nominal_trend['pct_per_year']:+.2f}%/year   p={nominal_trend['p_value']:.2f}")
print(f"  real sales trend      {real_trend['pct_per_year']:+.2f}%/year   p={real_trend['p_value']:.2f}")
print("""
  Read this carefully, because it is easy to overclaim. The point estimate flips
  sign: flat in euros, shrinking in goods. But p = 0.16 means we cannot call the
  real decline statistically significant either — 143 weeks is not enough to
  separate a -1.7%/year drift from noise. The honest sentence is "nominal
  stability may be masking a real decline, and this data cannot settle it",
  which is a more useful thing to tell a board than a fabricated growth rate.""")

figure = go.Figure()
figure.add_trace(go.Scatter(x=sales.index, y=sales.values, mode="lines",
                            name="nominal", line=dict(color=BLUE, width=1.5)))
figure.add_trace(go.Scatter(x=real_sales.index, y=real_sales.values, mode="lines",
                            name="real (2010 prices)", line=dict(color=ORANGE, width=1.5)))
figure.update_layout(height=420, title="<b>The same sales, in euros and in goods</b>",
                     yaxis_title="weekly sales")
show(figure)

# SECOND: promotions. Any real upward movement in a saturated market with stable
# competition comes from campaigns and pricing decisions — internal knowledge, not
# a statistical property of the past. This dataset has a promotion column, and it
# comes with a trap that is far more instructive than the column itself.

markdowns = [c for c in dept_panel.columns if c.startswith("MarkDown")]
recorded = dept_panel.dropna(subset=markdowns, how="all")["Date"]
weeks_without = sales.index[sales.index < recorded.min()].size

print("\nPROMOTIONS")
print("-" * 78)
print(f"  promotion columns     {', '.join(markdowns)}")
print(f"  first week recorded   {recorded.min().date()}")
print(f"  weeks with no data    {weeks_without} of {len(sales)}  ({weeks_without / len(sales):.0%})")
print("""
  This is a structural break, not missing data. The columns are not absent
  because someone lost them; promotion tracking did not exist before that date.
  Imputing zeros would tell every model "we ran no campaigns for two years",
  which is false. The usable options are to model only the later period, or to
  treat "was promotion data recorded" as a feature in its own right — never to
  fill the gap and forget it happened.""")

# %% 6. STATIONARITY
# ==============================================================================
# A stationary series has a constant mean and variance: it wanders around a level
# rather than drifting away from it. It matters because ARIMA's middle letter is
# a fix for non-stationarity, and if a series is already stationary that fix
# costs accuracy for nothing.
#
# The Augmented Dickey-Fuller test's null hypothesis is "there IS a unit root",
# i.e. non-stationary. A small p-value rejects that.

show_source(tc.adf_report)

adf_level = tc.adf_report(sales)
adf_diff = tc.adf_report(sales.diff().dropna())

print("\nAUGMENTED DICKEY-FULLER")
print("-" * 78)
print(f"  level            statistic {adf_level['statistic']:>8.2f}   p={adf_level['p_value']:.2e}   "
      f"stationary={adf_level['stationary']}")
print(f"  first difference statistic {adf_diff['statistic']:>8.2f}   p={adf_diff['p_value']:.2e}   "
      f"stationary={adf_diff['stationary']}")
print("""
  Already stationary at the level, which follows directly from having no trend:
  there is nothing pulling the mean away. So differencing is optional here, and
  the SARIMA fitted in section 10 uses d=0 for the non-seasonal part.

  One caveat worth carrying: strong seasonality can make this test say
  "stationary" about a series nobody would call flat, because the seasonal swings
  keep dragging it back to the same level. The test answers a narrower question
  than most people read into it. Always look at the plot as well.""")

# %% 7. AUTOCORRELATION
# ==============================================================================
# How much does an observation resemble the ones before it?
#
#   ACF (autocorrelation)          correlation with lag k, everything included
#   PACF (partial autocorrelation) the same, with the intermediate lags removed
#
# The difference matters. If today depends on yesterday and yesterday on the day
# before, the ACF shows a correlation at lag 2 that is entirely inherited. The
# PACF strips that inheritance out and shows only what lag 2 adds directly, which
# is what tells you how many AR terms a model actually needs.

correlations = tc.autocorrelation(sales, nlags=60)

figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                       subplot_titles=("ACF — autocorrelation", "PACF — partial autocorrelation"))
confidence = 1.96 / np.sqrt(len(sales))
for row, column in enumerate(["acf", "pacf"], start=1):
    figure.add_trace(
        go.Bar(x=correlations["lag"], y=correlations[column], marker_color=BLUE, name=column),
        row=row, col=1,
    )
    for sign in (1, -1):
        figure.add_hline(y=sign * confidence, line=dict(color=RED, dash="dash", width=1),
                         row=row, col=1)
figure.update_layout(height=600, showlegend=False,
                     title="<b>Autocorrelation of weekly chain sales</b> (dashed = 95% significance)")
figure.update_xaxes(title_text="lag (weeks)", row=2, col=1)
show(figure)

print("\nAUTOCORRELATION")
print("-" * 78)
for lag in (1, 2, 4, 13, 26, 52):
    value = correlations.loc[correlations.lag == lag, "acf"].iloc[0]
    marker = "  <- one year" if lag == 52 else ""
    print(f"  lag {lag:>2}   acf {value:+.2f}{marker}")
print(f"""
  Lag 52 is the strongest correlation in the series ({report['acf_at_period']:+.2f}) — stronger than
  lag 1. Last year's same week tells you more about this week than last week
  does. That single fact is why the baseline in section 9 is so hard to beat,
  and why every model here is built around a 52-week season.""")

# %% 8. SPLITTING, AND THE FOUR ERROR MEASURES
# ==============================================================================
# A time series is split by *time*: the last weeks are held out, and the model
# never sees them. Shuffling would let the model learn from March in order to
# predict February — a score that measures interpolation on a task that is
# extrapolation.

show_source(tc.split_series)

train, test = tc.split_series(sales, test_weeks=tc.TEST_WEEKS)

print("\nTHE SPLIT")
print("-" * 78)
print(f"  train  {len(train):>3} weeks   {train.index[0].date()} to {train.index[-1].date()}")
print(f"  test   {len(test):>3} weeks   {test.index[0].date()} to {test.index[-1].date()}")

# Four ways to average the same errors. They disagree on purpose.
example = tc.seasonal_naive_forecast(train, len(test))
metrics = tc.forecast_metrics(test, example)

print("\nFOUR ERROR MEASURES, ONE FORECAST")
print("-" * 78)
print(f"  MAE   {metrics['MAE']:>18,.0f}   average miss, in euros")
print(f"  MSE   {metrics['MSE']:>18,.0f}   squared euros — no interpretable unit")
print(f"  RMSE  {metrics['RMSE']:>18,.0f}   back in euros, but big misses count more")
print(f"  MAPE  {metrics['MAPE']:>18.2f}   average miss as a percentage")
print(f"  R2    {metrics['R2']:>18.2f}   share of variance explained")
print("""
  Choosing between them is a business decision, not a technical one:

    MAE   treats a EUR 100k miss as exactly twice a EUR 50k miss. Use it when the
          cost of being wrong is proportional to how wrong you are.
    RMSE  squares first, so one catastrophic week outweighs several mediocre
          ones. Use it when a single large miss is what actually hurts — a
          stockout at Christmas, say.
    MAPE  is unitless, so series of different sizes become comparable. That is
          why it is the default in forecasting competitions. It has one serious
          failure mode: it divides by the actual value. forecasting.py shows it
          producing errors above 100% on the same data, at a finer grain.""")

# %% 9. BASELINES — THE BAR EVERY MODEL HAS TO CLEAR
# ==============================================================================
# Before any model, two forecasts that take no fitting at all. Reporting a model
# without comparing it to these is the most common way to present a result that
# means nothing.

show_source(tc.seasonal_naive_forecast)

results = {
    "naive (last value)": tc.naive_forecast(train, len(test)),
    "seasonal naive (t-52)": tc.seasonal_naive_forecast(train, len(test)),
    "mean of history": np.repeat(train.mean(), len(test)),
}

print("\nBASELINES")
print("-" * 78)
for name, prediction in results.items():
    print(f"  {name:<24} MAPE {tc.forecast_metrics(test, prediction)['MAPE']:>6.2f}%")
print("""
  The gap between the two naive forecasts is the value of knowing about
  seasonality, and nothing else. Same data, same zero parameters; one repeats
  last week, the other repeats last year.""")

# %% 10. EXPONENTIAL SMOOTHING
# ==============================================================================
# The first actual model, and the formula is one line:
#
#     forecast[t+1] = alpha * actual[t] + (1 - alpha) * forecast[t]
#
# Every forecast is a blend of what just happened and what you already believed.
# Unrolling it shows the weight on an observation k steps back is
# alpha * (1-alpha)^k — older data fades rather than disappearing. alpha sets how
# fast: high chases the latest point, low smooths through noise.
#
# Written out in full first, because it is short enough to read and believe.

show_source(tc.simple_exponential_smoothing)

print("\nSIMPLE EXPONENTIAL SMOOTHING")
print("-" * 78)
for alpha in (0.2, 0.4, 0.8):
    smoothed = tc.simple_exponential_smoothing(sales, alpha=alpha)
    error = tc.forecast_metrics(sales.iloc[1:], smoothed.iloc[1:])
    print(f"  alpha={alpha}   one-step-ahead MAE {error['MAE']:>12,.0f}   MAPE {error['MAPE']:.2f}%")

print("""
  Three very different values of alpha, and barely a tenth of a percent between
  them. That is the result, not a disappointing preliminary to one: no amount of
  tuning helps, because the thing this series is made of is seasonality and
  simple exponential smoothing has no seasonal term to put it in.

  It has no trend term either, so beyond one step it predicts a flat line. That
  is precisely why Holt added a trend equation and Winters added a seasonal one —
  each new term exists because the previous model could not represent something.""")

# Holt-Winters: the same smoothing idea, three times over — level, trend, season.
timer = time.perf_counter()
ets_model, ets_forecast = fm.fit_ets(train, len(test))
ets_seconds = time.perf_counter() - timer
results["Holt-Winters (ETS)"] = ets_forecast.values

print(f"\n  Holt-Winters   MAPE {tc.forecast_metrics(test, ets_forecast.values)['MAPE']:.2f}%"
      f"   fitted in {ets_seconds:.2f}s")
print("""
  statsmodels may warn about too few observations to estimate starting values.
  That warning is section 3's "2.75 cycles" arriving in person: 52 seasonal terms
  from 130 observations is thin, and the library is right to say so.""")

# %% 11. ARIMA / SARIMA
# ==============================================================================
# Three separate ideas behind one acronym, one parameter for each:
#
#     AR(p)   autoregression — today as a weighted sum of the last p values
#     I(d)    integration — difference d times to remove a trend
#     MA(q)   moving average — today also depends on the last q forecast errors
#
# The seasonal version repeats all three at the seasonal lag, written
# (P, D, Q, m). Here (0, 1, 1, 52): difference against the same week last year and
# carry one seasonal error term. The non-seasonal d stays 0 because section 6
# showed the series is already stationary.

show_source(fm.fit_sarima)

timer = time.perf_counter()
sarima_model, sarima_forecast = fm.fit_sarima(train, len(test))
sarima_seconds = time.perf_counter() - timer
results["SARIMA(1,0,1)(0,1,1,52)"] = sarima_forecast.values

print(f"  SARIMA   MAPE {tc.forecast_metrics(test, sarima_forecast.values)['MAPE']:.2f}%"
      f"   fitted in {sarima_seconds:.1f}s")
print(f"""
  Worth pausing on the cost. SARIMA took {sarima_seconds:.0f} seconds; Holt-Winters took
  {ets_seconds:.2f}. That is roughly {sarima_seconds / max(ets_seconds, 1e-9):.0f}x the compute, because 52 seasonal lags mean a
  state vector of length ~52 to be estimated at every step. Whether the result
  justifies it is a question the comparison below answers, and the answer is
  not obviously yes.""")

# %% 12. CROSS-VALIDATION, AND THE CHRISTMAS PROBLEM
# ==============================================================================
# One split is one sample. It can flatter a model by landing on an easy stretch,
# and the only defence is to score several splits.
#
# For a time series the folds must respect time. Rolling-origin (expanding
# window) cross-validation trains on everything up to a point and tests on what
# follows, then moves the point forward. Standard k-fold would hold out the
# middle of the series and train on both sides of the hole — the future leaking
# into the past, which reliably reports a smaller error than the model will ever
# achieve in production.

show_source(tc.rolling_origin_splits)

folds = tc.rolling_origin_splits(sales, n_folds=3, horizon=tc.TEST_WEEKS)

print("\nROLLING-ORIGIN CROSS-VALIDATION")
print("-" * 78)
print(f"  {'fold':<6}{'train':>7}{'test window':>28}{'seas.naive':>12}{'ETS':>9}")
cv_scores = {"seasonal naive": [], "Holt-Winters": []}
for i, (fold_train, fold_test) in enumerate(folds):
    baseline = tc.seasonal_naive_forecast(fold_train, len(fold_test))
    _, fold_forecast = fm.fit_ets(fold_train, len(fold_test))
    baseline_mape = tc.forecast_metrics(fold_test, baseline)["MAPE"]
    ets_mape = tc.forecast_metrics(fold_test, fold_forecast.values)["MAPE"]
    cv_scores["seasonal naive"].append(baseline_mape)
    cv_scores["Holt-Winters"].append(ets_mape)
    window = f"{fold_test.index[0].date()} to {fold_test.index[-1].date()}"
    print(f"  {i:<6}{len(fold_train):>7}{window:>28}{baseline_mape:>11.2f}%{ets_mape:>8.2f}%")

for name, scores in cv_scores.items():
    print(f"  {name:<22} mean {np.mean(scores):.2f}%   spread {min(scores):.2f}-{max(scores):.2f}%")

# Now the finding that governs how much any of this can be trusted.
week_of_year = sales.index.isocalendar().week.to_numpy()
christmas_weeks = sales.index[np.isin(week_of_year, [51, 52])]
earliest_testable = sales.index[104]  # ETS(52) needs two full cycles to fit

print(f"""
  THE CHRISTMAS PROBLEM
  ---------------------
  Every fold above tests a quiet stretch. None of them contains Christmas, and
  that is not a choice — it is forced:

    Christmas weeks in this data   {', '.join(str(d.date()) for d in christmas_weeks)}
    earliest week a 52-season model can be tested from   {earliest_testable.date()}

  A seasonal model needs two complete cycles before it can be fitted at all, and
  two cycles is already {104 / 52:.0f} of the {report['cycles']:.2f} available. By the time enough history
  exists to fit, every Christmas is already inside it. The hardest weeks of the
  retail year cannot be evaluated at all with this dataset.""")

# What we *can* measure: the baseline, which needs only 52 weeks of history, on a
# window that does contain Christmas. Take the latest Christmas that still leaves
# the baseline a full year of history to copy from.
candidates = [
    sales.index.get_loc(week) - 6
    for week in christmas_weeks
    if sales.index.get_loc(week) - 6 > 52
]
christmas_start = max(candidates) if candidates else 0
christmas_test = sales.iloc[christmas_start : christmas_start + 13]
christmas_train = sales.iloc[:christmas_start]
if len(christmas_train) > 52:
    christmas_baseline = tc.seasonal_naive_forecast(christmas_train, len(christmas_test))
    christmas_mape = tc.forecast_metrics(christmas_test, christmas_baseline)["MAPE"]
    quiet_mape = tc.forecast_metrics(test, tc.seasonal_naive_forecast(train, len(test)))["MAPE"]
    print(f"""  The seasonal-naive baseline only needs 52 weeks, so it can be scored there:

    quiet window     ({test.index[0].date()} to {test.index[-1].date()})   MAPE {quiet_mape:.2f}%
    Christmas window ({christmas_test.index[0].date()} to {christmas_test.index[-1].date()})   MAPE {christmas_mape:.2f}%

  The same method, {christmas_mape / quiet_mape:.1f}x worse on the weeks that matter most. Every headline
  number in this lesson comes from the quiet window and is flattered by it. Say
  so when you present one.""")

# %% 13. COMPARISON
# ==============================================================================

table = tc.metrics_table({name: tc.forecast_metrics(test, p) for name, p in results.items()})

print("\nALL MODELS, SAME 13-WEEK TEST WINDOW")
print("-" * 78)
print(table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

best = table.iloc[0]
baseline_mape = table.loc[table.model == "seasonal naive (t-52)", "MAPE"].iloc[0]
if best["model"] == "seasonal naive (t-52)":
    print(f"""
  The baseline won. Nothing fitted here beat "the same week last year" ({baseline_mape:.2f}%),
  which is a result rather than a failure: it says the series is close to purely
  seasonal, and that the honest recommendation is the forecast that costs nothing
  to build and nothing to maintain. Report it that way.""")
else:
    print(f"""
  {best['model']} wins at {best['MAPE']:.2f}% against the seasonal-naive baseline's {baseline_mape:.2f}%.
  A {100 * (1 - best['MAPE'] / baseline_mape):.0f}% reduction in error over a forecast that has no parameters at
  all — real, but worth keeping in proportion.""")

figure = go.Figure()
figure.add_trace(go.Scatter(x=train.index[-52:], y=train.values[-52:], mode="lines",
                            name="history", line=dict(color=GREY, width=1.5)))
figure.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines+markers",
                            name="actual", line=dict(color=BLUE, width=2.5)))
for name in ("seasonal naive (t-52)", "Holt-Winters (ETS)", "SARIMA(1,0,1)(0,1,1,52)"):
    figure.add_trace(go.Scatter(x=test.index, y=results[name], mode="lines",
                                name=name, line=dict(width=1.8, dash="dot")))
figure.add_vline(x=train.index[-1], line=dict(color=RED, dash="dash", width=1.5))
figure.update_layout(height=480, title="<b>Forecasts against the held-out quarter</b>",
                     yaxis_title="weekly sales")
show(figure)

# Residuals: what the best model still cannot explain.
residuals = test.values - results[best["model"]]
figure = go.Figure()
figure.add_trace(go.Bar(x=test.index, y=residuals, marker_color=BLUE))
figure.add_hline(y=0, line=dict(color=RED, width=1.5))
figure.update_layout(height=360, title=f"<b>Residuals — {best['model']}</b>",
                     yaxis_title="actual - forecast")
show(figure)

# %% 14. WHAT THE SLIDES CLAIM, AND WHAT THIS DATA SHOWS
# ==============================================================================

print("\nSLIDES VS DATA")
print("-" * 78)
print(f"""
  "Decomposition into trend, seasonality and residuals"
      Holds, with an asterisk. The decomposition works, but on this series the
      trend component is {trend_swing / seasonal_swing:.2f}x the size of the seasonal one and is not
      statistically distinguishable from noise. Three components does not mean
      three *useful* components.

  "Stationarity means constant mean and variance (= no trend)"
      Holds, and here it arrives by an unusual route: this series is stationary
      because it never had a trend, not because we differenced one away.

  "ARIMA ... often better than exponential smoothing"
      Not here. SARIMA scored {tc.forecast_metrics(test, results['SARIMA(1,0,1)(0,1,1,52)'])['MAPE']:.2f}% against Holt-Winters' {tc.forecast_metrics(test, results['Holt-Winters (ETS)'])['MAPE']:.2f}% —
      a difference far smaller than the spread between cross-validation folds,
      for {sarima_seconds / max(ets_seconds, 1e-9):.0f}x the compute. On a series this seasonal and this short,
      they are the same model wearing different clothes.

  "MAE, RMSE and MAPE are indispensable"
      Holds. But choosing between them is a business decision, and MAPE in
      particular has a failure mode this grain of data hides — see
      forecasting.py, section 12.

  The transferable habit: run the suitability report before the model, and
  report the baseline next to every result.""")

print("\n" + "=" * 78)
print(f"Done. Analysed {len(sales)} weeks of chain sales; best MAPE {best['MAPE']:.2f}% "
      f"({best['model']})  (data: {SOURCE})")
print("=" * 78)
print("\nNext lesson: forecasting.py — Prophet and XGBoost on the same series.")
