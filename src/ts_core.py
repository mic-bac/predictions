"""
ts_core.py
----------
The time-series primitives shared by ``timeseries.py`` (which teaches them) and
``forecasting.py`` (which needs the same splits and the same error measures, so
that the numbers printed by the two lessons can be laid side by side).

They live here rather than inside ``timeseries.py`` for one reason: importing a
``# %%`` lesson script executes the whole lesson. ``forecasting.py`` wants the
*functions*, not another 400 lines of output. One definition, two lessons, no
drift — and both lessons print the real source of these functions as they run, so
nothing is hidden from the reader.

Everything here is UI-free and returns data. No printing, no plotting.

A note on the shape of the data
-------------------------------
The univariate helpers take a **pandas Series indexed by date**; the panel
helpers take a DataFrame with a ``Date`` column. Keeping those two shapes
straight is most of what goes wrong when people start writing forecasting code,
so the signatures say which is which rather than accepting both and guessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, pacf

# One quarter. Short enough that 143 weeks still leave a usable training set,
# long enough that a forecast has to survive more than a couple of steps.
TEST_WEEKS = 13

# Weekly data with a yearly cycle: 52 weeks to the season.
SEASONAL_PERIOD = 52


def as_series(
    frame: pd.DataFrame, value: str = "Weekly_Sales", date: str = "Date"
) -> pd.Series:
    """Turn a two-column frame into a date-indexed Series, sorted and named."""
    series = frame.set_index(date)[value].sort_index()
    series.index = pd.DatetimeIndex(series.index)
    return series


# ---------------------------------------------------------------------------
# Is this series even forecastable?
#
# The question almost nobody asks before fitting a model, and the one that
# decides whether any of the later numbers mean anything.
# ---------------------------------------------------------------------------


def trend_test(series: pd.Series) -> dict:
    """Fit a straight line through the series and report whether it is real.

    Returns the slope in units per period, the same slope as a percentage of the
    mean per year, the p-value, and r². The p-value is the part that matters: a
    slope of "+0.5% a year" with p=0.70 is not a small trend, it is **no trend**,
    and treating it as a small one leads directly to forecasts that drift.
    """
    y = np.asarray(series, dtype=float)
    fit = stats.linregress(np.arange(len(y)), y)
    return {
        "slope": fit.slope,
        "pct_per_year": 100 * fit.slope * SEASONAL_PERIOD / y.mean(),
        "p_value": fit.pvalue,
        "r_squared": fit.rvalue**2,
        "significant": bool(fit.pvalue < 0.05),
    }


def strength(series: pd.Series, period: int = SEASONAL_PERIOD) -> dict:
    """Measure how much of the series is seasonality and how much is trend.

    Both on a 0-1 scale, following the standard definition: how far the residual
    variance falls when the component is added back in. A seasonal strength near
    1 means the yearly pattern explains nearly everything and a model that gets
    the season right will look excellent regardless of what else it does.
    """
    parts = seasonal_decompose(series, model="additive", period=period)
    residual = parts.resid.dropna()

    def component(part: pd.Series) -> float:
        combined = part.reindex(residual.index) + residual
        return float(max(0.0, 1 - residual.var() / combined.var()))

    return {"seasonal": component(parts.seasonal), "trend": component(parts.trend)}


def adf_report(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test for stationarity.

    The null hypothesis is "this series has a unit root", i.e. it wanders and its
    mean is not constant. A small p-value rejects that and says the series is
    stationary — which for ARIMA means the ``d`` in (p,d,q) can stay at 0.

    Worth knowing: strong seasonality can make this test report "stationary" for
    a series that plainly is not flat, because the seasonal swings keep pulling
    it back to the same level. The test answers a narrower question than most
    people read into it.
    """
    statistic, p_value, lags, n_obs, critical, _ = adfuller(
        np.asarray(series, dtype=float), autolag="AIC"
    )
    return {
        "statistic": statistic,
        "p_value": p_value,
        "lags_used": lags,
        "n_obs": n_obs,
        "critical_values": critical,
        "stationary": bool(p_value < 0.05),
    }


def autocorrelation(
    series: pd.Series, nlags: int = 60
) -> pd.DataFrame:
    """ACF and PACF side by side, as a tidy frame of ``lag, acf, pacf``.

    The ACF asks "how much does today resemble the value k steps ago?" including
    everything in between; the PACF strips the in-between out. On weekly retail
    data the honest expectation is a modest lag-1 correlation and a large spike
    at lag 52 — last year's same week is the single best predictor available.
    """
    values = np.asarray(series, dtype=float)
    # PACF is only defined for lags below half the sample.
    pacf_lags = min(nlags, len(values) // 2 - 1)
    acf_values = acf(values, nlags=nlags)
    pacf_values = np.full(nlags + 1, np.nan)
    pacf_values[: pacf_lags + 1] = pacf(values, nlags=pacf_lags)
    return pd.DataFrame(
        {"lag": np.arange(nlags + 1), "acf": acf_values, "pacf": pacf_values}
    )


def suitability_report(
    series: pd.Series, period: int = SEASONAL_PERIOD
) -> dict:
    """Everything worth knowing before choosing a model. Returns a flat dict.

    This is the function the whole first lesson hangs on. It answers, in order:
    is the time axis regular, is there enough history for the seasonal cycle we
    intend to model, is there a trend, how much of the series is season, and is
    it stationary.

    ``cycles`` is the one people skip. A yearly seasonal model estimates 52
    separate seasonal terms; with under two full cycles it cannot be fitted at
    all, and with under three each term rests on two or three observations. That
    single number caps how much any seasonal model here can be trusted.
    """
    gaps = series.index.to_series().diff().dropna().dt.days
    trend = trend_test(series)
    strengths = strength(series, period=period)
    adf = adf_report(series)
    correlations = autocorrelation(series, nlags=min(period + 8, len(series) - 2))

    return {
        "n": len(series),
        "start": series.index[0],
        "end": series.index[-1],
        "regular_grid": bool(gaps.nunique() == 1),
        "step_days": int(gaps.mode().iloc[0]) if len(gaps) else 0,
        "cycles": len(series) / period,
        "trend_pct_per_year": trend["pct_per_year"],
        "trend_p_value": trend["p_value"],
        "trend_r_squared": trend["r_squared"],
        "trend_significant": trend["significant"],
        "seasonal_strength": strengths["seasonal"],
        "trend_strength": strengths["trend"],
        "adf_p_value": adf["p_value"],
        "stationary": adf["stationary"],
        "acf_at_period": float(correlations.loc[correlations.lag == period, "acf"].iloc[0]),
        "strongest_lag": int(correlations.loc[correlations.lag > 2, "acf"].idxmax()),
    }


def decompose(
    series: pd.Series, period: int = SEASONAL_PERIOD, model: str = "additive"
) -> pd.DataFrame:
    """Split a series into trend, seasonality and residual. Returns a tidy frame.

    Additive by default, meaning the parts sum back to the original. The
    alternative, multiplicative, assumes the seasonal swing grows with the level
    — the right choice when a business doubles in size and its Christmas peak
    doubles with it. Here the level is flat, so the two barely differ and the
    additive form is easier to read off a chart.
    """
    parts = seasonal_decompose(series, model=model, period=period)
    return pd.DataFrame(
        {
            "Date": series.index,
            "observed": parts.observed.to_numpy(),
            "trend": parts.trend.to_numpy(),
            "seasonal": parts.seasonal.to_numpy(),
            "residual": parts.resid.to_numpy(),
        }
    )


# ---------------------------------------------------------------------------
# Splitting — the one place a time series is unlike every other dataset
# ---------------------------------------------------------------------------


def split_series(
    series: pd.Series, test_weeks: int = TEST_WEEKS
) -> tuple[pd.Series, pd.Series]:
    """Hold out the **last** ``test_weeks`` observations. Never shuffle.

    A random split would let the model learn from March 2012 in order to predict
    February 2012. The resulting score is not optimistic, it is meaningless: it
    measures interpolation on a task that is extrapolation. Everything about
    evaluating a forecast follows from respecting the arrow of time.
    """
    return series.iloc[:-test_weeks], series.iloc[-test_weeks:]


def split_panel(
    frame: pd.DataFrame, test_weeks: int = TEST_WEEKS, date: str = "Date"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same idea for a panel: split on the date, so every series is cut at once.

    Splitting each series independently would put different stores' futures at
    different calendar dates, and a model trained on one store's future could
    then help predict another's past.
    """
    cutoff = np.sort(frame[date].unique())[-test_weeks]
    return frame[frame[date] < cutoff].copy(), frame[frame[date] >= cutoff].copy()


def rolling_origin_splits(
    series: pd.Series, n_folds: int = 3, horizon: int = TEST_WEEKS
) -> list[tuple[pd.Series, pd.Series]]:
    """Expanding-window cross-validation: the time-series answer to k-fold.

    Fold *i* trains on everything up to a point and tests on the ``horizon``
    weeks that follow; the next fold moves that point forward. The training
    window only ever grows, so no fold is ever asked to predict a week that an
    earlier fold trained on.

    Why not plain k-fold: holding out the middle of a series leaves the model
    training on both sides of the hole, which is the future leaking into the
    past. It reliably reports a smaller error than the model will ever achieve
    in production, which is the worst possible direction for an error to be
    wrong in.
    """
    folds = []
    for i in range(n_folds):
        end = len(series) - (n_folds - 1 - i) * horizon
        folds.append((series.iloc[: end - horizon], series.iloc[end - horizon : end]))
    return folds


# ---------------------------------------------------------------------------
# Error measures
# ---------------------------------------------------------------------------


def forecast_metrics(y_true, y_pred) -> dict:
    """MAE, MSE, RMSE, MAPE and R² for one forecast.

    All four of the deck's error measures, computed together because the
    interesting thing about them is how they disagree:

    * **MAE** — average euro miss. Reads directly in the unit of the business.
    * **MSE / RMSE** — square the errors first, so one catastrophic week counts
      for more than five mediocre ones. RMSE is back in euros; MSE is not in any
      unit anyone can interpret, and exists mainly as the thing RMSE is the root
      of.
    * **MAPE** — average miss as a percentage, which makes series of different
      sizes comparable. It has one serious failure mode: it divides by the
      actual value, so a series that passes near zero produces percentage errors
      in the hundreds. Values ``<= 0`` are excluded here rather than silently
      producing infinities, and ``forecasting.py`` shows what that failure looks
      like at department grain.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    mse = float(np.mean(errors**2))
    positive = y_true > 0
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE": float(100 * np.mean(np.abs(errors[positive] / y_true[positive]))),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """Stack ``{name: metrics}`` into one frame, best MAPE first."""
    table = pd.DataFrame(results).T
    table.index.name = "model"
    return table.sort_values("MAPE").reset_index()


# ---------------------------------------------------------------------------
# Baselines — the bar every model has to clear
# ---------------------------------------------------------------------------


def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Repeat the last observed value. The cheapest forecast that exists.

    On a strongly seasonal series it is also a terrible one, and that is why it
    is here: it sets the floor. A model that cannot beat this has learned
    nothing at all.
    """
    return np.repeat(float(train.iloc[-1]), horizon)


def seasonal_naive_forecast(
    train: pd.Series, horizon: int, period: int = SEASONAL_PERIOD
) -> np.ndarray:
    """Predict each week with the same week one year ago. The bar that matters.

    This is the baseline that embarrasses people. It has no parameters, takes no
    time to fit, and on seasonal retail data it is genuinely hard to beat —
    because "the same week last year" already contains the season, the holiday
    calendar and the store's rough size. Any model reported without a comparison
    to this number should be treated as unevaluated.

    Only ``train`` is read, so there is no leakage as long as
    ``horizon <= period``.
    """
    if horizon > period:
        raise ValueError(
            f"horizon {horizon} exceeds one season ({period}); the forecast would "
            "need values it has not been given"
        )
    start = len(train) - period
    return np.asarray(train.iloc[start : start + horizon], dtype=float)


def simple_exponential_smoothing(series: pd.Series, alpha: float = 0.4) -> pd.Series:
    """One-step-ahead forecasts by exponential smoothing, written out in full.

    Six lines, no library, because the formula is the entire idea and it is
    easier to believe once you have seen it run::

        forecast[t+1] = alpha * actual[t] + (1 - alpha) * forecast[t]

    Every forecast is a blend of what just happened and what you previously
    believed. Unrolling the recursion shows that the weight on an observation
    ``k`` steps back is ``alpha * (1 - alpha)**k`` — older data never disappears,
    it fades. ``alpha`` sets how fast: high values chase the last point, low
    values smooth through noise.

    Note what it cannot do. There is no trend term and no seasonal term, so the
    forecast beyond one step is a flat line. That is not a bug to be fixed with a
    better ``alpha`` — it is the reason Holt and then Winters added the terms
    that bear their names.
    """
    values = np.asarray(series, dtype=float)
    smoothed = np.empty(len(values))
    smoothed[0] = values[0]
    for t in range(1, len(values)):
        smoothed[t] = alpha * values[t - 1] + (1 - alpha) * smoothed[t - 1]
    return pd.Series(smoothed, index=series.index, name=f"ses_alpha_{alpha}")


# ---------------------------------------------------------------------------
# Features for the models that need a table rather than a series
# ---------------------------------------------------------------------------


def calendar_features(dates: pd.DatetimeIndex | pd.Series) -> pd.DataFrame:
    """Turn a date column into the numbers a tree can split on.

    Trees cannot read a timestamp. They can, however, split on "week 51", which
    is how a gradient booster ends up representing Christmas.

    The sine and cosine pair is the part worth explaining. Week 52 and week 1 are
    one week apart in reality and 51 apart as integers, so a model given only the
    week number sees the new year as an enormous jump. Projecting the week onto a
    circle removes that seam: ``sin`` and ``cos`` together identify the position
    in the year uniquely, and adjacent weeks stay adjacent.
    """
    index = pd.DatetimeIndex(dates)
    week = index.isocalendar().week.to_numpy().astype(int)
    return pd.DataFrame(
        {
            "year": index.year,
            "month": index.month,
            "week": week,
            "quarter": index.quarter,
            "week_sin": np.sin(2 * np.pi * week / SEASONAL_PERIOD),
            "week_cos": np.cos(2 * np.pi * week / SEASONAL_PERIOD),
        },
        index=index,
    )


def lag_features(
    frame: pd.DataFrame,
    value: str = "Weekly_Sales",
    by: str | list[str] | None = None,
    lags: tuple[int, ...] = (1, 2, SEASONAL_PERIOD),
    windows: tuple[int, ...] = (4, 13),
) -> pd.DataFrame:
    """Add lagged and rolling-mean columns — the features that carry the level.

    A lag is simply the target, shifted. ``lag_52`` hands the model last year's
    same week, which on this data is most of the answer; the rolling means smooth
    recent weeks into a "where are we lately" summary.

    Every rolling window is computed on ``shift(1)`` first, so the mean for week
    *t* never includes week *t* itself. Forgetting that shift is the most common
    way to leak the target into its own features, and it produces spectacular
    training scores followed by a model that fails the moment it is asked to
    predict a week nobody has observed yet.

    Note the cost, which the lessons make a point of: a lag of 52 destroys the
    first 52 rows of every series. On 143 weekly points that is more than a third
    of the data gone before a model sees anything.
    """
    out = frame.copy()
    grouped = out.groupby(by)[value] if by is not None else out[value]

    for lag in lags:
        out[f"lag_{lag}"] = grouped.shift(lag)
    for window in windows:
        shifted = grouped.shift(1)
        if by is not None:
            shifted = shifted.groupby([out[column] for column in np.atleast_1d(by)])
            out[f"roll_{window}"] = shifted.transform(
                lambda s, w=window: s.rolling(w).mean()
            )
        else:
            out[f"roll_{window}"] = shifted.rolling(window).mean()

    return out
