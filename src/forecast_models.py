"""
forecast_models.py
------------------
Thin, uniform wrappers around the six model families the lessons compare:
exponential smoothing, SARIMA, Prophet, XGBoost, Random Forest, and the
regression-plus-booster hybrid.

Each wrapper does three things and no more: set the arguments that are decisions
worth teaching, avoid the traps that make time-series code silently wrong, and
return ``(fitted_model, forecast)`` so the lesson scripts read as a comparison
rather than as six pages of boilerplate.

Everything here is UI-free and returns data. No printing, no plotting.

The traps, since they are the reason several of these wrappers exist
--------------------------------------------------------------------
* **Prophet on weekly data.** Weekly and daily seasonality cannot be estimated
  from weekly observations — there is one point per week, so there is no
  within-week pattern to find. Asking for them anyway produces components fitted
  to noise. Both are off here.
* **Prophet regressors it cannot extrapolate.** ``Year`` as a regressor works
  beautifully in-sample and then has to predict 2013 from a coefficient fitted on
  2010-2012. Calendar integers are excluded; genuine external series are not.
* **The confidence interval.** ``yhat_lower`` and ``yhat_upper`` are different
  columns. Passing one where the other belongs collapses the interval to a line
  and hides exactly the uncertainty the interval exists to show.
* **Early stopping that does not stop early.** Supplying ``eval_set`` alone does
  nothing; ``early_stopping_rounds`` is what makes the booster watch the
  validation curve and stop when it turns.
* **The validation split.** For a time series it must be the *last* slice of the
  training window, never a random sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from .ts_core import SEASONAL_PERIOD

RANDOM_STATE = 42

# One shared booster configuration, so that every XGBoost number in the lessons
# differs because of the *data* it was given and not because of a stray
# hyper-parameter. Shallow trees and a low learning rate suit a few thousand
# rows; `n_estimators` is deliberately generous because early stopping decides
# where to actually stop.
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)


def future_index(train: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """The ``horizon`` dates that follow a series, on its own weekly grid."""
    step = train.index[-1] - train.index[-2]
    return pd.DatetimeIndex([train.index[-1] + step * (i + 1) for i in range(horizon)])


# ---------------------------------------------------------------------------
# Classical models
# ---------------------------------------------------------------------------


def fit_ets(
    train: pd.Series,
    horizon: int,
    seasonal_periods: int = SEASONAL_PERIOD,
    trend: str | None = "add",
    seasonal: str | None = "add",
) -> tuple[object, pd.Series]:
    """Holt-Winters exponential smoothing. Returns ``(fitted, forecast)``.

    Three smoothing equations running at once — one for the level, one for the
    trend, one for the seasonal figure — each updating the same way simple
    exponential smoothing does: a weighted blend of what just happened and what
    was previously believed.

    ``seasonal_periods=52`` means 52 seasonal terms are estimated, so the fit
    needs at least two complete cycles to exist and considerably more to be
    stable. With 130 weekly observations each of those terms rests on two or
    three numbers. It still produces the best forecast in these lessons, which
    says more about how dominant the seasonality is than about how well-estimated
    the model is.
    """
    fitted = ExponentialSmoothing(
        train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    ).fit()
    forecast = pd.Series(
        np.asarray(fitted.forecast(horizon), dtype=float),
        index=future_index(train, horizon),
        name="ets",
    )
    return fitted, forecast


def fit_sarima(
    train: pd.Series,
    horizon: int,
    order: tuple[int, int, int] = (1, 0, 1),
    seasonal_order: tuple[int, int, int, int] = (0, 1, 1, SEASONAL_PERIOD),
) -> tuple[object, pd.Series]:
    """Seasonal ARIMA. Returns ``(fitted, forecast)``.

    The acronym is three separate ideas bolted together, and the ``(p, d, q)``
    notation names one parameter for each:

        AR(p)  autoregression — today as a weighted sum of the last p values
        I(d)   integration — difference the series d times to remove a trend
        MA(q)  moving average — today also depends on the last q *errors*

    The seasonal half repeats all three at a lag of ``m``, so
    ``seasonal_order=(0, 1, 1, 52)`` means: difference against the same week last
    year, and carry one seasonal error term.

    A warning about cost. Fitting 52 seasonal lags means the state-space
    representation carries a state vector of length ~52, and this single call
    takes roughly 25 seconds — a hundred times slower than Holt-Winters on the
    same data, for a slightly worse forecast. That trade is worth seeing once.
    """
    fitted = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=0)
    forecast = pd.Series(
        np.asarray(fitted.forecast(horizon), dtype=float),
        index=future_index(train, horizon),
        name="sarima",
    )
    return fitted, forecast


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------


def fit_prophet(
    train: pd.Series,
    horizon: int,
    holidays: pd.DataFrame | None = None,
    changepoint_prior_scale: float = 0.05,
    yearly_seasonality: bool | int = True,
    country_holidays: str | None = None,
    interval_width: float = 0.95,
) -> tuple[Prophet, pd.DataFrame]:
    """Fit Prophet and predict history plus ``horizon`` future weeks.

    Returns ``(model, forecast)`` where ``forecast`` spans the training window
    *and* the future, because Prophet's value is not only the prediction — it is
    the decomposition, and the components only make sense across history.

    Prophet is an additive model in the literal sense::

        y(t) = trend(t) + seasonality(t) + holidays(t) + noise

    Each term can be pulled out and plotted, which is why a business audience can
    be shown *why* a number moved rather than only that it did.

    ``changepoint_prior_scale`` is the one knob worth understanding. Prophet
    models the trend as piecewise-linear and places candidate changepoints
    through the history; this value sets how freely the slope may change at them.
    Raise it and the trend bends to follow every wobble, which looks like a
    superb fit and then extrapolates whatever slope it happened to end on —
    the overshooting failure. Lower it and the trend is nearly straight.

    Weekly and daily seasonality are off: with one observation per week there is
    no within-week signal to estimate.
    """
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=holidays,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=interval_width,
    )
    if country_holidays:
        model.add_country_holidays(country_name=country_holidays)

    model.fit(pd.DataFrame({"ds": train.index, "y": np.asarray(train, dtype=float)}))
    future = model.make_future_dataframe(periods=horizon, freq="W-FRI")
    return model, model.predict(future)


def prophet_forecast_only(forecast: pd.DataFrame, dates) -> pd.DataFrame:
    """The rows of a Prophet forecast covering ``dates``, with the interval intact.

    ``yhat_lower`` and ``yhat_upper`` are carried through as separate columns —
    see the module docstring for why that is worth a function.
    """
    wanted = pd.DatetimeIndex(dates)
    subset = forecast[forecast["ds"].isin(wanted)]
    return subset[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)


def prophet_components(model: Prophet, forecast: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pull Prophet's additive components out as plain frames, ready to plot.

    Prophet ships its own ``plot_components``, which draws matplotlib figures.
    These lessons are Plotly throughout, and more importantly the components are
    *data* — extracting them keeps the plotting in the lesson where a reader can
    see it, and lets the trend be drawn with its changepoints overlaid.
    """
    parts: dict[str, pd.DataFrame] = {
        "trend": forecast[["ds", "trend", "trend_lower", "trend_upper"]].copy()
    }
    for name in ("yearly", "holidays", "extra_regressors_additive"):
        if name in forecast.columns:
            parts[name] = forecast[["ds", name]].copy()

    parts["changepoints"] = pd.DataFrame(
        {
            "ds": model.changepoints,
            # One delta per changepoint: how much the slope moved there.
            "delta": np.asarray(model.params["delta"]).mean(axis=0),
        }
    )
    return parts


# ---------------------------------------------------------------------------
# Tree ensembles
# ---------------------------------------------------------------------------


def time_ordered_validation_split(
    X: pd.DataFrame, y: pd.Series, val_fraction: float = 0.2
):
    """Cut the **last** slice off a training set for early stopping to watch.

    ``sklearn.train_test_split(..., shuffle=True)`` is the wrong tool here and is
    the single most common bug in time-series notebooks. It hands the booster
    validation rows drawn from the middle of the training window, so the
    validation error it monitors measures interpolation between known weeks. That
    error keeps falling long after the model has stopped being able to predict a
    genuinely unseen future, and early stopping consequently stops too late.
    """
    cut = int(len(X) * (1 - val_fraction))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    early_stopping_rounds: int | None = 50,
    **params,
) -> XGBRegressor:
    """Fit a gradient-boosted tree ensemble, with early stopping when possible.

    Boosting builds trees in sequence, each one fitted to the errors the previous
    trees have not explained yet. That is powerful and it is also why it
    overfits: given enough rounds it will eventually memorise the training set.

    Early stopping is the defence. With a validation set and
    ``early_stopping_rounds``, training halts once the validation error has
    failed to improve for that many consecutive rounds, and the best iteration is
    kept. Passing ``eval_set`` *without* ``early_stopping_rounds`` — which is
    what a surprising amount of published code does — merely prints a curve while
    training runs to completion regardless.
    """
    settings = {**XGB_PARAMS, **params}
    if X_val is not None and early_stopping_rounds:
        settings["early_stopping_rounds"] = early_stopping_rounds

    model = XGBRegressor(**settings)
    if X_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)
    return model


def fit_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 300, **params
) -> RandomForestRegressor:
    """Fit a random forest — the other way to combine many trees.

    A forest grows its trees *independently*, each on a bootstrap sample and a
    random subset of features, then averages them. Boosting grows them in
    sequence, each correcting the last. Averaging independent trees mainly
    reduces variance and is hard to overfit; boosting reduces bias too and needs
    early stopping to stay honest.

    It shares the boosters' hard limit: a forest also predicts by averaging
    training values, so it cannot extrapolate beyond the range it has seen.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# The hybrid: a straight line for the level, a booster for everything else
# ---------------------------------------------------------------------------


def fit_hybrid(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    target: str = "Weekly_Sales",
    time_column: str = "t",
    group: str | None = None,
    **params,
) -> tuple[dict, np.ndarray]:
    """Linear trend plus a booster on the residual. Returns ``(trends, predictions)``.

    The reasoning is sound and worth knowing. A tree cannot predict a value
    outside the range it was trained on, so on a growing series every forecast is
    capped at the highest level in the training data. Fitting a straight line
    first, subtracting it, and letting the booster model only what remains gives
    the tree a target with the growth taken out — something it *can* represent —
    while the line supplies the extrapolation the tree cannot.

    One line is fitted per ``group`` (per store, say), because a chain's stores
    do not share a growth rate.

    When this helps and when it does not is the whole point of the section that
    uses it. Two conditions have to hold: the series must actually trend, and the
    trend must be roughly linear. Where the features already include a lag of the
    target, the lag is *already* carrying the level and there is nothing left for
    the line to add — at which point it contributes only its own extrapolation
    error, and the hybrid does worse than the plain model.
    """
    train, test = train.copy(), test.copy()
    keys = [(None, train.index)] if group is None else list(train.groupby(group).groups.items())

    trends: dict = {}
    train["_trend"] = np.nan
    for key, index in keys:
        block = train.loc[index]
        line = LinearRegression().fit(block[[time_column]], block[target])
        trends[key] = line
        train.loc[index, "_trend"] = line.predict(block[[time_column]])

    if group is None:
        test["_trend"] = trends[None].predict(test[[time_column]])
    else:
        test["_trend"] = [
            trends[key].predict(np.array([[value]]))[0]
            for key, value in zip(test[group], test[time_column])
        ]

    booster = XGBRegressor(**{**XGB_PARAMS, **params})
    booster.fit(train[feature_columns], train[target] - train["_trend"])

    return trends, np.asarray(test["_trend"]) + booster.predict(test[feature_columns])
