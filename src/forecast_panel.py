"""
forecast_panel.py
-----------------
The feature table and the panel evaluation used by ``forecasting.py`` — moved out
of the lesson script so the teaching app can import the very same code.

Everything here builds on ``ts_core`` (splits, metrics, calendar and lag
features) and ``forecast_models`` (the model wrappers). UI-free, returns data.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import forecast_models as fm
from . import ts_core as tc

# The columns a tree model gets. CALENDAR says *when*, LAGS carry the *level*.
CALENDAR = ["week", "month", "quarter", "week_sin", "week_cos", "t"]
LAGS = ["lag_1", "lag_52", "roll_4"]


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
    On 143 weekly points that is more than a third of the data, and section 8 of
    ``forecasting.py`` is entirely about what that costs.
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


def panel_folds(panel: pd.DataFrame, n_folds: int = 3, horizon: int = tc.TEST_WEEKS):
    """Expanding-window folds over a panel, cut on the date so all stores align."""
    dates = np.sort(panel["Date"].unique())
    for i in range(n_folds):
        end = len(dates) - (n_folds - 1 - i) * horizon
        cutoff, stop = dates[end - horizon], dates[end - 1]
        yield (panel[panel["Date"] < cutoff],
               panel[(panel["Date"] >= cutoff) & (panel["Date"] <= stop)])


def score_panel(
    panel: pd.DataFrame, feature_columns: list[str], hybrid: bool = False, n_folds: int = 3
) -> list[float]:
    """MAPE on each fold for one feature set, optionally via the detrending hybrid."""
    scores = []
    for fold_train, fold_test in panel_folds(panel, n_folds=n_folds):
        if hybrid:
            _, prediction = fm.fit_hybrid(fold_train, fold_test, feature_columns, group="Store")
        else:
            a, b, c, d = fm.time_ordered_validation_split(
                fold_train[feature_columns], fold_train["Weekly_Sales"]
            )
            prediction = fm.fit_xgboost(a, c, b, d).predict(fold_test[feature_columns])
        scores.append(tc.forecast_metrics(fold_test["Weekly_Sales"], prediction)["MAPE"])
    return scores


def ets_per_store(store_panel: pd.DataFrame, test_weeks: int = tc.TEST_WEEKS) -> dict:
    """The classical alternative to one booster: one Holt-Winters model per store.

    Returns ``{"mape", "seconds", "models"}`` — the accuracy *and* the operational
    cost, because "forecasting at scale" is an argument about the second.
    A store with too little history simply opts out.
    """
    start = time.perf_counter()
    errors = []
    for _store, group in store_panel.groupby("Store"):
        series = tc.as_series(group)
        store_train, store_test = tc.split_series(series, test_weeks=test_weeks)
        try:
            _, forecast = fm.fit_ets(store_train, len(store_test))
            errors.append(np.abs(store_test.to_numpy() - forecast.to_numpy()) / store_test.to_numpy())
        except Exception:  # noqa: BLE001
            continue
    return {
        "mape": 100 * float(np.mean(np.concatenate(errors))),
        "seconds": time.perf_counter() - start,
        "models": len(errors),
    }
