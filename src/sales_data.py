"""
sales_data.py
-------------
Shared data access for the two forecasting lessons: ``timeseries.py`` (is this
series forecastable, and what do classical models make of it?) and
``forecasting.py`` (Prophet and XGBoost).

Both start from the same **weekly sales log** — one row per store, department and
week — because the grain you forecast at is a decision, not a given, and this
dataset lets us make that decision three times and watch the answer change.

Dataset: https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast
Download ``train.csv``, ``stores.csv`` and ``features.csv`` into ``data/sales/``.

Shape of the data
-----------------
421,570 store-department-weeks from 45 Walmart stores, 2010-02-05 to 2012-10-26 —
143 consecutive Fridays, a perfectly regular weekly grid with no gaps. That
regularity is worth noticing: half of real forecasting work is repairing a time
axis, and this one arrives clean.

    train.csv       Store, Dept, Date, Weekly_Sales, IsHoliday
    stores.csv      Store, Type (A/B/C), Size
    features.csv    Store, Date, Temperature, Fuel_Price, MarkDown1-5,
                    CPI, Unemployment, IsHoliday

``test.csv`` also ships with the download and is **never used here**: it has no
``Weekly_Sales`` column, because on Kaggle it is the hidden answer. You cannot
measure an error against values you do not have, so every split in these lessons
is carved out of ``train.csv``.

Two quirks that matter more than they look
------------------------------------------
``MarkDown1-5`` (promotion spend) is entirely missing before **2011-11-11**. That
is not noise to be imputed away — it is a *structural break*, a feature that did
not exist for the first 92 of 143 weeks. And ``CPI`` rises 5.3% across the window,
which is why ``deflate`` exists: a nominal series that looks flat can be a real
series that is shrinking.

Graceful degradation
--------------------
``data/`` is git-ignored, so a fresh clone has no CSV. ``load_sales`` then falls
back to a **synthetic** log with the same schema and the same measured dynamics —
flat chain total hiding stores that grow and shrink, seasonality that dwarfs the
trend, near-empty departments, the MarkDown break. Every lesson reports which
source produced its numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Paths are anchored to this file, not the working directory, so the lessons run
# the same from anywhere (`python timeseries.py`, an IDE cell, a notebook).
SALES_DIR = Path(__file__).resolve().parents[1] / "data" / "sales"
TRAIN_CSV = SALES_DIR / "train.csv"
STORES_CSV = SALES_DIR / "stores.csv"
FEATURES_CSV = SALES_DIR / "features.csv"

# The first week promotion spend was recorded. Before this date all five
# MarkDown columns are NaN for every store — see the module docstring.
MARKDOWN_START = pd.Timestamp("2011-11-11")

MARKDOWN_COLUMNS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

# The four trading events Walmart itself flags. Prophet takes them as a holidays
# frame; `lower_window`/`upper_window` widen each one into the days around it,
# because a Thanksgiving effect is not confined to Thanksgiving — shoppers move
# spend into the days before and the weekend after.
#
# Dates run to 2013 so a forecast can extend past the end of the sales data.
WALMART_HOLIDAYS = pd.DataFrame(
    {
        "holiday": np.repeat(
            ["Superbowl", "Labor Day", "Thanksgiving", "Christmas"], 4
        ),
        "ds": pd.to_datetime(
            [
                "2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08",  # Superbowl
                "2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06",  # Labor Day
                "2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29",  # Thanksgiving
                "2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27",  # Christmas
            ]
        ),
        "lower_window": -3,
        "upper_window": 2,
    }
)

# Columns the store-level panel carries. Kept as a constant because both lessons
# and the feature-set comparison in `forecasting.py` need to agree on it.
EXTERNAL_FEATURES = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]


# ---------------------------------------------------------------------------
# Loading (with a synthetic fallback so the lessons always run)
# ---------------------------------------------------------------------------


def load_sales(
    path: Path | str = SALES_DIR, random_state: int = 42
) -> tuple[pd.DataFrame, str]:
    """Load the three CSVs and join them into one long table; returns ``(frame, source)``.

    ``source`` is ``"walmart"`` or ``"synthetic"``, so each lesson can be honest
    on screen about which data produced its numbers.

    The join itself contains one trap worth naming. ``IsHoliday`` appears in
    *both* ``train.csv`` and ``features.csv``, so a naive merge yields
    ``IsHoliday_x`` and ``IsHoliday_y`` and every later reference to
    ``IsHoliday`` raises a ``KeyError``. Dropping the duplicate before merging is
    a one-line fix that is much easier to apply than to debug.
    """
    path = Path(path)
    train_csv, stores_csv, features_csv = (
        path / "train.csv",
        path / "stores.csv",
        path / "features.csv",
    )
    if not (train_csv.exists() and stores_csv.exists() and features_csv.exists()):
        return make_synthetic_sales(random_state=random_state), "synthetic"

    sales = pd.read_csv(train_csv, parse_dates=["Date"])
    stores = pd.read_csv(stores_csv)
    features = pd.read_csv(features_csv, parse_dates=["Date"])

    frame = sales.merge(stores, on="Store", how="left").merge(
        features.drop(columns=["IsHoliday"]), on=["Store", "Date"], how="left"
    )
    return frame.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True), "walmart"


def make_synthetic_sales(random_state: int = 42) -> pd.DataFrame:
    """Generate a sales log with the same schema and the same measured dynamics.

    The dynamics are the point. A fallback that emitted a smooth trending
    sinusoid would quietly rewrite the conclusions of both lessons — the whole
    argument of ``timeseries.py`` is that this chain has *no* usable trend and
    that its seasonality is everything. So the generator reproduces, by
    construction, the four properties the lessons actually lean on:

    * **A flat chain total hiding stores that move.** Each store gets its own
      annual growth rate; the rates are centred so that summing the chain
      cancels them out. That is what the real data does, and it is what makes
      "the aggregate is stable" a misleading sentence.
    * **Seasonality that dwarfs the trend**, with sharp Thanksgiving and
      Christmas spikes rather than a gentle wave.
    * **Departments of wildly different size**, including near-empty ones, so
      the section on MAPE breaking down has something to break on.
    * **The MarkDown structural break** at :data:`MARKDOWN_START`.

    Roughly 3,300 store-department series over 143 weeks, i.e. ~470,000 rows.
    """
    rng = np.random.default_rng(random_state)

    dates = pd.date_range("2010-02-05", periods=143, freq="W-FRI")
    n_weeks = len(dates)
    week_of_year = dates.isocalendar().week.to_numpy()
    t = np.arange(n_weeks)

    # --- stores -----------------------------------------------------------
    n_stores = 45
    store_ids = np.arange(1, n_stores + 1)
    types = np.array(["A"] * 22 + ["B"] * 17 + ["C"] * 6)
    rng.shuffle(types)
    size_by_type = {"A": 175_000, "B": 115_000, "C": 40_000}
    sizes = np.array(
        [int(rng.normal(size_by_type[ty], size_by_type[ty] * 0.15)) for ty in types]
    )
    base_level = sizes * rng.uniform(4.5, 6.5, n_stores)

    # Per-store annual growth, then re-centred so the chain total comes out flat.
    growth = rng.normal(0.0, 0.08, n_stores)
    growth[rng.choice(n_stores, 3, replace=False)] -= 0.12  # a few dying stores
    growth -= np.average(growth, weights=base_level)
    trend = 1.0 + np.outer(growth, t / 52.0)  # (store, week)

    # --- seasonality ------------------------------------------------------
    # A mild yearly wave plus the two spikes that dominate retail: the
    # Thanksgiving week and the fortnight before Christmas.
    season = 1.0 + 0.06 * np.sin(2 * np.pi * (week_of_year - 6) / 52.0)
    spike = np.zeros(n_weeks)
    spike[np.isin(week_of_year, [47])] += 0.22  # Thanksgiving
    spike[np.isin(week_of_year, [50])] += 0.28
    spike[np.isin(week_of_year, [51])] += 0.62  # the Christmas week
    spike[np.isin(week_of_year, [52])] += 0.15
    season = season + spike

    store_week = base_level[:, None] * trend * season[None, :]
    store_week *= rng.normal(1.0, 0.02, store_week.shape)

    # --- departments ------------------------------------------------------
    # Department shares are lognormal, so a handful of departments carry most of
    # the store and a long tail sits near zero. Those small ones are not padding:
    # they are what makes percentage errors explode at department grain.
    n_depts = 80
    dept_ids = np.arange(1, n_depts + 1)
    # sigma is high on purpose: in the real data the smallest departments turn
    # over single-digit euros a week while the largest turn over tens of
    # thousands. That four-order-of-magnitude spread is what breaks MAPE, so a
    # fallback with tidy same-sized departments would hide the lesson.
    dept_share = rng.lognormal(0.0, 2.0, n_depts)
    dept_share /= dept_share.sum()
    # Not every store carries every department.
    carries = rng.random((n_stores, n_depts)) < 0.82

    store_idx, dept_idx = np.nonzero(carries)
    n_series = len(store_idx)

    weights = dept_share[dept_idx] * rng.normal(1.0, 0.25, n_series).clip(0.2)
    values = store_week[store_idx][:, :] * weights[:, None]
    values *= rng.normal(1.0, 0.18, values.shape)  # weekly noise

    frame = pd.DataFrame(
        {
            "Store": np.repeat(store_ids[store_idx], n_weeks),
            "Dept": np.repeat(dept_ids[dept_idx], n_weeks),
            "Date": np.tile(dates.to_numpy(), n_series),
            "Weekly_Sales": values.ravel().round(2),
        }
    )

    # A small share of non-positive rows — returns and corrections — so
    # `clean_sales` still has something real to remove.
    negatives = rng.random(len(frame)) < 0.003
    frame.loc[negatives, "Weekly_Sales"] *= -0.4

    # --- store metadata and external features -----------------------------
    frame = frame.merge(
        pd.DataFrame({"Store": store_ids, "Type": types, "Size": sizes}),
        on="Store",
        how="left",
    )

    holiday_weeks = {6, 36, 47, 52}  # Superbowl, Labor Day, Thanksgiving, Christmas
    per_week = pd.DataFrame(
        {
            "Date": dates,
            "IsHoliday": np.isin(week_of_year, list(holiday_weeks)),
            # CPI climbs 5.3% across the window, as it does in the real data.
            "CPI": 167.7 * (1 + 0.053 * t / (n_weeks - 1)),
            "Unemployment": 8.3 - 1.1 * t / (n_weeks - 1) + rng.normal(0, 0.05, n_weeks),
            "Fuel_Price": 2.75 + 0.6 * t / (n_weeks - 1) + rng.normal(0, 0.08, n_weeks),
        }
    )
    frame = frame.merge(per_week, on="Date", how="left")

    # Temperature is per store (they are spread across the country) and seasonal.
    store_offset = rng.normal(0, 9, n_stores)
    temp = 60 + 22 * -np.cos(2 * np.pi * (week_of_year - 2) / 52.0)
    temp_map = pd.DataFrame(
        {
            "Store": np.repeat(store_ids, n_weeks),
            "Date": np.tile(dates.to_numpy(), n_stores),
            "Temperature": (temp[None, :] + store_offset[:, None]).ravel().round(2),
        }
    )
    frame = frame.merge(temp_map, on=["Store", "Date"], how="left")

    # The structural break: no promotion spend was recorded before this date.
    for i, column in enumerate(MARKDOWN_COLUMNS, start=1):
        values = rng.lognormal(6.5 + 0.3 * i, 1.1, len(frame)).round(2)
        frame[column] = np.where(frame["Date"] >= MARKDOWN_START, values, np.nan)

    return frame.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean_sales(frame: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop the rows that cannot be forecast, and say what was dropped.

    Only one rule is applied: **non-positive weekly sales go**. About 1,300 rows
    (0.3%) record a week where returns outweighed purchases for a department.
    They are real bookkeeping, but a negative "sales" figure breaks percentage
    errors and confuses every seasonal model, and at 0.3% they cannot move an
    aggregate. Removing them is a judgement call, so it is made once, here,
    where it can be seen — not scattered through two lesson scripts.

    Missing values are deliberately *not* imputed. ``MarkDown1-5`` is missing
    because promotion tracking did not exist yet (see :data:`MARKDOWN_START`),
    and filling a structural break with zeros invents a fact.
    """
    before = len(frame)
    cleaned = frame[frame["Weekly_Sales"] > 0].reset_index(drop=True)

    if verbose:
        dropped = before - len(cleaned)
        print(f"  rows in                 {before:>10,}")
        print(f"  non-positive sales      {dropped:>10,}  ({dropped / before:.1%})")
        print(f"  rows out                {len(cleaned):>10,}")
        print(
            f"  weeks {cleaned['Date'].nunique()}   stores {cleaned['Store'].nunique()}"
            f"   store-department series {cleaned.groupby(['Store', 'Dept']).ngroups:,}"
        )
        missing = cleaned[MARKDOWN_COLUMNS].isna().all(axis=1).sum()
        print(
            f"  no promotion data       {missing:>10,}  "
            f"({missing / len(cleaned):.0%} of rows, all before {MARKDOWN_START.date()})"
        )

    return cleaned


# ---------------------------------------------------------------------------
# Three grains
#
# The same sales, aggregated three ways. Which one you forecast changes which
# model wins, and that is one of the two or three things worth remembering from
# these lessons.
# ---------------------------------------------------------------------------


def chain_total(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per week: the whole chain's sales. 143 rows.

    The external features are averaged across stores, which is the honest
    aggregate for a rate (CPI, unemployment, fuel price) and a defensible one for
    temperature. ``IsHoliday`` is a property of the week, so it just carries over.
    """
    grouped = frame.groupby("Date", as_index=False).agg(
        Weekly_Sales=("Weekly_Sales", "sum"),
        IsHoliday=("IsHoliday", "max"),
        **{column: (column, "mean") for column in EXTERNAL_FEATURES},
    )
    return grouped.sort_values("Date").reset_index(drop=True)


def store_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per store and week: 45 parallel series, 6,435 rows.

    This is the grain where the store attributes stop being constants. On the
    chain total, ``Size`` and ``Type`` are single numbers and cannot explain
    anything; here they vary across 45 stores and a model can finally use them.
    """
    grouped = frame.groupby(["Store", "Date"], as_index=False).agg(
        Weekly_Sales=("Weekly_Sales", "sum"),
        IsHoliday=("IsHoliday", "max"),
        Type=("Type", "first"),
        Size=("Size", "first"),
        **{column: (column, "mean") for column in EXTERNAL_FEATURES},
    )
    return grouped.sort_values(["Store", "Date"]).reset_index(drop=True)


def dept_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per store, department and week: ~3,300 series, ~420,000 rows.

    The finest grain, the actual Kaggle task — and the one where a percentage
    error stops meaning anything, because a department that sells €40 in a week
    can be wrong by 300% while being wrong by €120. ``forecasting.py`` uses it to
    make exactly that point.
    """
    return frame.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)


def deflate(nominal: pd.Series, cpi: pd.Series) -> pd.Series:
    """Convert a nominal series into a real one, in first-period prices.

    Sales are counted in euros, and a euro in 2012 buys less than a euro in 2010.
    Dividing by the price index rebases everything onto the purchasing power of
    the first week, which is the only way to ask whether the chain sold *more
    goods* rather than merely *more euros*. On this dataset the two answers point
    in opposite directions, which is the entire point of the section that uses it.
    """
    return nominal / (cpi / cpi.iloc[0])


def prepare(
    path: Path | str = SALES_DIR, random_state: int = 42, verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """The one call each lesson makes: load, clean, and cut the three grains.

    Returns ``(chain, stores, depts, source)``.
    """
    frame, source = load_sales(path, random_state=random_state)
    if verbose:
        print(f"  source: {source}")
    cleaned = clean_sales(frame, verbose=verbose)
    return (
        chain_total(cleaned),
        store_panel(cleaned),
        dept_panel(cleaned),
        source,
    )
