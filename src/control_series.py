"""
control_series.py
-----------------
One extra time series, used by ``forecasting.py`` for a single purpose: to show
what a method looks like when it *works*.

Why a second series at all
--------------------------
The Walmart chain total has no trend. Not a small one — none: the fitted slope is
+0.5% a year with p=0.70 and r²=0.001, which is a straight line through noise.
That is a perfectly good property for a series to have, and most of what the
forecasting lessons teach works fine on it.

But it makes one important lesson impossible to demonstrate. Tree-based models
cannot extrapolate: a decision tree predicts by averaging training observations,
so it can never output a value above the highest level it was trained on. The
standard fix is to model the trend separately and let the tree learn only what is
left. On a series with no trend, that fix has nothing to remove, and measuring it
on Walmart shows it doing slight harm — which teaches the opposite of the truth.

So the lesson pairs the two: the fix measured on Walmart, where it hurts and the
reason is instructive, and the same code on a series that genuinely trends, where
it is several times more accurate.

Why this series
---------------
``statsmodels`` ships the Mauna Loa atmospheric CO₂ record, weekly since 1958.
It is used here for three reasons and no others:

1. It is **real data**, so nothing about the result can have been tuned by us —
   which matters when the point of the section is a measured comparison.
2. It **ships offline** with a package the project already depends on. No
   download, no Kaggle account, no new dependency.
3. Its trend is overwhelming (r²=0.948 against Walmart's 0.001) and its yearly
   seasonality is textbook-clean, so the contrast is unmistakable.

It is not sales data and it is not pretending to be. It is a measuring stick.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

# 1980 onwards: about 1,150 weekly points. Long enough for 22 seasonal cycles
# (against Walmart's 2.75), short enough that every fit in the lesson stays fast.
DEFAULT_START = "1980"


def load_control_series(start: str = DEFAULT_START) -> pd.Series:
    """Weekly atmospheric CO₂ in ppm, date-indexed and gap-free.

    The raw record has scattered missing weeks — the instrument was not always
    running. They are linearly interpolated, which for a series this smooth is
    uncontroversial and keeps the weekly grid regular, as every seasonal model
    here assumes.
    """
    frame = sm.datasets.co2.load_pandas().data
    series = frame["co2"].interpolate().dropna()
    series.index = pd.DatetimeIndex(series.index)
    return series.loc[start:].rename("co2")
