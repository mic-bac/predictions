"""
churn_data.py
-------------
Shared data access for the two churn lessons: ``propensity.py`` (classification)
and ``survival.py`` (time-to-event).

Both lessons start from the same Kaggle "Customer Churn" dataset, so loading and
cleaning live here once instead of being copy-pasted into each script.

Dataset: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

Columns
-------
CustomerID, Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay,
Subscription Type, Contract Length, Total Spend, Last Interaction, Churn

``Tenure`` (months with the company) is what makes this dataset usable for *both*
lessons: the propensity model predicts **whether** ``Churn`` happens, the survival
model predicts **when** — using ``Tenure`` as the observed time.
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Paths are anchored to this file, not to the working directory, so the scripts
# run the same from anywhere (`python propensity.py`, an IDE cell, a notebook).
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "churn"
TRAIN_CSV = DATA_DIR / "customer_churn_dataset-training-master.csv"
TEST_CSV = DATA_DIR / "customer_churn_dataset-testing-master.csv"


def load_churn(sample_size: int | None = None, random_state: int = 42) -> pd.DataFrame:
    """Load and clean the churn dataset.

    The Kaggle download ships as two files (a "training" and a "testing" master).
    They are two halves of one customer base, not a modelling split, so we
    concatenate them and make our own split later, inside each lesson.

    ``sample_size`` draws a **stratified** subsample (same churn rate as the full
    data). The full set is ~505,000 rows; hyperparameter searches over that take
    hours, which is not what a lesson needs. Pass ``None`` for everything.
    """
    df = pd.concat([pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)]).reset_index(drop=True)

    # The two files number their customers independently, so the IDs collide.
    # Renumber before anything else — a duplicated key is worse than no key.
    df["CustomerID"] = range(len(df))

    # Exactly one row of the Kaggle file is entirely empty; it is why every
    # numeric column reads as float64.
    df = df.dropna().reset_index(drop=True)
    df["Churn"] = df["Churn"].astype(int)

    if sample_size is not None and sample_size < len(df):
        df = (
            df.groupby("Churn", group_keys=False)
            .sample(frac=sample_size / len(df), random_state=random_state)
            .sample(frac=1, random_state=random_state)  # shuffle the strata back together
            .reset_index(drop=True)
        )

    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Split into features/target and label-encode the categorical columns.

    Returns ``(X, y, encoders)``. The encoders are kept so a prediction for a new
    customer can be encoded the same way the training data was — forgetting this
    is one of the classic ways a model silently breaks in production.

    ``CustomerID`` is dropped: it is a key, not a feature. Left in, a tree model
    will happily memorise it.
    """
    data = df.drop(columns=["CustomerID"]).copy()

    encoders = {}
    for col in data.select_dtypes(include=["object", "string"]).columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder

    return data.drop(columns="Churn"), data["Churn"], encoders
