from typing import Tuple, List
import pandas as pd
import numpy as np

from .base import Dataset, DATA_ROOT

# Local copy of the UCI Bike Sharing dataset (id=275), hourly data.
# Loaded offline from the repo's data/ directory instead of ucimlrepo.
CSV_PATH = DATA_ROOT / "Bike-Sharing-Dataset" / "hour.csv"

# Feature metadata mirroring the UCI repository's variables table (id=275).
ALL_FEATURES = [
    "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
]
NUMERICAL_FEATURES = ["temp", "atemp", "hum", "windspeed"]
CATEGORICAL_FEATURES = [
    "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit",
]


def _load_hourly() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def prepare_bike_sharing():
    df = _load_hourly()
    X_df, y_ser = preprocess_bike_sharing(df=df)
    return X_df, y_ser


def preprocess_bike_sharing(df: pd.DataFrame):
    df = df.drop(["instant", "dteday", "casual", "registered", "atemp"], axis=1)

    # Standarize X
    X_df = df.drop(["cnt"], axis=1)
    X_df = (X_df - X_df.mean()) / X_df.std()

    # Standarize Y
    y_df = df["cnt"]
    y_df = (y_df - y_df.mean()) / y_df.std()

    return X_df, y_df


class BikeSharing(Dataset):
    def fetch(self):
        df = _load_hourly()

        self.X = df[ALL_FEATURES].copy()
        self.y = df["cnt"]

        self.all_features = list(ALL_FEATURES)
        self.numerical_features = list(NUMERICAL_FEATURES)
        self.categorical_features = list(CATEGORICAL_FEATURES)

        return self

    def preprocess(self):
        feats_to_drop = ["dteday", "atemp"]
        self.X = self.X.drop(feats_to_drop, axis=1)
        for f in feats_to_drop:
            if f in self.all_features:
                self.all_features.remove(f)
            if f in self.numerical_features:
                self.numerical_features.remove(f)
            if f in self.categorical_features:
                self.categorical_features.remove(f)

        # random shuffle
        shuffled_index = np.random.permutation(self.X.shape[0])
        self.X = self.X.iloc[shuffled_index]
        self.y = self.y.iloc[shuffled_index]
        self.X.reset_index(drop=True, inplace=True)
        self.y.reset_index(drop=True, inplace=True)

        return self

    def get_Xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.X, self.y

    def get_feature_names(self) -> Tuple[List[str], List[str], List[str]]:
        return self.all_features, self.numerical_features, self.categorical_features
