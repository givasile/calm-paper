from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

from .base import Dataset, DATA_ROOT

# Local copy of the MIMIC2 dataset as distributed by nodegam
# (nodegam.data.fetch_MIMIC2), stored under data/calm-data/mimic2/.
MIMIC2_DIR = DATA_ROOT / "calm-data" / "mimic2"


def fetch_MIMIC2(path=None, fold: int = 0) -> dict:
    """Load the MIMIC2 dataset from disk.

    Offline re-implementation of ``nodegam.data.fetch_MIMIC2`` (no download).

    Args:
        path: Directory containing the ``mimic2/`` folder. Defaults to
            ``DATA_ROOT / "calm-data"``.
        fold: Which data fold to use. Choose from [0, 1, 2, 3, 4].

    Returns:
        dict with keys 'X_train', 'y_train', 'X_test', 'y_test', 'problem',
        'cat_features', 'metric'.
    """
    assert 0 <= fold <= 4, "fold is only allowed btw 0 and 4, but get %d" % fold

    mimic2_dir = MIMIC2_DIR if path is None else Path(path) / "mimic2"
    if not (mimic2_dir / "mimic2.data").exists():
        raise FileNotFoundError(
            f"MIMIC-II is not redistributed (PhysioNet data use agreement); obtain it "
            f"through PhysioNet and place the NODE-GAM files (mimic2.data, "
            f"train0-4.txt, test0-4.txt) in {mimic2_dir}. See data/README.md."
        )

    cols = [
        "Age", "GCS", "SBP", "HR", "Temperature",
        "PFratio", "Renal", "Urea", "WBC", "CO2", "Na", "K",
        "Bilirubin", "AdmissionType", "AIDS",
        "MetastaticCancer", "Lymphoma", "HospitalMortality",
    ]

    table = pd.read_csv(mimic2_dir / "mimic2.data", delimiter=" ", header=None)
    table.columns = cols

    X_df = table.iloc[:, :-1]
    y_df = table.iloc[:, -1].values.astype(np.int64)

    train_idx = pd.read_csv(mimic2_dir / ("train%d.txt" % fold), header=None)[0].values
    test_idx = pd.read_csv(mimic2_dir / ("test%d.txt" % fold), header=None)[0].values

    cat_features = [
        "GCS", "Temperature", "AdmissionType", "AIDS",
        "MetastaticCancer", "Lymphoma", "Renal",
    ]
    for c in cat_features:
        X_df[c] = X_df[c].astype(object)

    return dict(
        X_train=X_df.iloc[train_idx], y_train=y_df[train_idx],
        X_test=X_df.iloc[test_idx], y_test=y_df[test_idx],
        problem="classification",
        cat_features=cat_features,
        metric="negative_auc",
    )


class MIMIC2(Dataset):
    def fetch(self):
        mimic = fetch_MIMIC2()

        self.X = pd.concat([mimic["X_train"], mimic["X_test"]], ignore_index=True)
        self.y = pd.Series(
            np.concatenate([mimic["y_train"], mimic["y_test"]]), name="target"
        )

        self.all_features = self.X.columns.tolist()
        self.categorical_features = mimic["cat_features"]
        self.numerical_features = self.X.columns.difference(
            self.categorical_features
        ).tolist()

        return self

    def preprocess(self):
        # random shuffle
        shuffled_index = np.random.permutation(self.X.shape[0])
        self.X = self.X.iloc[shuffled_index]
        self.y = self.y.iloc[shuffled_index]
        self.X.reset_index(drop=True, inplace=True)
        self.y.reset_index(drop=True, inplace=True)

        return self

    def get_Xy(self):
        return self.X, self.y

    def get_feature_names(self) -> Tuple[List[str], List[str], List[str]]:
        return self.all_features, self.numerical_features, self.categorical_features
