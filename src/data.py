"""Loading the competition data and the column semantics taken from `data_description.txt`."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

TARGET = "SalePrice"

# Shared 5-point quality scale, Poor -> Excellent. 0 is reserved for "feature absent".
QUALITY_SCALE = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

ORDINAL_MAPPINGS = {
    "ExterQual": QUALITY_SCALE, "ExterCond": QUALITY_SCALE,
    "BsmtQual": QUALITY_SCALE, "BsmtCond": QUALITY_SCALE,
    "HeatingQC": QUALITY_SCALE, "KitchenQual": QUALITY_SCALE,
    "FireplaceQu": QUALITY_SCALE, "GarageQual": QUALITY_SCALE,
    "GarageCond": QUALITY_SCALE, "PoolQC": QUALITY_SCALE,
    "BsmtExposure": {"No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4,
                   "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8},
    "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
    "PavedDrive": {"N": 1, "P": 2, "Y": 3},
    "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4},
    "CentralAir": {"N": 0, "Y": 1},
}

# Columns where a missing value means the house has no such feature, rather than unknown.
ABSENT_MEANS_NONE = {
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "PoolQC",
}

STRUCTURAL_NONE = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "MasVnrType",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
]

STRUCTURAL_ZERO = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
]

NOMINAL_STRUCTURAL = [c for c in STRUCTURAL_NONE if c not in ORDINAL_MAPPINGS]

# Above this size the dataset's author recommends discarding the rows (training only).
OUTLIER_GRLIVAREA = 4000


def load_raw(raw_dir: Path = RAW_DIR):
    """Return (X, y, test, test_ids) with `Id` dropped and the target log-transformed.

    `MSSubClass` is cast to string: it holds dwelling-type codes, not quantities, so leaving
    it numeric would imply an ordering between unrelated dwelling types.
    """
    train = pd.read_csv(raw_dir / "train.csv").drop(columns="Id")
    test = pd.read_csv(raw_dir / "test.csv")

    test_ids = test["Id"].copy()
    test = test.drop(columns="Id")

    for frame in (train, test):
        frame["MSSubClass"] = frame["MSSubClass"].astype(str)

    X = train.drop(columns=TARGET)
    y = np.log1p(train[TARGET])
    return X, y, test, test_ids
