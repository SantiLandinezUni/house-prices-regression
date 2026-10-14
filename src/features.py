"""Transformers turning the raw frame into a model-ready matrix.

Split by one question: does the step learn anything from the data?

`DomainCleaner` and `FeatureEngineer` do not — they apply constants from the data dictionary
and row-wise arithmetic — so they cannot leak. `SkewFixer` *chooses* its columns from the data,
so that choice lives in `fit` and is made on training rows only.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .data import (ABSENT_MEANS_NONE, NOMINAL_STRUCTURAL, ORDINAL_MAPPINGS, STRUCTURAL_ZERO)


class DomainCleaner(BaseEstimator, TransformerMixin):
    """Deterministic fixes taken from `data_description.txt`."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Ordinal categories -> integers; absent features become 0, which ranks
        # "no basement" below "poor basement" on a single scale.
        for col, mapping in ORDINAL_MAPPINGS.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
                if col in ABSENT_MEANS_NONE:
                    X[col] = X[col].fillna(0)

        for col in NOMINAL_STRUCTURAL:
            if col in X.columns:
                X[col] = X[col].fillna("None")

        for col in STRUCTURAL_ZERO:
            if col in X.columns:
                X[col] = X[col].fillna(0)

        # A garage is normally built with the house.
        if "GarageYrBlt" in X.columns:
            X["GarageYrBlt"] = X["GarageYrBlt"].fillna(X["YearBuilt"])

        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Combinations of existing columns. Row-wise arithmetic, so nothing is learned.

    Measured effect on the neural network was within fold-to-fold noise: the information is
    already present in the components, and a linear layer can form sums by itself. Kept because
    the full combination scored best, not because these features carried the improvement.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
        X["TotalBath"] = (X["FullBath"] + 0.5 * X["HalfBath"]
                          + X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"])
        X["TotalPorchSF"] = (X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"]
                             + X["ScreenPorch"] + X["WoodDeckSF"])

        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        X["RemodAge"] = X["YrSold"] - X["YearRemodAdd"]
        X["IsRemodeled"] = (X["YearRemodAdd"] != X["YearBuilt"]).astype(int)
        X["OverallGrade"] = X["OverallQual"] * X["OverallCond"]

        X["HasBsmt"] = (X["TotalBsmtSF"] > 0).astype(int)
        X["Has2ndFlr"] = (X["2ndFlrSF"] > 0).astype(int)
        X["HasGarage"] = (X["GarageArea"] > 0).astype(int)
        X["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)
        X["HasPool"] = (X["PoolArea"] > 0).astype(int)

        return X


class SkewFixer(BaseEstimator, TransformerMixin):
    """log1p the strongly skewed numeric columns.

    Standardising cannot fix skew: it is a linear transformation, so it recentres a
    distribution without changing its shape. log1p is nonlinear and does change it.
    """

    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        numeric = X.select_dtypes(include="number")
        self.cols_ = numeric.columns[numeric.skew().abs() > self.threshold].tolist()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols_:
            if col in X.columns:
                X[col] = np.log1p(X[col].clip(lower=0))
        return X


def column_groups(X):
    """Numeric and nominal column names after cleaning and feature engineering.

    `pd.api.types.is_numeric_dtype` rather than `dtype == object`: pandas 3.0 gave strings
    their own dtype, so the older idiom silently matches nothing.
    """
    prepared = FeatureEngineer().fit_transform(DomainCleaner().fit_transform(X))
    numeric = [c for c in prepared.columns if pd.api.types.is_numeric_dtype(prepared[c])]
    nominal = [c for c in prepared.columns if not pd.api.types.is_numeric_dtype(prepared[c])]
    return numeric, nominal
