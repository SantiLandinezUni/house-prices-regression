"""Reusable pieces of the House Prices pipeline.

The notebook `notebooks/01_house_prices.ipynb` is deliberately self-contained, because it is
submitted on its own. This package mirrors the same code for reuse outside the notebook.
"""

from .data import ABSENT_MEANS_NONE, ORDINAL_MAPPINGS, STRUCTURAL_NONE, STRUCTURAL_ZERO, load_raw
from .features import DomainCleaner, FeatureEngineer, SkewFixer, column_groups
from .model import TabularMLP, TorchMLPRegressor

__all__ = [
    "load_raw", "ORDINAL_MAPPINGS", "ABSENT_MEANS_NONE", "STRUCTURAL_NONE", "STRUCTURAL_ZERO",
    "DomainCleaner", "FeatureEngineer", "SkewFixer", "column_groups",
    "TabularMLP", "TorchMLPRegressor",
]
