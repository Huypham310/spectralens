"""Datasets used by the experiments (EuroSAT, CIFAR-10) with leakage-free splits."""

from spectralens.data.datasets import (
    SPECS,
    DataSpec,
    Splits,
    build_datasets,
    fixed_subset,
    make_loaders,
)
from spectralens.data.splits import stratified_split
from spectralens.data.stats import channel_stats

__all__ = [
    "SPECS",
    "DataSpec",
    "Splits",
    "build_datasets",
    "channel_stats",
    "fixed_subset",
    "make_loaders",
    "stratified_split",
]
