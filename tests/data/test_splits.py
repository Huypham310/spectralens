import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from spectralens.data import stratified_split


def test_parts_are_disjoint_and_cover_everything():
    labels = np.repeat(np.arange(10), 270)  # EuroSAT-like: 10 classes
    parts = stratified_split(labels, (0.8, 0.1, 0.1), seed=0)
    joined = np.concatenate(parts)
    assert len(joined) == len(labels)
    assert len(np.unique(joined)) == len(labels)


def test_same_seed_same_split_other_seed_different():
    labels = np.repeat(np.arange(5), 100)
    a = stratified_split(labels, (0.8, 0.2), seed=1)
    b = stratified_split(labels, (0.8, 0.2), seed=1)
    c = stratified_split(labels, (0.8, 0.2), seed=2)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not np.array_equal(a[0], c[0])


def test_rejects_bad_fractions():
    with pytest.raises(ValueError, match="sum to 1"):
        stratified_split([0, 1], (0.5, 0.4))
    with pytest.raises(ValueError, match="positive"):
        stratified_split([0, 1], (1.2, -0.2))


@settings(max_examples=50, deadline=None)
@given(
    counts=st.lists(st.integers(min_value=10, max_value=300), min_size=2, max_size=8),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_every_part_keeps_the_class_balance(counts, seed):
    labels = np.concatenate([np.full(n, c) for c, n in enumerate(counts)])
    parts = stratified_split(labels, (0.8, 0.1, 0.1), seed=seed)
    for part, frac in zip(parts, (0.8, 0.1, 0.1), strict=True):
        got = np.bincount(labels[part], minlength=len(counts))
        # per-class rounding can move at most one sample per cut
        assert np.all(np.abs(got - np.asarray(counts) * frac) <= 1.0)
