"""Deterministic stratified splits."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def stratified_split(
    labels: Sequence[int] | np.ndarray,
    fractions: Sequence[float],
    seed: int = 0,
) -> list[np.ndarray]:
    """Split sample indices into ``len(fractions)`` parts with the same class balance.

    Every class is shuffled with its own slice of one seeded RNG and cut by ``fractions``, so
    the result only depends on ``(labels, fractions, seed)``. The last part takes the rounding
    remainder, which keeps the parts disjoint and covering all indices.
    """
    labels = np.asarray(labels)
    fractions = np.asarray(fractions, dtype=float)
    if labels.ndim != 1:
        raise ValueError("labels must be 1-D")
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError(f"fractions must be positive and sum to 1, got {fractions.tolist()}")

    rng = np.random.default_rng(seed)
    parts: list[list[np.ndarray]] = [[] for _ in fractions]
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        # cumulative cut points, rounded per class
        cuts = np.round(np.cumsum(fractions)[:-1] * len(idx)).astype(int)
        for part, chunk in zip(parts, np.split(idx, cuts), strict=True):
            part.append(chunk)

    # sort so loaders see a stable order before their own shuffling
    return [np.sort(np.concatenate(p)) for p in parts]
