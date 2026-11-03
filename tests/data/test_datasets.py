import json

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from torchvision.datasets import FakeData

from spectralens.data import build_datasets, channel_stats, fixed_subset, make_loaders
from spectralens.data.datasets import (
    InMemoryImages,
    TransformSubset,
    _to_tensor,
    labels_of,
    preload,
)


class ConstImages(Dataset):
    """Images whose pixel values are known exactly, to check the statistics."""

    def __init__(self, values: list[float], size: int = 4):
        self.values = values
        self.size = size

    def __len__(self):
        return len(self.values)

    def __getitem__(self, i):
        v = self.values[i]
        img = torch.stack(
            [
                torch.full((self.size, self.size), v),
                torch.zeros(self.size, self.size),
                torch.full((self.size, self.size), 1.0),
            ]
        )
        return img, i % 2


def test_channel_stats_match_closed_form():
    ds = ConstImages([0.0, 0.5, 1.0, 0.5])
    mean, std = channel_stats(ds, range(len(ds)), batch_size=3)
    assert mean == pytest.approx([0.5, 0.0, 1.0])
    # values 0, .5, 1, .5 -> population variance 0.125
    assert std == pytest.approx([0.125**0.5, 0.0, 0.0])


def test_channel_stats_only_use_the_given_indices():
    ds = ConstImages([0.0, 1.0, 1.0, 1.0])
    mean, _ = channel_stats(ds, [1, 2, 3])
    assert mean[0] == pytest.approx(1.0)


@pytest.fixture
def fake_eurosat(tmp_path):
    pool = FakeData(size=400, image_size=(3, 16, 16), num_classes=4, random_offset=7)
    return build_datasets("eurosat", tmp_path, seed=0, pool=pool), tmp_path


def test_eurosat_style_split_sizes_and_no_overlap(fake_eurosat):
    splits, _ = fake_eurosat
    assert len(splits.train) + len(splits.val) + len(splits.test) == 400
    assert len(splits.val) == pytest.approx(40, abs=4)
    sets = [set(s.indices.tolist()) for s in (splits.train, splits.val, splits.test)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


def test_normalization_comes_from_train_only_and_is_cached(fake_eurosat):
    splits, root = fake_eurosat
    train_raw = TransformSubset(splits.train.base, splits.train.indices, _to_tensor())
    expected_mean, _ = channel_stats(train_raw, range(len(train_raw)))
    assert splits.mean == pytest.approx(expected_mean, abs=1e-6)
    cached = json.loads((root / "eurosat_stats_seed0.json").read_text())
    assert cached["mean"] == pytest.approx(splits.mean)


def test_eval_splits_are_not_augmented(fake_eurosat):
    splits, _ = fake_eurosat
    x1, _ = splits.val[0]
    x2, _ = splits.val[0]
    assert torch.equal(x1, x2)
    assert x1.shape == (3, 16, 16) and x1.dtype == torch.float32


def test_cifar_style_uses_the_official_test_set(tmp_path):
    pool = FakeData(size=300, image_size=(3, 8, 8), num_classes=3, random_offset=1)
    test = FakeData(size=50, image_size=(3, 8, 8), num_classes=3, random_offset=2)
    splits = build_datasets("cifar10", tmp_path, pool=pool, test_pool=test)
    assert len(splits.test) == 50
    assert len(splits.train) + len(splits.val) == 300
    assert len(splits.val) == pytest.approx(30, abs=3)


def test_fixed_subset_is_small_balanced_and_repeatable(fake_eurosat):
    splits, _ = fake_eurosat
    a = fixed_subset(splits.train, 40, seed=3)
    b = fixed_subset(splits.train, 40, seed=3)
    assert np.array_equal(a.indices, b.indices)
    assert len(a) == pytest.approx(40, abs=4)
    assert set(a.indices.tolist()) <= set(splits.train.indices.tolist())
    counts = np.bincount(labels_of(splits.train.base)[a.indices], minlength=4)
    assert counts.min() >= 5  # every class present


def test_train_loader_shuffle_is_seeded(fake_eurosat):
    splits, _ = fake_eurosat
    first = [next(iter(make_loaders(splits, batch_size=16, seed=5)["train"]))[1] for _ in range(2)]
    other = next(iter(make_loaders(splits, batch_size=16, seed=6)["train"]))[1]
    assert torch.equal(first[0], first[1])
    assert not torch.equal(first[0], other)


def test_preload_matches_decoding_on_the_fly_and_caches(tmp_path):
    pool = FakeData(size=30, image_size=(3, 8, 8), num_classes=3, random_offset=4)
    cache = tmp_path / "pool.pt"
    mem = preload(pool, cache)
    assert isinstance(mem, InMemoryImages) and cache.exists()
    for i in (0, 17, 29):
        on_the_fly = _to_tensor()(pool[i][0])
        assert torch.equal(_to_tensor()(mem[i][0]), on_the_fly)
        assert mem[i][1] == int(pool[i][1])
    again = preload(FakeData(size=1), cache)  # served from the cache, not the new dataset
    assert len(again) == 30


def test_preload_uses_cifar_style_arrays_directly():
    class ArrayDataset(Dataset):
        data = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
        targets = [1, 0]

        def __len__(self):
            return 2

        def __getitem__(self, i):
            raise AssertionError("should not decode one by one")

    mem = preload(ArrayDataset())
    assert mem.images.shape == (2, 3, 4, 4)
    assert mem.images[1, 2, 0, 1] == ArrayDataset.data[1, 0, 1, 2]


def test_in_memory_and_lazy_splits_agree(tmp_path):
    pool = FakeData(size=120, image_size=(3, 8, 8), num_classes=4, random_offset=9)
    lazy = build_datasets("eurosat", tmp_path / "a", pool=pool, in_memory=False)
    fast = build_datasets("eurosat", tmp_path / "b", pool=pool, in_memory=True)
    assert lazy.mean == pytest.approx(fast.mean)
    assert torch.allclose(lazy.val[3][0], fast.val[3][0])


def test_unknown_dataset(tmp_path):
    with pytest.raises(KeyError):
        build_datasets("mnist", tmp_path, pool=FakeData(size=10))
