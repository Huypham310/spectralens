"""EuroSAT and CIFAR-10 with train / val / test splits and train-only normalization.

The same underlying images are shared by every split; each split is a view with its own
transform, so augmentation never leaks into validation or test.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, tv_tensors
from torchvision.transforms import v2

from spectralens.data.splits import stratified_split
from spectralens.data.stats import channel_stats


@dataclass(frozen=True)
class DataSpec:
    name: str
    num_classes: int
    image_size: int
    in_channels: int = 3


SPECS: dict[str, DataSpec] = {
    "eurosat": DataSpec("eurosat", num_classes=10, image_size=64),
    "cifar10": DataSpec("cifar10", num_classes=10, image_size=32),
}


class TransformSubset(Dataset):
    """``Subset`` that applies its own transform to the raw (PIL) samples."""

    def __init__(self, base: Dataset, indices: Sequence[int], transform: Callable | None):
        self.base = base
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, target = self.base[int(self.indices[i])]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class InMemoryImages(Dataset):
    """Decoded images as one uint8 tensor (N, C, H, W) plus their labels.

    Decoding EuroSAT's PNGs with PIL on every access caps the loader at ~1k images/s on one
    worker; keeping the decoded pixels in RAM (27k x 3 x 64 x 64 bytes = 330 MB) removes that.
    """

    def __init__(self, images: torch.Tensor, targets: Sequence[int]):
        if images.dtype != torch.uint8 or images.ndim != 4:
            raise ValueError("images must be a uint8 tensor of shape (N, C, H, W)")
        self.images = images
        self.targets = list(targets)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, i: int):
        return tv_tensors.Image(self.images[i]), self.targets[i]


def preload(ds: Dataset, cache: Path | None = None, batch_size: int = 512) -> InMemoryImages:
    """Decode every image of ``ds`` once. Uses ``cache`` (a .pt file) if it already exists."""
    if cache is not None and cache.exists():
        saved = torch.load(cache, weights_only=True)
        return InMemoryImages(saved["images"], saved["targets"].tolist())

    data = getattr(ds, "data", None)
    if isinstance(data, np.ndarray) and data.ndim == 4:
        # CIFAR keeps everything as an (N, H, W, C) array already, no decoding needed
        images = torch.from_numpy(data).permute(0, 3, 1, 2).contiguous()
        targets = labels_of(ds).tolist()
    else:
        decode = TransformSubset(ds, np.arange(len(ds)), v2.ToImage())
        chunks, targets = [], []
        for x, y in DataLoader(decode, batch_size=batch_size):
            chunks.append(x)
            targets.extend(int(t) for t in y)
        images = torch.cat(chunks)

    if cache is not None:
        torch.save({"images": images, "targets": torch.tensor(targets)}, cache)
    return InMemoryImages(images, targets)


@dataclass
class Splits:
    spec: DataSpec
    train: TransformSubset
    val: TransformSubset
    test: TransformSubset
    mean: list[float]
    std: list[float]


def labels_of(ds: Dataset) -> np.ndarray:
    """Class label of every sample, without decoding images when the dataset exposes targets."""
    targets = getattr(ds, "targets", None)
    if targets is not None:
        return np.asarray(targets)
    # int(): some datasets (e.g. FakeData) return 0-d tensors as targets
    return np.asarray([int(ds[i][1]) for i in range(len(ds))])


def _to_tensor() -> v2.Transform:
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


def train_augment(name: str) -> list[v2.Transform]:
    if name == "eurosat":
        # overhead imagery has no "up": flips and 90-degree turns are all valid views
        return [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomChoice([v2.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
        ]
    if name == "cifar10":
        return [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()]
    return []


def _load_raw(name: str, root: Path, download: bool) -> tuple[Dataset, Dataset | None]:
    """Return (pool to split, official test set or None)."""
    if name == "eurosat":
        # EuroSAT has no official split, everything goes into the pool
        return datasets.EuroSAT(str(root), download=download), None
    if name == "cifar10":
        train = datasets.CIFAR10(str(root), train=True, download=download)
        test = datasets.CIFAR10(str(root), train=False, download=download)
        return train, test
    raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(SPECS)}")


def _cached_stats(
    path: Path, compute: Callable[[], tuple[list[float], list[float]]]
) -> tuple[list[float], list[float]]:
    if path.exists():
        cached = json.loads(path.read_text())
        return cached["mean"], cached["std"]
    mean, std = compute()
    path.write_text(json.dumps({"mean": mean, "std": std}, indent=2))
    return mean, std


def build_datasets(
    name: str,
    root: str | Path,
    seed: int = 0,
    download: bool = False,
    augment: bool = True,
    in_memory: bool = True,
    pool: Dataset | None = None,
    test_pool: Dataset | None = None,
) -> Splits:
    """Build the three splits for ``name``.

    EuroSAT: stratified 80 / 10 / 10. CIFAR-10: official test set, the official training set is
    split 90 / 10 into train / val. Normalization statistics come from the train split only and
    are cached next to the data per seed. With ``in_memory`` the decoded images are kept in RAM
    (and cached on disk). ``pool`` / ``test_pool`` let tests pass synthetic data.
    """
    spec = SPECS[name]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    real = pool is None
    if real:
        pool, test_pool = _load_raw(name, root, download)
    if in_memory:
        # only real datasets get an on-disk cache; synthetic test data is rebuilt every time
        pool = preload(pool, root / f"{name}_pool_uint8.pt" if real else None)
        if test_pool is not None:
            test_pool = preload(test_pool, root / f"{name}_test_uint8.pt" if real else None)

    labels = labels_of(pool)
    if test_pool is None:
        train_idx, val_idx, test_idx = stratified_split(labels, (0.8, 0.1, 0.1), seed)
        test_base = pool
    else:
        train_idx, val_idx = stratified_split(labels, (0.9, 0.1), seed)
        test_idx = np.arange(len(test_pool))
        test_base = test_pool

    def compute() -> tuple[list[float], list[float]]:
        raw = TransformSubset(pool, train_idx, _to_tensor())
        return channel_stats(raw, range(len(raw)))

    mean, std = _cached_stats(root / f"{name}_stats_seed{seed}.json", compute)
    normalize = [_to_tensor(), v2.Normalize(mean, std)]
    eval_tf = v2.Compose(normalize)
    train_tf = v2.Compose([*(train_augment(name) if augment else []), *normalize])

    return Splits(
        spec=spec,
        train=TransformSubset(pool, train_idx, train_tf),
        val=TransformSubset(pool, val_idx, eval_tf),
        test=TransformSubset(test_base, test_idx, eval_tf),
        mean=mean,
        std=std,
    )


def fixed_subset(split: TransformSubset, n: int, seed: int = 0) -> TransformSubset:
    """A small, class-balanced, always-identical subset (e.g. for Hessian estimates)."""
    labels = labels_of(split.base)[split.indices]
    if n >= len(split):
        return split
    picked, _ = stratified_split(labels, (n / len(split), 1 - n / len(split)), seed)
    return TransformSubset(split.base, split.indices[picked], split.transform)


def make_loaders(
    splits: Splits,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 0,
) -> dict[str, DataLoader]:
    """Train loader shuffles with its own seeded generator; val / test keep their order."""
    gen = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    return {
        "train": DataLoader(splits.train, shuffle=True, generator=gen, drop_last=True, **common),
        "val": DataLoader(splits.val, shuffle=False, **common),
        "test": DataLoader(splits.test, shuffle=False, **common),
    }
