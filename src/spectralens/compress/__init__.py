"""Low-rank compression of trained networks (truncated SVD, Tucker-2) and its cost model."""

from spectralens.compress.apply import LayerChoice, compress_model, eligible_layers
from spectralens.compress.decompose import (
    rank_for_energy,
    svd_conv,
    svd_linear,
    tucker2_conv,
    tucker2_factors,
    tucker2_ranks,
)
from spectralens.compress.flops import count_macs, count_params

__all__ = [
    "LayerChoice",
    "compress_model",
    "count_macs",
    "count_params",
    "eligible_layers",
    "rank_for_energy",
    "svd_conv",
    "svd_linear",
    "tucker2_conv",
    "tucker2_factors",
    "tucker2_ranks",
]
