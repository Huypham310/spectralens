"""Spectral analysis of trained networks."""

from spectralens.analysis.spectra import (
    LayerSpectrum,
    hill_alpha,
    layer_matrices,
    mp_edge,
    spectrum_of,
)

__all__ = ["LayerSpectrum", "hill_alpha", "layer_matrices", "mp_edge", "spectrum_of"]
