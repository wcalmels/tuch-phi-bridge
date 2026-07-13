"""
tuch-phi-bridge — Spectral Φ adapter for TUCH stacks.

Wraps / mirrors ConsciousAI's spectral integrated-information metric and
exposes a tiny sync API for sentinel-edge, book_agent, HTTP, and PX4.
"""

from .monitor import SpectralPhiMonitor, SpectralPhiResult, SpectralLevel
from .matrix_phi import phi_from_signal_matrix

__all__ = [
    "SpectralPhiMonitor",
    "SpectralPhiResult",
    "SpectralLevel",
    "phi_from_signal_matrix",
]

__version__ = "0.2.0"
