#!/usr/bin/env python3
"""
Spectral Φ core (engineering approximation, not exact IIT, not Phi_TTH).

Formula (ConsciousAI README):
  Φ(data, C) = H(eigenvalues(Cov(data))) × |positive eigs| × mean(|C|)
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional, Tuple

import numpy as np


class SpectralLevel(IntEnum):
    UNCONSCIOUS = 0
    MINIMAL = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5

    @classmethod
    def from_phi(cls, phi: float) -> "SpectralLevel":
        # Absolute cutoffs only useful after normalize; keep ConsciousAI-compatible bands
        # when phi is already scaled toward [0, 1+]. Raw eigenvalue-entropy Φ can be > 1.
        x = float(phi)
        if x < 0.1:
            return cls.UNCONSCIOUS
        if x < 0.3:
            return cls.MINIMAL
        if x < 0.5:
            return cls.LOW
        if x < 0.7:
            return cls.MODERATE
        if x < 0.9:
            return cls.HIGH
        return cls.VERY_HIGH


def pearson_connectivity(data: np.ndarray) -> np.ndarray:
    """(T, N) → (N, N) absolute-correlation connectivity (diag 0)."""
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        n = x.shape[1] if x.ndim == 2 else 1
        return np.eye(n, dtype=np.float64)
    # Column-wise standardize
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    z = (x - mu) / sd
    c = (z.T @ z) / max(x.shape[0] - 1, 1)
    c = np.abs(c)
    np.fill_diagonal(c, 0.0)
    return c


def spectral_phi(
    data: np.ndarray,
    connectivity: Optional[np.ndarray] = None,
    *,
    tol: float = 1e-10,
    eps: float = 1e-12,
) -> float:
    """
    Compute spectral Φ for a window of shape (T, N).

    This is the same engineering approximation published in ConsciousAI,
    implemented here so the bridge works even when the upstream package
    cannot be imported cleanly.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    t, n = x.shape
    if t < 2 or n < 1:
        return 0.0

    if connectivity is None:
        connectivity = pearson_connectivity(x)
    conn_strength = float(np.mean(np.abs(connectivity))) if connectivity.size else 1.0
    conn_strength = max(conn_strength, eps)

    # Covariance across components
    if n == 1:
        # Degenerate: use variance-based proxy
        v = float(np.var(x[:, 0]))
        return float(max(0.0, np.log1p(v) * conn_strength))

    cov = np.cov(x, rowvar=False)
    if cov.ndim == 0:
        return 0.0
    cov = np.nan_to_num(cov, nan=0.0)
    try:
        eigs = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return 0.0

    pos = np.abs(eigs)
    mask = pos > tol
    if not np.any(mask):
        return 0.0
    total = float(pos[mask].sum())
    if total < tol:
        return 0.0
    p = pos[mask] / total
    # Shannon entropy of eigenvalue spectrum
    ent = float(-np.sum(p * np.log(p + eps)))
    n_pos = int(mask.sum())
    phi = max(0.0, ent * n_pos * conn_strength)
    return float(phi)


def normalize_phi_series(phis: np.ndarray) -> np.ndarray:
    """Map a Φ series to roughly [0, 1] via robust max for level bands."""
    a = np.asarray(phis, dtype=np.float64)
    if a.size == 0:
        return a
    lo = float(np.percentile(a, 5))
    hi = float(np.percentile(a, 95))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def phi_from_signal_matrix(
    matrix: np.ndarray,
    *,
    connectivity: Optional[np.ndarray] = None,
) -> Tuple[float, SpectralLevel]:
    """One-shot Φ + level for a (rows=units, cols=channels) signal matrix."""
    # Treat rows as time-like samples of channel vectors
    phi = spectral_phi(matrix, connectivity)
    # For one-shot document audits, also report band using soft log-scale
    # mapped into ConsciousAI level cuts when raw phi is large.
    scaled = phi / (1.0 + phi)  # (0,1)
    level = SpectralLevel.from_phi(scaled)
    return phi, level
