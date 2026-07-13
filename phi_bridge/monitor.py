#!/usr/bin/env python3
"""Online spectral Φ monitor with optional ConsciousAI backend."""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np

from .core import SpectralLevel, normalize_phi_series, pearson_connectivity, spectral_phi

# Sibling clone of https://github.com/wcalmels/consciousai
_DEFAULT_CAI = Path(__file__).resolve().parents[2] / "consciousai"


@dataclass
class SpectralPhiResult:
    phi_spectral: float
    phi_norm: float
    level: SpectralLevel
    level_name: str
    alert: bool
    critical: bool
    backend: str
    n_channels: int
    window: int
    note: str = ""


@dataclass
class SpectralPhiMonitor:
    """
    Sliding-window multivariate Φ monitor.

    Preferred tag in all reports: ``phi_spectral`` (never confuse with
    PhiCS C_phi or book Phi_TTH).
    """

    window: int = 40
    channel_order: Optional[List[str]] = None
    alert_pct: float = 10.0
    critical_pct: float = 2.0
    history_len: int = 200
    prefer_consciousai: bool = True
    consciousai_root: Path = _DEFAULT_CAI

    _buf: Deque[Dict[str, float]] = field(default_factory=deque, repr=False)
    _phi_hist: Deque[float] = field(default_factory=deque, repr=False)
    _backend: str = "builtin"
    _engine: Optional[object] = field(default=None, repr=False)
    _labels: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._buf = deque(maxlen=max(2, self.window))
        self._phi_hist = deque(maxlen=max(20, self.history_len))
        if self.prefer_consciousai:
            self._try_bind_consciousai()

    def _try_bind_consciousai(self) -> None:
        root = Path(self.consciousai_root)
        if not root.exists():
            self._backend = "builtin"
            return
        try:
            r = str(root.resolve())
            if r not in sys.path:
                sys.path.insert(0, r)
            # Import only connectivity helper; Φ uses builtin formula that
            # matches ConsciousAI README to avoid pulling security monitor threads.
            from src.core.connectivity import ConnectivityLearner  # type: ignore

            self._engine = ConnectivityLearner
            self._backend = "consciousai+builtin"
        except Exception:
            self._backend = "builtin"
            self._engine = None

    @property
    def backend(self) -> str:
        return self._backend

    def reset(self) -> None:
        self._buf.clear()
        self._phi_hist.clear()
        self._labels = []

    def push(self, readings: Dict[str, float]) -> Optional[SpectralPhiResult]:
        """Append one multivariate observation; return Φ once window is full."""
        if not readings:
            return None
        if not self._labels:
            self._labels = list(self.channel_order or sorted(readings.keys()))
        self._buf.append({k: float(readings.get(k, 0.0)) for k in self._labels})
        if len(self._buf) < min(self.window, max(8, len(self._labels) + 2)):
            return None
        return self._score_buffer()

    def score_matrix(self, matrix: np.ndarray, labels: Optional[Sequence[str]] = None) -> SpectralPhiResult:
        """Batch score a (T, N) or (units, channels) matrix."""
        x = np.asarray(matrix, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("matrix must be 2-D")
        conn = None
        if self._engine is not None:
            try:
                learner = self._engine(method="pearson")
                conn = learner.fit(x)
            except Exception:
                conn = pearson_connectivity(x)
        else:
            conn = pearson_connectivity(x)
        phi = spectral_phi(x, conn)
        self._phi_hist.append(phi)
        return self._pack(phi, n_channels=x.shape[1], window=x.shape[0])

    def _score_buffer(self) -> SpectralPhiResult:
        labels = self._labels
        rows = [[obs[k] for k in labels] for obs in self._buf]
        x = np.asarray(rows, dtype=np.float64)
        return self.score_matrix(x, labels)

    def _pack(self, phi: float, *, n_channels: int, window: int) -> SpectralPhiResult:
        hist = np.asarray(list(self._phi_hist) or [phi], dtype=np.float64)
        # Normalized level using local history
        if hist.size >= 5:
            phi_norm = float(normalize_phi_series(hist)[-1])
        else:
            phi_norm = float(phi / (1.0 + phi))

        alert = critical = False
        note = ""
        if hist.size >= 15:
            thr_a = float(np.percentile(hist, self.alert_pct))
            thr_c = float(np.percentile(hist, self.critical_pct))
            alert = phi < thr_a
            critical = phi < thr_c
            note = f"thr_alert={thr_a:.4f} thr_crit={thr_c:.4f}"
        level = SpectralLevel.from_phi(phi_norm)
        return SpectralPhiResult(
            phi_spectral=float(phi),
            phi_norm=phi_norm,
            level=level,
            level_name=level.name,
            alert=alert,
            critical=critical,
            backend=self._backend,
            n_channels=n_channels,
            window=window,
            note=note,
        )
