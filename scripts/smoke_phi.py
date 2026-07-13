#!/usr/bin/env python3
"""Smoke test for tuch-phi-bridge (builtin backend only)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from phi_bridge import SpectralPhiMonitor, phi_from_signal_matrix


def main() -> int:
    rng = np.random.default_rng(0)
    mon = SpectralPhiMonitor(window=20, prefer_consciousai=False)
    last = None
    for _ in range(30):
        last = mon.push({
            "a": float(60 + rng.normal()),
            "b": float(0.5 + 0.05 * rng.normal()),
            "c": float(200 + 3 * rng.normal()),
            "d": float(10 + rng.normal()),
        })
    assert last is not None, "monitor never scored"
    print(f"online phi_spectral={last.phi_spectral:.4f} level={last.level_name} backend={last.backend}")

    M = rng.random((12, 4))
    phi, level = phi_from_signal_matrix(M)
    print(f"matrix  phi_spectral={phi:.4f} level={level.name}")
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
