#!/usr/bin/env python3
"""Ensure tuch-phi-bridge is importable from sibling consumers."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_bridge_path(bridge_root: Path | None = None) -> Path:
    root = Path(bridge_root) if bridge_root else Path(__file__).resolve().parents[1]
    # This file lives in tuch-phi-bridge/phi_bridge/_path.py → parents[1] = package root
    # When copied into consumers, pass absolute Escritorio/tuch-phi-bridge
    if not (root / "phi_bridge").exists():
        # caller may pass consumer helper that lives elsewhere
        guess = Path(__file__).resolve().parents[2] / "tuch-phi-bridge"
        if (guess / "phi_bridge").exists():
            root = guess
    s = str(root.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)
    return root
