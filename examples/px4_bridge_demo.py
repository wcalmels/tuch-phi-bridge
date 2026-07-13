#!/usr/bin/env python3
"""
PX4 connector → spectral Φ (+ optional POST to local API).

  py -3 examples/px4_bridge_demo.py
  py -3 examples/px4_bridge_demo.py --steps 80
  py -3 examples/px4_bridge_demo.py --mode mavlink --connection udp:127.0.0.1:14550
  py -3 examples/px4_bridge_demo.py --api http://127.0.0.1:8787
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phi_bridge import SpectralPhiMonitor  # noqa: E402
from phi_bridge.connectors import Px4Connector, SourceMode  # noqa: E402


def post_api(base: str, readings: dict, session_id: str) -> dict:
    payload = json.dumps({"readings": readings, "session_id": session_id}).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/phi-spectral",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode())


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="PX4/sim → phi_spectral")
    p.add_argument("--mode", choices=["sim", "mavlink"], default="sim")
    p.add_argument("--connection", default="udp:127.0.0.1:14550")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--api", default="", help="Optional API base e.g. http://127.0.0.1:8787")
    p.add_argument("--session-id", default="px4-demo")
    args = p.parse_args(argv)

    mon = SpectralPhiMonitor(window=args.window, prefer_consciousai=False)
    print()
    print("=" * 68)
    print("  TUCH phi-bridge — PX4 connector demo")
    print(f"  mode={args.mode}  steps={args.steps}")
    if args.api:
        print(f"  API={args.api}")
    print("=" * 68)

    try:
        conn = Px4Connector(
            mode=SourceMode(args.mode),
            connection=args.connection,
            seed=args.seed,
            poll_hz=args.hz,
        )
        conn.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR connecting: {exc}")
        return 1

    last = None
    try:
        for i, frame in enumerate(conn.stream(steps=args.steps)):
            r = mon.push(frame.readings)
            api_note = ""
            if args.api:
                try:
                    remote = post_api(args.api, frame.readings, args.session_id)
                    if remote.get("ready"):
                        api_note = f"  api_phi={remote.get('phi_spectral', 0):.4f}"
                    else:
                        api_note = "  api=warming"
                except Exception as exc:  # noqa: BLE001
                    api_note = f"  api_err={exc}"
            if r is None:
                if i % 10 == 0:
                    print(f"  t={frame.t:6.2f}s  source={frame.source}  warming…{api_note}")
                continue
            last = r
            if i % 5 == 0 or r.alert:
                print(
                    f"  t={frame.t:6.2f}s  phi={r.phi_spectral:8.4f}  "
                    f"level={r.level_name:<12}  alert={r.alert}{api_note}"
                )
    finally:
        conn.close()

    print("-" * 68)
    if last:
        print(f"  Last phi_spectral={last.phi_spectral:.4f} level={last.level_name}")
    else:
        print("  No score yet — increase --steps or lower --window")
    print("  Note: sim by default; mavlink needs PX4 SITL + pymavlink")
    print("=" * 68)
    print()
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    raise SystemExit(main())
