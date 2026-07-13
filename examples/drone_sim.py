#!/usr/bin/env python3
"""
Drone simulation — spectral Φ → safety action (RTL / land).

Engineering demonstration only. Not flight-certified. Not "robot consciousness".
Tags: phi_spectral (integration) — never equated with Phi_TTH.

Usage:
  py -3 examples/drone_sim.py
  py -3 examples/drone_sim.py --steps 120 --fault gps_dropout
  py -3 examples/drone_sim.py --fault motor_desync --seed 7
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phi_bridge import SpectralPhiMonitor, SpectralPhiResult  # noqa: E402


CHANNELS = [
    "gps_lat_rate",
    "gps_lon_rate",
    "alt_m",
    "imu_ax",
    "imu_ay",
    "imu_az",
    "gyro_yaw",
    "motor_fl",
    "motor_fr",
    "motor_rl",
    "motor_rr",
    "battery_v",
    "link_rssi",
]


class Fault(str, Enum):
    NONE = "none"
    GPS_DROPOUT = "gps_dropout"
    MOTOR_DESYNC = "motor_desync"
    LINK_FADE = "link_fade"
    WIND_GUST = "wind_gust"


class Action(str, Enum):
    CRUISE = "CRUISE"
    CONSERVATIVE = "CONSERVATIVE"
    RTL = "RTL"           # return to launch
    EMERGENCY_LAND = "EMERGENCY_LAND"


@dataclass
class Tick:
    t: int
    readings: Dict[str, float]
    phi: Optional[SpectralPhiResult]
    action: Action
    fault_active: bool
    phase: str = "warmup"  # warmup | calibrate | armed


def simulate_tick(
    t: int,
    *,
    rng: np.random.Generator,
    fault: Fault,
    fault_start: int,
) -> Tuple[Dict[str, float], bool]:
    """Synthetic UAV multi-sensor snapshot (normalized-ish engineering units)."""
    cruise = 0.02 * np.sin(t / 18.0)
    active = fault != Fault.NONE and t >= fault_start

    readings = {
        "gps_lat_rate": float(cruise + rng.normal(0, 0.01)),
        "gps_lon_rate": float(0.015 * np.cos(t / 22.0) + rng.normal(0, 0.01)),
        "alt_m": float(42.0 + 0.4 * np.sin(t / 30.0) + rng.normal(0, 0.05)),
        "imu_ax": float(rng.normal(0, 0.03)),
        "imu_ay": float(rng.normal(0, 0.03)),
        "imu_az": float(9.81 + rng.normal(0, 0.04)),
        "gyro_yaw": float(0.05 * np.sin(t / 40.0) + rng.normal(0, 0.01)),
        "motor_fl": float(0.55 + cruise + rng.normal(0, 0.01)),
        "motor_fr": float(0.55 + cruise + rng.normal(0, 0.01)),
        "motor_rl": float(0.55 + cruise + rng.normal(0, 0.01)),
        "motor_rr": float(0.55 + cruise + rng.normal(0, 0.01)),
        "battery_v": float(15.8 - 0.004 * t + rng.normal(0, 0.01)),
        "link_rssi": float(-62.0 + rng.normal(0, 0.8)),
    }

    if active:
        if fault == Fault.GPS_DROPOUT:
            # GPS rates collapse / NaN-like freeze → decorrelated from IMU
            readings["gps_lat_rate"] = float(rng.normal(0, 0.4))
            readings["gps_lon_rate"] = float(rng.normal(0, 0.4))
        elif fault == Fault.MOTOR_DESYNC:
            readings["motor_fl"] = float(0.85 + rng.normal(0, 0.08))
            readings["motor_rr"] = float(0.25 + rng.normal(0, 0.08))
            readings["imu_ax"] = float(rng.normal(0, 0.35))
            readings["gyro_yaw"] = float(rng.normal(0, 0.25))
        elif fault == Fault.LINK_FADE:
            readings["link_rssi"] = float(-95.0 + rng.normal(0, 2.0))
            # command delay proxy: motors lag + noise
            lag = 0.15 * np.sin(t / 3.0)
            for m in ("motor_fl", "motor_fr", "motor_rl", "motor_rr"):
                readings[m] = float(readings[m] + lag + rng.normal(0, 0.05))
        elif fault == Fault.WIND_GUST:
            readings["imu_ax"] = float(rng.normal(0.2, 0.4))
            readings["imu_ay"] = float(rng.normal(-0.1, 0.4))
            readings["alt_m"] = float(readings["alt_m"] + rng.normal(0, 1.2))
            for m in ("motor_fl", "motor_fr", "motor_rl", "motor_rr"):
                readings[m] = float(readings[m] + rng.normal(0, 0.12))

    return readings, active


def decide_action(
    phi: float,
    *,
    calib: List[float],
    latched: Action,
    armed: bool,
    streak: int,
) -> Tuple[Action, int]:
    """
    Relative safety ladder vs cruise calibration (not absolute Φ bands).

    Requires `streak` consecutive out-of-band samples before RTL/LAND.
    Returns (action, updated_streak).
    """
    if latched in (Action.RTL, Action.EMERGENCY_LAND):
        return latched, streak
    if not armed or len(calib) < 10:
        return Action.CRUISE, 0

    thr_cons = float(np.percentile(calib, 20))
    thr_rtl = float(np.percentile(calib, 8))
    thr_land = float(np.percentile(calib, 2))
    med = float(np.median(calib))
    iqr = float(np.percentile(calib, 75) - np.percentile(calib, 25)) + 1e-9

    land_hit = phi < thr_land or phi > med + 5.5 * iqr
    rtl_hit = phi < thr_rtl or phi > med + 3.8 * iqr
    cons_hit = phi < thr_cons

    if land_hit:
        streak += 1
        if streak >= 2:
            return Action.EMERGENCY_LAND, streak
        return Action.CONSERVATIVE, streak
    if rtl_hit:
        streak += 1
        if streak >= 2:
            return Action.RTL, streak
        return Action.CONSERVATIVE, streak
    if cons_hit:
        return Action.CONSERVATIVE, 0
    return Action.CRUISE, 0


def run_mission(
    *,
    steps: int,
    fault: Fault,
    fault_start: int,
    seed: int,
    window: int,
    calib_ticks: int,
) -> List[Tick]:
    rng = np.random.default_rng(seed)
    mon = SpectralPhiMonitor(
        window=window,
        channel_order=CHANNELS,
        history_len=max(120, steps),
        prefer_consciousai=False,
    )
    latched = Action.CRUISE
    calib: List[float] = []
    ticks: List[Tick] = []
    scored = 0
    streak = 0

    for t in range(steps):
        readings, active = simulate_tick(t, rng=rng, fault=fault, fault_start=fault_start)
        phi_r = mon.push(readings)
        if phi_r is None:
            ticks.append(
                Tick(
                    t=t, readings=readings, phi=None, action=Action.CRUISE,
                    fault_active=active, phase="warmup",
                )
            )
            continue

        scored += 1
        # Keep calibrating until just before fault injection (cleaner demo).
        # With fault=none, use fixed calib_ticks then arm for the rest.
        if fault != Fault.NONE:
            armed = t >= fault_start
            phase = "armed" if armed else ("calibrate" if scored else "warmup")
            if not armed:
                calib.append(phi_r.phi_spectral)
        else:
            armed = scored > calib_ticks
            phase = "armed" if armed else "calibrate"
            if not armed:
                calib.append(phi_r.phi_spectral)

        action, streak = decide_action(
            phi_r.phi_spectral, calib=calib, latched=latched, armed=armed, streak=streak,
        )
        latched = action
        ticks.append(
            Tick(
                t=t, readings=readings, phi=phi_r, action=action,
                fault_active=active, phase=phase,
            )
        )
    return ticks


def print_report(ticks: List[Tick], fault: Fault, fault_start: int) -> None:
    print()
    print("=" * 72)
    print("  TUCH phi-bridge — UAV spectral-Φ safety demo")
    print("  (simulation · not flight-certified)")
    print("=" * 72)
    print(f"  Fault: {fault.value}  starting at t={fault_start}")
    print(f"  Channels: {len(CHANNELS)}")
    print()
    print(f"{'t':>4}  {'phase':<9}  {'fault':<5}  {'phi':>8}  action")
    print("-" * 72)

    rtl_at = land_at = None
    last_action = None
    for tk in ticks:
        if tk.phi is None:
            continue
        changed = tk.action != last_action
        last_action = tk.action
        if not changed and tk.t % 12 != 0:
            continue
        print(
            f"{tk.t:4d}  {tk.phase:<9}  {'ON' if tk.fault_active else '-':<5}  "
            f"{tk.phi.phi_spectral:8.4f}  {tk.action.value}"
        )
        if tk.action == Action.RTL and rtl_at is None:
            rtl_at = tk.t
        if tk.action == Action.EMERGENCY_LAND and land_at is None:
            land_at = tk.t

    scored = [tk for tk in ticks if tk.phi is not None]
    print("-" * 72)
    if not scored:
        print("  No Φ scores (increase --steps or reduce --window).")
        return
    phis = [tk.phi.phi_spectral for tk in scored if tk.phi]
    print(f"  Scored ticks: {len(scored)}")
    print(f"  phi_spectral range: {min(phis):.4f} … {max(phis):.4f}")
    print(f"  Final action: {ticks[-1].action.value}")
    if rtl_at is not None:
        print(f"  RTL triggered at t={rtl_at}")
    if land_at is not None:
        print(f"  EMERGENCY_LAND at t={land_at}")
    print()
    print("  Policy: calibrate cruise baseline → drop/spike → RTL/LAND (latched)")
    print("  Note: phi_spectral ≠ Phi_TTH; dual-gate with PhiCS in Sentinel.")
    print("=" * 72)
    print()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="UAV spectral-Φ → RTL demo")
    p.add_argument("--steps", type=int, default=110)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--calib-ticks", type=int, default=28)
    p.add_argument(
        "--fault",
        choices=[f.value for f in Fault],
        default=Fault.MOTOR_DESYNC.value,
    )
    p.add_argument("--fault-start", type=int, default=55)
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args(argv)

    min_start = args.window + args.calib_ticks + 2
    if args.fault_start < min_start:
        print(f"NOTE: raising --fault-start to {min_start} (finish calibrate first)")
        args.fault_start = min_start

    ticks = run_mission(
        steps=args.steps,
        fault=Fault(args.fault),
        fault_start=args.fault_start,
        seed=args.seed,
        window=args.window,
        calib_ticks=args.calib_ticks,
    )
    print_report(ticks, Fault(args.fault), args.fault_start)

    after = [tk for tk in ticks if tk.t >= args.fault_start and tk.phi is not None]
    escalated = any(
        tk.action in (Action.RTL, Action.EMERGENCY_LAND, Action.CONSERVATIVE)
        for tk in after
    )
    if Fault(args.fault) != Fault.NONE and not escalated:
        print("WARN: fault injected but no escalate — try another --seed")
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    raise SystemExit(main())
