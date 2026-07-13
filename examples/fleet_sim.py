#!/usr/bin/env python3
"""
Fleet simulation — per-drone phi_spectral + collective fleet Φ.

Engineering demo only. Not flight-certified.

Each UAV has its own SpectralPhiMonitor and safety ladder (CRUISE→RTL→LAND).
A fleet monitor reads the vector of per-drone Φ each tick and can escalate
fleet-wide HOLD / RTL_ALL / LAND_ALL when collective integration breaks.

Usage:
  py -3 examples/fleet_sim.py
  py -3 examples/fleet_sim.py --n 10 --seed 11
  py -3 examples/fleet_sim.py --n 8 --steps 100 --fault-start 50
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLES))

from phi_bridge import SpectralPhiMonitor  # noqa: E402

from drone_sim import (  # noqa: E402
    CHANNELS,
    Action,
    Fault,
    decide_action,
    simulate_tick,
)


class FleetAction(str):
    HOLD = "HOLD"          # keep formation / mission
    HOLD_PATTERN = "HOLD_PATTERN"
    RTL_ALL = "RTL_ALL"
    LAND_ALL = "LAND_ALL"


DEFAULT_SCENARIOS: List[Fault] = [
    Fault.NONE,
    Fault.NONE,
    Fault.GPS_DROPOUT,
    Fault.NONE,
    Fault.MOTOR_DESYNC,
    Fault.NONE,
    Fault.WIND_GUST,
    Fault.NONE,
    Fault.LINK_FADE,
    Fault.NONE,
]


@dataclass
class DroneState:
    drone_id: str
    fault: Fault
    mon: SpectralPhiMonitor
    calib: List[float] = field(default_factory=list)
    latched: Action = Action.CRUISE
    streak: int = 0
    last_phi: Optional[float] = None
    last_action: Action = Action.CRUISE
    rtl_at: Optional[int] = None
    land_at: Optional[int] = None


@dataclass
class FleetTick:
    t: int
    drone_phis: Dict[str, float]
    drone_actions: Dict[str, Action]
    fleet_phi: Optional[float]
    fleet_action: str
    n_fault_active: int


def fleet_decide(
    fleet_phi: float,
    *,
    calib: List[float],
    latched: str,
    armed: bool,
    streak: int,
    distress_frac: float,
) -> Tuple[str, int]:
    """Collective policy from fleet Φ + fraction of drones already RTL/LAND."""
    if latched in (FleetAction.RTL_ALL, FleetAction.LAND_ALL):
        return latched, streak
    if not armed or len(calib) < 8:
        return FleetAction.HOLD, 0

    # Many vehicles already aborting → pull the rest
    if distress_frac >= 0.50:
        return FleetAction.LAND_ALL, streak
    if distress_frac >= 0.30:
        return FleetAction.RTL_ALL, streak

    med = float(np.median(calib))
    iqr = float(np.percentile(calib, 75) - np.percentile(calib, 25)) + 1e-9
    thr_hold = float(np.percentile(calib, 15))
    thr_rtl = float(np.percentile(calib, 6))
    thr_land = float(np.percentile(calib, 2))

    up = fleet_phi > med + 4.5 * iqr
    land_hit = fleet_phi < thr_land or fleet_phi > med + 7.0 * iqr
    rtl_hit = fleet_phi < thr_rtl or up
    soft = fleet_phi < thr_hold

    if land_hit:
        streak += 1
        if streak >= 3:
            return FleetAction.LAND_ALL, streak
        if streak >= 2:
            return FleetAction.RTL_ALL, streak
        return FleetAction.HOLD_PATTERN, streak
    if rtl_hit:
        streak += 1
        if streak >= 2:
            return FleetAction.RTL_ALL, streak
        return FleetAction.HOLD_PATTERN, streak
    if soft:
        return FleetAction.HOLD_PATTERN, 0
    return FleetAction.HOLD, 0


def run_fleet(
    *,
    n: int,
    steps: int,
    fault_start: int,
    seed: int,
    window: int,
) -> Tuple[List[DroneState], List[FleetTick]]:
    rng = np.random.default_rng(seed)
    scenarios = (DEFAULT_SCENARIOS * ((n // len(DEFAULT_SCENARIOS)) + 1))[:n]

    drones = [
        DroneState(
            drone_id=f"UAV-{i+1:02d}",
            fault=scenarios[i],
            mon=SpectralPhiMonitor(
                window=window,
                channel_order=CHANNELS,
                history_len=max(120, steps),
                prefer_consciousai=False,
            ),
        )
        for i in range(n)
    ]

    fleet_mon = SpectralPhiMonitor(
        window=max(12, window // 2),
        channel_order=[d.drone_id for d in drones],
        history_len=max(120, steps),
        prefer_consciousai=False,
    )
    fleet_calib: List[float] = []
    fleet_latched = FleetAction.HOLD
    fleet_streak = 0
    history: List[FleetTick] = []

    for t in range(steps):
        drone_phis: Dict[str, float] = {}
        drone_actions: Dict[str, Action] = {}
        n_fault = 0

        for i, d in enumerate(drones):
            # Independent noise stream per drone
            d_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) ^ (i * 10007 + t))
            readings, active = simulate_tick(
                t, rng=d_rng, fault=d.fault, fault_start=fault_start,
            )
            if active:
                n_fault += 1
            phi_r = d.mon.push(readings)
            if phi_r is None:
                drone_actions[d.drone_id] = d.latched
                continue

            armed = t >= fault_start
            if t < fault_start:
                d.calib.append(phi_r.phi_spectral)
                action, d.streak = Action.CRUISE, 0
            elif d.fault == Fault.NONE:
                # Healthy UAVs stay in CRUISE until fleet-level abort
                action, d.streak = Action.CRUISE, 0
            else:
                action, d.streak = decide_action(
                    phi_r.phi_spectral,
                    calib=d.calib,
                    latched=d.latched,
                    armed=True,
                    streak=d.streak,
                )
            d.latched = action
            d.last_phi = phi_r.phi_spectral
            d.last_action = action
            drone_phis[d.drone_id] = phi_r.phi_spectral
            drone_actions[d.drone_id] = action
            if action == Action.RTL and d.rtl_at is None:
                d.rtl_at = t
            if action == Action.EMERGENCY_LAND and d.land_at is None:
                d.land_at = t

        fleet_phi_val: Optional[float] = None
        if len(drone_phis) == n:
            # Fill missing with last known / median proxy
            snapshot = {d.drone_id: float(drone_phis.get(d.drone_id, d.last_phi or 0.3)) for d in drones}
            fr = fleet_mon.push(snapshot)
            if fr is not None:
                fleet_phi_val = fr.phi_spectral
                if t < fault_start:
                    fleet_calib.append(fleet_phi_val)
                    fleet_action, fleet_streak = FleetAction.HOLD, 0
                else:
                    distress = sum(
                        1
                        for a in drone_actions.values()
                        if a in (Action.RTL, Action.EMERGENCY_LAND)
                    ) / max(n, 1)
                    fleet_action, fleet_streak = fleet_decide(
                        fleet_phi_val,
                        calib=fleet_calib,
                        latched=fleet_latched,
                        armed=True,
                        streak=fleet_streak,
                        distress_frac=distress,
                    )
                    fleet_latched = fleet_action
            else:
                fleet_action = fleet_latched
        else:
            fleet_action = fleet_latched

        # Propagate fleet RTL/LAND to still-cruising drones (mission abort)
        if fleet_action == FleetAction.LAND_ALL:
            for d in drones:
                if d.latched not in (Action.EMERGENCY_LAND,):
                    d.latched = Action.EMERGENCY_LAND
                    d.last_action = Action.EMERGENCY_LAND
                    drone_actions[d.drone_id] = Action.EMERGENCY_LAND
                    if d.land_at is None:
                        d.land_at = t
        elif fleet_action == FleetAction.RTL_ALL:
            for d in drones:
                if d.latched == Action.CRUISE or d.latched == Action.CONSERVATIVE:
                    d.latched = Action.RTL
                    d.last_action = Action.RTL
                    drone_actions[d.drone_id] = Action.RTL
                    if d.rtl_at is None:
                        d.rtl_at = t

        history.append(
            FleetTick(
                t=t,
                drone_phis=dict(drone_phis),
                drone_actions=dict(drone_actions),
                fleet_phi=fleet_phi_val,
                fleet_action=fleet_action,
                n_fault_active=n_fault,
            )
        )

    return drones, history


def print_report(
    drones: List[DroneState],
    history: List[FleetTick],
    *,
    fault_start: int,
) -> None:
    print()
    print("=" * 78)
    print("  TUCH phi-bridge — UAV FLEET spectral-Φ demo")
    print("  (simulation · not flight-certified)")
    print("=" * 78)
    print(f"  Fleet size: {len(drones)}   fault_start: t={fault_start}")
    print()
    print(f"{'ID':<8} {'assigned_fault':<14} {'final_action':<16} {'phi_last':>8}  event")
    print("-" * 78)
    for d in drones:
        ev = []
        if d.rtl_at is not None:
            ev.append(f"RTL@{d.rtl_at}")
        if d.land_at is not None:
            ev.append(f"LAND@{d.land_at}")
        print(
            f"{d.drone_id:<8} {d.fault.value:<14} {d.last_action.value:<16} "
            f"{(d.last_phi or 0):8.4f}  {', '.join(ev) or '-'}"
        )

    print()
    print(f"{'t':>4}  {'faults':>6}  {'fleet_phi':>10}  {'fleet_action':<14}  distress")
    print("-" * 78)
    last_fa = None
    for tk in history:
        if tk.fleet_phi is None:
            continue
        distress = sum(
            1
            for a in tk.drone_actions.values()
            if a in (Action.RTL, Action.EMERGENCY_LAND)
        )
        changed = tk.fleet_action != last_fa
        last_fa = tk.fleet_action
        if not changed and tk.t % 15 != 0 and tk.t < fault_start:
            continue
        if not changed and tk.t % 10 != 0 and tk.t >= fault_start:
            continue
        print(
            f"{tk.t:4d}  {tk.n_fault_active:6d}  {tk.fleet_phi:10.4f}  "
            f"{tk.fleet_action:<14}  {distress}/{len(drones)}"
        )

    scored = [tk for tk in history if tk.fleet_phi is not None]
    print("-" * 78)
    if scored:
        fps = [tk.fleet_phi for tk in scored if tk.fleet_phi is not None]
        print(f"  Fleet Φ range: {min(fps):.4f} … {max(fps):.4f}")
        print(f"  Final fleet action: {history[-1].fleet_action}")
    n_abort = sum(
        1 for d in drones if d.last_action in (Action.RTL, Action.EMERGENCY_LAND)
    )
    n_faulted = sum(1 for d in drones if d.fault != Fault.NONE)
    print(f"  Drones with injected fault: {n_faulted}/{len(drones)}")
    print(f"  Drones aborted (RTL/LAND):  {n_abort}/{len(drones)}")
    print()
    print("  Per-drone: calibrate → relative Φ policy")
    print("  Fleet: Φ of {drone Φ channels} + distress fraction → HOLD/RTL_ALL/LAND_ALL")
    print("  Note: phi_spectral ≠ Phi_TTH")
    print("=" * 78)
    print()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="UAV fleet spectral-Φ demo")
    p.add_argument("--n", type=int, default=10, help="Fleet size")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--window", type=int, default=18)
    p.add_argument("--fault-start", type=int, default=50)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args(argv)

    min_start = args.window + 12
    if args.fault_start < min_start:
        print(f"NOTE: raising --fault-start to {min_start}")
        args.fault_start = min_start

    # Import path handled at module load
    drones, history = run_fleet(
        n=args.n,
        steps=args.steps,
        fault_start=args.fault_start,
        seed=args.seed,
        window=args.window,
    )
    print_report(drones, history, fault_start=args.fault_start)
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    raise SystemExit(main())
