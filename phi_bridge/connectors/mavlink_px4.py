#!/usr/bin/env python3
"""
PX4 / MAVLink telemetry → channel dict for SpectralPhiMonitor.

Modes:
  sim   — synthetic MAVLink-like channels (no hardware)
  mavlink — live UDP/TCP via pymavlink (optional extra)

Mapped channels align with examples/drone_sim.py for policy reuse.
Not flight-certified.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, Optional

import numpy as np

# Same names as drone_sim for drop-in safety demos
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


class SourceMode(str, Enum):
    SIM = "sim"
    MAVLINK = "mavlink"


@dataclass
class TelemetryFrame:
    t: float
    readings: Dict[str, float]
    source: str
    raw: Dict[str, float] = field(default_factory=dict)


@dataclass
class Px4Connector:
    """
    Stream of UAV channel dicts.

    sim: always works.
    mavlink: requires `pip install pymavlink` and a vehicle/SITL on connection.
    """

    mode: SourceMode = SourceMode.SIM
    connection: str = "udp:127.0.0.1:14550"
    seed: int = 0
    poll_hz: float = 10.0
    _rng: np.random.Generator = field(init=False, repr=False)
    _t0: float = field(init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)
    _master: object = field(default=None, init=False, repr=False)
    _last: Dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = SourceMode(self.mode)
        self._rng = np.random.default_rng(self.seed)
        self._t0 = time.time()
        self._last = {c: 0.0 for c in CHANNELS}
        self._last.update(
            {
                "alt_m": 42.0,
                "imu_az": 9.81,
                "motor_fl": 0.55,
                "motor_fr": 0.55,
                "motor_rl": 0.55,
                "motor_rr": 0.55,
                "battery_v": 15.8,
                "link_rssi": -62.0,
            }
        )

    def connect(self) -> "Px4Connector":
        if self.mode == SourceMode.SIM:
            return self
        try:
            from pymavlink import mavutil  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pymavlink required for mode=mavlink — pip install pymavlink"
            ) from exc
        self._master = mavutil.mavlink_connection(self.connection)
        self._master.wait_heartbeat(timeout=10)
        return self

    def close(self) -> None:
        master = self._master
        self._master = None
        if master is not None and hasattr(master, "close"):
            try:
                master.close()
            except Exception:
                pass

    def __enter__(self) -> "Px4Connector":
        return self.connect()

    def __exit__(self, *_) -> None:
        self.close()

    def read(self) -> TelemetryFrame:
        if self.mode == SourceMode.SIM:
            return self._read_sim()
        return self._read_mavlink()

    def stream(self, steps: Optional[int] = None) -> Iterator[TelemetryFrame]:
        n = 0
        dt = 1.0 / max(self.poll_hz, 0.1)
        while steps is None or n < steps:
            yield self.read()
            n += 1
            time.sleep(dt)

    def _read_sim(self) -> TelemetryFrame:
        self._step += 1
        t = self._step
        cruise = 0.02 * math.sin(t / 18.0)
        r = {
            "gps_lat_rate": float(cruise + self._rng.normal(0, 0.01)),
            "gps_lon_rate": float(0.015 * math.cos(t / 22.0) + self._rng.normal(0, 0.01)),
            "alt_m": float(42.0 + 0.4 * math.sin(t / 30.0) + self._rng.normal(0, 0.05)),
            "imu_ax": float(self._rng.normal(0, 0.03)),
            "imu_ay": float(self._rng.normal(0, 0.03)),
            "imu_az": float(9.81 + self._rng.normal(0, 0.04)),
            "gyro_yaw": float(0.05 * math.sin(t / 40.0) + self._rng.normal(0, 0.01)),
            "motor_fl": float(0.55 + cruise + self._rng.normal(0, 0.01)),
            "motor_fr": float(0.55 + cruise + self._rng.normal(0, 0.01)),
            "motor_rl": float(0.55 + cruise + self._rng.normal(0, 0.01)),
            "motor_rr": float(0.55 + cruise + self._rng.normal(0, 0.01)),
            "battery_v": float(15.8 - 0.004 * t + self._rng.normal(0, 0.01)),
            "link_rssi": float(-62.0 + self._rng.normal(0, 0.8)),
        }
        self._last = r
        return TelemetryFrame(t=time.time() - self._t0, readings=r, source="sim")

    def _read_mavlink(self) -> TelemetryFrame:
        """Non-blocking drain of available MAVLink messages into CHANNELS."""
        master = self._master
        assert master is not None
        deadline = time.time() + (1.0 / max(self.poll_hz, 0.1))
        raw: Dict[str, float] = {}
        while time.time() < deadline:
            msg = master.recv_match(blocking=False)
            if msg is None:
                break
            mtype = msg.get_type()
            if mtype == "BAD_DATA":
                continue
            if mtype == "ATTITUDE":
                raw["roll"] = float(msg.roll)
                raw["pitch"] = float(msg.pitch)
                raw["yaw"] = float(msg.yaw)
                raw["yawspeed"] = float(msg.yawspeed)
            elif mtype == "HIGHRES_IMU":
                raw["ax"] = float(msg.xacc)
                raw["ay"] = float(msg.yacc)
                raw["az"] = float(msg.zacc)
            elif mtype == "RAW_IMU":
                # millig → rough m/s^2
                raw["ax"] = float(msg.xacc) * 9.81 / 1000.0
                raw["ay"] = float(msg.yacc) * 9.81 / 1000.0
                raw["az"] = float(msg.zacc) * 9.81 / 1000.0
            elif mtype == "GLOBAL_POSITION_INT":
                raw["alt"] = float(msg.relative_alt) / 1000.0
                raw["vx"] = float(msg.vx) / 100.0
                raw["vy"] = float(msg.vy) / 100.0
            elif mtype == "SYS_STATUS":
                raw["battery_v"] = float(msg.voltage_battery) / 1000.0
            elif mtype == "RADIO_STATUS":
                raw["rssi"] = float(msg.rssi)
            elif mtype == "SERVO_OUTPUT_RAW":
                # Normalize PWM ~1000-2000 → 0-1
                for name, attr in (
                    ("m1", "servo1_raw"),
                    ("m2", "servo2_raw"),
                    ("m3", "servo3_raw"),
                    ("m4", "servo4_raw"),
                ):
                    pwm = float(getattr(msg, attr, 1500))
                    raw[name] = max(0.0, min(1.0, (pwm - 1000.0) / 1000.0))

        r = dict(self._last)
        if "vx" in raw:
            r["gps_lat_rate"] = raw["vx"]
        if "vy" in raw:
            r["gps_lon_rate"] = raw["vy"]
        if "alt" in raw:
            r["alt_m"] = raw["alt"]
        if "ax" in raw:
            r["imu_ax"] = raw["ax"]
        if "ay" in raw:
            r["imu_ay"] = raw["ay"]
        if "az" in raw:
            r["imu_az"] = raw["az"]
        if "yawspeed" in raw:
            r["gyro_yaw"] = raw["yawspeed"]
        if "m1" in raw:
            r["motor_fl"] = raw["m1"]
            r["motor_fr"] = raw.get("m2", raw["m1"])
            r["motor_rl"] = raw.get("m3", raw["m1"])
            r["motor_rr"] = raw.get("m4", raw["m1"])
        if "battery_v" in raw and raw["battery_v"] > 0:
            r["battery_v"] = raw["battery_v"]
        if "rssi" in raw:
            # 0-255 → rough dBm-like
            r["link_rssi"] = -100.0 + 0.15 * raw["rssi"]

        self._last = r
        return TelemetryFrame(
            t=time.time() - self._t0,
            readings=r,
            source="mavlink",
            raw=raw,
        )
