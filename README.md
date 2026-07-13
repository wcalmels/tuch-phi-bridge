# tuch-phi-bridge

Local-first **spectral Φ** adapter for TUCH stacks.

- Upstream math: [consciousai](https://github.com/wcalmels/consciousai)
- Built-in fallback implements the same formula so consumers always work
- Tags: `phi_spectral` ≠ PhiCS `C_phi` ≠ book `Phi_TTH`

## Layout

```
Escritorio/
  consciousai/        # optional sibling
  tuch-phi-bridge/    # this package
  sentinel-edge/      # consumer
  book_agent/         # consumer
```

## Quick start

```python
from phi_bridge import SpectralPhiMonitor
import numpy as np

mon = SpectralPhiMonitor(window=40)
for t in range(50):
    r = mon.push({
        "motor_temp": 60 + np.random.randn(),
        "vibration": 0.5 + 0.1 * np.random.randn(),
        "pressure": 200 + 5 * np.random.randn(),
    })
    if r:
        print(r.phi_spectral, r.level_name, r.alert)
```

One-shot matrix (book / vision signals):

```python
from phi_bridge import phi_from_signal_matrix
import numpy as np

M = np.random.rand(12, 4)  # 12 chapters × 4 channels
phi, level = phi_from_signal_matrix(M)
```

## HTTP API (`POST /phi-spectral`)

```powershell
py -3 -m phi_bridge.server --port 8787
```

```powershell
# health
curl http://127.0.0.1:8787/health

# one-shot matrix
curl -X POST http://127.0.0.1:8787/phi-spectral -H "Content-Type: application/json" `
  -d "{\"matrix\": [[0.1,0.2],[0.2,0.3],[0.15,0.25],[0.2,0.2]]}"

# online push (session)
curl -X POST http://127.0.0.1:8787/phi-spectral -H "Content-Type: application/json" `
  -d "{\"session_id\": \"edge-1\", \"readings\": {\"a\": 1.0, \"b\": 2.0, \"c\": 1.5}}"
```

## PX4 / MAVLink connector

```powershell
# synthetic telemetry (default)
py -3 examples\px4_bridge_demo.py --steps 60

# live SITL / vehicle (needs: pip install pymavlink)
py -3 examples\px4_bridge_demo.py --mode mavlink --connection udp:127.0.0.1:14550

# stream into the HTTP API
py -3 -m phi_bridge.server --port 8787
py -3 examples\px4_bridge_demo.py --api http://127.0.0.1:8787
```

## Install

```powershell
cd C:\Users\wcalm\OneDrive\Escritorio\tuch-phi-bridge
py -3 -m pip install -e .
py -3 -m pip install -e ".[mavlink]"   # optional pymavlink
```

## Examples

```powershell
py -3 examples\drone_sim.py
py -3 examples\fleet_sim.py --n 10
py -3 examples\px4_bridge_demo.py
py -3 scripts\smoke_phi.py
```

| Demo | Policy |
|------|--------|
| Single UAV | CRUISE → RTL → EMERGENCY_LAND |
| Fleet | HOLD → RTL_ALL / LAND_ALL |
| PX4 connector | telemetry → `phi_spectral` (+ optional API) |

Simulation / SITL helpers are **not flight-certified**.

See [INTEGRATION_MAP.md](INTEGRATION_MAP.md) for domains.
