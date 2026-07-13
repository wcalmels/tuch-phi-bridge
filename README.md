# tuch-phi-bridge

Local-first **spectral Φ** adapter for TUCH stacks.

- Upstream math: [consciousai](https://github.com/wcalmels/consciousai) (cloned beside this repo)
- Built-in fallback implements the same formula so consumers always work
- Tags: `phi_spectral` ≠ PhiCS `C_phi` ≠ book `Phi_TTH`

## Layout

```
Escritorio/
  consciousai/        # git clone
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

## Install (editable)

```powershell
cd C:\Users\wcalm\OneDrive\Escritorio\tuch-phi-bridge
py -3 -m pip install -e .
```

Or just add this folder to `PYTHONPATH` / `sys.path` (sentinel & book_agent helpers do that).

## Cloud later

Wrap `SpectralPhiMonitor.score_matrix` / `push` behind `POST /phi-spectral` without changing consumers — see `INTEGRATION_MAP.md`.

## Examples

```powershell
cd C:\Users\wcalm\OneDrive\Escritorio\tuch-phi-bridge
py -3 examples\drone_sim.py
py -3 examples\drone_sim.py --fault gps_dropout --fault-start 40
py -3 examples\drone_sim.py --fault wind_gust --steps 120
```

UAV demo maps `phi_spectral` → **CRUISE / CONSERVATIVE / RTL / EMERGENCY_LAND**.
Simulation only — not flight-certified.