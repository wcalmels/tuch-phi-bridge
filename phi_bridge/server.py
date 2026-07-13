#!/usr/bin/env python3
"""
HTTP API for spectral Φ — local cloud stub.

  GET  /health
  POST /phi-spectral          one-shot matrix or online push
  POST /phi-spectral/reset    clear a session

Stdlib only (no FastAPI required). Optional cloud later: same routes behind a reverse proxy.

Example:
  py -3 -m phi_bridge.server --port 8787
  curl -s http://127.0.0.1:8787/health
  curl -s -X POST http://127.0.0.1:8787/phi-spectral -H "Content-Type: application/json" ^
    -d "{\"matrix\": [[0.1,0.2,0.3],[0.2,0.2,0.4],[0.15,0.25,0.35],[0.1,0.3,0.3]]}"
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import numpy as np

from .monitor import SpectralPhiMonitor

_SESSIONS: Dict[str, SpectralPhiMonitor] = {}
_LOCK = threading.Lock()
_DEFAULT_WINDOW = 40


def _result_dict(r) -> Dict[str, Any]:
    return {
        "phi_spectral": r.phi_spectral,
        "phi_norm": r.phi_norm,
        "level": r.level_name,
        "alert": r.alert,
        "critical": r.critical,
        "backend": r.backend,
        "n_channels": r.n_channels,
        "window": r.window,
        "note": "phi_spectral ≠ Phi_TTH ≠ PhiCS C_phi",
    }


def _get_session(session_id: str, window: int) -> SpectralPhiMonitor:
    with _LOCK:
        mon = _SESSIONS.get(session_id)
        if mon is None:
            mon = SpectralPhiMonitor(window=window, prefer_consciousai=False)
            _SESSIONS[session_id] = mon
        return mon


def handle_phi_spectral(body: Dict[str, Any]) -> Dict[str, Any]:
    session_id = str(body.get("session_id") or "default")
    window = int(body.get("window") or _DEFAULT_WINDOW)

    if "matrix" in body:
        mat = np.asarray(body["matrix"], dtype=np.float64)
        if mat.ndim != 2:
            return {"ok": False, "error": "matrix must be 2-D (T, N)"}
        mon = SpectralPhiMonitor(window=max(window, mat.shape[0]), prefer_consciousai=False)
        r = mon.score_matrix(mat)
        return {"ok": True, "mode": "matrix", **_result_dict(r)}

    readings = body.get("readings")
    if isinstance(readings, dict) and readings:
        mon = _get_session(session_id, window)
        r = mon.push({k: float(v) for k, v in readings.items()})
        if r is None:
            return {
                "ok": True,
                "mode": "push",
                "session_id": session_id,
                "ready": False,
                "buffer": len(mon._buf),
                "window": mon.window,
                "message": "warming up — need more ticks",
            }
        return {"ok": True, "mode": "push", "session_id": session_id, "ready": True, **_result_dict(r)}

    return {
        "ok": False,
        "error": "Provide 'matrix' [[...]] or 'readings' {channel: value}",
    }


class PhiHandler(BaseHTTPRequestHandler):
    server_version = "TuchPhiBridge/0.1"

    def log_message(self, fmt: str, *args) -> None:  # quieter
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in ("/health", "/"):
            with _LOCK:
                n = len(_SESSIONS)
            self._send(200, {"ok": True, "service": "tuch-phi-bridge", "sessions": n})
            return
        self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/")
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send(400, {"ok": False, "error": "invalid JSON"})
            return

        if path == "/phi-spectral":
            try:
                self._send(200, handle_phi_spectral(body if isinstance(body, dict) else {}))
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"ok": False, "error": str(exc)})
            return

        if path == "/phi-spectral/reset":
            sid = str((body or {}).get("session_id") or "default")
            with _LOCK:
                _SESSIONS.pop(sid, None)
            self._send(200, {"ok": True, "reset": sid})
            return

        self._send(404, {"ok": False, "error": "not found"})


def serve(host: str = "127.0.0.1", port: int = 8787) -> None:
    httpd = ThreadingHTTPServer((host, port), PhiHandler)
    print(f"tuch-phi-bridge API on http://{host}:{port}")
    print("  GET  /health")
    print("  POST /phi-spectral")
    print("  POST /phi-spectral/reset")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        httpd.server_close()


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Spectral Φ HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    args = p.parse_args(argv)
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
