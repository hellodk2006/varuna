"""
Best-effort background sync to the Docker backend (server/).

The app's detection flow is unchanged and works with no server at all -
every run is still saved locally to results/ (used by the Reports page and
PDF export). This module ADDITIONALLY tries to upload a copy of each result
to the Dockerized server, so it lands in the shared Postgres database too.

Design choices:
  - Runs on its own QThread so it can never freeze the UI.
  - Silent on failure (server not running, Docker not started, no network) -
    the app must work exactly the same with the backend off.
  - Fire-and-forget: nothing else in the app waits on this.

Point the app at a different backend with the VARUNA_API environment
variable, e.g.  $env:VARUNA_API="http://192.168.1.5:8000"
or an ngrok URL for a demo across machines/networks:
  $env:VARUNA_API="https://your-tunnel.ngrok-free.dev"
"""
from __future__ import annotations

import os
from pathlib import Path

import requests
from PySide6.QtCore import QThread, Signal

DEFAULT_API_BASE = "http://127.0.0.1:8000"
API_BASE = os.getenv("VARUNA_API", DEFAULT_API_BASE).rstrip("/")

# A shared session so every request carries this header automatically.
# Free ngrok URLs (*.ngrok-free.app / *.ngrok-free.dev) show an HTML
# "you're about to visit..." interstitial page to any client that doesn't
# send it - including requests, not just browsers. Without this, every
# call below would get that HTML page back instead of JSON and fail to
# parse. Harmless against a plain localhost/LAN server too.
_session = requests.Session()
_session.headers.update({"ngrok-skip-browser-warning": "true"})


def backend_available(timeout: float = 1.0) -> bool:
    try:
        r = _session.get(f"{API_BASE}/health", timeout=timeout)
        return bool(r.ok and r.json().get("model_loaded") is True)
    except (requests.exceptions.RequestException, ValueError):
        return False


class SyncWorker(QThread):
    """Uploads one already-completed local result to the Docker backend.

    Note: this sends the ORIGINAL image and lets the backend re-run
    detection, rather than uploading the already-computed result, so the
    two entries (local JSON, database row) are each produced by their own
    full pipeline run - simple and consistent, at the cost of running
    detection twice. For a typical single sonar image this is a small
    price for a shared history that survives a restart.
    """
    done = Signal(dict)   # {"saved": bool, "id": int|None, "reason": str}

    def __init__(self, image_path: str, confidence: float, mission: str = "Mission Alpha"):
        super().__init__()
        self.image_path = image_path
        self.confidence = confidence
        self.mission = mission

    def run(self):
        try:
            with open(self.image_path, "rb") as f:
                r = _session.post(
                    f"{API_BASE}/upload",
                    files={"file": (Path(self.image_path).name, f)},
                    data={"confidence_threshold": self.confidence, "mission": self.mission},
                    timeout=(2, 180),
                )
        except requests.exceptions.RequestException as e:
            self.done.emit({"saved": False, "id": None, "reason": f"backend unreachable ({type(e).__name__})"})
            return

        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            self.done.emit({"saved": False, "id": None, "reason": detail})
            return

        data = r.json()
        self.done.emit({"saved": bool(data.get("saved")), "id": data.get("id"), "reason": ""})
