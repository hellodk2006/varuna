"""
VARUNA - Mission Map.

Replaces the earlier static placeholder with a real Leaflet map. Reads
straight from the results/ folder every other part of the app already
writes to - no separate store to keep in sync.

Two kinds of geodata get plotted:

1. Single-image analyses that were manually tagged with a device/location
   lat & lon on the Analyze page (see AnalyzePage's Location fields) -
   plotted as one pin per image.
2. XTF missions (results/xtf/<mission>/mission_results.json) - plotted as
   the ship's actual surveyed track (real navigation data pulled straight
   from the XTF file by xtf_mission.XTFMissionReader.full_track()), with a
   start marker, an end marker, and one pin per accepted detection along
   the route.

Leaflet + its tiles are loaded from a CDN inside the embedded page, so
this needs internet access to actually render - there is no offline tile
cache bundled here.
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QUrl, Qt
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QSizePolicy,
)

from sonar_integration import DEFAULT_RESULTS_DIR
from marine_app import PageShell, label, make_button, MUTED, BORDER

XTF_RESULTS_DIR = DEFAULT_RESULTS_DIR / "xtf"

_SEVERITY_COLOR = {"high": "#C0392B", "medium": "#D4860A", "low": "#2471A3", "none": "#2E9B72"}


def _load_image_pins() -> list[dict]:
    """Single-image results/*.json files carrying a manually-entered
    device location (see AnalyzePage's Location fields)."""
    pins = []
    if not DEFAULT_RESULTS_DIR.exists():
        return pins
    for p in DEFAULT_RESULTS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        lat, lon = data.get("latitude"), data.get("longitude")
        if lat is None or lon is None:
            continue
        pins.append({
            "file": data.get("image", p.stem),
            "lat": lat,
            "lon": lon,
            "detections": data.get("detection_count", len(data.get("detections", []))),
            "severity": (data.get("severity") or {}).get("level", "none"),
            "timestamp": data.get("timestamp", ""),
        })
    return pins


def _load_xtf_missions() -> list[dict]:
    """Every completed XTF mission under results/xtf/, each carrying its
    own journey_start / journey_end / journey_track (from run_xtf_mission)
    plus its geotagged detections."""
    missions = []
    if not XTF_RESULTS_DIR.exists():
        return missions
    for mission_dir in sorted(XTF_RESULTS_DIR.iterdir()):
        json_path = mission_dir / "mission_results.json"
        if not json_path.exists():
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "journey_start" not in data:
            continue  # a run from before journey tracking was added
        missions.append(data)
    return missions


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _build_html(image_pins: list[dict], missions: list[dict], selected_mission: str | None) -> str:
    markers_js = []
    all_coords: list[tuple[float, float]] = []

    for pin in image_pins:
        color = _SEVERITY_COLOR.get(pin["severity"], "#2471A3")
        popup = _esc(
            f"<b>{pin['file']}</b><br>"
            f"Detections: {pin['detections']}<br>"
            f"Severity: {pin['severity'].upper()}<br>"
            f"{pin['timestamp']}"
        )
        markers_js.append(f"""
            L.circleMarker([{pin['lat']}, {pin['lon']}], {{
                radius: 8, fillColor: '{color}', color: '#fff', weight: 2, fillOpacity: 0.9
            }}).addTo(map).bindPopup('{popup}');
        """)
        all_coords.append((pin["lat"], pin["lon"]))

    for mission in missions:
        if selected_mission and mission.get("mission") != selected_mission:
            continue

        mission_name = _esc(mission.get("mission", "XTF mission"))

        track = mission.get("journey_track", [])
        if track:
            coords = [[t["latitude"], t["longitude"]] for t in track]
            markers_js.append(
                f"L.polyline({json.dumps(coords)}, "
                f"{{color:'#4A4742', weight:3, opacity:0.85}}).addTo(map);"
            )
            all_coords.extend((c[0], c[1]) for c in coords)

        start = mission.get("journey_start")
        if start:
            markers_js.append(f"""
                L.circleMarker([{start['latitude']}, {start['longitude']}], {{
                    radius: 9, fillColor: '#2E9B72', color:'#fff', weight:2, fillOpacity:0.95
                }}).addTo(map).bindPopup('<b>Journey start</b><br>{mission_name}');
            """)
            all_coords.append((start["latitude"], start["longitude"]))

        end = mission.get("journey_end")
        if end:
            markers_js.append(f"""
                L.circleMarker([{end['latitude']}, {end['longitude']}], {{
                    radius: 9, fillColor: '#C0392B', color:'#fff', weight:2, fillOpacity:0.95
                }}).addTo(map).bindPopup('<b>Journey end</b><br>{mission_name}');
            """)
            all_coords.append((end["latitude"], end["longitude"]))

        for det in mission.get("detections", []):
            lat, lon = det.get("latitude"), det.get("longitude")
            if lat is None or lon is None:
                continue
            popup = _esc(
                f"<b>{det.get('class_name', 'unknown')}</b><br>"
                f"{float(det.get('confidence', 0)) * 100:.1f}% confidence"
            )
            markers_js.append(f"""
                L.circleMarker([{lat}, {lon}], {{
                    radius: 6, fillColor:'#7A2020', color:'#fff', weight:1.5, fillOpacity:0.9
                }}).addTo(map).bindPopup('{popup}');
            """)
            all_coords.append((lat, lon))

    if all_coords:
        center_lat = sum(c[0] for c in all_coords) / len(all_coords)
        center_lon = sum(c[1] for c in all_coords) / len(all_coords)
        zoom = 13 if len(all_coords) > 1 else 15
    else:
        # Falls back to roughly the Konkan coast / Arabian Sea so the map
        # isn't centered on (0,0) when nothing has been geotagged yet.
        center_lat, center_lon, zoom = 15.4, 73.7, 6

    markers_block = "\n".join(markers_js)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{ width:100%; height:100%; }}
  #map {{ width:100%; height:100%; }}
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
</head><body>
<div id="map"></div>
<script>
var map = L.map('map', {{ center: [{center_lat}, {center_lon}], zoom: {zoom} }});
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '&copy; OpenStreetMap contributors', maxZoom: 19
}}).addTo(map);
{markers_block}
</script>
</body></html>"""


class MapPage(PageShell):
    """Real mission map: single-image device pins + XTF ship tracks with
    start/end markers and per-detection pins, all read from results/."""

    def __init__(self):
        super().__init__(
            "Mission Intelligence",
            "Mission Map",
            "Device locations and XTF ship tracks, plotted from every saved analysis."
        )

        toolbar = QHBoxLayout()
        toolbar.addWidget(label("XTF mission", 12, MUTED, True))
        self.mission_combo = QComboBox()
        self.mission_combo.setFixedHeight(38)
        self.mission_combo.setMinimumWidth(260)
        self.mission_combo.currentIndexChanged.connect(self._on_mission_changed)
        toolbar.addWidget(self.mission_combo)
        toolbar.addStretch(1)

        refresh_btn = make_button("Refresh", "secondary", 38)
        refresh_btn.clicked.connect(self.refresh)
        toolbar.addWidget(refresh_btn)
        self.body.addLayout(toolbar)

        self._webview = QWebEngineView()
        self._webview.setMinimumHeight(500)
        self._webview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._webview.setStyleSheet(f"border:1px solid {BORDER}; border-radius:10px;")
        self.body.addWidget(self._webview)

        self._missions: list[dict] = []
        self.refresh()

    def refresh(self):
        """Re-scan results/ and results/xtf/ and rebuild the map. Called
        on startup and whenever the user navigates to this tab, so a
        mission analyzed while on another page still shows up here."""
        self._missions = _load_xtf_missions()

        current = self.mission_combo.currentData()
        self.mission_combo.blockSignals(True)
        self.mission_combo.clear()
        self.mission_combo.addItem("All missions", None)
        for m in self._missions:
            self.mission_combo.addItem(m.get("mission", "XTF mission"), m.get("mission"))
        idx = self.mission_combo.findData(current)
        self.mission_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.mission_combo.blockSignals(False)

        self._reload_map()

    def _on_mission_changed(self, _index):
        self._reload_map()

    def _reload_map(self):
        selected = self.mission_combo.currentData()
        html = _build_html(_load_image_pins(), self._missions, selected)
        self._webview.setHtml(html, QUrl("about:blank"))
