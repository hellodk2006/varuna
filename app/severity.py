"""
Maps detected object classes to a safety/priority severity level, so the UI,
sound alerts, and PDF reports can all agree on what counts as a "high
alert" vs routine detection.

Severity is intentionally data-driven (sonar_model/config/severity.json)
rather than hardcoded, so the mapping can be tuned without touching UI code.
"""

import json
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
SEVERITY_PATH = APP_DIR.parent / "sonar_model" / "config" / "severity.json"

# level -> (background, text, dot) colors used consistently across the app.
LEVEL_COLORS = {
    "high":   {"bg": "#FDECEC", "text": "#C62828", "solid": "#E53935"},
    "medium": {"bg": "#FFF4E5", "text": "#B15E00", "solid": "#F58A27"},
    "low":    {"bg": "#EAF7F1", "text": "#1E7A57", "solid": "#2E9B72"},
    "none":   {"bg": "#EAF7F1", "text": "#1E7A57", "solid": "#2E9B72"},
}

_LEVEL_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}

_DEFAULT_ENTRY = {"level": "low", "label": "INFO", "reason": "Unclassified object"}


def _load_config() -> dict:
    try:
        with open(SEVERITY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_CONFIG = _load_config()


def class_severity(class_name: str) -> dict:
    """Returns {'level', 'label', 'reason'} for a single detection class."""
    return _CONFIG.get(class_name, _DEFAULT_ENTRY)


def mission_severity(detections: list) -> dict:
    """
    Given a list of detection dicts (each with a 'class_name'), returns the
    single highest severity level found, e.g. one mine_cylinder among ten
    crab_pots still reports as 'high' - the point is not to bury a real
    threat inside a routine-looking summary.
    """
    if not detections:
        return {"level": "none", "label": "CLEAR", "reason": "No objects detected"}

    best = {"level": "none", "label": "CLEAR", "reason": "No objects detected"}
    for d in detections:
        entry = class_severity(d.get("class_name", ""))
        if _LEVEL_RANK.get(entry["level"], 0) >= _LEVEL_RANK.get(best["level"], 0):
            best = entry
    return best


def colors_for(level: str) -> dict:
    return LEVEL_COLORS.get(level, LEVEL_COLORS["low"])
