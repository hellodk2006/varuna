"""
Bridges the MarineVision GUI to the trained sonar detection pipeline that
lives in ../sonar_model (copied from the SIH-Varuna-Model repo: NLM
denoising -> YOLO11s -> confidence filtering -> JSON + overlay image).

The GUI only ever calls run_detection() - it doesn't need to know that the
underlying pipeline is Ultralytics/NLM/etc. If the model package changes,
only this file should need updating.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from severity import mission_severity

# sonar_model/ sits next to app/, not inside it, and its internal modules
# use bare imports like "from preprocessing.nlm_preprocessing import ..."
# and "from inference.detector import ...", so sonar_model/ itself (not
# app/) needs to be on sys.path.
SONAR_MODEL_DIR = Path(__file__).resolve().parent.parent / "sonar_model"
if str(SONAR_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(SONAR_MODEL_DIR))

DEFAULT_MODEL_PATH = SONAR_MODEL_DIR / "models" / "varuna_final.pt"
DEFAULT_CLASSES_PATH = SONAR_MODEL_DIR / "config" / "classes.json"

# Cache pipelines by (model_path, classes_path) so switching the confidence
# slider doesn't reload the YOLO weights from disk every click.
_pipeline_cache: dict = {}


def _pick_device() -> str:
    """Use a GPU if available, otherwise fall back to CPU.

    The underlying SonarDetector defaults to device=0 (a CUDA GPU), which
    throws on a machine with no GPU - most laptops. Auto-detecting keeps
    the app usable everywhere.
    """
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_pipeline(model_path: str, classes_path: str):
    from inference.inference_pipeline import SonarInferencePipeline

    key = (str(model_path), str(classes_path))
    if key not in _pipeline_cache:
        _pipeline_cache[key] = SonarInferencePipeline(
            model_path=str(model_path),
            classes_path=str(classes_path),
            confidence=0.25,
            iou=0.45,
            device=_pick_device(),
        )
    return _pipeline_cache[key]


# Where annotated overlays and per-run JSON results are written by default.
# Kept centralized (rather than dropped next to the source image) so a demo
# run leaves a clean, reviewable trail of everything the app has detected.
DEFAULT_RESULTS_DIR = SONAR_MODEL_DIR.parent / "results"


def run_detection(
    image_path: str,
    model_path: str = str(DEFAULT_MODEL_PATH),
    classes_path: str = str(DEFAULT_CLASSES_PATH),
    confidence: float = 0.25,
    iou: float = 0.45,
    overlay_dir: Optional[str] = None,
) -> dict:
    """
    Runs the full NLM -> YOLO11s -> confidence-filtering pipeline on one
    image. Returns the same structured dict the repo's CLI produces
    (image size, detections list with bbox/class/confidence, etc.), plus
    'overlay_path' and 'results_json_path' keys pointing at the two files
    this run wrote to disk.
    """
    from inference.visualization import draw_detections
    from inference.inference_pipeline import save_results

    pipeline = get_pipeline(model_path, classes_path)
    # These are read fresh on every predict() call, so updating them here
    # (rather than rebuilding the pipeline) is enough to apply new values.
    pipeline.detector.confidence = confidence
    pipeline.detector.iou = iou

    results = pipeline.run(image_path)

    now = datetime.now()
    results["timestamp"] = now.isoformat(timespec="seconds")
    results["severity"] = mission_severity(results.get("detections", []))

    out_dir = Path(overlay_dir) if overlay_dir else DEFAULT_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp-suffixed filenames (rather than just the image stem) so
    # re-analyzing the same image - e.g. at a different confidence - adds a
    # new history entry instead of silently overwriting the previous run.
    stem = f"{Path(image_path).stem}_{now.strftime('%Y%m%d_%H%M%S')}"

    overlay_path = out_dir / f"{stem}_overlay.png"
    draw_detections(image_path, results, str(overlay_path))
    results["overlay_path"] = str(overlay_path)

    json_path = out_dir / f"{stem}_detection.json"
    save_results(results, str(json_path))
    results["results_json_path"] = str(json_path)

    return results


def refilter_overlay(results: dict, min_confidence_percent: float, output_path: str) -> str:
    """
    Redraws the overlay image keeping only detections at or above
    min_confidence_percent (0-100), WITHOUT re-running the model.

    This only works as a pure filter, not a re-run: the model was already
    called once at the run's original confidence threshold, so detections
    below that threshold were never returned and can't be recovered here.
    A UI slider using this should therefore range from the run's original
    threshold up to 100, not below it.
    """
    from inference.visualization import draw_detections

    filtered = [
        d for d in results.get("detections", [])
        if d.get("confidence_percent", 0) >= min_confidence_percent
    ]
    filtered_results = dict(results)
    filtered_results["detections"] = filtered
    filtered_results["detection_count"] = len(filtered)

    draw_detections(results["image_path"], filtered_results, output_path)
    return output_path
