"""
Runs the trained sonar detector for the API server.

Pipeline (from ../sonar_model):  NLM denoising -> YOLO11s -> confidence filter

- The model is loaded ONCE and reused for every request.
- A lock makes sure only one image is processed at a time (the pipeline keeps
  its confidence setting on a shared object, so two requests at once could mix
  their settings up).
- Only this file touches sonar_model, the same idea as app/sonar_integration.py.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
SONAR_MODEL_DIR = SERVER_DIR.parent / "sonar_model"

# sonar_model's own files use imports like "from inference.detector import ...",
# so the sonar_model folder itself has to be on Python's search path.
if str(SONAR_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(SONAR_MODEL_DIR))

MODEL_PATH = Path(os.getenv(
    "SONAR_MODEL_PATH",
    SONAR_MODEL_DIR / "models" / "varuna_final.pt",
))
CLASSES_PATH = SONAR_MODEL_DIR / "config" / "classes.json"

UPLOADS_DIR = SERVER_DIR / "uploads"   # images received from clients
RESULTS_DIR = SERVER_DIR / "results"   # annotated overlay images + JSON

_lock = threading.RLock()
_pipeline = None
_device = None


def _pick_device():
    """GPU if there is one, otherwise CPU (the model code defaults to GPU 0,
    which crashes on laptops without one)."""
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_model():
    """Load the model the first time it is needed, then reuse it."""
    global _pipeline, _device
    with _lock:
        if _pipeline is None:
            from inference.inference_pipeline import SonarInferencePipeline
            _device = _pick_device()
            _pipeline = SonarInferencePipeline(
                model_path=str(MODEL_PATH),
                classes_path=str(CLASSES_PATH),
                confidence=0.25,
                iou=0.45,
                device=_device,
            )
    return _pipeline


def is_loaded() -> bool:
    return _pipeline is not None


def device_name() -> str:
    return "not loaded" if _device is None else ("GPU" if _device != "cpu" else "CPU")


def detect(image_path: str, confidence: float = 0.25, iou: float = 0.45) -> dict:
    """Run the full pipeline on one image file.

    Returns the pipeline's result dict plus 'overlay_file': the name of the
    annotated picture saved in RESULTS_DIR.
    """
    from inference.visualization import draw_detections
    from inference.inference_pipeline import save_results

    pipeline = load_model()

    with _lock:  # one image at a time
        pipeline.detector.confidence = confidence
        pipeline.detector.iou = iou
        results = pipeline.run(image_path)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    overlay_name = f"{stem}_overlay.png"
    draw_detections(image_path, results, str(RESULTS_DIR / overlay_name))
    results["overlay_file"] = overlay_name

    save_results(results, str(RESULTS_DIR / f"{stem}_detection.json"))
    return results
