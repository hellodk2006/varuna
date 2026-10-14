
import json
from pathlib import Path

import cv2

from preprocessing.nlm_preprocessing import nlm_preprocess
from inference.detector import SonarDetector
from inference.visualization import draw_detections

DEFAULT_CLASSES = {
    0: "crab_pot",
    1: "submarine_pipeline",
    2: "shipwreck",
    3: "ghost_net",
    4: "mine_cylinder",
}


class SonarInferencePipeline:
    """
    Complete inference pipeline:

    Input sonar image
        ↓
    NLM preprocessing
        ↓
    YOLO detection
        ↓
    Confidence filtering
        ↓
    Structured detection output
    """

    def __init__(
        self,
        model_path: str,
        classes_path: str = "config/classes.json",
        confidence: float = 0.25,
        iou: float = 0.45,
        device: int | str = 0,
    ):
        self.classes = self._load_classes(classes_path)

        self.detector = SonarDetector(
            model_path=model_path,
            confidence=confidence,
            iou=iou,
            device=device,
        )

    @staticmethod
    def _load_classes(classes_path: str) -> dict:
        path = Path(classes_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Classes file not found: {path}"
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {int(k): v for k, v in data.items()}

    def run(self, image_path: str) -> dict:
        """
        Run the complete inference pipeline.

        Parameters
        ----------
        image_path : str
            Path to the sonar image.

        Returns
        -------
        dict
            Structured detection results.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Input image not found: {image_path}"
            )

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        original_height, original_width = image.shape[:2]

        # Apply the same NLM preprocessing used during training.
        processed_image = nlm_preprocess(image)

        # Run YOLO.
        results = self.detector.predict(processed_image)

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())

                xyxy = boxes.xyxy[i].tolist()

                x1, y1, x2, y2 = [
                    round(float(value), 2)
                    for value in xyxy
                ]

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": self.classes.get(
                            class_id,
                            f"class_{class_id}"
                        ),
                        "confidence": round(confidence, 4),
                        "confidence_percent": round(
                            confidence * 100,
                            2
                        ),
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "width": round(x2 - x1, 2),
                            "height": round(y2 - y1, 2),
                        },
                    }
                )

        return {
            "image": image_path.name,
            "image_path": str(image_path),
            "image_width": original_width,
            "image_height": original_height,
            "preprocessing": "NLM",
            "model": str(self.detector.model_path),
            "confidence_threshold": self.detector.confidence,
            "iou_threshold": self.detector.iou,
            "detections": detections,
            "detection_count": len(detections),
        }


def save_results(results: dict, output_path: str) -> None:
    """
    Save inference results as JSON.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=4,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run underwater sonar object detection."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input sonar image.",
    )

    parser.add_argument(
        "--model",
        default="models/yolo11s_baseline_640_best.pt",
        help="Path to YOLO model.",
    )

    parser.add_argument(
        "--classes",
        default="config/classes.json",
        help="Path to class configuration.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold.",
    )

    parser.add_argument(
        "--device",
        default="0",
        help="Inference device, e.g. 0 or cpu.",
    )

    parser.add_argument(
        "--output",
        default="examples/sample_output/detection.json",
        help="Output JSON path.",
    )

    args = parser.parse_args()

    pipeline = SonarInferencePipeline(
        model_path=args.model,
        classes_path=args.classes,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
    )

    results = pipeline.run(args.image)

    save_results(
        results,
        args.output,
    )

    overlay_output = (
        Path(args.output).parent
        / f"{Path(args.image).stem}_overlay.png"
    )

    draw_detections(
        args.image,
        results,
        str(overlay_output),
    )

    print(json.dumps(results, indent=4))
    print(f"\nResults saved to: {args.output}")
    print(f"Overlay saved to: {overlay_output}")