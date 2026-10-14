import cv2
from pathlib import Path

# BGR (OpenCV order) colors per severity level, kept in sync with
# app/severity.py's LEVEL_COLORS (which uses the same palette in RGB/hex
# for the Qt UI and PDF report, so a mine detection reads the same red
# everywhere: overlay image, alert banner, and report table).
_SEVERITY_BGR = {
    "high": (53, 57, 229),    # #E53935
    "medium": (39, 138, 245),  # #F58A27
    "low": (114, 155, 46),     # #2E9B72
    "none": (114, 155, 46),
}


def _severity_lookup():
    """Best-effort import of the severity map. Falls back to plain green
    if app/severity.py isn't importable (e.g. this module used standalone
    outside the MarineVision app), so this file has no hard dependency
    on the app/ package."""
    try:
        from severity import class_severity
        return class_severity
    except Exception:
        return None


def draw_detections(
    image_path: str,
    results: dict,
    output_path: str,
) -> None:
    """
    Draw YOLO detections on the original sonar image. Boxes are
    color-coded by severity (red for high-priority classes like
    mine_cylinder, orange for medium, green for routine/low) when the
    severity config is available, and plain green otherwise.

    Parameters
    ----------
    image_path : str
        Path to the original sonar image.

    results : dict
        Detection results returned by SonarInferencePipeline.

    output_path : str
        Path where the annotated image will be saved.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    class_severity = _severity_lookup()

    for detection in results.get("detections", []):
        bbox = detection["bbox"]

        x1 = int(round(bbox["x1"]))
        y1 = int(round(bbox["y1"]))
        x2 = int(round(bbox["x2"]))
        y2 = int(round(bbox["y2"]))

        class_name = detection["class_name"]
        confidence = detection["confidence_percent"]

        label = f"{class_name} {confidence:.1f}%"

        level = class_severity(class_name)["level"] if class_severity else "low"
        color = _SEVERITY_BGR.get(level, _SEVERITY_BGR["low"])

        # Draw bounding box.
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        # Calculate label size.
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )

        label_y1 = max(0, y1 - text_height - baseline - 6)
        label_y2 = y1

        # Label background.
        cv2.rectangle(
            image,
            (x1, label_y1),
            (x1 + text_width + 6, label_y2),
            color,
            -1,
        )

        # Label text.
        cv2.putText(
            image,
            label,
            (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # Add summary in the top-left corner.
    count = results.get("detection_count", 0)

    summary = f"Detections: {count}"

    cv2.putText(
        image,
        summary,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:
        raise IOError(
            f"Could not save overlay: {output_path}"
        )