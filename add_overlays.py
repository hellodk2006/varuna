import json
from pathlib import Path
import cv2

MISSION_DIR = Path(r"E:\varuna\results\xtf\VARUNA_DEMO_MISSION")
JSON_PATH = MISSION_DIR / "mission_results.json"
FRAMES_DIR = MISSION_DIR / "frames"
OVERLAYS_DIR = MISSION_DIR / "overlays"

OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)

with JSON_PATH.open("r", encoding="utf-8") as f:
    mission = json.load(f)

detections = mission.get("detections", [])

# Group detections by source frame.
by_frame = {}

for det in detections:
    source_frame = det.get("source_frame")

    if not source_frame:
        event_number = int(det.get("event_number", 0))
        source_frame = f"event_{event_number:04d}.png"

    by_frame.setdefault(source_frame, []).append(det)

detection_frames = []

for source_frame, frame_detections in sorted(by_frame.items()):

    frame_path = FRAMES_DIR / source_frame

    if not frame_path.exists():
        print(f"[WARN] Frame not found: {frame_path}")
        continue

    image = cv2.imread(str(frame_path))

    if image is None:
        print(f"[WARN] Could not read: {frame_path}")
        continue

    for det in frame_detections:

        class_name = str(det.get("class_name", "unknown"))
        confidence = float(det.get("confidence", 0.0)) * 100.0

        x1 = int(round(float(det["bbox_x1"])))
        y1 = int(round(float(det["bbox_y1"])))
        x2 = int(round(float(det["bbox_x2"])))
        y2 = int(round(float(det["bbox_y2"])))

        # Clamp coordinates to image boundaries.
        x1 = max(0, min(x1, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        x2 = max(0, min(x2, image.shape[1] - 1))
        y2 = max(0, min(y2, image.shape[0] - 1))

        # Bounding box.
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        label = f"{class_name} {confidence:.1f}%"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        # Put the label above the box when possible.
        label_x = x1
        label_y = y1 - 8

        if label_y - text_h - baseline < 0:
            label_y = y1 + text_h + baseline + 8

        # Label background.
        bg_x1 = label_x
        bg_y1 = label_y - text_h - baseline - 4
        bg_x2 = label_x + text_w + 8
        bg_y2 = label_y + 4

        cv2.rectangle(
            image,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            (0, 255, 0),
            -1
        )

        # Black text.
        cv2.putText(
            image,
            label,
            (label_x + 4, label_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    output_path = OVERLAYS_DIR / source_frame

    cv2.imwrite(str(output_path), image)

    event_number = frame_detections[0].get("event_number")

    detection_frames.append({
        "event_number": event_number,
        "frame_path": str(frame_path),
        "overlay_path": str(output_path),
        "detections": frame_detections
    })

    print(
        f"Event {event_number}: "
        f"{len(frame_detections)} detection(s) -> "
        f"{output_path.name}"
    )

mission["detection_frames"] = detection_frames
mission["overlays_dir"] = str(OVERLAYS_DIR)

with JSON_PATH.open("w", encoding="utf-8") as f:
    json.dump(mission, f, indent=2)

print()
print("=" * 60)
print("VARUNA OVERLAYS COMPLETE")
print("=" * 60)
print(f"Detection frames: {len(detection_frames)}")
print(f"Overlay directory: {OVERLAYS_DIR}")
print(f"Updated JSON:       {JSON_PATH}")
