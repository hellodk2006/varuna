from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2

from xtf_mission import XTFMissionReader, geotag_detection
from sonar_integration import (
    run_detection,
    DEFAULT_MODEL_PATH,
    DEFAULT_CLASSES_PATH,
)


# ------------------------------------------------------------------
# VARUNA class-specific confidence thresholds
#
# YOLO itself stays at 0.25 so that low-confidence ghost-net
# candidates are not discarded before class-specific filtering.
# ------------------------------------------------------------------

CLASS_THRESHOLDS = {
    "crab_pot": None,
    "submarine_pipeline": 0.65,
    "shipwreck": 0.55,
    "ghost_net": 0.25,
    "mine_cylinder": 0.35,
}


def passes_class_threshold(detection: dict) -> bool:
    """
    Apply VARUNA's class-specific confidence threshold.

    A None threshold means that no class-specific filtering is
    currently applied for that class.
    """

    class_name = detection.get("class_name", "")
    confidence = float(detection.get("confidence", 0.0))

    threshold = CLASS_THRESHOLDS.get(class_name)

    if threshold is None:
        return True

    return confidence >= threshold


def _bbox_as_list(detection: dict) -> list[float]:
    """
    Convert the existing sonar pipeline's bbox dictionary into
    the [x1, y1, x2, y2] format expected by geotag_detection().
    """

    bbox = detection["bbox"]

    return [
        float(bbox["x1"]),
        float(bbox["y1"]),
        float(bbox["x2"]),
        float(bbox["y2"]),
    ]


def run_xtf_mission(
    xtf_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    classes_path: str | Path = DEFAULT_CLASSES_PATH,
    confidence: float = 0.25,
    iou: float = 0.45,
    output_dir: str | Path | None = None,
) -> dict:
    """
    Run the complete VARUNA pipeline over an XTF mission.

    Pipeline:

        XTF
          ↓
        reconstructed sonar frame
          ↓
        NLM + YOLO11s
          ↓
        class-specific confidence filtering
          ↓
        XTF navigation + sonar geometry
          ↓
        geotagged detections
          ↓
        aggregate mission JSON

    Parameters
    ----------
    xtf_path:
        Path to the XTF mission.

    model_path:
        YOLO model weights.

    classes_path:
        VARUNA class mapping.

    confidence:
        Base YOLO confidence threshold.
        Keep this at 0.25 for the current calibration.

    iou:
        YOLO NMS IoU threshold.

    output_dir:
        Directory for mission results.
    """

    xtf_path = Path(xtf_path)

    if not xtf_path.exists():
        raise FileNotFoundError(
            f"XTF mission not found: {xtf_path}"
        )

    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent
            / "results"
            / "xtf"
            / xtf_path.stem
        )

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mission = XTFMissionReader(xtf_path)

    event_numbers = mission.event_numbers()

    # Ship's actual surveyed path - independent of which frames were
    # reconstructed, so this reflects the real start/end of the run even
    # if only a subset of events produced usable frames above.
    journey_start, journey_end = mission.journey_bounds()
    full_track = [
        {"latitude": p.latitude, "longitude": p.longitude, "ping_number": p.ping_number}
        for p in mission.full_track()
    ]

    mission_detections = []
    frame_results = []

    total_frames = len(event_numbers)

    for frame_index, event_number in enumerate(event_numbers, start=1):

        frame = mission.get_frame(event_number)

        # ----------------------------------------------------------
        # Save reconstructed XTF frame temporarily.
        #
        # The existing sonar pipeline expects an image file, so we
        # bridge XTF -> PNG here without changing the existing
        # NLM/YOLO implementation.
        # ----------------------------------------------------------

        frame_path = frames_dir / (
            f"event_{event_number:04d}.png"
        )

        if not cv2.imwrite(
            str(frame_path),
            frame.image,
        ):
            raise RuntimeError(
                f"Could not save reconstructed frame: {frame_path}"
            )

        # ----------------------------------------------------------
        # Existing VARUNA image pipeline:
        #
        # NLM -> YOLO11s -> structured detections
        # ----------------------------------------------------------

        results = run_detection(
            str(frame_path),
            model_path=str(model_path),
            classes_path=str(classes_path),
            confidence=confidence,
        )

        raw_detections = results.get(
            "detections",
            [],
        )

        accepted_detections = []

        for detection in raw_detections:

            if not passes_class_threshold(
                detection
            ):
                continue

            # geotag_detection() expects a list bbox.
            geo_input = dict(detection)
            geo_input["bbox"] = _bbox_as_list(
                detection
            )

            geo = geotag_detection(
                frame,
                geo_input,
            )

            geo_dict = geo.to_dict()

            # Add mission/frame metadata.
            geo_dict.update(
                {
                    "frame_index": frame_index,
                    "frame_count": total_frames,
                    "source_frame": frame_path.name,
                    "confidence_threshold_base": confidence,
                    "class_threshold": CLASS_THRESHOLDS.get(
                        detection.get("class_name")
                    ),
                    "navigation_source": "synthetic_demo",
                }
            )

            accepted_detections.append(
                geo_dict
            )

            mission_detections.append(
                geo_dict
            )

        frame_results.append(
            {
                "event_number": event_number,
                "frame_index": frame_index,
                "frame_count": total_frames,
                "width": frame.width,
                "height": frame.height,
                "ping_count": len(frame.pings),
                "channel": frame.channel_type,
                "ground_range_m": frame.ground_range_m,
                "slant_range_m": frame.slant_range_m,
                "source_frame": frame_path.name,
                "raw_detection_count": len(raw_detections),
                "accepted_detection_count": len(
                    accepted_detections
                ),
                "detections": accepted_detections,
            }
        )

        print(
            f"[{frame_index}/{total_frames}] "
            f"Event {event_number}: "
            f"{len(raw_detections)} raw → "
            f"{len(accepted_detections)} accepted"
        )

    # --------------------------------------------------------------
    # Mission-level summary
    # --------------------------------------------------------------

    class_counts = {}

    for detection in mission_detections:

        class_name = detection["class_name"]

        class_counts[class_name] = (
            class_counts.get(class_name, 0) + 1
        )

    mission_result = {
        "mission": xtf_path.name,
        "source_file": str(xtf_path),
        "navigation_source": "synthetic_demo",
        "geotag_method": (
            "XTF ping navigation + sonar pixel geometry"
        ),
        "model": str(model_path),
        "classes": str(classes_path),
        "base_confidence": confidence,
        "iou": iou,
        "class_thresholds": CLASS_THRESHOLDS,
        "event_count": total_frames,
        "processed_frame_count": len(frame_results),
        "detection_count": len(mission_detections),
        "class_counts": class_counts,
        "detections": mission_detections,
        "frames": frame_results,
        "journey_start": {
            "latitude": journey_start.latitude,
            "longitude": journey_start.longitude,
            "ping_number": journey_start.ping_number,
        },
        "journey_end": {
            "latitude": journey_end.latitude,
            "longitude": journey_end.longitude,
            "ping_number": journey_end.ping_number,
        },
        "journey_track": full_track,
    }

        # --------------------------------------------------------------
    # JSON report
    # --------------------------------------------------------------

    output_json = output_dir / "mission_results.json"

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            mission_result,
            f,
            indent=4,
        )

    # --------------------------------------------------------------
    # CSV report
    #
    # One row = one accepted detection.
    # --------------------------------------------------------------

    output_csv = output_dir / "mission_results.csv"

    csv_fields = [
        "detection_id",
        "event_number",
        "frame_index",
        "source_frame",
        "class_id",
        "class_name",
        "confidence",
        "latitude",
        "longitude",
        "center_x",
        "center_y",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_width",
        "bbox_height",
        "sensor_depth_m",
        "altitude_m",
        "heading_deg",
        "across_track_offset_m",
        "ground_range_from_sensor_m",
        "slant_range_from_sensor_m",
        "channel",
        "ping_number",
        "navigation_source",
        "geotag_method",
    ]

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        for detection_id, detection in enumerate(
            mission_detections,
            start=1,
        ):

            bbox = detection.get("bbox", {})

            row = {
                "detection_id": detection_id,
                "event_number": detection.get(
                    "event_number"
                ),
                "frame_index": detection.get(
                    "frame_index"
                ),
                "source_frame": detection.get(
                    "source_frame"
                ),
                "class_id": detection.get(
                    "class_id"
                ),
                "class_name": detection.get(
                    "class_name"
                ),
                "confidence": detection.get(
                    "confidence"
                ),
                "latitude": detection.get(
                    "latitude"
                ),
                "longitude": detection.get(
                    "longitude"
                ),
                "center_x": detection.get(
                    "center_x"
                ),
                "center_y": detection.get(
                    "center_y"
                ),
                "bbox_x1": bbox.get("x1"),
                "bbox_y1": bbox.get("y1"),
                "bbox_x2": bbox.get("x2"),
                "bbox_y2": bbox.get("y2"),
                "bbox_width": detection.get(
                    "bbox_width"
                ),
                "bbox_height": detection.get(
                    "bbox_height"
                ),
                "sensor_depth_m": detection.get(
                    "sensor_depth_m"
                ),
                "altitude_m": detection.get(
                    "altitude_m"
                ),
                "heading_deg": detection.get(
                    "heading_deg"
                ),
                "across_track_offset_m": detection.get(
                    "across_track_offset_m"
                ),
                "ground_range_from_sensor_m": detection.get(
                    "ground_range_from_sensor_m"
                ),
                "slant_range_from_sensor_m": detection.get(
                    "slant_range_from_sensor_m"
                ),
                "channel": detection.get(
                    "channel"
                ),
                "ping_number": detection.get(
                    "ping_number"
                ),
                "navigation_source": detection.get(
                    "navigation_source"
                ),
                "geotag_method": detection.get(
                    "geotag_method"
                ),
            }

            writer.writerow(row)

    # --------------------------------------------------------------
    # Add output paths to returned result
    # --------------------------------------------------------------

    mission_result["results_json_path"] = str(
        output_json
    )

    mission_result["results_csv_path"] = str(
        output_csv
    )

    print()
    print("=" * 60)
    print("VARUNA XTF MISSION COMPLETE")
    print("=" * 60)
    print(f"Mission:       {xtf_path.name}")
    print(f"Frames:        {total_frames}")
    print(
        f"Detections:    {len(mission_detections)}"
    )
    print(f"Class counts:  {class_counts}")
    print(f"JSON:          {output_json}")
    print(f"CSV:           {output_csv}")
    print("=" * 60)

    return mission_result