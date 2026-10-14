"""
VARUNA / MarineVision mission report generator.

Supports:
1. Existing single-image detection reports.
2. Consolidated XTF mission reports.

The generator is GUI-independent and can be called from:
- Analyze page
- Reports page
- XTF batch processing
- CLI/demo scripts
"""

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    HRFlowable,
    PageBreak,
)

from severity import mission_severity, class_severity


NAVY = colors.HexColor("#100F3F")
MUTED = colors.HexColor("#5B5E7E")
BORDER = colors.HexColor("#D9D8E2")

SEVERITY_PDF_COLORS = {
    "high": colors.HexColor("#E53935"),
    "medium": colors.HexColor("#F58A27"),
    "low": colors.HexColor("#2E9B72"),
    "none": colors.HexColor("#2E9B72"),
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _confidence_percent(detection: dict) -> float:
    """
    Supports both:
        confidence = 0.91
    and:
        confidence_percent = 91.0
    """
    if detection.get("confidence_percent") is not None:
        return float(detection["confidence_percent"])

    return float(detection.get("confidence", 0.0)) * 100.0


def _bbox_values(detection: dict):
    """
    Supports both bbox formats:

        {"x1": ..., "y1": ..., "x2": ..., "y2": ...}

    and:

        [x1, y1, x2, y2]
    """
    bbox = detection.get("bbox")

    if isinstance(bbox, dict):
        return (
            bbox.get("x1"),
            bbox.get("y1"),
            bbox.get("x2"),
            bbox.get("y2"),
        )

    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return (
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3],
        )

    return (None, None, None, None)


def _bbox_text(detection: dict) -> str:
    x1, y1, x2, y2 = _bbox_values(detection)

    def fmt(value):
        if value is None:
            return "?"

        try:
            return f"{float(value):.1f}"
        except Exception:
            return str(value)

    return (
        f"{fmt(x1)}, {fmt(y1)}, "
        f"{fmt(x2)}, {fmt(y2)}"
    )


def _safe_datetime(results: dict) -> str:
    timestamp = results.get("timestamp")

    if timestamp:
        try:
            return datetime.fromisoformat(
                timestamp
            ).strftime("%d %b %Y, %H:%M")
        except Exception:
            return str(timestamp)

    return "Not recorded"


def _build_styles():
    styles = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "VARUNA_Title",
            parent=styles["Title"],
            textColor=NAVY,
            fontSize=22,
            spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "VARUNA_Subtitle",
            parent=styles["Normal"],
            textColor=MUTED,
            fontSize=10,
            spaceAfter=14,
        ),
        "section": ParagraphStyle(
            "VARUNA_Section",
            parent=styles["Heading2"],
            textColor=NAVY,
            fontSize=13,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "VARUNA_Body",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
        ),
        "small": ParagraphStyle(
            "VARUNA_Small",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=MUTED,
        ),
        "banner": ParagraphStyle(
            "VARUNA_Banner",
            parent=styles["Normal"],
            fontSize=13,
            leading=16,
            textColor=colors.white,
            alignment=1,
        ),
    }


def _severity_banner(story, severity, styles):
    level = severity.get("level", "none")

    sev_color = SEVERITY_PDF_COLORS.get(
        level,
        SEVERITY_PDF_COLORS["low"],
    )

    banner_text = (
        f"{severity.get('label', 'INFO')} "
        f"&nbsp;·&nbsp; "
        f"{severity.get('reason', '')}"
    )

    banner_table = Table(
        [[Paragraph(banner_text, styles["banner"])]],
        colWidths=[170 * mm],
    )

    banner_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    sev_color,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
            ]
        )
    )

    story.append(banner_table)
    story.append(Spacer(1, 14))


# ----------------------------------------------------------------------
# Existing single-image report
# ----------------------------------------------------------------------

def _generate_single_image_report(
    results: dict,
    output_path: Path,
    mission_name: str = None,
    gps: str = None,
) -> str:

    detections = results.get("detections", [])

    severity = mission_severity(detections)

    styles = _build_styles()

    story = []

    story.append(
        Paragraph(
            "MarineVision — Mission Report",
            styles["title"],
        )
    )

    story.append(
        Paragraph(
            "AI-assisted underwater sonar detection · "
            "runs fully offline, on-device",
            styles["subtitle"],
        )
    )

    _severity_banner(
        story,
        severity,
        styles,
    )

    image_name = results.get(
        "image",
        Path(
            results.get(
                "image_path",
                "unknown",
            )
        ).name,
    )

    story.append(
        Paragraph(
            "Mission Details",
            styles["section"],
        )
    )

    info_rows = [
        [
            "Mission / File",
            mission_name or image_name,
        ],
        [
            "Analyzed",
            _safe_datetime(results),
        ],
        [
            "Model",
            "YOLO11s Sonar Detector "
            "(NLM preprocessing)",
        ],
        [
            "Confidence threshold",
            str(
                results.get(
                    "confidence_threshold",
                    "—",
                )
            ),
        ],
        [
            "Detections found",
            str(
                results.get(
                    "detection_count",
                    len(detections),
                )
            ),
        ],
        [
            "GPS coordinates",
            gps or "Not available",
        ],
    ]

    info_table = Table(
        info_rows,
        colWidths=[
            45 * mm,
            125 * mm,
        ],
    )

    info_table.setStyle(
        TableStyle(
            [
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (0, -1),
                    MUTED,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "LINEBELOW",
                    (0, 0),
                    (-1, -2),
                    0.5,
                    BORDER,
                ),
            ]
        )
    )

    story.append(info_table)

    # Overlay
    overlay_path = results.get("overlay_path")

    if (
        overlay_path
        and Path(overlay_path).exists()
    ):
        story.append(
            Paragraph(
                "Annotated Detection",
                styles["section"],
            )
        )

        try:
            img = Image(overlay_path)

            max_w = 170 * mm
            max_h = 110 * mm

            if (
                img.imageWidth > 0
                and img.imageHeight > 0
            ):
                scale = min(
                    1.0,
                    max_w / img.imageWidth,
                    max_h / img.imageHeight,
                )

                img.drawWidth = (
                    img.imageWidth * scale
                )
                img.drawHeight = (
                    img.imageHeight * scale
                )

            story.append(img)

        except Exception:
            story.append(
                Paragraph(
                    "(Overlay image could not "
                    "be embedded.)",
                    styles["body"],
                )
            )

        story.append(Spacer(1, 10))

    # Detection table
    story.append(
        Paragraph(
            "Detected Objects",
            styles["section"],
        )
    )

    if detections:

        header = [
            "#",
            "Class",
            "Confidence",
            "Severity",
            "Bounding box",
        ]

        rows = [header]

        for i, detection in enumerate(
            detections,
            start=1,
        ):

            entry = class_severity(
                detection.get(
                    "class_name",
                    "",
                )
            )

            rows.append(
                [
                    str(i),
                    detection.get(
                        "class_name",
                        "unknown",
                    ),
                    f"{_confidence_percent(detection):.1f}%",
                    entry["label"],
                    _bbox_text(detection),
                ]
            )

        det_table = Table(
            rows,
            colWidths=[
                10 * mm,
                38 * mm,
                24 * mm,
                26 * mm,
                72 * mm,
            ],
        )

        det_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#F2F1F6"
                        ),
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        9,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        BORDER,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                ]
            )
        )

        story.append(det_table)

    else:

        story.append(
            Paragraph(
                "No objects were detected above "
                "the configured confidence threshold.",
                styles["body"],
            )
        )

    story.append(Spacer(1, 18))

    story.append(
        HRFlowable(
            width="100%",
            color=BORDER,
            thickness=0.5,
        )
    )

    story.append(Spacer(1, 6))

    story.append(
        Paragraph(
            "Generated by VARUNA — Offline AI "
            "Sonar Detection System (SIH 2026). "
            "This report is generated automatically "
            "from model output and should be verified "
            "by a qualified operator before action "
            "is taken.",
            styles["small"],
        )
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )

    doc.build(story)

    return str(output_path)


# ----------------------------------------------------------------------
# XTF mission report
# ----------------------------------------------------------------------

def _generate_xtf_mission_report(
    results: dict,
    output_path: Path,
    mission_name: str = None,
) -> str:

    detections = results.get(
        "detections",
        [],
    )

    frames = results.get(
        "frames",
        [],
    )

    severity = mission_severity(
        detections
    )

    styles = _build_styles()

    story = []

    # --------------------------------------------------------------
    # Header
    # --------------------------------------------------------------

    story.append(
        Paragraph(
            "VARUNA — Sonar Mission Report",
            styles["title"],
        )
    )

    story.append(
        Paragraph(
            "AI-powered automated underwater "
            "marine debris and anomaly detection",
            styles["subtitle"],
        )
    )

    _severity_banner(
        story,
        severity,
        styles,
    )

    # --------------------------------------------------------------
    # Mission details
    # --------------------------------------------------------------

    story.append(
        Paragraph(
            "Mission Details",
            styles["section"],
        )
    )

    mission_file = results.get(
        "mission",
        mission_name or "Unknown mission",
    )

    model_path = results.get(
        "model",
        "Not recorded",
    )

    model_name = Path(
        str(model_path)
    ).name

    class_thresholds = results.get(
        "class_thresholds",
        {},
    )

    threshold_text = ", ".join(
        f"{name}: "
        f"{value if value is not None else 'none'}"
        for name, value in class_thresholds.items()
    )

    info_rows = [
        [
            "Mission / XTF file",
            mission_file,
        ],
        [
            "Frames processed",
            str(
                results.get(
                    "processed_frame_count",
                    len(frames),
                )
            ),
        ],
        [
            "Total detections",
            str(
                results.get(
                    "detection_count",
                    len(detections),
                )
            ),
        ],
        [
            "Model",
            model_name,
        ],
        [
            "Base confidence",
            str(
                results.get(
                    "base_confidence",
                    0.25,
                )
            ),
        ],
        [
            "NMS IoU",
            str(
                results.get(
                    "iou",
                    0.45,
                )
            ),
        ],
        [
            "Navigation source",
            results.get(
                "navigation_source",
                "Not recorded",
            ),
        ],
        [
            "Geotag method",
            results.get(
                "geotag_method",
                "Not recorded",
            ),
        ],
        [
            "Class thresholds",
            threshold_text,
        ],
    ]

    info_table = Table(
        info_rows,
        colWidths=[
            45 * mm,
            125 * mm,
        ],
    )

    info_table.setStyle(
        TableStyle(
            [
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (0, -1),
                    MUTED,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "LINEBELOW",
                    (0, 0),
                    (-1, -2),
                    0.5,
                    BORDER,
                ),
            ]
        )
    )

    story.append(info_table)

    # --------------------------------------------------------------
    # Class summary
    # --------------------------------------------------------------

    story.append(
        Paragraph(
            "Detection Summary",
            styles["section"],
        )
    )

    class_counts = results.get(
        "class_counts",
        {},
    )

    summary_rows = [
        [
            "Class",
            "Detections",
        ]
    ]

    for class_name, count in class_counts.items():
        summary_rows.append(
            [
                class_name,
                str(count),
            ]
        )

    if len(summary_rows) == 1:
        summary_rows.append(
            [
                "No detections",
                "0",
            ]
        )

    summary_table = Table(
        summary_rows,
        colWidths=[
            110 * mm,
            60 * mm,
        ],
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#F2F1F6"
                    ),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    BORDER,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    story.append(summary_table)

    # --------------------------------------------------------------
    # Geotagged detections
    # --------------------------------------------------------------

    story.append(
        Paragraph(
            "Geotagged Detections",
            styles["section"],
        )
    )

    if detections:

        header = [
            "#",
            "Class",
            "Confidence",
            "Latitude",
            "Longitude",
            "Event",
            "Depth (m)",
        ]

        rows = [header]

        for index, detection in enumerate(
            detections,
            start=1,
        ):

            latitude = detection.get(
                "latitude"
            )

            longitude = detection.get(
                "longitude"
            )

            rows.append(
                [
                    str(index),
                    detection.get(
                        "class_name",
                        "unknown",
                    ),
                    f"{_confidence_percent(detection):.1f}%",
                    (
                        f"{float(latitude):.6f}"
                        if latitude is not None
                        else "—"
                    ),
                    (
                        f"{float(longitude):.6f}"
                        if longitude is not None
                        else "—"
                    ),
                    str(
                        detection.get(
                            "event_number",
                            "—",
                        )
                    ),
                    (
                        f"{float(detection.get('sensor_depth_m')):.1f}"
                        if detection.get(
                            "sensor_depth_m"
                        ) is not None
                        else "—"
                    ),
                ]
            )

        geo_table = Table(
            rows,
            colWidths=[
                8 * mm,
                32 * mm,
                23 * mm,
                31 * mm,
                31 * mm,
                17 * mm,
                28 * mm,
            ],
            repeatRows=1,
        )

        geo_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#F2F1F6"
                        ),
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        7.5,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        BORDER,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        4,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        4,
                    ),
                ]
            )
        )

        story.append(geo_table)

    else:

        story.append(
            Paragraph(
                "No accepted detections were produced.",
                styles["body"],
            )
        )

    # --------------------------------------------------------------
    # Detection details
    # --------------------------------------------------------------

    if detections:

        story.append(
            PageBreak()
        )

        story.append(
            Paragraph(
                "Detection Details",
                styles["section"],
            )
        )

        detail_rows = [
            [
                "#",
                "Class",
                "Confidence",
                "Event",
                "Ping",
                "Heading",
                "Range (m)",
                "Channel",
            ]
        ]

        for index, detection in enumerate(
            detections,
            start=1,
        ):

            detail_rows.append(
                [
                    str(index),
                    detection.get(
                        "class_name",
                        "unknown",
                    ),
                    f"{_confidence_percent(detection):.1f}%",
                    str(
                        detection.get(
                            "event_number",
                            "—",
                        )
                    ),
                    str(
                        detection.get(
                            "ping_number",
                            "—",
                        )
                    ),
                    (
                        f"{float(detection.get('heading_deg')):.1f}°"
                        if detection.get(
                            "heading_deg"
                        ) is not None
                        else "—"
                    ),
                    (
                        f"{float(detection.get('slant_range_from_sensor_m')):.2f}"
                        if detection.get(
                            "slant_range_from_sensor_m"
                        ) is not None
                        else "—"
                    ),
                    detection.get(
                        "channel",
                        "—",
                    ),
                ]
            )

        detail_table = Table(
            detail_rows,
            colWidths=[
                8 * mm,
                31 * mm,
                23 * mm,
                17 * mm,
                25 * mm,
                22 * mm,
                25 * mm,
                19 * mm,
            ],
            repeatRows=1,
        )

        detail_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#F2F1F6"
                        ),
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        7.5,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        BORDER,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        4,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        4,
                    ),
                ]
            )
        )

        story.append(detail_table)

    # --------------------------------------------------------------
    # Annotated frames
    #
    # The XTF runner will populate:
    # detection_frames = [
    #     {"event_number": ..., "path": "..."}
    # ]
    # --------------------------------------------------------------

    detection_frames = results.get(
        "detection_frames",
        [],
    )

    if detection_frames:

        story.append(
            PageBreak()
        )

        story.append(
            Paragraph(
                "Annotated Sonar Frames",
                styles["section"],
            )
        )

        story.append(
            Paragraph(
                "Frames containing accepted detections "
                "are included below.",
                styles["body"],
            )
        )

        for frame_entry in detection_frames:

            frame_path = Path(
                frame_entry.get(
                    "path",
                    "",
                )
            )

            if not frame_path.exists():
                continue

            event_number = frame_entry.get(
                "event_number",
                "—",
            )

            story.append(
                Paragraph(
                    f"Event {event_number}",
                    styles["section"],
                )
            )

            try:

                img = Image(
                    str(frame_path)
                )

                max_w = 170 * mm
                max_h = 210 * mm

                if (
                    img.imageWidth > 0
                    and img.imageHeight > 0
                ):

                    scale = min(
                        1.0,
                        max_w / img.imageWidth,
                        max_h / img.imageHeight,
                    )

                    img.drawWidth = (
                        img.imageWidth * scale
                    )

                    img.drawHeight = (
                        img.imageHeight * scale
                    )

                story.append(img)

            except Exception:

                story.append(
                    Paragraph(
                        "Annotated frame could "
                        "not be embedded.",
                        styles["body"],
                    )
                )

            story.append(
                Spacer(1, 10)
            )

    # --------------------------------------------------------------
    # Synthetic navigation note
    # --------------------------------------------------------------

    if results.get(
        "navigation_source"
    ) == "synthetic_demo":

        story.append(
            PageBreak()
        )

        story.append(
            Paragraph(
                "Navigation / Geolocation Note",
                styles["section"],
            )
        )

        story.append(
            Paragraph(
                "This demonstration mission uses synthetic "
                "XTF navigation data. Detection coordinates "
                "are derived from XTF ping navigation and "
                "sonar pixel geometry for demonstration "
                "purposes. They must not be interpreted as "
                "survey-grade real-world coordinates.",
                styles["body"],
            )
        )

    # --------------------------------------------------------------
    # Footer
    # --------------------------------------------------------------

    story.append(
        Spacer(1, 18)
    )

    story.append(
        HRFlowable(
            width="100%",
            color=BORDER,
            thickness=0.5,
        )
    )

    story.append(
        Spacer(1, 6)
    )

    story.append(
        Paragraph(
            "Generated by VARUNA — Offline AI Sonar "
            "Detection System (SIH 2026). "
            "This report is generated automatically "
            "from model output and should be verified "
            "by a qualified operator before action is taken.",
            styles["small"],
        )
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )

    doc.build(story)

    return str(output_path)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def generate_mission_report(
    results: dict,
    output_path: str,
    mission_name: str = None,
    gps: str = None,
) -> str:
    """
    Generate a VARUNA mission PDF.

    Automatically selects the XTF mission format when the result
    contains XTF mission-level fields such as 'frames' and
    'navigation_source'.

    Otherwise, preserves the existing single-image report format.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # XTF mission result
    if (
        "frames" in results
        and "navigation_source" in results
    ):
        return _generate_xtf_mission_report(
            results,
            output_path,
            mission_name=mission_name,
        )

    # Existing image-analysis result
    return _generate_single_image_report(
        results,
        output_path,
        mission_name=mission_name,
        gps=gps,
    )