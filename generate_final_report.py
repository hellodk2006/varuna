import json
from pathlib import Path
import cv2

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

BASE = Path(r"E:\varuna\results\xtf\VARUNA_DEMO_MISSION")
JSON = BASE / "mission_results.json"
PDF = BASE / "VARUNA_Final_Mission_Report.pdf"

with open(JSON, encoding="utf-8") as f:
    data = json.load(f)

detections = data.get("detections", [])
frames = data.get("detection_frames", [])

styles = getSampleStyleSheet()
doc = SimpleDocTemplate(
    str(PDF),
    pagesize=A4,
    rightMargin=15*mm,
    leftMargin=15*mm,
    topMargin=15*mm,
    bottomMargin=15*mm
)

story = []

story.append(Paragraph("VARUNA", styles["Title"]))
story.append(Paragraph(
    "AI-Powered Underwater Marine Debris and Anomaly Detection",
    styles["Heading2"]
))
story.append(Spacer(1, 10))

# Summary
story.append(Paragraph("Mission Summary", styles["Heading2"]))

counts = {}
for d in detections:
    name = d["class_name"]
    counts[name] = counts.get(name, 0) + 1

summary = [
    ["Frames processed", str(data.get("frame_count", 7))],
    ["Accepted detections", str(len(detections))],
]

for name, count in counts.items():
    summary.append([name, str(count)])

t = Table(summary, colWidths=[80*mm, 80*mm])
t.setStyle(TableStyle([
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("BACKGROUND", (0,0), (0,-1), colors.lightgrey),
    ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
    ("PADDING", (0,0), (-1,-1), 6),
]))
story.append(t)

# Detection table
story.append(Spacer(1, 15))
story.append(Paragraph("Geotagged Detections", styles["Heading2"]))

rows = [["Class", "Conf.", "Latitude", "Longitude", "Depth", "Event"]]

for d in detections:
    rows.append([
        d["class_name"],
        f'{d["confidence"]*100:.1f}%',
        f'{d["latitude"]:.6f}',
        f'{d["longitude"]:.6f}',
        f'{d["sensor_depth_m"]:.1f} m',
        str(d["event_number"])
    ])

t = Table(rows, colWidths=[35*mm, 20*mm, 30*mm, 30*mm, 22*mm, 18*mm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#26354A")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
    ("FONTSIZE", (0,0), (-1,-1), 7),
    ("PADDING", (0,0), (-1,-1), 4),
]))
story.append(t)

# Annotated images
story.append(PageBreak())
story.append(Paragraph("Annotated Sonar Frames", styles["Title"]))
story.append(Spacer(1, 10))

for frame in frames:
    path = Path(frame["overlay_path"])

    story.append(
        Paragraph(
            f'Event {frame["event_number"]}',
            styles["Heading2"]
        )
    )

    if path.exists():
        img = cv2.imread(str(path))

        if img is not None:
            h, w = img.shape[:2]

            max_w = 175 * mm
            max_h = 105 * mm
            scale = min(max_w / w, max_h / h)

            story.append(Image(
                str(path),
                width=w * scale,
                height=h * scale
            ))
        else:
            story.append(Paragraph("Unable to read image.", styles["Normal"]))
    else:
        story.append(Paragraph(
            f"Image not found: {path}",
            styles["Normal"]
        ))

    story.append(Spacer(1, 10))

story.append(PageBreak())
story.append(Paragraph("Technical Notes", styles["Heading2"]))
story.append(Paragraph(
    "This demonstration uses a synthetic but structurally valid XTF mission. "
    "Navigation coordinates are demonstration data and are not survey-grade. "
    "Geotagging combines XTF ping navigation with sonar pixel geometry.",
    styles["Normal"]
))

doc.build(story)

print("=" * 55)
print("VARUNA FINAL PDF GENERATED")
print("=" * 55)
print(f"PDF: {PDF}")
print(f"Detections: {len(detections)}")
print(f"Images: {len(frames)}")
print(f"Size: {PDF.stat().st_size / 1024:.1f} KB")