import json
from pathlib import Path

from app.report_generator import generate_mission_report

MISSION_DIR = Path(r"E:\varuna\results\xtf\VARUNA_DEMO_MISSION")
JSON_PATH = MISSION_DIR / "mission_results.json"
PDF_PATH = MISSION_DIR / "VARUNA_Mission_Report.pdf"

with JSON_PATH.open("r", encoding="utf-8") as f:
    results = json.load(f)

generate_mission_report(
    results,
    str(PDF_PATH),
    mission_name="VARUNA_DEMO_MISSION"
)

print()
print("=" * 60)
print("PDF GENERATED")
print("=" * 60)
print(PDF_PATH)
print(f"Size: {PDF_PATH.stat().st_size / 1024:.1f} KB")
