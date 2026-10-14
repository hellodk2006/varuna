"""
Varuna detection server.

  POST /upload        image in  ->  runs NLM + YOLO11s  ->  result out (and saved to Postgres)
  GET  /reports       every saved result, newest first
  GET  /reports/{id}  one saved result in full (all boxes)
  GET  /results/...   the annotated pictures (bounding boxes drawn on)
  GET  /health        quick self-check (model loaded? database reachable?)

Start it from this folder:   uvicorn main:app --reload
"""
import os
import re
import traceback
import uuid
from contextlib import asynccontextmanager, closing
from datetime import date
from pathlib import Path

import cv2
import psycopg2
from psycopg2.extras import Json
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

import detection

# ------------------------------------------------------------------
# Database connection: Docker container "vibrant_kare" (PostGIS image).
# It was created with only POSTGRES_PASSWORD=pass, so the database and
# user are the PostgreSQL defaults: "postgres" / "postgres".
# ------------------------------------------------------------------
DB = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "pass"),  # change here if yours differs
}

MAX_UPLOAD_MB = 50

# The table is created on first run. The ALTER lines upgrade a table made by
# the earlier server version, so your existing rows are kept.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS varuna_reports (
    id           SERIAL PRIMARY KEY,
    mission      TEXT        NOT NULL,
    report_date  DATE        NOT NULL,
    detections   INTEGER     NOT NULL,
    confidence   REAL        NOT NULL,
    status       TEXT        NOT NULL,
    source_file  TEXT,
    cleaned_file TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE varuna_reports ADD COLUMN IF NOT EXISTS overlay_file  TEXT;
ALTER TABLE varuna_reports ADD COLUMN IF NOT EXISTS class_summary JSONB;
ALTER TABLE varuna_reports ADD COLUMN IF NOT EXISTS result_json   JSONB;
"""


def db_connect():
    return psycopg2.connect(connect_timeout=3, **DB)


def ensure_schema(cur):
    cur.execute(SCHEMA_SQL)


def init_db():
    try:
        with closing(db_connect()) as conn:
            with conn, conn.cursor() as cur:
                ensure_schema(cur)
        print("Database ready: table 'varuna_reports'")
    except Exception as e:
        print(f"WARNING: database not reachable ({e}).")
        print("Is the Docker container running?  ->  docker start vibrant_kare")


@asynccontextmanager
async def lifespan(app):
    init_db()
    try:
        print("Loading detection model (first start can take a few seconds)...")
        detection.load_model()
        print(f"Model ready on {detection.device_name()}: {detection.MODEL_PATH.name}")
    except Exception as e:
        print(f"WARNING: model could not be loaded: {type(e).__name__}: {e}")
        print("Uploads will fail until this is fixed.")
    yield


app = FastAPI(title="Varuna detection server", lifespan=lifespan)

# Serve the annotated pictures at /results/<name>
detection.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
detection.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=detection.RESULTS_DIR), name="results")


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def safe_name(name):
    name = Path(name or "upload").name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "upload"


def class_counts(detections):
    counts = {}
    for d in detections:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    return counts


def build_response(results, mission, source_name):
    """Turn the model's output into the JSON the app receives.

    Mission / Date / Detections / Confidence / Status match the columns of the
    app's Reports table. The rest (boxes, sizes, overlay picture) is extra
    detail for the Analyze page.
    """
    dets = results["detections"]
    confs = [d["confidence_percent"] for d in dets]
    return {
        "Mission": mission,
        "Date": date.today().isoformat(),
        "Detections": len(dets),
        "Confidence": round(sum(confs) / len(confs), 1) if confs else 0.0,
        "Status": "Completed",
        "image": source_name,
        "image_width": results["image_width"],
        "image_height": results["image_height"],
        "preprocessing": results["preprocessing"],
        "confidence_threshold": results["confidence_threshold"],
        "detection_count": len(dets),
        "class_summary": class_counts(dets),
        "detections": dets,
        "overlay_url": f"/results/{results['overlay_file']}",
    }


def save_report(resp, overlay_file):
    with closing(db_connect()) as conn:
        with conn, conn.cursor() as cur:
            ensure_schema(cur)
            cur.execute(
                """
                INSERT INTO varuna_reports
                    (mission, report_date, detections, confidence, status,
                     source_file, overlay_file, class_summary, result_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    resp["Mission"], resp["Date"], resp["Detections"],
                    resp["Confidence"], resp["Status"], resp["image"],
                    overlay_file, Json(resp["class_summary"]), Json(resp),
                ),
            )
            return cur.fetchone()[0]


def row_to_dict(r):
    return {
        "id": r[0],
        "Mission": r[1],
        "Date": r[2].isoformat(),
        "Detections": r[3],
        "Confidence": float(r[4]),
        "Status": r[5],
        "class_summary": r[7] or {},
        "overlay_url": f"/results/{r[6]}" if r[6] else None,
    }


# ------------------------------------------------------------------
# endpoints
# ------------------------------------------------------------------
@app.post("/upload")
def upload(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.25),
    mission: str = Form("Mission Alpha"),
):
    if not 0.0 <= confidence_threshold <= 1.0:
        raise HTTPException(400, "confidence_threshold must be between 0 and 1.")

    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")

    name = safe_name(file.filename)
    saved = detection.UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{name}"
    saved.write_bytes(data)

    if cv2.imread(str(saved)) is None:
        saved.unlink(missing_ok=True)
        raise HTTPException(
            400, "That file is not a readable image (use PNG, JPG, BMP or TIFF)."
        )

    try:
        results = detection.detect(str(saved), confidence=confidence_threshold)
    except FileNotFoundError as e:
        raise HTTPException(503, f"Model file not found: {e}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Detection failed: {type(e).__name__}: {e}")

    resp = build_response(results, mission.strip()[:80] or "Mission Alpha", name)

    try:
        resp["id"] = save_report(resp, results["overlay_file"])
        resp["saved"] = True
    except Exception as e:
        print(f"WARNING: could not save to database: {e}")
        resp["saved"] = False

    return resp


@app.get("/reports")
def list_reports(limit: int = 200):
    """All saved results, newest first."""
    try:
        with closing(db_connect()) as conn:
            with conn, conn.cursor() as cur:
                ensure_schema(cur)
                cur.execute(
                    """
                    SELECT id, mission, report_date, detections, confidence,
                           status, overlay_file, class_summary
                    FROM varuna_reports
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(503, f"Database not reachable: {e}")
    return [row_to_dict(r) for r in rows]


@app.get("/reports/{report_id}")
def get_report(report_id: int):
    """One saved result including every bounding box."""
    try:
        with closing(db_connect()) as conn:
            with conn, conn.cursor() as cur:
                ensure_schema(cur)
                cur.execute(
                    """
                    SELECT id, mission, report_date, detections, confidence,
                           status, overlay_file, class_summary, result_json
                    FROM varuna_reports WHERE id = %s
                    """,
                    (report_id,),
                )
                r = cur.fetchone()
    except Exception as e:
        raise HTTPException(503, f"Database not reachable: {e}")
    if r is None:
        raise HTTPException(404, f"No report with id {report_id}.")
    out = row_to_dict(r)
    out["result"] = r[8]  # full detection JSON (None for very old rows)
    return out


@app.get("/health")
def health():
    db_ok = True
    try:
        with closing(db_connect()) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception:
        db_ok = False
    return {
        "status": "ok",
        "model_loaded": detection.is_loaded(),
        "device": detection.device_name(),
        "database": db_ok,
    }
