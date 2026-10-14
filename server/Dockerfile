# Varuna detection server
#
# Build context must be the "varuna" folder (the one with app/, sonar_model/
# and server/), NOT the server/ folder itself, so this image can copy
# sonar_model/ and app/severity.py. docker-compose.yml already sets this up;
# if building by hand, run it from the "varuna" folder:
#   docker build -f server/Dockerfile -t varuna-server .
FROM python:3.12-slim

# libgl1/libglib2.0-0: required by opencv-python-headless at import time
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/requirements.txt ./requirements.txt
# CPU-only torch build: much smaller than the default CUDA wheels, and this
# runs fine without a GPU. Delete this line to get GPU support instead.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# The trained model + pipeline code (unchanged from the desktop app).
COPY sonar_model/ ./sonar_model/
# Only severity.py is used from app/ - see server/detection.py. The rest of
# app/ (PySide6 code) is never imported and does not need to be in the image.
COPY app/severity.py ./app/severity.py
COPY sonar_model/config/severity.json ./sonar_model/config/severity.json

COPY server/*.py ./server/
WORKDIR /app/server

ENV SONAR_DATA_DIR=/data
VOLUME ["/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
