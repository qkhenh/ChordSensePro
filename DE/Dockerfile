# ─────────────────────────────────────────────────────────────
# ChordSensePro DE — Custom Airflow image with audio libs
# Base: apache/airflow:2.10.3-python3.11
# ─────────────────────────────────────────────────────────────

FROM apache/airflow:2.10.3-python3.11

USER root

# ── System deps for audio processing ──────────────────────
# ffmpeg: required by yt-dlp and demucs for audio conversion
# libsndfile1: required by soundfile for WAV I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# ── Python deps ───────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Application code ──────────────────────────────────────
# Copied here so import paths work without sys.path hacks
COPY src/ /opt/airflow/src/
COPY dags/ /opt/airflow/dags/
