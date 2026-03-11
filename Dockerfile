FROM python:3.11-slim

# System deps for psycopg2, Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libjpeg-dev \
    zlib1g-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# /tmp is ephemeral on Railway — jobs live here during review only
# Images are uploaded to R2 on save, so /tmp loss only affects in-progress jobs
RUN mkdir -p /tmp/examside_jobs

EXPOSE 8000

# Single worker — Railway free tier has 512MB RAM
# --timeout-keep-alive 75 matches Railway's 75s idle timeout
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "75", \
     "--log-level", "info"]