FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libjpeg-dev \
    zlib1g-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# --timeout 120: wait longer per chunk
# --retries 5: retry on timeout
# --no-cache-dir: saves space
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

COPY . .

RUN mkdir -p /tmp/examside_jobs

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
