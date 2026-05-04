FROM python:3.11.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
# Build dependencies required by scientific Python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
# Constrain packages that commonly break due to Python/wheel mismatches.
RUN printf "numpy==1.26.4\ntensorflow-cpu==2.16.1\n" > /app/constraints.txt \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt -c /app/constraints.txt
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
