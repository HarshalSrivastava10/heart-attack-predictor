# Heart disease API — multi-stage not required; train at build for self-contained image
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY data ./data

RUN pip install --no-cache-dir -e .

# Bundle model into image (reproducible from shipped CSV)
RUN python -m heart_mlops.train --fast

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
