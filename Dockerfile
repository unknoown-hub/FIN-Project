# ── Stage 1: dependency installation ─────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy only the files the application needs
COPY FINd_optimized.py matrix.py ./
COPY find_api/ ./find_api/

# Switch to non-root user
USER appuser

EXPOSE 8945

CMD ["uvicorn", "find_api.app:app", "--host", "0.0.0.0", "--port", "8945"]
