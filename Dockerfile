# syntax=docker/dockerfile:1

# Pinned to the Debian variant, not just `3.12-slim`, which floats across Debian
# releases and has broken builds on that alone.
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# README.md is a build input: pyproject declares it as the long description, so
# the image cannot build without it.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .


FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    KEEL_PROJECT_ROOT=/app \
    KEEL_DATABASE_URL=sqlite+aiosqlite:////app/data/keel.db

RUN useradd --create-home --uid 10001 keel

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv

# Migration scripts are not part of the wheel, so they are copied separately for
# `keel db`. KEEL_PROJECT_ROOT above points at this directory.
COPY alembic.ini ./
COPY alembic ./alembic

RUN mkdir -p /app/data && chown -R keel:keel /app
USER keel

EXPOSE 8000

# urllib rather than curl, which is not in the slim image.
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
  CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

CMD ["sh", "-c", "keel db head && exec uvicorn keel.api.app:app --host 0.0.0.0 --port 8000"]