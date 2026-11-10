# CPU image for re-running the analyses on saved checkpoints (training itself wants a GPU).
#
#   docker build -t spectralens .
#   docker run --rm -v ./checkpoints:/app/checkpoints -v ./reports:/app/reports \
#       spectralens python -m spectralens.spectra
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# torch first, in its own layer: it's the slow, big one and changes least often
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install .

# non-root user; the mounted folders must be writable by it
RUN useradd --create-home spectralens && mkdir -p checkpoints reports data \
    && chown -R spectralens /app
USER spectralens

ENTRYPOINT []
CMD ["spectralens", "info"]
