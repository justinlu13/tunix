# syntax=docker/dockerfile:1
# ==============================================================================
# Tunix Container Image (`FROM python:3.12-slim`)
#
# Why we build from `python:3.12-slim` instead of `FROM vllm/vllm-tpu`:
#   Upstream `vllm/vllm-tpu` (`vllm-project/tpu-inference/docker/Dockerfile`)
#   starts from `python:3.12-slim-bookworm`, installs packages globally via
#   standard `pip` (without `/opt/venv` or `uv`), resolves `vllm`'s
#   `requirements/tpu.txt` + `lm-eval` + `tpu-inference`'s `requirements.txt`
#   and `requirements_benchmarking.txt` without `--no-deps`, and freezes a
#   single `vllm` + `tpu-inference` commit pair (`ENTRYPOINT ["/entrypoint.sh"]`)
#   oriented around standalone inference serving.
#
#   Building directly from `python:3.12-slim` with `uv` into `/opt/venv` gives
#   explicit control over layer ordering and dependency resolution:
#     1. Isolates the heavy vLLM + TPU-Inference + Tunix runtime layer from
#        `maxtext_requirements.txt`, so MaxText or Tunix source updates never
#        invalidate the base runtime layer.
#     2. Splits MaxText into a cached `--deps-only` wheel layer and a ~15-second
#        `--source-only` (`--no-deps`) layer triggered by GitHub's `.atom` commit
#        feed (`ADD https://github.com/AI-Hypercomputer/maxtext/commits/${MAXTEXT_REF}.atom`).
#     3. Prevents transitive dependency clobbering (`protobuf>=7.35.1` and
#        `numpy==2.3.5` are locked after `tensorflow`/`tensorflow-datasets` and
#        `maxtext`).
#     4. Zero model weights, datasets, or checkpoints are baked into the image
#        so `tunix/mlperf` stays generic and fast to pull (~13s container init);
#        CI workflows mount cached weights/datasets into `/root/.cache` at runtime.
# ==============================================================================

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV VLLM_TARGET_DEVICE=tpu

# Install OS build dependencies and initialize `/opt/venv` with `uv`.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      git \
      libnuma-dev \
      libomp-dev \
      libopenmpi-dev \
      ninja-build \
      python3 \
      python3-pip \
      python3-venv && \
    rm -rf /var/lib/apt/lists/* && \
    python3.12 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip uv

WORKDIR /app

# ---------------------------------------------------------------------------
# 1. Core Tunix + vLLM + TPU-Inference Runtime Layer
# ---------------------------------------------------------------------------
# Copy ONLY `requirements.txt`, `special_requirements.txt`, and
# `install_tunix_vllm_requirement.sh` so edits to `maxtext_requirements.txt` or
# `install_maxtext.sh` never invalidate this ~15-minute base layer.
COPY scripts/install_tunix_vllm_requirement.sh /app/scripts/install_tunix_vllm_requirement.sh
COPY requirements/requirements.txt /app/requirements/requirements.txt
COPY requirements/special_requirements.txt /app/requirements/special_requirements.txt
COPY pyproject.toml README.md /app/
RUN mkdir -p /app/tunix && touch /app/tunix/__init__.py

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    bash /app/scripts/install_tunix_vllm_requirement.sh --full

# ---------------------------------------------------------------------------
# 2. Optional Kubernetes CLI Tools (`INSTALL_K8S_TOOLS=true`)
# ---------------------------------------------------------------------------
ARG INSTALL_K8S_TOOLS=false
COPY scripts/install_k8s_tools.sh /app/scripts/install_k8s_tools.sh
RUN if [ "$INSTALL_K8S_TOOLS" = "true" ]; then \
      bash /app/scripts/install_k8s_tools.sh; \
    fi

# ---------------------------------------------------------------------------
# 3. Optional MaxText + Adapter Split-Layer Caching (`INSTALL_MAXTEXT=true`)
# ---------------------------------------------------------------------------
ARG INSTALL_MAXTEXT=false
ARG MAXTEXT_REF=63d5db085447274e9cd8769334866fe5268459a9

COPY scripts/install_maxtext.sh /app/scripts/install_maxtext.sh
COPY requirements/maxtext_requirements.txt /app/requirements/maxtext_requirements.txt

# 3a. Cached static third-party MaxText wheels (not invalidated when MAXTEXT_REF advances).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_MAXTEXT" = "true" ]; then \
      bash /app/scripts/install_maxtext.sh --deps-only; \
    fi

# 3b. Selective cache invalidation for MaxText + adapter source (`~15s` rebuild).
# TODO(maxtext-dev): Replace this GitHub Atom feed cache-bust + `--no-deps`
# source layer once MaxText publishes versioned post-training wheels to PyPI or
# Artifact Registry.
ADD https://github.com/AI-Hypercomputer/maxtext/commits/${MAXTEXT_REF}.atom /tmp/maxtext_commits.atom
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_MAXTEXT" = "true" ]; then \
      MAXTEXT_REF="${MAXTEXT_REF}" \
      bash /app/scripts/install_maxtext.sh --source-only; \
    fi

# Install zstd for fast GitHub Actions cache compression/decompression and
# re-pin packages that MaxText's requirements may have overridden. Placed
# before `COPY . /app` so source edits do not invalidate this layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    apt-get update && \
    apt-get install -y --no-install-recommends zstd && \
    rm -rf /var/lib/apt/lists/* && \
    uv pip install "numpy==2.3.5" "datasets>=3.0.0" pytest-asyncio

# ---------------------------------------------------------------------------
# 4. Copy Tunix Repository & Run Recipe/Component Installers
# ---------------------------------------------------------------------------
COPY . /app

# Optional DeepSWE recipe dependencies (`examples/deepswe/install_deepswe.sh`).
ARG INSTALL_DEEPSWE_DEPS=false
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_DEEPSWE_DEPS" = "true" ]; then \
      bash /app/examples/deepswe/install_deepswe.sh; \
    fi

# Optional Raiden (`tpu_sync_jax`) installation (`scripts/install_raiden.sh`).
# Uses `./raiden_wheels/*.whl` if present in the build context or fetches
# `RAIDEN_WHEEL_URL` via GCE metadata server / `gcloud`.
ARG INSTALL_RAIDEN=false
ARG RAIDEN_WHEEL_DIR=/app/raiden_wheels
ARG RAIDEN_WHEEL_URL=""
ARG RAIDEN_WHEEL_SHA256=""
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_RAIDEN" = "true" ]; then \
      RAIDEN_WHEEL_DIR="${RAIDEN_WHEEL_DIR}" \
      RAIDEN_WHEEL_URL="${RAIDEN_WHEEL_URL}" \
      RAIDEN_WHEEL_SHA256="${RAIDEN_WHEEL_SHA256}" \
      bash /app/scripts/install_raiden.sh; \
    fi

# Compile the explicit gRPC protobuf definition for distributed orchestrator
# discovery and install Tunix in editable mode (`--no-deps`).
RUN test -f /app/tunix/experimental/distributed/runtime/discovery/discovery_service.proto && \
    python3 -m grpc_tools.protoc -I/app --python_out=/app --grpc_python_out=/app \
      /app/tunix/experimental/distributed/runtime/discovery/discovery_service.proto && \
    uv pip install --no-deps -e /app

# Build-time environment verification (`scripts/verify_environment.py`).
RUN JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 python3 /app/scripts/verify_environment.py \
      --check-tunix-version \
      --packages jax jaxlib flax optax orbax.checkpoint qwix datasets vllm tpu_inference tunix \
      tunix.experimental.distributed.runtime.discovery.discovery_service_pb2 \
      $(if [ "$INSTALL_MAXTEXT" = "true" ]; then echo "maxtext maxtext_vllm_adapter"; fi) \
      $(if [ "$INSTALL_RAIDEN" = "true" ]; then echo "tpu_sync"; fi)

CMD ["bash"]
