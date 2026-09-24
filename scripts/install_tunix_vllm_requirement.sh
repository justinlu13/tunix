#!/usr/bin/env bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Installs Tunix, vLLM, and TPU-Inference runtime dependencies.
#
# Modes:
#   - Default (`bash scripts/install_tunix_vllm_requirement.sh`):
#     Auto-detects whether the pre-compiled vLLM + TPU-Inference base stack is
#     already present (e.g. `FROM vllm/vllm-tpu:nightly-...`). If missing or if
#     `OVERWRITE_VLLM_REQUIREMENTS=true` / `--full` is passed, installs the full
#     `requirements/requirements.txt` and `requirements/special_requirements.txt`
#     stack before applying the Tunix runtime overlay.
#   - Overlay-only (`bash scripts/install_tunix_vllm_requirement.sh --overlay-only`):
#     Assumes the base vLLM + TPU-Inference stack is already present and only
#     installs Tunix + runtime overlay dependencies and locks `numpy==2.3.5`.
#   - `--skip-if-installed`:
#     If `/opt/venv/.tunix_vllm_overlay_installed` is present (built via
#     `Dockerfile`), only links the current repository checkout (`pip install --no-deps -e .`).

set -euo pipefail

MODE="auto"
SKIP_IF_INSTALLED="false"
for arg in "$@"; do
  case "$arg" in
    --overlay-only)
      MODE="overlay-only"
      ;;
    --full)
      MODE="full"
      ;;
    --skip-if-installed)
      SKIP_IF_INSTALLED="true"
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--overlay-only | --full] [--skip-if-installed]" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MARKER_FILE="/opt/venv/.tunix_vllm_overlay_installed"

if command -v uv >/dev/null 2>&1; then
  PIP_INSTALL=(uv pip install)
else
  PIP_INSTALL=(python3 -m pip install)
fi

unset PIP_NO_CACHE_DIR

if [[ "${SKIP_IF_INSTALLED}" == "true" && -f "${MARKER_FILE}" ]]; then
  echo "Tunix vLLM runtime overlay already installed in container (${MARKER_FILE}); linking editable checkout only..."
  if [[ -f "${REPO_ROOT}/pyproject.toml" ]]; then
    "${PIP_INSTALL[@]}" --no-deps -e "${REPO_ROOT}"
  fi
  exit 0
fi

NEED_FULL_VLLM_STACK="false"
if [[ "${MODE}" == "full" || "${OVERWRITE_VLLM_REQUIREMENTS:-false}" == "true" ]]; then
  NEED_FULL_VLLM_STACK="true"
elif [[ "${MODE}" == "auto" ]]; then
  if ! python3 -c "import vllm, tpu_inference" >/dev/null 2>&1; then
    NEED_FULL_VLLM_STACK="true"
  fi
fi

export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-tpu}"

if [[ "${NEED_FULL_VLLM_STACK}" == "true" ]]; then
  echo "Installing base vLLM and TPU-Inference requirements (VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE})..."
  if command -v uv >/dev/null 2>&1; then
    rm -rf /root/.cache/uv/git-v0/checkouts/*/*/build /root/.cache/uv/git-v0/checkouts/*/*/.deps 2>/dev/null || true
    "${PIP_INSTALL[@]}" \
      --refresh-package vllm \
      -r "${REPO_ROOT}/requirements/requirements.txt" \
      -r "${REPO_ROOT}/requirements/special_requirements.txt" \
      --override "${REPO_ROOT}/requirements/special_requirements.txt" \
      --torch-backend=cpu
  else
    "${PIP_INSTALL[@]}" --extra-index-url https://download.pytorch.org/whl/cpu \
      -r "${REPO_ROOT}/requirements/requirements.txt" \
      -r "${REPO_ROOT}/requirements/special_requirements.txt"
  fi
else
  echo "Detected pre-installed vLLM base stack; applying special_requirements.txt and Tunix overlay..."
  # Always apply `special_requirements.txt` (`--no-deps`, ~5s) so that any
  # `tpu-inference` commit bump ahead of the base container tag takes effect.
  "${PIP_INSTALL[@]}" --no-deps -r "${REPO_ROOT}/requirements/special_requirements.txt"
fi

# TODO(tunix-dev): Consolidate these Tunix/SFT/TFDS overlay packages into
# `pyproject.toml` optional-dependencies extras (e.g., `.[vllm,tpu]`) so that
# `Dockerfile` and CI workflows install a single declarative extra rather than
# maintaining a separate package list here.
#
# Note: `tensorflow`, `tensorflow-datasets`, `array-record`, `kagglehub`, and
# `kagglesdk` MUST be installed BEFORE `numpy==2.3.5` below so `tensorflow` does
# not upgrade NumPy past the `==2.3.5` pin required by `tpu-inference` / Numba.
echo "Installing Tunix runtime & TPU CI overlay dependencies..."
"${PIP_INSTALL[@]}" \
  "git+https://github.com/ayaka14732/jax-smi.git" \
  qwix \
  gcsfs \
  wandb \
  --upgrade flax \
  torchax \
  aqtp \
  tokamax \
  math_verify \
  drjax \
  tensorflow \
  tensorflow-datasets \
  array-record \
  kagglehub \
  kagglesdk \
  grpcio-tools

if [[ -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Installing Tunix in editable mode..."
  "${PIP_INSTALL[@]}" -e "${REPO_ROOT}"
fi

# TODO(tunix-dev): Remove this explicit `numpy==2.3.5` re-pin once
# `tpu-inference` and `numba` support `numpy>=2.4`.
echo "Locking numpy==2.3.5 for tpu-inference / Numba compatibility..."
"${PIP_INSTALL[@]}" "numpy==2.3.5"

if [[ -d "/opt/venv" && -w "/opt/venv" ]]; then
  touch "${MARKER_FILE}"
fi
