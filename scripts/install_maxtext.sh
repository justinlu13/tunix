#!/bin/bash
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

# Unified MaxText + maxtext-vllm-adapter installer for Tunix.
#
# Modes:
#   --deps-only         Install only third-party wheels from
#                       `requirements/maxtext_requirements.txt` (excluding
#                       `git+https://` source packages). Used as a cached Docker
#                       layer so advancing `MAXTEXT_REF` does not rebuild wheels.
#   --source-only       Install only `maxtext` and `maxtext-vllm-adapter` with
#                       `--no-deps`, plus `protobuf>=7.35.1` and `numpy==2.3.5`.
#   --all               (Default) Run both `--deps-only` and `--source-only`.
#   --skip-if-installed Exit 0 immediately if MaxText is already installed in
#                       the container (`/opt/venv/.tunix_maxtext_installed`).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAXTEXT_REQ_FILE="${REPO_ROOT}/requirements/maxtext_requirements.txt"
MARKER_FILE="/opt/venv/.tunix_maxtext_installed"

MODE="all"
SKIP_IF_INSTALLED=false
for arg in "$@"; do
  case "${arg}" in
    --deps-only) MODE="deps-only" ;;
    --source-only) MODE="source-only" ;;
    --all) MODE="all" ;;
    --skip-if-installed) SKIP_IF_INSTALLED=true ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

if [[ "${SKIP_IF_INSTALLED}" == "true" ]] && { [[ -f "${MARKER_FILE}" ]] || python3 -c "import maxtext, maxtext_vllm_adapter" >/dev/null 2>&1; }; then
  echo "=== MaxText & maxtext-vllm-adapter already installed; skipping ==="
  exit 0
fi

MAXTEXT_REF="${MAXTEXT_REF:-63d5db085447274e9cd8769334866fe5268459a9}"

pip_install() {
  if command -v uv >/dev/null 2>&1 && [[ -n "${VIRTUAL_ENV:-}" || -d "/opt/venv" ]]; then
    uv pip install "$@"
  else
    python3 -m pip install "$@"
  fi
}

if [[ "${MODE}" == "deps-only" || "${MODE}" == "all" ]]; then
  echo "=== Installing MaxText third-party dependencies from ${MAXTEXT_REQ_FILE} ==="
  grep -v -E '^\s*#|^\s*$|git\+https://' "${MAXTEXT_REQ_FILE}" > /tmp/maxtext_pypi_deps.txt
  pip_install -r /tmp/maxtext_pypi_deps.txt
  rm -f /tmp/maxtext_pypi_deps.txt
fi

if [[ "${MODE}" == "source-only" || "${MODE}" == "all" ]]; then
  echo "=== Installing MaxText & maxtext-vllm-adapter (AI-Hypercomputer/maxtext@${MAXTEXT_REF}) ==="
  # TODO(maxtext-dev): Remove `--no-deps` once MaxText decouples its post-training
  # wheel dependencies from conflicting `jax`/`torch` pins.
  pip_install --no-deps --force-reinstall --no-cache-dir \
    "maxtext @ git+https://github.com/AI-Hypercomputer/maxtext.git@${MAXTEXT_REF}" \
    "maxtext-vllm-adapter @ git+https://github.com/AI-Hypercomputer/maxtext.git@${MAXTEXT_REF}#subdirectory=src/maxtext/integration/vllm"
  # Ensure `protobuf>=7.35.1` (for Tunix gRPC runtime discovery) and
  # `numpy==2.3.5` (for `tpu-inference`/Numba) remain locked.
  pip_install "protobuf>=7.35.1" "numpy==2.3.5"
  if [[ -d "/opt/venv" ]]; then
    touch "${MARKER_FILE}" || true
  fi
fi
