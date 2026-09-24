# Tunix Container & Dependency Installation Architecture

This directory contains the shared dependency installation and environment
verification scripts used by both `Dockerfile` and GitHub Actions CI
(`.github/workflows/build_and_test_tunix.yml`, `.github/workflows/cpu-tests.yml`,
`.github/workflows/tpu-tests.yml`, and
`.github/workflows/tpu-nightly-regression.yml`).

---

## 1. Why `Dockerfile` Builds `FROM python:3.12-slim` Instead of `vllm/vllm-tpu`

### How Upstream `vllm/vllm-tpu` Is Built

Upstream `vllm/vllm-tpu` (`vllm-project/tpu-inference/docker/Dockerfile`) starts
`FROM python:3.12-slim-bookworm` and installs all packages directly into the
global system Python via `pip` (without `uv` or `/opt/venv`):

1. Clones `vllm-project/vllm` at `VLLM_COMMIT_HASH`, removes `torch` pins from
   `requirements/tpu.txt`, and runs `pip install -r requirements/tpu.txt`,
   `pip install -e . --no-build-isolation`,
   `pip install lm-eval[api,math]==0.4.12`, and `pip install depyf`.
2. Copies `tpu-inference` into `/workspace/tpu_inference` and runs
   `pip install -r requirements.txt`,
   `pip install -r requirements_benchmarking.txt`, and `pip install -e .`
   (without `--no-deps`).
3. Sets `ENTRYPOINT ["/entrypoint.sh"]`.

### Why Our Split-Layer `FROM python:3.12-slim` Architecture Is Superior

1. **Isolated `/opt/venv` & `/app` Layout**: Installs all runtime packages into
   `/opt/venv` (`PATH="/opt/venv/bin:$PATH"`) with `WORKDIR /app` and
   `CMD ["bash"]`, without inheriting `/entrypoint.sh` or locking into a single
   static `vllm` + `tpu-inference` commit pair.
2. **Explicit Split-Layer Dependency Caching (`uv` + `--deps-only` vs
   `--source-only`)**: Instead of resolving a monolithic pip tree on every
   source edit, `Dockerfile` splits installation into independent cached layers
   with `--mount=type=cache,target=/root/.cache/uv`:
   - **Layer 1 (Base Runtime)**: `requirements/requirements.txt` +
     `requirements/special_requirements.txt` + `vllm` + `tpu-inference` +
     `tensorflow`/`tensorflow-datasets` + `numpy==2.3.5` inside `/opt/venv`.
   - **Layer 2 (MaxText Heavy PyPI Wheels)**: `install_maxtext.sh --deps-only`
     installs all third-party wheels from
     `requirements/maxtext_requirements.txt` and stays cached even when
     `MAXTEXT_REF` or Tunix source changes.
   - **Layer 3 (MaxText Source Only, ~15s)**:
     `ADD https://github.com/AI-Hypercomputer/maxtext/commits/${MAXTEXT_REF}.atom`
     selectively invalidates ONLY `install_maxtext.sh --source-only`
     (`--no-deps`), rebuilding MaxText in **~15 seconds** without `--no-cache`.
   - **Layer 4 (Tunix Source & Distributed Discovery Proto)**: Compiles
     `tunix/experimental/distributed/runtime/discovery/discovery_service.proto`
     explicitly and runs `uv pip install --no-deps -e /app`.
3. **Zero Transitive Version Clobbering**: Explicitly enforces
   `protobuf>=7.35.1` (required by Tunix gRPC runtime discovery) and
   `numpy==2.3.5` (required by `tpu-inference` / Numba) **after** all
   third-party wheel layers so `tensorflow` or `maxtext` never silently
   upgrade/downgrade them.

---

## 2. Single Sources of Truth (Where to Update Pins)

| Component | Single Source of Truth | Installer Script |
| :--- | :--- | :--- |
| **vLLM & TPU-Inference** | `requirements/requirements.txt` & `requirements/special_requirements.txt` | [`install_tunix_vllm_requirement.sh`](install_tunix_vllm_requirement.sh) |
| **MaxText & Adapter** | `Dockerfile` & [`install_maxtext.sh`](install_maxtext.sh) + `requirements/maxtext_requirements.txt` | [`install_maxtext.sh`](install_maxtext.sh) |
| **Raiden (`tpu_sync_jax`)** | [`install_raiden.sh`](install_raiden.sh) (`DEFAULT_RAIDEN_WHEEL_URL` & `DEFAULT_RAIDEN_WHEEL_SHA256`) | [`install_raiden.sh`](install_raiden.sh) |
| **Kubernetes CLI Tools** | [`install_k8s_tools.sh`](install_k8s_tools.sh) (`INSTALL_K8S_TOOLS=true`) | [`install_k8s_tools.sh`](install_k8s_tools.sh) |
| **DeepSWE & Agent Sandbox** | `examples/deepswe/install_deepswe.sh` (`INSTALL_DEEPSWE_DEPS=true`) | `examples/deepswe/install_deepswe.sh` |

---

## 3. Script Reference

### [`install_tunix_vllm_requirement.sh`](install_tunix_vllm_requirement.sh)

Installs the Tunix + vLLM + TPU-Inference runtime and locks `numpy==2.3.5`
(required by `tpu-inference` / Numba) **after** installing `tensorflow`,
`tensorflow-datasets`, and `array-record` so TFDS never upgrades NumPy mid-run.

- **`--full`**: Used by `Dockerfile` (`FROM python:3.12-slim`) to build the
  complete `/opt/venv` stack from `requirements/requirements.txt` and
  `requirements/special_requirements.txt`.
- **`--overlay-only --skip-if-installed`**: Used by `tpu-tests.yml` (`run_dev`).
  Links `-e .` in ~1 second when `/opt/venv/.tunix_vllm_overlay_installed` is
  present.

### [`install_maxtext.sh`](install_maxtext.sh)

Installs `AI-Hypercomputer/maxtext` at `MAXTEXT_REF` and
`MAXTEXT_VLLM_ADAPTER_REPO` at `MAXTEXT_VLLM_ADAPTER_REF`.

- **`--deps-only`**: Installs only the third-party PyPI wheels from
  `requirements/maxtext_requirements.txt` (excluding `git+https://` lines).
- **`--source-only`**: Installs `maxtext` (`AI-Hypercomputer/maxtext`) and
  `maxtext-vllm-adapter` with `--no-deps`, plus `protobuf>=7.35.1` and
  `numpy==2.3.5` (~15s).
- **`--skip-if-installed`**: Skips reinstallation when
  `/opt/venv/.tunix_maxtext_installed` is present.

### [`install_raiden.sh`](install_raiden.sh)

Installs the native TPU weight-synchronization wheel (`tpu_sync_jax`) from
`./raiden_wheels/*.whl` (`RAIDEN_WHEEL_DIR`) or `RAIDEN_WHEEL_URL` and compiles
`tunix/experimental/distributed/runtime/discovery/discovery_service.proto`.

### [`install_k8s_tools.sh`](install_k8s_tools.sh)

Installs `google-cloud-cli`, `google-cloud-cli-gke-gcloud-auth-plugin`,
`kubectl`, `k9s`, and debugging utilities when `INSTALL_K8S_TOOLS=true`.

### [`verify_environment.py`](verify_environment.py)

Validates Python packages and `tunix.__version__` during the Docker build
(`Dockerfile`).

---

## 4. Building, Resolving & Publishing Container Images

```bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg INSTALL_MAXTEXT=true \
  -t tunix-mlperf:latest .
```

In `.github/workflows/build_and_test_tunix.yml` (`build_tunix_docker`), every CI
run resolves or builds an immutable `tunix/mlperf:<github.sha>` container image
before launching `tunix_cpu_unit_tests` (`run_vllm`) and `tunix_tpu_unit_tests`
(`run_dev`):

1. **Content-Addressed Environment Hash (`:env-<hash>` Fast Path, ~5s)**:
   Computes a SHA-256 hash (`ENV_HASH`) over all container-defining files
   (`Dockerfile`, `pyproject.toml`, `requirements/*.txt`,
   `scripts/install_*.sh`, `scripts/verify_environment.py`,
   `examples/deepswe/install_deepswe.sh`, and `discovery_service.proto`). On
   pull requests where `:env-${ENV_HASH}` already exists in Artifact Registry,
   `build_tunix_docker` aliases `:env-${ENV_HASH}` to `:${GITHUB_SHA}` and
   `:${SHORT_SHA}` in ~5s without rebuilding; downstream jobs overlay the PR's
   Tunix code in ~1s at runtime.
2. **Build & Multi-Region Publish Path**: When any container-defining file
   changes (or on `main` / `schedule` / `workflow_dispatch`),
   `build_tunix_docker` builds and verifies `Dockerfile`
   (`INSTALL_MAXTEXT=true`) via BuildKit with Artifact Registry layer caching
   (`type=registry,ref=.../tunix/mlperf:buildcache`) and publishes
   `:${GITHUB_SHA}`, `:${SHORT_SHA}`, and `:env-${ENV_HASH}` to:
   - `us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/mlperf`
   - `europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/mlperf`
3. **Immutable Image Passing & Concurrency Safety**: Both `cpu-tests.yml`
   (`run_vllm`) and `tpu-tests.yml` (`run_dev`) receive
   `needs.build_tunix_docker.outputs.image_uri` (`:${GITHUB_SHA}`), guaranteeing
   every test job in a workflow run executes against the exact same verified
   container image. The `:latest` tag is only updated on `main` branch runs,
   preventing concurrent PRs from overwriting `:latest`.

---

## 5. CI Caching & Test Throughput Optimizations

### Container-Compatible Model, Dataset & Checkpoint Caching (Zero Image Bloat)

To keep `tunix/mlperf` generic for Trellis and MLPerf workloads (and keep
container pull + initialization on TPU runners at ~13s), **zero model weights,
datasets, or checkpoints are baked into the Docker image**. Instead,
`.github/workflows/tpu-tests.yml` mounts and restores cached artifacts into the
container's `/root/.cache` hierarchy at runtime:

- **Pinned Container Cache Roots**:
  - `HF_HOME=/root/.cache/huggingface`: Hugging Face Hub weights & tokenizers.
  - `KAGGLEHUB_CACHE=/root/.cache/kagglehub`: Kaggle model weights (e.g.
    `gemma2-2b-it` used in SFT smoke tests).
  - `TFDS_DATA_DIR=/root/.cache/tensorflow_datasets`: Pre-built TFDS datasets
    (e.g. `gsm8k`).
  - `JAX_COMPILATION_CACHE_DIR=/root/.cache/jax_compilation_cache`: Persistent
    XLA TPU/CPU compiled executables across runs.
  - `ARTIFACT_ROOT=/root/.cache/tunix_artifacts/qwen3_dist_gsm8k`: Caches both
    the downloaded `Qwen/Qwen3-0.6B` safetensors (`models/`) and the
    pre-converted MaxText Orbax checkpoint (`maxtext_models/`), allowing
    `ci_smoke_gsm8k_qwen3_0p6b_maxtext.sh` to skip `convert_hf_to_maxtext` on
    warm runs while writing ephemeral training checkpoints to `/tmp`.
- **Selective Cache Exclusions (`xet/` and `>7B` Models)**:
  - `HF_XET_HIGH_PERFORMANCE=1` accelerates cold downloads via Xet chunk
    streaming, but writes duplicate staging chunks to
    `/root/.cache/huggingface/xet` alongside the final `hub/` files. Excluding
    `!/root/.cache/huggingface/xet` prevents archiving duplicate chunks.
  - `Qwen/Qwen2.5-7B` (~15 GB raw / ~6.5 GB compressed) streams from the Xet CDN
    to a GCE `v6e-8` VM in `us-central1` in ~7s, whereas restoring a 6.5 GB
    archive from GitHub Actions cache (Azure Blob Storage) takes ~52s and
    consumes most of GitHub's 10 GB repository cache quota. Excluding
    `!/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B*` keeps the
    `run_dev` cache compact (~1.5 GB) and fast to restore (~10s).

### Fast Dependency Resolution (`uv`) & Parallel Test Execution (`pytest-xdist`)

- **`uv pip install --system` vs. `pip install`**: In
  `.github/workflows/cpu-tests.yml`, replacing `python -m pip install` with
  `uv pip install --system` reduces wheel resolution and installation on
  ephemeral Ubuntu runners from ~70–127s to ~10–14s. When combining PyPI with
  `--extra-index-url https://download.pytorch.org/whl/cpu` (in `run_dev`),
  `--index-strategy unsafe-best-match` is passed so `uv` can select newer PyPI
  versions (such as `requests>=2.32.2` required by `datasets>=3.0.0`) rather
  than pinning to older versions hosted on the PyTorch CPU index
  (`requests==2.28.1`).
- **`pytest -n auto --dist=loadfile`**: Parallelizes multi-file CPU unit test
  suites across all available runner vCPUs while keeping all tests within a
  given module on the same worker (`--dist=loadfile`), avoiding redundant
  per-worker initialization of module-scoped JAX models and meshes.
