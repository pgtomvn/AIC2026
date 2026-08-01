# AIC 2026 multimodal video retrieval

An early-stage retrieval system for the Ho Chi Minh City AI Challenge. The repository is
organized around Textual Known Item Search (KIS), Visual Question Answering, and Temporal
Alignment of Key Events (TRAKE).

> [!IMPORTANT]
> This is an executable scaffold, not a completed retrieval engine. The lightweight tests
> exercise constructors and similarity utilities; many model, indexing, retrieval, and
> evaluation methods still raise `NotImplementedError`. The Streamlit application displays
> placeholder results until those pipelines are connected.

## Repository layout

```text
app/        Streamlit application and presentation components
config/     Central paths, model settings, service endpoints, and retrieval weights
scripts/    Environment helpers plus extraction, indexing, embedding, and evaluation CLIs
src/        Extraction, embedding, indexing, retrieval, VLM, and shared utilities
tests/      Lightweight unit tests; model/service integration is not required by default
checksums/  Integrity manifest for the immutable source archives
```

The two root ZIP files are immutable source data. They are intentionally ignored by Git and
must never be rewritten in place. Extracted data, model weights, indexes, logs, outputs,
service state, local tools, environments, and secrets are also ignored.

## Prerequisites

- Windows 10/11 with PowerShell 5.1 or newer for the primary local workflow. PowerShell 7 is
  recommended and is used by Linux CI.
- Internet access for the first bootstrap. The bootstrap downloads the pinned uv 0.11.16
  binary, verifies its published SHA-256 checksum, and lets uv install a project-local Python
  3.11 runtime.
- Docker Desktop or Docker Engine with Compose v2 only when running Qdrant and Meilisearch.
- NVIDIA tooling is optional. The default environment does not install PyTorch, model
  weights, PaddleOCR, Whisper, OpenCLIP, or vLLM.

No global Python, uv, Conda, CUDA, or model installation is required for the lightweight
workflow.

## First-time setup

Run these commands from the repository root:

```powershell
# Optional quick inventory. Omit -SkipArchiveHash for full archive verification.
.\scripts\preflight.ps1 -SkipArchiveHash

# Install pinned uv/Python locally, create .env when absent, and sync uv.lock.
.\scripts\bootstrap.ps1

# Run lint and the lightweight unit tests.
.\scripts\check.ps1
```

`bootstrap.ps1` never overwrites an existing `.env`. For a new file it copies the documented
settings and generates independent high-entropy `QDRANT_API_KEY` and
`MEILISEARCH_MASTER_KEY` values. Keep that file local.

The pinned uv executable is under `.tools/uv/0.11.16/`. On the primary Windows workflow:

```powershell
$Uv = Resolve-Path .tools/uv/0.11.16/uv.exe
& $Uv run --frozen streamlit run app/streamlit_app.py
```

On Linux, use the same command with `.tools/uv/0.11.16/uv`.

## Routine commands

```powershell
# Recreate the Python environment and test/lint caches; retain the package cache.
.\scripts\reset-env.ps1

# Also discard the downloadable uv package cache when explicitly required.
.\scripts\reset-env.ps1 -PurgePackageCache

# Validate Docker Compose without starting services.
.\scripts\check.ps1 -IncludeCompose

# Start local search services, then verify their health.
docker compose up -d
.\scripts\check.ps1 -IncludeServices

# Include a headless Streamlit startup/health smoke test.
.\scripts\check.ps1 -IncludeStreamlit

# Validate and CRC-read the complete source archives (slow).
.\scripts\check.ps1 -IncludeArchives

# Stop services without deleting their persisted data.
docker compose stop
```

Qdrant listens only on `127.0.0.1:6333` (REST) and `127.0.0.1:6334` (gRPC).
Meilisearch listens only on `127.0.0.1:7700`. Their persistent bind mounts live under
`docker_data/`.

The safe extraction CLI supports inspection, one-video sampling, and an explicit full run:

```powershell
$Uv = Resolve-Path .tools/uv/0.11.16/uv.exe
& $Uv run --frozen python scripts/extract_data.py --inspect
& $Uv run --frozen python scripts/extract_data.py --sample L21_V001
& $Uv run --frozen python scripts/extract_data.py --full
```

Existing extracted files are skipped by default. Use `--force` only when replacement is
intentional. A normal environment reset never removes archives, extracted data, indexes,
models, logs, outputs, `.env`, or Docker state.

## Environment and dependency policy

`pyproject.toml` and `uv.lock` define the supported lightweight environment. Always use
`--frozen` for normal commands so dependency drift is detected rather than silently resolved.
`environment.legacy.yml` records the original mixed Conda/pip proposal for reference only;
do not use it for reproducible setup.

Heavy ML backends are deliberately deferred because their CUDA, platform, and model choices
must be made separately. In particular, vLLM is a Linux-oriented service and the proposed
Qwen2-VL-7B model is not appropriate for a 4 GiB local GPU. Keep that workload remote or
design a smaller-model profile before adding it to the lockfile.

## Troubleshooting

- If PowerShell blocks local scripts, use `Set-ExecutionPolicy -Scope Process Bypass`; do not
  weaken the machine-wide policy.
- Windows PowerShell 5.1 can display UTF-8 poorly. The project scripts configure UTF-8 for
  their process. For manual commands, set
  `[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)` and
  `$env:PYTHONUTF8 = "1"`.
- Run `.\scripts\preflight.ps1` for versions, GPU/Docker availability, ports, disk space,
  stale PATH entries, and archive integrity. It is read-only.
- A missing `uv.lock` is an incomplete checkout. Do not let bootstrap resolve dependencies
  implicitly; restore or regenerate and review the lockfile as an intentional maintenance
  change.
