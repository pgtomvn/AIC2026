# AIC 2026 — Multimodal Video Retrieval

[![CI](https://github.com/pgtomvn/AIC2026/actions/workflows/ci.yml/badge.svg)](https://github.com/pgtomvn/AIC2026/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A multimodal video search system for the **Ho Chi Minh City AI Challenge 2026** (AIC 2026).
Given a natural-language description (Vietnamese or English), it finds the exact video moment,
answers questions about a frame, and aligns multi-step events across a video timeline.

## Features

The system supports three competition workflows over a shared runtime:

| Workflow | What it does |
|---|---|
| **Textual KIS** (Known-Item Search) | Hybrid search over visual embeddings, OCR, ASR, objects, and metadata; fuses channels with weighted Reciprocal Rank Fusion, aggregates by shot/time window, and refines the top windows against the raw video to recover an exact frame index. Supports interactive refinement (positive/negative feedback, pinning, neighbouring frames). |
| **Visual Q&A** | Two-stage *retrieve-then-reason*: retrieve five candidate frames for an event description, let the operator pick one, then ask a vision-language model about exactly that frame. Gemini is the primary provider (with explicit per-session consent); a local Qwen2.5-VL model served by Ollama is the offline fallback. |
| **TRAKE** (temporal alignment) | Takes a global description plus 2–20 ordered stages and finds the videos where those stages occur in order, using a dynamic-programming alignment over full-timeline trajectories. Returns the top 100 distinct videos with per-stage frames. |

Other highlights:

- **Vietnamese-first querying** — offline query translation/routing and accent-folded text search.
- **Cross-batch search** — query Batch 1, Batch 2, or both at once with rank-level fusion.
- **Strict frame identity** — every result carries a canonical decoded `source_frame_idx`;
  incompatible embedding spaces (organizer OpenCLIP 512-D vs. SigLIP2 768-D) are never mixed.
- **Validated submission export** — CSV exports for KIS, Q&A, and TRAKE are checked for IDs,
  ranks, duplicates, and limits before download.
- **Resumable ingestion** — ZIP-streamed extraction with a SQLite job ledger, so archives are
  never fully expanded and interrupted runs resume safely.
- **Laptop-friendly** — the default profile targets a 4 GiB GPU / 16 GiB RAM machine; an
  accelerated profile is available for 12–24 GiB GPUs.

## Architecture

```text
 Keyframe/video ZIPs ──► catalog + resumable SQLite ledger
                               │
          ┌────────────────────┼─────────────────────┐
          ▼                    ▼                     ▼
   visual embeddings      OCR / ASR text      objects / metadata
   (SigLIP2, OpenCLIP)    (PaddleOCR, VietOCR, Whisper)
          │                    │                     │
          ▼                    └──────────┬──────────┘
       Qdrant                         Meilisearch
          └──────────────┬───────────────┘
                         ▼
            weighted RRF fusion + shot/window aggregation
                         │
      ┌──────────────────┼──────────────────────┐
      ▼                  ▼                      ▼
 Textual KIS        Visual Q&A               TRAKE
 (exact-frame       (1 frame → Gemini /      (ordered-stage
  refinement)        local Qwen2.5-VL)        DP alignment)
      └──────────────────┼──────────────────────┘
                         ▼
          FastAPI runtime ◄── Streamlit UI (Vietnamese)
                         │
                validated CSV export
```

See the [operations guide](docs/guides/operations.md) for detailed per-workflow diagrams,
canonical data contracts, and the full HTTP API.

## Tech stack

Python 3.11 · FastAPI · Streamlit · Qdrant · Meilisearch · PyTorch · SigLIP2 / OpenCLIP ·
PaddleOCR / VietOCR · Faster-Whisper · Ollama (Qwen2.5-VL) · Google Gemini · uv · Docker Compose

## Repository layout

```text
app/          Streamlit competition console and UI components
config/       settings.yaml (models, indexes, retrieval policy, limits) and organizer baselines
src/
  contracts.py  Canonical records and versioned competition specifications
  api.py        FastAPI transport over the shared retrieval runtime
  ingest/       Archive catalog, SQLite ledger/artifacts, reference and native pipelines
  extraction/   Media probing, adaptive keyframes, OCR, ASR
  embedding/    SigLIP2, Vietnamese text, and OpenCLIP adapters
  indexing/     Qdrant and Meilisearch adapters
  query/        Query decomposition, translation routing, binding plans
  retrieval/    Textual KIS, Visual Q&A, TRAKE, fusion, reranking, interactive search
  vlm/          Gemini / Ollama / Florence-2 providers and GPU coordination
  evaluation/   KIS, Q&A, and TRAKE evaluators
  submission/   Competition export adapters (see also export/)
scripts/      Bootstrap, catalog, ingest, evaluate, serve, drill, and export CLIs
notebooks/    Colab/Kaggle workers for cloud feature extraction and ASR
tests/        CPU-only, offline test suite
checksums/    Expected hashes for archives and prepared runtime assets
docs/         Guides, design notes, and batch release notes
```

Imports and constructors are side-effect free: they never create data, load models, download
assets, or connect to services.

## Getting started

The supported workflow is **Windows PowerShell** with Python 3.11 managed by a pinned, project-local
[uv](https://github.com/astral-sh/uv). CI also runs the checks on Ubuntu.

```powershell
git clone https://github.com/pgtomvn/AIC2026.git
cd AIC2026

# Read-only environment inventory
.\scripts\preflight.ps1 -SkipArchiveHash

# Install uv/Python locally, create .env with fresh local service keys, sync the lockfile
.\scripts\bootstrap.ps1

# Lint + CPU/offline test suite
.\scripts\check.ps1
```

### Running the full stack

Searching real data additionally requires the competition dataset (keyframe/video archives and
the organizer reference packs, obtained from the official AIC source), local services, and
prepared models:

```powershell
# Start Qdrant (6333/6334) and Meilisearch (7700), bound to localhost
docker compose up -d

# Install the competition runtime (PyTorch, OCR, ASR, ...) into its own environment
.\scripts\bootstrap_runtime.ps1 -Profile laptop

# Start FastAPI (8080) and the Streamlit UI (8504)
.\scripts\run_app.ps1 -Profile laptop -BatchId aic25-b1-reference -ApiPort 8080 -Port 8504
```

Model preparation, reference-pack import, and native ingestion are explicit steps described in:

- [Operations guide](docs/guides/operations.md) — models, data operations, API, acceptance gates
- [Hướng dẫn cài đặt cho thành viên (Tiếng Việt)](docs/guides/team_setup_vi.md) — step-by-step setup
- [Cloud feature extraction](docs/guides/cloud_feature_extraction.md) — Colab/Kaggle workers

### Configuration

Non-secret defaults live in [`config/settings.yaml`](config/settings.yaml). Credentials and local
overrides go in an untracked `.env` (see [`.env.example`](.env.example)):

| Variable | Purpose |
|---|---|
| `QDRANT_API_KEY`, `MEILISEARCH_MASTER_KEY` | Local service keys (generated by `bootstrap.ps1`) |
| `GEMINI_API_KEY` | Optional cloud Visual Q&A provider; kept server-side |
| `OLLAMA_URL` | Local vision-language model endpoint (loopback only) |
| `AIC_PROFILE` | `laptop` (default) or `accelerated` |
| `AIC_BATCH_ID` | Default competition batch for the launcher and UI |

## Testing

```powershell
.\scripts\check.ps1                                    # lint + offline unit tests
.\scripts\check.ps1 -IncludeCompose -IncludeServices   # also check Docker services
.\scripts\check.ps1 -IncludeLocalVLM                   # explicit local-model integration
```

The default suite is CPU-only and offline. Tests that need Docker, a GPU, a model, the dataset,
or network access are opt-in integration checks.

## Documentation

All documentation is indexed in [docs/README.md](docs/README.md).

## AI assistance

This project was built with the help of AI assistants. **ChatGPT** (OpenAI) and **Claude**
(Anthropic) were used to discuss ideas and design options, and to help write, review, and debug
code and documentation. All AI-assisted output was reviewed, tested, and integrated by the
project team, who remain responsible for the final code.

## Acknowledgements

Built for the Ho Chi Minh City AI Challenge 2026.
Competition data and baseline material belong to the organizers and are not redistributed here.
Bundled UI fonts are distributed under the SIL Open Font License (see `app/static/fonts/`).

## License

Released under the [MIT License](LICENSE).
