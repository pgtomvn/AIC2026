# AIC 2026 multimodal video retrieval

This repository implements the Textual Known Item Search (Textual KIS) foundation, a Phase 2
retrieve-then-reason Visual Q&A subsystem, and a Phase 3 TRAKE temporal-alignment engine for the
Ho Chi Minh City AI Challenge. They share canonical frame records, source-frame semantics, API
runtime, readiness probes, and the Vietnamese competition UI. Model preparation, corpus indexes,
and competition-day data provisioning remain explicit operational steps rather than a completed
turnkey deployment.

> [!IMPORTANT]
> The configured adapters implement historical Textual KIS interval scoring and historical
> numeric-only Visual Q&A exact matching, a provisional conservative general-text QA mode, and
> the `historical_2025_provisional` TRAKE adapter.
> They are defaults, not claims about the final AIC 2026 rules. Replace the versioned specs and
> their tests when the organizer publishes the 2026 schema.

## Textual KIS architecture

```text
Stable ZIPs or loose videos
        |
        +--> archive catalog + resumable SQLite job ledger
        |        |
        |        +--> streamed organizer maps/features/objects/metadata
        |        +--> probe -> shots/anchors -> thumbnails -> visual/OCR/ASR
        |                                      |
        |                                      v
        |                         canonical frame/text artifacts
        |                              |                 |
        |                              v                 v
        |                       Qdrant vectors    Meilisearch text
        |                              \                 /
        |                               weighted RRF (k=60)
        |                                      |
        |                         shot/window aggregation
        |                                      |
        |                    local raw-video exact-frame refinement
        |                                      |
        +--------------------------> ranked top-100 hits
                                               |
                         FastAPI / Streamlit / validated CSV export
```

The laptop default uses `google/siglip2-base-patch16-224` for native 768-D visual vectors.
Organizer `ViT-B-32-quickgelu/openai` 512-D features, Vietnamese semantic text, and the optional
accelerated visual encoder use separate collections; incompatible embeddings are never mixed.
OCR, ASR, semantic text, metadata, and visual results are fused by rank rather than by
incomparable raw scores.

Temporal aggregation operates in timestamps and respects shot boundaries. The top windows can
be decoded from raw video at 6 FPS within a +/-4 second neighborhood to recover an exact decoded
`source_frame_idx`. Temporal NMS and per-video quotas preserve top-100 coverage instead of
spending ranks on near-duplicates.

## Visual Q&A architecture

```text
event description -> Textual KIS (refine_raw=false) -> 20 diverse temporal clusters
                                                        |
                                                        v
                                  3--5 canonical frames within +/-4 seconds
                                                        |
                           bounded ZIP/local resolver -> normalized inline images
                                                        |
                       Gemini 3.6 Flash -> optional cache-only Qwen3-VL fallback
                                                        |
                         normalized answer + evidence positions + safe provenance
                                                        |
                  representative-first / neighbor-round-robin top-100 predictions
                                                        |
                       SQLite session/cache -> API/UI -> validated QA CSV
```

Interactive searches reason over five clusters and expand idempotently to 10, 15, or 20;
automatic searches perform the same bounded cascade to 20. Reference ZIP members are streamed
without extraction. The resolver accepts only persisted canonical frame identities, applies EXIF
orientation and bounded resizing, and hashes the normalized bytes used by provider/cache calls.

## TRAKE architecture

```text
global description + 2--20 ordered stages
                         |
              one compatible dense space
          organizer OpenCLIP (reference batch)
                    or SigLIP2 (native batch)
                         |
       grouped top-300 retrieval per prompt -> at most 200 videos
                         |
       one warm-up scan -> compact exact full-timeline trajectory cache
          (canonical source frames, timestamps, normalized vectors)
                         |
       one batched cosine pass + strict prefix-max DP in O(stages x frames)
                         |
       one best alignment per video -> ranked top-100 distinct videos
                         |
              API / Vietnamese timeline / validated CSV
```

TRAKE never mixes organizer 512-D vectors with native 768-D vectors. Each result contains strictly
increasing canonical `source_frame_idx` values; `frame_ids` is only the public serialization alias.
Warm-up streams the selected compatible collection once, rejects malformed vectors and mappings,
merges duplicate organizer ordinals by canonical source frame, and atomically publishes immutable
NumPy trajectories. Stage-B then evaluates every canonical frame of each candidate video; it does
not approximate alignment with grouped-search seed frames. Videos with fewer frames than stages
remain infeasible.
The evaluator assigns zero to a wrong video and partial credit for correctly localized stages in
the correct video, then reports the best R-Score at cutoffs 1, 5, 20, 50, and 100.

## Canonical contracts

The validated contracts live in `src/contracts.py` and are JSON-friendly.

| Contract | Important fields |
|---|---|
| `FrameRecord` | Batch/video identity, organizer ordinal, decoded source-frame index, PTS/time base, timestamp, FPS rational, shot, sampling reasons, URIs, source hash, extractor version |
| `TextSegment` | OCR/ASR/metadata interval, optional source frame, raw/NFC/accent-folded text, confidence, model version |
| `QueryRequest` | Original text, automatic/interactive mode, batch filters, before/current/after clauses, feedback IDs, result limit |
| `SearchHit` | Rank, video, decoded source frame, timestamp, shot, fused score, channel ranks, evidence, thumbnail |
| `CompetitionSpec` | Versioned output columns, video normalization, frame semantics, result maximum, cutoffs, interval rules |
| `QAQueryRequest` | Event-only retrieval text, visual question, mode, batch filters, answer-spec ID, result limit |
| `QAReasoningResult` | Raw/normalized answer, answerability, evidence positions, provider/model/prompt/cache provenance |
| `QAPrediction` | Canonical batch/video/source frame, answer, ranking proxy, answer source, window and evidence |
| `QACompetitionSpec` | Independent numeric/general answer normalization, three-column export, intervals and cutoffs |
| `TRAKEQueryRequest` | One batch, global description, 2--20 ordered stages, and top-100 bound |
| `TRAKEResponseItem` | One video, strict chronological stage matches, confidence proxy, and public `frame_ids` alias |
| `TRAKECompetitionSpec` | Versioned provisional no-header export and best-at-cutoff interval scoring |

`SearchHit` is also the public API response contract. Internal retrieval dictionaries use
`score`/`image_uri`, but FastAPI validates and normalizes them to `fused_score`/`thumbnail_uri`
with contiguous ranks before returning a response.

Internally, `source_frame_idx` always means the zero-based decoded ordinal in the source video.
`keyframe_ordinal` is the organizer/extractor image position and must never be submitted as a
frame index. Only the competition adapter emits `<video_name>,<frame_idx>`.

## Repository layout

```text
app/                  Streamlit competition console and UI components
config/settings.yaml Laptop defaults, model pins, index names, retrieval policy, limits
scripts/              Bootstrap, catalog, ingest, evaluate, serve, drill, and export CLIs
src/contracts.py      Canonical records and versioned competition specification
src/ingest/           Archive catalog, SQLite ledger/artifacts, reference and native pipelines
src/extraction/       Media probe, adaptive keyframes, OCR, ASR, calibration
src/embedding/        SigLIP2, Vietnamese text, and OpenCLIP adapters
src/indexing/         Lazy Qdrant and Meilisearch adapters
src/retrieval/        KIS, QA clustering/media/sessions, and strict TRAKE temporal alignment
src/vlm/              Lazy Gemini/Qwen reasoners, normalization, and the compatibility facade
src/evaluation/       Textual KIS, Visual Q&A, and provisional TRAKE evaluators
src/api.py            FastAPI transport over the shared retrieval runtime
tests/                 CPU-only/offline default tests plus explicit integration boundaries
checksums/             Immutable archive and prepared-runtime asset manifests
```

Imports and constructors do not create data, load models, download assets, or connect to
services. Those actions occur only through explicit CLI/application startup calls.

## Execution profiles

### Laptop profile (default)

`config/settings.yaml` is sized for an RTX 2050 4 GiB and 16 GiB system RAM:

- SigLIP2 Base, FP16, starting batch size 8 with CUDA OOM batch reduction.
- One GPU-heavy stage at a time: core visual indexing first, then Whisper ASR.
- PaddleOCR and Vietnamese text embedding on CPU.
- Scene candidates at threshold 0.30, at most two seconds between retained frames, and at least
  0.5 seconds between candidates.
- 512-pixel WebP thumbnails; full-resolution frames are decoded only for inspection/refinement.
- 12 GiB maximum scratch use and a mandatory 30 GiB free-space reserve.

### Accelerated profile (optional)

For a 12--24 GiB NVIDIA GPU, set `profile: accelerated`. This enables the profile-gated
OpenCLIP channel and also permits the pinned cache-only
`Qwen/Qwen3-VL-4B-Instruct` fallback after Gemini. Qwen acquires the shared GPU lease, unloads
dense encoders, reasons over one five-frame window at a time, then releases VRAM before the next
dense retrieval. The laptop profile remains Gemini-only for Visual Q&A.

## Bootstrap and services

The supported workflow is Windows PowerShell with Python 3.11 managed locally by pinned uv
0.11.16. Run from the repository root:

```powershell
# Quick read-only inventory. Omit the flag when full archive hashing is relevant.
.\scripts\preflight.ps1 -SkipArchiveHash

# Install uv/Python locally, preserve or create .env, and sync the default lock.
.\scripts\bootstrap.ps1

# Required before normal Python handoff: lint and CPU/offline tests.
.\scripts\check.ps1
```

`bootstrap.ps1` does not overwrite `.env`. A new file receives independent local Qdrant and
Meilisearch keys. Keep it uncommitted. Configure a newly rotated Gemini credential only as
`GEMINI_API_KEY` in that file; never reuse a key pasted into chat, source, logs, or command
history. Qdrant binds to `127.0.0.1:6333/6334` and Meilisearch to
`127.0.0.1:7700`. Qdrant uses Docker-managed Linux volumes so Windows bind-mount semantics cannot
break atomic segment renames; Meilisearch state and any retained Qdrant migration rollback copy
remain under `docker_data/`.

Runtime selection is deterministic for both `AIC_PROFILE` and `AIC_BATCH_ID`: an explicit
process environment value wins, then the untracked `.env` value, then `config/settings.yaml`.
The supported profiles are `laptop` and `accelerated`; the configured default batch is
`aic25-b1-reference`. The canonical launcher sets both values for its readiness check and the
Streamlit process, so the banner, searches, and exports use the same batch.

The default dependency set can run tests, the evaluator, API/UI startup, and lightweight
adapters. Install the locked competition ML group explicitly before model preparation or native
inference:

```powershell
$Uv = Resolve-Path .tools/uv/0.11.16/uv.exe
& $Uv sync --frozen --group competition

docker compose up -d
.\scripts\check.ps1 -IncludeCompose -IncludeServices

# Create or validate all configured Qdrant collections and lexical indexes.
& $Uv run --frozen --group competition python scripts/init_indexes.py --batch-id AIC26-B2
```

This group includes PyAV, CUDA PyTorch 2.5.1/torchvision 0.20.1, Transformers, OpenCLIP,
PaddleOCR/PaddlePaddle, Faster-Whisper, and the Hugging Face client. It does not download model
weights merely by being installed.

## Prepare pinned models and prove offline readiness

Model preparation is an explicit, networked, one-time operation. Revisions for SigLIP2,
Vietnamese embeddings, and Faster-Whisper are pinned in `config/settings.yaml`; PaddleOCR assets
are acquired into the configured project cache.

```powershell
$Uv = Resolve-Path .tools/uv/0.11.16/uv.exe

# Download all configured core models and write checksums/runtime-assets.sha256.
& $Uv run --frozen --group competition python scripts/prepare_models.py

# Or prepare selected families. Repeat --only as needed in a single invocation.
& $Uv run --frozen --group competition python scripts/prepare_models.py `
  --only visual --only organizer --only accelerated --only text --only asr --only ocr --only qwen

# Add other prepared runtime assets or tool directories to a checksum manifest.
& $Uv run --frozen python scripts/freeze_assets.py models .tools/my-runtime-tool
```

The preparation commands create material generated state under `models/` and a runtime checksum
manifest. They have not been run merely because this repository was checked out.

Native ingestion also requires the pinned local FFmpeg/ffprobe manifest configured at
`.tools/ffmpeg/active.json`. Obtain a trusted Windows FFmpeg ZIP and its independently published
SHA-256, then install only the two required executables:

```powershell
$FfmpegArchive = Resolve-Path D:\installers\ffmpeg-release.zip
$TrustedFfmpegSha256 = "<64-character SHA-256 from the trusted source>"
& $Uv run --frozen python scripts/install_ffmpeg.py `
  --archive $FfmpegArchive --sha256 $TrustedFfmpegSha256

# Run after model and tool preparation so the manifest covers both sets.
& $Uv run --frozen python scripts/freeze_assets.py models .tools/ffmpeg
```

The installer rejects a digest mismatch, unsafe ZIP members, missing executables, destinations
outside project `.tools`, and conflicting existing versions. Native ingestion verifies the active
tool manifest before probing any video.

Verify the actual competition machine with networking disabled (or disconnected):

```powershell
# Checks the asset manifest, disk reserve, and local search-service health.
& $Uv run --frozen python scripts/competition_drill.py

# Also verify a prepared batch and execute one offline retrieval.
& $Uv run --frozen python scripts/competition_drill.py `
  --batch-id aic25-b1-reference --query "một người mặc áo đỏ đi qua cây cầu"

# Full logical drill: feedback, temporary export, client reconnect, and recovery fixture.
& $Uv run --frozen python scripts/competition_drill.py --full `
  --batch-id aic25-b1-reference --query "một người mặc áo đỏ đi qua cây cầu"

# Explicit mutating competition-day drill: verified relocation, bounded core ingestion,
# query/refine/export, physical Qdrant/Meilisearch restart, reconnect, and recovery.
& $Uv run --frozen --group competition python scripts/competition_drill.py `
  --full --operational --incoming D:\incoming\aic-batch-2 `
  --batch-id AIC26-B2 --query "một người mặc áo đỏ đi qua cây cầu"
```

`--skip-services` and `--skip-assets` are diagnostic escape hatches, not a complete competition
drill. Normal and `--full` modes never mutate source archives or search indexes. They verify
manifest entries, pinned FFmpeg, disk reserve, service health, optional batch status, and an
optional real query. `--full` additionally exercises Rocchio feedback, validated export in a
temporary directory, a fresh runtime/service-client reconnect, and interrupted-stage recovery in
a disposable ledger. `--operational` is deliberately separate because it moves verified stable
arrivals into the canonical archive tree, updates their checksum manifest, core-ingests them, and
physically restarts the persisted local services. It refuses protected archives and still uses the
12 GiB scratch/30 GiB reserve safeguards.

## Data operations

All examples use project-local uv. Full-corpus operations are opt-in.

### 1. Catalog stable archives

The catalog ignores `.crdownload` files, requires ZIPs to be unchanged for 120 seconds by
default, classifies actual members rather than filenames, and can perform full CRC/SHA passes.
Without `--relocate` it never extracts or moves archives.

```powershell
$Uv = Resolve-Path .tools/uv/0.11.16/uv.exe

# Fast content inventory.
& $Uv run --frozen python scripts/catalog.py D:\incoming\aic-batch-2

# Full verification and persistence in the resumable ledger.
& $Uv run --frozen python scripts/catalog.py D:\incoming\aic-batch-2 `
  --verify-crc --sha256 --ledger data/state/jobs.sqlite3

# Explicit arrival workflow: force CRC/SHA, then atomically sort verified archives.
& $Uv run --frozen python scripts/catalog.py D:\incoming\aic-batch-2 `
  --relocate --competition aic2026 --batch-id AIC26-B2
```

Relocation cannot reduce the 120-second stability gate. It fully CRC-reads and hashes every
candidate before the first move, then uses a same-volume atomic rename into
`data/archives/<competition>/<batch>/{videos,keyframes,reference}`. It refuses existing
destinations, unsafe competition/batch components, and the two protected development archives.
On success it updates `checksums/archives.sha256` and `data/state/jobs.sqlite3`; use
`--archive-root`, `--checksum-manifest`, or `--ledger` only for an intentional alternate path.
Relocation never extracts, overwrites, or deletes archive content.

The already-imported Batch-1 development packs are fully CRC/SHA cataloged in place because
their active canonical thumbnail URIs reference those paths; they are not silently migrated.
Competition-day/new batch arrivals use the relocation path above before ingestion.

### 2. Validate or import organizer reference packs

Reference ZIPs are streamed and are not expanded. With no indexing flags, the command validates
the expected 873 videos and 177,321 aligned map/feature/object records.

```powershell
# Validate only; default paths target the AIC25-B1 reference packs at repository root.
& $Uv run --frozen python scripts/import_reference.py

# Import verified organizer vectors and lexical object/metadata documents.
& $Uv run --frozen python scripts/import_reference.py `
  --index-vectors --index-objects --index-metadata

# If vectors already exist, import canonical maps and activate the reference batch.
& $Uv run --frozen python scripts/import_reference.py `
  --import-canonical --activate-existing-vectors

# Verify only and atomically write the required model/fingerprint gate record.
& $Uv run --frozen --group competition python scripts/import_reference.py --verify-encoder

# Destructive only to the named index: recreate it, and independently verify the encoder.
& $Uv run --frozen --group competition python scripts/import_reference.py `
  --index-vectors --recreate --verify-encoder
```

Organizer vector search should be used only after the 16-image verification meets the configured
median cosine threshold of 0.999. This workspace passed with the exact pinned
`ViT-B-32-quickgelu/openai` encoder. Runtime still omits the channel unless the generated
`data/state/organizer-clip-verification.json` matches the configured model, fingerprint, sample
count, and threshold. A new machine must rerun verification before trusting or recreating that
index. Organizer 512-D vectors remain isolated from native SigLIP2 vectors. Vector import also
persists canonical reference frames and every ordinal-to-source mapping; duplicate source
identities remain separate Qdrant points but one canonical frame.

### 3. Ingest raw videos

The native pipeline accepts loose videos, directories, and verified canonical ZIPs. ZIP inputs
must first pass the catalog relocation workflow above (the two immutable protected archives may
remain in place after full CRC/SHA cataloging). ZIP processing materializes at most one video
member in bounded scratch. Per-video stages are fingerprinted in SQLite; reruns
resume safely, interrupted stages can be reclaimed, and one failed video does not block active
completed videos.

```powershell
# Fastest competition-day core: probe, adaptive frames, SigLIP2, atomic activation.
& $Uv run --frozen --group competition python scripts/ingest.py D:\incoming\batch2 `
  --batch-id AIC26-B2 --core-only

# Full enrichment adds OCR, ASR, Vietnamese semantic text, and lexical indexing.
& $Uv run --frozen --group competition python scripts/ingest.py D:\incoming\batch2 `
  --batch-id AIC26-B2 --cpu-workers 2 --queue-size 4
```

Use `--recreate-indexes` only when deliberately replacing the configured collections. By
default the pipeline recovers interrupted stages; `--no-recover` disables that behavior for
diagnostics. Do not point native ingestion at incomplete downloads.

The earlier extraction helper remains available for protected development data:

```powershell
& $Uv run --frozen python scripts/extract_data.py --inspect
& $Uv run --frozen python scripts/extract_data.py --sample L21_V001
```

Use `--full` only when explicitly required; existing targets are skipped unless `--force` is
deliberately supplied.

Calibrate adaptive extraction only on explicitly named raw videos that also have organizer
reference timestamps in the canonical artifact database. The command evaluates the configured
`scene_threshold_grid x max_gap_grid_seconds`, selects the smallest extraction reaching 95%
coverage within one second, and otherwise records the documented `0.30/2s` fallback. Generated
calibration thumbnails live in a temporary directory and are removed; the JSON decision is
persisted without editing `settings.yaml` or extracting the corpus. `--raw-root` may contain
loose videos and/or ZIPs: selected video members are streamed for SHA-256 and materialized one at
a time through the same 12 GiB scratch/30 GiB reserve guard used by ingestion. The report records
each source URI/content digest plus the complete calibration configuration fingerprint.

```powershell
& $Uv run --frozen --group competition python scripts/calibrate_extraction.py `
  --reference-batch aic25-b1-reference `
  --raw-root data/raw/videos `
  --video-id L21_V001 --video-id L21_V002 `
  --output outputs/calibration/extraction-selection.json
```

The configured `ingestion.calibration_record` points to that atomic report. If it exists and
passes schema/fingerprint validation, `build_native_ingestor` applies its selected scene
threshold and maximum gap. If absent, native ingestion retains the documented YAML defaults.

For manual recovery or offline transfer, the lower-level artifact CLIs remain available:

```powershell
# Materialize canonical SigLIP2 matrices and mappings from the artifact database.
& $Uv run --frozen --group competition python scripts/build_embeddings.py `
  --batch-id AIC26-B2 --output-dir data/embeddings/AIC26-B2

# Import those matrices into the configured native Qdrant collection.
& $Uv run --frozen python scripts/index_vectors.py `
  --embeddings-dir data/embeddings/AIC26-B2 --collection native_frames --batch-id AIC26-B2

# Rebuild one canonical lexical channel from SQLite.
& $Uv run --frozen python scripts/index_text.py --batch-id AIC26-B2 --kind ocr
& $Uv run --frozen python scripts/index_text.py --batch-id AIC26-B2 --kind asr
```

Normal `scripts/ingest.py` runs these stages directly; the lower-level commands are not required
for the standard path. `scripts/init_indexes.py --recreate-empty` is destructive to every
configured collection and must be used only before ingestion.

### 4. Evaluate historical Textual KIS rankings

Predictions are evaluated by exact video and submitted source frame inside the inclusive ground
truth interval. The report includes mean historical Top-k score, R@1/5/20/50/100, MRR, median
rank, per-query metrics, intent classes when provided, and optional latency statistics.

```powershell
& $Uv run --frozen python scripts/run_evaluation.py `
  --ground-truth data/evaluation/ground-truth.csv `
  --predictions outputs/predictions.json `
  --latencies outputs/latencies.json `
  --output outputs/evaluation/report.json
```

Pass `--spec path/to/spec.json` to test another validated `CompetitionSpec`, or `--compact` to
omit per-query details from displayed JSON. Historical labeled queries/intervals are required for
fusion tuning; the documented defaults remain active when they are unavailable.

Tune RRF using cached per-channel rankings, so the grid never calls a model or search service.
The objective is Mean Top-k R-Score, then R@1, then cached p95 channel latency. `--depth` and
`--weight` are repeatable grid overrides; the output also reports the cumulative channel
ablation in configured channel order. With no `--ground-truth`, the command deliberately writes
the configured defaults and records that no tuning was performed.

```powershell
# Explicit no-label/default record.
& $Uv run --frozen python scripts/tune_retrieval.py

# Labeled cached-ranking tuning.
& $Uv run --frozen python scripts/tune_retrieval.py `
  --rankings outputs/evaluation/cached-rankings.json `
  --ground-truth data/evaluation/ground-truth.csv `
  --rrf-k 20 --rrf-k 40 --rrf-k 60 --rrf-k 80 `
  --depth visual=200,300 --depth ocr=100,200 `
  --weight visual=0.8,1.0 --weight ocr=0.6,0.8,1.0
```

Cached-ranking JSON uses `{"schema_version":1,"queries":[...]}`. Each query has `query_id` and
a `channels` object; each channel contains `results` (canonical `video_id` plus
`source_frame_idx`) and optional measured `latency_ms`. The grid is capped at 50,000 candidates
by default to catch accidental combinatorial expansions.
The default output is the configured `retrieval.tuning_record`; runtime construction validates
and applies its selected k/depth/weight override only when labeled tuning actually ran. A
no-label/default report is retained for auditability but never overrides current YAML defaults.

## API and interactive UI

Start the automatic API and the interactive console in separate terminals:

```powershell
$Uv = Resolve-Path .tools/uv/0.11.16/uv.exe
& $Uv run --frozen python scripts/serve.py --host 127.0.0.1 --port 8080

# One-time, pinned competition runtime (kept separate from the test .venv).
.\scripts\bootstrap_runtime.ps1 -Profile laptop

# Canonical UI launch: validates the runtime stamp, port, and selected batch.
.\scripts\run_app.ps1 -Profile laptop -BatchId aic25-b1-reference -Port 8501
```

The launcher never starts Docker unless requested, kills a port owner, or changes ports. It uses
a lightweight batch/runtime preflight so Streamlit liveness is available immediately. The exact
service, index, and canonical-image audit then runs in the single-flight background warm-up;
all search controls remain locked until that audit and required model warm-up finish.
`python scripts/runtime_doctor.py --batch-id aic25-b1-reference` still prints the complete
secret-free readiness report without starting the application. Add `--live-gemini` only for an
explicit network integration check with the rotated key in untracked `.env`.

The Textual KIS tab exposes editable clauses, evidence/channel ranks, neighboring frames,
Rocchio feedback, pinning, and validated export. The isolated Visual Q&A tab retrieves by event
description, displays chronological five-frame windows with selected evidence emphasized,
supports manual or automatic five-cluster expansion, safe provider traces, raw/normalized
answers, persisted audited operator edits, pinning, session restore, and validated QA export.
The TRAKE tab provides a Vietnamese global description editor, 2--20 ordered stage controls,
top-100 distinct-video results, a horizontal stage timeline with canonical source frames, and a
validated no-header CSV download.
Download and server-copy actions require one active batch and use persisted ordinal-to-source
evidence in addition to ID, rank, duplicate, and limit checks. Missing services, models, mappings,
or indexes surface as errors rather than fabricated matches.

### HTTP endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/v1/kis/search` | Automatic or interactive initial hybrid search |
| `POST` | `/v1/kis/refine` | Fresh search with positive/negative `VIDEO_ID:SOURCE_FRAME_IDX` feedback IDs |
| `GET` | `/v1/frames/{batch}/{video}/{source_frame_idx}/neighbors` | Timestamp-based canonical neighborhood (`radius_ms`, `limit`) |
| `POST` | `/v1/qa/search` | Initial five-cluster QA search; automatic mode continues in five-cluster batches to 20 |
| `POST` | `/api/v1/search/qa` | Deprecated compatibility alias for `/v1/qa/search` |
| `POST` | `/v1/qa/search/{search_id}/expand` | Idempotently expand an interactive search to 10, 15, or 20 clusters |
| `GET` | `/v1/qa/search/{search_id}` | Restore cumulative predictions, traces, failures, and audited edits |
| `GET` | `/v1/frames/{batch}/{video}/{source_frame_idx}/image` | Stream a bounded canonical image without accepting a client URI |
| `POST` | `/v1/qa/submissions/export` | Validate an active batch and return headerless `video_name,frame_idx,answer` CSV |
| `POST` | `/v1/trake/search` | Retrieve and strictly align 2--20 ordered event stages in one compatible dense space |
| `POST` | `/api/v1/search/trake` | Deprecated compatibility alias for `/v1/trake/search` |
| `POST` | `/v1/trake/submissions/export` | Validate an active batch and return `<query-id>-trake-submission.csv` |
| `POST` | `/v1/submissions/export` | Validate an active batch and return a top-100 CSV |
| `GET` | `/v1/ingest/status?batch_id=...` | Readiness/ETA, disk/scratch/RAM/VRAM snapshot, and first-ten-video core projection; accepts optional `preload_window_seconds` |
| `GET` | `/v1/readiness?batch_id=...` | Batch-aware KIS/QA readiness, channel applicability, and sanitized canonical-launch identity |
| `GET` | `/v1/health/status?batch_id=...` | Sanitized KIS/QA/TRAKE, ingest, warm-up, and aggregate channel health |
| `GET` | `/api/v1/health/status?batch_id=...` | Deprecated compatibility alias for `/v1/health/status` |
| `GET` | `/healthz` | Process health |

Minimal search request:

```json
{
  "query_id": "kis-001",
  "original_text": "một người mặc áo đỏ đi qua cây cầu",
  "mode": "automatic",
  "batch_filters": ["AIC26-B2"],
  "limit": 100
}
```

Search responses contain the preserved/generated query plan, ranked canonical hits, channel
evidence, and any channel-specific failures. Export requests require `batch_id`, `filename`, and
the canonical `hits` list; the API rejects malformed video IDs, payloads that expose only
`keyframe_ordinal`, organizer ordinals mislabeled as source frames when organizer provenance is
present, duplicates, inactive batches, unsafe filenames, and more than 100 rows. Provenance-aware
validation avoids misclassifying a legitimate refined source-frame number that merely equals an
unrelated ordinal. The evidence is persisted during reference/native ingestion.

Minimal Visual Q&A request:

```json
{
  "query_id": "qa-001",
  "event_description": "một người đang cầm ba quả táo",
  "question": "Người đó đang cầm bao nhiêu quả táo?",
  "mode": "interactive",
  "answer_spec_id": "aic-2024-visual-qa-numeric-v1",
  "batch_filters": ["AIC26-B2"],
  "limit": 100
}
```

The QA response contains no credentials, request headers, image payloads, client-provided URIs,
or chain-of-thought. `confidence_score` is a deterministic ranking proxy, not a calibrated
probability. Provider failures are reduced to stable operational codes.

Minimal TRAKE request:

```json
{
  "query_id": "trake-001",
  "global_description": "một người chọn hàng, thanh toán rồi rời cửa hàng",
  "stages": [
    {"stage_index": 1, "description": "người chọn một món hàng"},
    {"stage_index": 2, "description": "người thanh toán tại quầy"},
    {"stage_index": 3, "description": "người rời khỏi cửa hàng"}
  ],
  "batch_filters": ["aic25-b1-reference"],
  "top_k": 100
}
```

For headless export from a JSON hit list:

```powershell
& $Uv run --frozen python scripts/export_submission.py `
  --results outputs/canonical-hits.json `
  --output outputs/submissions/textual-kis.csv `
  --batch-id aic25-b1-reference
```

The CLI can also accept `--spec path/to/spec.json`; like the API and UI, it requires persisted
batch activation and frame-map evidence.

Visual Q&A has parallel export and evaluation CLIs:

```powershell
& $Uv run --frozen python scripts/export_qa_submission.py `
  --results outputs/qa-predictions.json `
  --output outputs/submissions/visual-qa.csv `
  --batch-id aic25-b1-reference

& $Uv run --frozen python scripts/run_qa_evaluation.py `
  --ground-truth data/evaluation/qa-ground-truth.json `
  --predictions outputs/qa-rankings.json `
  --output outputs/evaluation/qa-report.json
```

The QA evaluator reports end-to-end exact recall, MRR and median rank, retrieval-only oracle
recall, answer accuracy conditional on localization, latency, cache rate, and provider usage.

Measure TRAKE cold/warm latency in fresh processes without recording the query text in output:

```powershell
& $Uv run --frozen --group competition python scripts/benchmark_trake_latency.py `
  --batch-id aic25-b1-reference `
  --global-description "một người chọn hàng, thanh toán rồi rời đi" `
  --stage "người chọn một món hàng" `
  --stage "người thanh toán tại quầy" `
  --stage "người rời cửa hàng" `
  --iterations 50 `
  --output outputs/evaluation/trake-latency.json
```

## Verification and acceptance

Run the normal validation before handing off Python changes:

```powershell
.\scripts\check.ps1
.\scripts\check.ps1 -IncludeCompose -IncludeServices -IncludeStreamlit -IncludeStreamlitRuntime

# Explicit integrations only; neither runs during bootstrap/default checks.
.\scripts\check.ps1 -IncludeGemini  # rotated key + network
.\scripts\check.ps1 -IncludeQwen    # prepared cache + nominal 12 GiB CUDA GPU
```

By default, `scripts/check_gemini.py` makes exactly one multimodal provider attempt using a tiny
synthetic image. Its extended diagnostic requires `--allow-multiple-provider-calls`; all stages
share the same 13-RPM governor and 4.5-second pacing. It prints only allowlisted operational
metadata. Sending real competition frames is a separate, explicit one-call check:

```powershell
.\.venv-runtime\Scripts\python.exe scripts\check_gemini.py

.\.venv-runtime\Scripts\python.exe scripts\check_qa_live.py `
  --live-gemini --allow-external-frame-transfer `
  --batch-id aic25-b1-reference `
  --event-description "a person holding a cat" `
  --question "What animal is visible?" `
  --answer-spec general
```

Neither checker runs during import, bootstrap, startup, readiness, or the default offline suite.

Use `-IncludeArchives` only when a complete CRC read of the protected source archives is
relevant. The default offline suite covers contract validation, interval evaluation/export,
archive classification and failures, source-frame/VFR mapping, ledger idempotence and recovery,
artifact conflicts, ingestion isolation and OOM reduction, Unicode OCR/ASR joins, RRF/window
logic, temporal diversity, raw refinement, relevance feedback, store adapters, and API routes.
It also covers QA normalization/contracts, ZIP/local image safety, provider retry/redaction,
session restart/cache/concurrent expansion, ranking/export/evaluation, and KIS route regressions.

Environment/corpus acceptance remains a separate operation. Before competition day, verify:

- Every expected reference map/feature/object count agrees and every active video has matching
  canonical artifacts and index counts.
- A restart after each ingestion stage creates no duplicate points and resumes from the last
  atomic completion marker.
- Warm automatic search p95 is at most two seconds before raw refinement and at most 12 seconds
  with top-window refinement.
- Laptop TRAKE warm-up completes within 90 seconds, and warm exact alignment p95 is at most five
  seconds for up to ten stages on the Batch-1 organizer collection.
- Laptop use remains below 3.8 GiB VRAM, 13 GiB RAM, 12 GiB scratch, and the 30 GiB disk reserve.
- Core ingestion projected after the first ten videos fits within 80% of the available preload
  window; otherwise use the validated accelerated profile.
- The offline drill succeeds with prepared model/tool checksums, healthy local services, an active
  batch, and a real query; `--full` also passes feedback, temporary export, reconnect, and recovery.
- The explicit `--operational` drill physically restarts local services and confirms persisted
  results after arrival cataloging and bounded core ingestion.
- A live Gemini check succeeds with a newly rotated key, and cached replay remains below two
  seconds locally. Initial/expansion batches never exceed five provider calls or 20 total.
- The accelerated profile is marked ready only after `-IncludeQwen` completes five-frame
  structured inference without OOM and reports peak VRAM on the target 12--24 GiB GPU.

These are acceptance gates to measure on the target machine; checkout, bootstrap, and unit tests
alone do not prove them.

## Protected data and generated-state policy

- `Keyframes_L21.zip` and `Videos_L21_a.zip` are immutable. Their expected hashes are in
  `checksums/archives.sha256`; never rename, relocate, rewrite, recompress, or delete them.
- `data/`, `docker_data/`, `models/`, `outputs/`, and `logs/` are user-owned generated state.
  Environment cleanup must not remove them.
- The 32 currently cataloged archives total approximately 106.7 GiB. Reference packs are
  streamed; do not expand the corpus.
- Native ZIP ingestion uses `data/cache/scratch`, enforces a 12 GiB limit and 30 GiB reserve, and
  materializes only one bounded member at a time.
- Persist only 512-pixel WebP thumbnails for newly extracted frames. Decode full-resolution raw
  neighborhoods on demand.
- Never commit `.env`, credentials, downloaded tools, virtual environments, model weights,
  derived data, or service state.

## Current limitations

- The final AIC 2026 Textual KIS output schema and scoring rules are not yet encoded; the 2025
  adapter is deliberately versioned and replaceable.
- Model snapshots, OCR assets, Docker services, and corpus indexes must be prepared locally. No
  download or full-corpus ingestion occurs during import, bootstrap, or tests.
- Fusion tuning and claims of improvement over the organizer baseline require historical labeled
  queries and ground-truth intervals that are not bundled automatically.
- Batch 2 has not arrived in this workspace, so its native collections, ten-video ingestion
  projection, raw-refinement p95, and end-to-end `--operational` drill remain data-dependent gates.
- Gemini live inference is always opt-in and requires a replacement credential stored only in
  untracked `.env`; the compromised pasted credential is never used. Normal tests use injected
  providers only, and real-frame transfer additionally requires its explicit consent flag.
- The accelerated 12–24 GiB Qwen profile is implemented but cannot be validated on this 4 GiB
  laptop; its readiness remains a target-machine acceptance gate.
- The default model preparation script covers the laptop core but not accelerated OpenCLIP or
  FFmpeg. Install FFmpeg from a trusted checksum-pinned ZIP and freeze all additional runtime
  assets explicitly.
- The TRAKE core, API, evaluator, provisional exporter, and UI are implemented, but the official
  AIC 2026 TRAKE schema/evaluator and Batch-2 latency gate are not yet available. Object-detector
  inference remains intentional scaffolding. Visual Q&A requires prepared canonical frame
  artifacts and either a rotated Gemini key or a validated accelerated Qwen cache.

For a clean Python environment while retaining all user data, use
`.\scripts\reset-env.ps1`. Add `-PurgePackageCache` only when deliberately discarding the
downloadable uv cache. If PowerShell blocks local scripts, use
`Set-ExecutionPolicy -Scope Process Bypass`; do not weaken the machine-wide policy.
