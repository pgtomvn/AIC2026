# Operations guide: AIC 2026 multimodal video retrieval

This guide covers the Textual Known Item Search (Textual KIS) engine, the retrieve-then-reason
Visual Q&A subsystem, and the TRAKE temporal-alignment engine built for the Ho Chi Minh City AI
Challenge. They share canonical frame records, source-frame semantics, API runtime, readiness
probes, and the Vietnamese competition UI. Model preparation, corpus indexing, and
competition-day data provisioning are explicit operational steps described below.

Scoring and export rules are encoded as versioned competition specifications in
`src/contracts.py` (Textual KIS interval scoring, Visual Q&A numeric and general-text answer
matching, and TRAKE stage scoring), each covered by its own tests.

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
event description -> Textual KIS (refine_raw=false) -> five canonical candidates
                                                        |
                                  UI/operator selects exactly one frame
                                                        |
                           bounded ZIP/local resolver -> one normalized image
                                                        |
                  Gemini 3.6 Flash -> bounded local Qwen2.5-VL/Ollama fallback
                                                        |
                       concise answer + exact single-frame provenance
                                                        |
                       SQLite session/cache -> API/UI -> validated QA CSV
```

Retrieval and answering are separate operations. Stage 1 never resolves media or calls a VLM;
Stage 2 accepts only one candidate frozen in the version-3 session and sends exactly one image to
the selected provider. A different candidate can be answered without rerunning retrieval. Legacy
version-2 cascade sessions remain readable but cannot be expanded or mutated. Reference ZIP
members are streamed without extraction. The resolver accepts only persisted canonical frame
identities, applies EXIF orientation and bounded resizing, and hashes the normalized bytes used by
provider/cache calls.

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
| `QACandidateRequest` | Retrieval-only event text, exactly one batch, and a candidate limit |
| `QACandidateFrame` | Stable candidate identity, canonical frame/timestamp, score, channel ranks, and safe evidence |
| `QASingleFrameAnswerRequest` | Candidate ID, question, answer spec, provider policy, and optional explicit Florence task |
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
src/vlm/              Gemini/Ollama providers, routing, normalization, and facade
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
OpenCLIP channel and can use a separately configured larger Ollama vision model. Visual Q&A is
local-first in both profiles. The runtime releases dense encoders before Ollama inference and
asks Ollama to release its model before the next KIS or TRAKE request.

## Bootstrap and services

For step-by-step Windows installation and daily startup in Vietnamese, see
[Hướng dẫn cài đặt và khởi động cho thành viên](team_setup_vi.md). It includes separate
Meilisearch/Qdrant commands, health checks, reference import, Ollama setup, and troubleshooting.

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
Meilisearch keys. Keep it uncommitted. `OLLAMA_URL` defaults to loopback; `GEMINI_API_KEY`
remains server-side and is used only after explicit per-session frame-transfer consent.
Qdrant binds to `127.0.0.1:6333/6334` and Meilisearch to
`127.0.0.1:7700`. Qdrant uses Docker-managed Linux volumes so Windows bind-mount semantics cannot
break atomic segment renames; Meilisearch state and any retained Qdrant migration rollback copy
remain under `docker_data/`.

Visual Q&A requires the official Ollama Windows build and exactly the pinned laptop model. The
canonical launcher uses `.tools/Ollama/ollama.exe` when the standalone build is installed there,
binds it to loopback, disables Ollama cloud, and stores generated model state under
`models/ollama`. The laptop launcher defaults to the bundled Vulkan backend because the RTX 2050
driver on the reference machine cannot execute the current CUDA-v12 kernels; override
`OLLAMA_LLM_LIBRARY` only after a live check. Provision only the 3B tag on the laptop:

```powershell
$env:OLLAMA_MODELS = (Resolve-Path models).Path + "\ollama"
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_NO_CLOUD = "1"
.\.tools\Ollama\ollama.exe pull qwen2.5vl:3b-q4_K_M
.\.venv-runtime\Scripts\python.exe scripts\check_vlm_offline.py
```

The configuration pins manifest digest prefix `fb90415cde1e`; do not substitute `latest`, which
selects the larger 7B artifact. No Ollama SDK is required because the provider uses the local REST
API directly.

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
PaddleOCR/PaddlePaddle, Faster-Whisper, CTranslate2, SentencePiece/Sacremoses, and the Hugging
Face client. It does not download model weights merely by being installed.

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
  --only visual --only organizer --only accelerated --only text --only asr --only ocr `
  --only translation

# Add other prepared runtime assets or tool directories to a checksum manifest.
& $Uv run --frozen python scripts/freeze_assets.py models .tools/my-runtime-tool
```

### Vietnamese query routing

Vietnamese text is routed by encoder instead of being globally replaced. Organizer and
accelerated OpenCLIP receive only the English translation; native SigLIP2 internally fuses the
Vietnamese and English variants as one logical channel; the Vietnamese semantic encoder and
OCR/ASR lexical indexes retain the original Vietnamese query. Metadata searches both variants.
Translation is fully offline through the pinned OPUS-MT vi→en checkpoint converted to
CTranslate2 CPU INT8. Missing or invalid translation assets degrade only English-centric
channels and never trigger Ollama or a cloud provider.

```powershell
# Prepare/checksum only the offline translator (existing manifest entries are preserved).
& $Uv run --frozen --group competition python scripts/prepare_models.py --only translation

# Build multilingual shadow indexes. This never promotes or changes the live v1 indexes.
& $Uv run --frozen python scripts/provision_meili_shadow.py --dry-run

# Unlabeled comparison: reports rank agreement/Jaccard, never "precision".
& $Uv run --frozen python scripts/test_vietnamese_retrieval.py `
  --batch-id aic25-b1-reference
```

The UI exposes `auto|vi|en` and an optional English override for KIS, QA retrieval, and every
TRAKE prompt. API responses include sanitized `query_processing` diagnostics; query text is not
written to operational logs.

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

Run the API-backed stack with the canonical launcher:

```powershell
# One-time, pinned competition runtime (kept separate from the test .venv).
.\scripts\bootstrap_runtime.ps1 -Profile laptop

# One stack: FastAPI owns RuntimeContext; Streamlit is an HTTP-only client.
.\scripts\run_app.ps1 -Profile laptop -BatchId aic25-b1-reference -ApiPort 8080 -Port 8504
```

The launcher never starts Docker unless requested, kills a port owner, or changes ports. It uses
a lightweight batch/runtime preflight so Streamlit liveness is available immediately. The exact
service, index, and canonical-image audit then runs in the single-flight background warm-up;
all search controls remain locked until that audit and required model warm-up finish.
`python scripts/runtime_doctor.py --batch-id aic25-b1-reference` still prints the complete
secret-free readiness report without starting the application. Add `--live-local-vlm` only for
an explicit Ollama inference check.

The Textual KIS tab exposes editable clauses, evidence/channel ranks, neighboring frames,
Rocchio feedback, pinning, and validated export. The isolated Visual Q&A tab retrieves five
canonical frames by event description, lets the operator choose one frame, and runs one explicit
single-image answer request. Candidates, answers, selection, pinning, session restore, and
validated QA export persist independently across UI reruns.
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
| `POST` | `/v1/qa/retrieve` | Retrieval-only Top-5 canonical candidates; never invokes a VLM |
| `POST` | `/v1/qa/search` | Deprecated retrieval-only compatibility alias |
| `POST` | `/api/v1/search/qa` | Deprecated retrieval-only compatibility alias |
| `POST` | `/v1/qa/search/{search_id}/answer` | Answer on exactly one persisted candidate frame |
| `POST` | `/v1/qa/search/{search_id}/expand` | Removed cascade endpoint; always returns `410 qa_expand_removed` |
| `GET` | `/v1/qa/search/{search_id}` | Restore candidates, selected frame, answers, failures, and audited edits |
| `POST` | `/v1/qa/search/{search_id}/edits` | Persist an audited operator answer edit through the API |
| `POST` | `/v1/qa/diagnostics/image` | Run a non-persistent single-image diagnostic (8 MiB maximum) |
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
| `GET` | `/v1/runtime/capabilities` | Sanitized runtime, workflow, and Visual Q&A provider capabilities |
| `GET` | `/healthz` | Process health |

Minimal search request:

```json
{
  "query_id": "kis-001",
  "original_text": "một người mặc áo đỏ đi qua cây cầu",
  "mode": "automatic",
  "batch_filters": ["AIC26-B2"],
  "limit": 100,
  "reasoning_policy": "local_only",
  "allow_external_frame_transfer": false
}
```

Search responses contain the preserved/generated query plan, ranked canonical hits, channel
evidence, and any channel-specific failures. Export requests require `batch_id`, `filename`, and
the canonical `hits` list; the API rejects malformed video IDs, payloads that expose only
`keyframe_ordinal`, organizer ordinals mislabeled as source frames when organizer provenance is
present, duplicates, inactive batches, unsafe filenames, and more than 100 rows. Provenance-aware
validation avoids misclassifying a legitimate refined source-frame number that merely equals an
unrelated ordinal. The evidence is persisted during reference/native ingestion.

Minimal Visual Q&A Stage-1 request:

```json
{
  "query_id": "qa-001",
  "event_description": "một người đang cầm ba quả táo",
  "batch_filters": ["AIC26-B2"],
  "top_k": 5
}
```

Stage 2 binds the question to one returned candidate:

```json
{
  "candidate_id": "AIC26-B2:L01_V001:120",
  "question": "Người đó đang cầm bao nhiêu quả táo?",
  "answer_spec_id": "aic-2024-visual-qa-numeric-v1",
  "reasoning_policy": "local_only",
  "allow_external_frame_transfer": false,
  "answer_engine": "qwen"
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

# Explicit local-model integration only; it does not run during default checks.
.\scripts\check.ps1 -IncludeLocalVLM
```

`scripts/check_vlm_offline.py` verifies the pinned Ollama tag, digest and vision capability,
then runs one cold and one warm synthetic-image inference. The two-second warm target is a
warning unless `--strict-latency` is supplied. Checking indexed media is separate:

```powershell
.\.venv-runtime\Scripts\python.exe scripts\check_vlm_offline.py

.\.venv-runtime\Scripts\python.exe scripts\check_qa_live.py `
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
- The pinned local Ollama model reports vision capability and completes cold/warm structured
  checks. Every answer request contains exactly one persisted canonical frame.
- Any accelerated model tag is admitted only after Ollama confirms vision capability and the
  target machine completes structured inference without OOM.

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

## Operational notes

- Model snapshots, OCR assets, Docker services, and corpus indexes are prepared locally. No
  download or full-corpus ingestion occurs during import, bootstrap, or tests.
- Gemini is the primary single-frame QA provider and requires an untracked server-side key plus
  explicit per-session consent. The local Qwen model is the offline fallback, and local-only mode
  makes no outbound provider request.
- The default model preparation script covers the laptop core. Install FFmpeg (and, for the
  accelerated profile, OpenCLIP weights) from trusted checksum-pinned sources and freeze those
  runtime assets explicitly.

For a clean Python environment while retaining all user data, use
`.\scripts\reset-env.ps1`. Add `-PurgePackageCache` only when deliberately discarding the
downloadable uv cache. If PowerShell blocks local scripts, use
`Set-ExecutionPolicy -Scope Process Bypass`; do not weaken the machine-wide policy.
