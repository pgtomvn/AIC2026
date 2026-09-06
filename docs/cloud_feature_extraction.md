# Cloud ZIP-streamed feature extraction

This runbook produces the B01 SigLIP2, OCR, and ASR artifacts on Google Colab or
Kaggle without expanding the keyframe or video archives. The same commands can
be resumed after a runtime disconnect.

## Outputs and storage layout

Keep the repository, immutable input ZIPs, and output directory separate:

```text
/content/aic2026/                            # ephemeral allowlisted source bundle
/content/aic2026-scratch/                    # ephemeral ASR audio, never Drive
/content/drive/MyDrive/AIC2026/B01/source/   # source ZIP, never credentials/data
/content/drive/MyDrive/AIC2026/B01/raw/      # input ZIPs
/content/drive/MyDrive/AIC2026/B01/out/      # shards, artifacts, download bundle
```

The output directory contains resumable shards below `cloud_work/`. Do not
rename or edit those shards. Finalization publishes these files atomically:

- `b01_siglip2_features.npy`
- `b01_siglip2_manifest.json`
- `b01_ocr_features.jsonl.gz`
- `b01_asr_transcripts.jsonl.gz`

The progressive schema-v3 manifest records `completed_channels`, stable
per-channel fingerprints, and `complete`. A SigLIP2-only publication has
`completed_channels=["siglip2"]` and `complete=false`; it does not create empty
OCR/ASR files. `complete=true` is reserved for a verified three-channel bundle.
Complete schema-v2 bundles remain readable for rollback.

## Prepare the source bundle and Drive inputs

Build the credential-free source ZIP from the current dirty worktree. The helper
uses a fixed allowlist (`src`, `scripts`, `config`, `checksums`, `notebooks`,
`pyproject.toml`, and `uv.lock`) and rejects credentials, generated data,
models, databases, archives, and Python caches:

```powershell
$Uv = Resolve-Path .tools/uv/0.11.16/uv.exe
& $Uv run --frozen python scripts/build_colab_source_bundle.py
```

Upload `outputs/colab/aic2026-colab-source.zip` to
`MyDrive/AIC2026/B01/source/`. Upload the fourteen `Keyframes_L*.zip` files,
`map-keyframes-aic25-b1.zip`, `Videos_L21_a.zip`, and the thirteen remaining
video ZIPs from `data/archives/aic25/b1/videos/` directly into
`MyDrive/AIC2026/B01/raw/`. Do not stage, rename, recompress, or extract the
corpus. The notebook verifies the exact archive names plus 873 video IDs,
177,321 mapping rows, and 176,707 canonical identities before inference.

## Colab setup

Open `notebooks/aic2026_cloud_feature_extraction.ipynb` and select a T4 or A100
GPU runtime. Enable High-RAM when Colab offers it. The notebook mounts Google
Drive, safely expands only the small source bundle into `/content/aic2026`, and
fails before dependency installation if the allocated GPU is not T4/A100 or
less than 20 GiB of ephemeral disk is free.

The equivalent shell setup is:

```bash
cd /content/aic2026
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --group competition --frozen
sudo apt-get update
sudo apt-get install -y ffmpeg
uv run --frozen python scripts/prepare_models.py --only visual
uv run --frozen python scripts/prepare_models.py --only reference-ocr
uv run --frozen python scripts/prepare_models.py --only reference-asr
```

Model preparation uses the revisions and RapidOCR checksums pinned in
`config/settings.yaml` and the repository checksum manifests. No credentials
are stored in the notebook.

## Publish the completed SigLIP2 channel immediately

When all 873 SigLIP2 receipts account for exactly 176,707 canonical frames,
finalize and package only that channel:

```bash
KEYFRAME_ARGS=()
for archive in "$RAW_ROOT"/Keyframes_L*.zip; do
  KEYFRAME_ARGS+=(--keyframes-zip "$archive")
done

uv run --frozen python scripts/colab_provision_zip.py \
  --maps-zip "$RAW_ROOT/map-keyframes-aic25-b1.zip" \
  "${KEYFRAME_ARGS[@]}" \
  --output-dir "$OUTPUT_ROOT" --batch-id aic25-b1-reference \
  --expected-videos 873 --expected-frames 176707 \
  --finalize-only --finalize-channels siglip2

uv run --frozen python scripts/import_cloud_artifacts.py \
  --artifact-dir "$OUTPUT_ROOT" --batch-id aic25-b1-reference \
  --channels siglip2 --validate-only

uv run --frozen python scripts/package_cloud_artifacts.py \
  --artifact-dir "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/bundles" \
  --batch-id aic25-b1-reference --channels siglip2
```

Download the ZIP and SHA-256 sidecar into the immutable local folder
`data/cloud_artifacts/aic25-b1-reference/<manifest-sha256>/`. Validate before
importing `--channels siglip2 --resume`. The partial import writes a receipt
under `data/checkpoints/cloud-import/aic25-b1-reference/`, never the Full
Multi-Modal completion receipt.

## Build deterministic OCR partitions

The coordinator creates one immutable plan using ZIP central directories and
canonical mappings. It never extracts or payload-hashes a corpus ZIP:

```bash
uv run --frozen python scripts/build_cloud_partition_plan.py \
  --maps-zip "$RAW_ROOT/map-keyframes-aic25-b1.zip" \
  "${KEYFRAME_ARGS[@]}" \
  --batch-id aic25-b1-reference --stage ocr \
  --max-frames 8000 --worker-count 2 \
  --output "$OUTPUT_ROOT/plans/ocr-partitions.json"
```

Open `notebooks/aic2026_cloud_partition_worker.ipynb` twice. Use worker slots 0
and 1 with `STAGE='ocr'`. Every video belongs to exactly one archive-local
partition; slots are balanced by canonical frames, not a fixed video count.
Each heartbeat is isolated at
`cloud_work/progress/ocr/<partition-id>.json(.jsonl)`. Existing valid flat OCR
receipts are checked first. Recover a revoked runtime by remounting Drive and
rerunning the same slot.

After both workers finish, strictly finalize `siglip2,ocr`, validate/package
those channels, and import only OCR. Health remains degraded with ASR missing.

## Publish a cumulative partial OCR checkpoint

When Colab quota or an operational pause makes it useful to activate the OCR
already produced, stop workers only after their current video receipt has been
atomically published. Record the partition-plan fingerprint, latest receipts,
deduplicated video/frame totals, and duplicate count in
`ocr_pause_checkpoint.json`. A video that started without a receipt is safely
processed again by the next `--resume` run.

Partial publication is always explicit. Finalize the cumulative receipt set,
then package only the manifest and OCR artifact; do not download or import the
SigLIP2 vector file again:

```bash
uv run --frozen python scripts/colab_provision_zip.py \
  --maps-zip "$RAW_ROOT/map-keyframes-aic25-b1.zip" \
  "${KEYFRAME_ARGS[@]}" \
  --output-dir "$OUTPUT_ROOT" --batch-id aic25-b1-reference \
  --expected-videos 873 --expected-frames 176707 \
  --finalize-only --finalize-channels siglip2,ocr --allow-partial

uv run --frozen python scripts/package_cloud_artifacts.py \
  --artifact-dir "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/bundles" \
  --batch-id aic25-b1-reference --channels ocr --allow-partial
```

Download the ZIP and sidecar to the immutable local directory
`data/cloud_artifacts/aic25-b1-reference/ocr-partial/<manifest-sha256>/`, then
validate and import with `--channels ocr --allow-partial --resume`. The manifest
records `channel_coverage` and stable per-video fingerprints. The importer
preflights hashes, batch/model identity, frame totals, deterministic IDs, and
the existing ledger before touching Meilisearch. Reimporting the same snapshot
is a no-op; a later cumulative snapshot imports only newly completed videos.

The channel receipt records `snapshot_import_complete=true` and
`coverage_complete=false`. It never creates the Full Multi-Modal completion
receipt. OCR is usable immediately at its configured retrieval weight while
health remains `DEGRADED · Multi-Modal Incomplete` and reports partial
coverage plus missing ASR. Operational rollback is non-destructive: set the OCR
retrieval weight to zero and restart API/Streamlit.

## Partition ASR after OCR

Create a separate plan using `--stage asr` plus all `--videos-zip` arguments.
Archive locality remains fixed; worker load is balanced by estimated mapped
duration. Run the same worker notebook with `STAGE='asr'`. Audio scratch stays
at `/content/aic2026-scratch`; Drive receives only completed shards/receipts.

For the B01 production run, use the immutable
`notebooks/aic2026_colab_asr_worker_0.ipynb` and
`notebooks/aic2026_colab_asr_worker_1.ipynb` in exactly two Colab GPU runtimes.
Their worker IDs are already pinned to `0` and `1`; keep `WORKER_COUNT=2`. The
notebook creates or validates one immutable duration-balanced ASR plan, downloads
the checksum-pinned `large-v3-turbo` Faster-Whisper snapshot, requires CUDA, and
runs `int8_float16` with per-video atomic receipts and `--resume`. It refuses to
fall back to CPU. Every mapped video must receive a receipt, including videos
where VAD finds no speech.

Do not run the production corpus on the local RTX 2050 while OCR is importing.
If Colab quota is exhausted, benchmark only 3–5 representative videos locally;
stop if the 4 GiB GPU runs out of memory or throughput is materially below
realtime. ASR local import starts only after the OCR ledger and exact filtered
Meilisearch count both equal 1,167,264 documents across 873 videos.

Finalize and package the ASR snapshot with the same production identity used by
the workers. The profile flag is mandatory; omitting it selects the lightweight
reference contract and correctly fails validation against production receipts:

```bash
uv run --frozen python scripts/colab_provision_zip.py \
  --maps-zip "$RAW_ROOT/map-keyframes-aic25-b1.zip" \
  "${KEYFRAME_ARGS[@]}" \
  "${VIDEO_ARGS[@]}" \
  --output-dir "$OUTPUT_ROOT" \
  --batch-id aic25-b1-reference \
  --expected-videos 873 \
  --expected-frames 176707 \
  --asr-profile production \
  --finalize-only --finalize-channels siglip2,ocr,asr

uv run --frozen python scripts/package_cloud_artifacts.py \
  --artifact-dir "$OUTPUT_ROOT" \
  --output-dir "$OUTPUT_ROOT/bundles" \
  --batch-id aic25-b1-reference \
  --channels asr \
  --asr-profile production
```

After OCR has passed its strict local acceptance checks, validate and import
only ASR. Do not re-import SigLIP2 or OCR:

```powershell
$Uv = Resolve-Path .tools/uv/0.11.16/uv.exe
& $Uv run --frozen python scripts/import_cloud_artifacts.py `
  --artifact-dir <versioned-asr-artifact-dir> `
  --batch-id aic25-b1-reference `
  --channels asr `
  --asr-profile production `
  --validate-only

& $Uv run --frozen python scripts/import_cloud_artifacts.py `
  --artifact-dir <versioned-asr-artifact-dir> `
  --batch-id aic25-b1-reference `
  --channels asr `
  --asr-profile production `
  --resume
```

ASR retrieval is query-gated. Automatic mode opens ASR for quoted speech,
dialogue/interview cues, names, or numeric terms and caps its channel weight at
0.7. Purely visual queries give ASR zero weight. `asr_only` searches transcript
segments directly, while `visual_only` provides the no-ASR baseline used for
held-out evaluation. Matching ASR segments receive a bounded bonus only when
their timestamps overlap the candidate frame; missing speech never penalizes a
video. Visual Q&A receives at most 4,000 characters of transcript from the
±30-second window around each candidate.

After strict ASR import, compare all three modes on held-out interval qrels:

```powershell
$Uv = Resolve-Path .tools/uv/0.11.16/uv.exe
& $Uv run --frozen python scripts/benchmark_asr_hybrid.py `
  --ground-truth data/evaluation/asr-heldout.json `
  --output outputs/evaluation/asr-hybrid-heldout.json
```

Do not publish Full Multi-Modal until the strict manifest has 873 ASR receipts,
no duplicate or mixed model fingerprints, its ASR document count matches the
external index, OCR remains exactly 1,167,264 documents, and SigLIP2 remains
exactly 176,707 points.

## Legacy single-worker commands

Each ZIP option is repeatable. Pass every keyframe/video pack that belongs to
B01. The examples use shell arrays so spaces in Drive paths remain safe.

```bash
KEYFRAME_ARGS=()
for archive in "$RAW_ROOT"/Keyframes_L*.zip; do
  KEYFRAME_ARGS+=(--keyframes-zip "$archive")
done

VIDEO_ARGS=()
for archive in "$RAW_ROOT"/Videos_L*.zip; do
  VIDEO_ARGS+=(--videos-zip "$archive")
done

uv run --frozen python scripts/colab_provision_zip.py \
  --maps-zip "$RAW_ROOT/map-keyframes-aic25-b1.zip" \
  "${KEYFRAME_ARGS[@]}" \
  --output-dir "$OUTPUT_ROOT" \
  --stages ocr \
  --device auto \
  --siglip-batch-size 128 \
  --progress-file "$OUTPUT_ROOT/cloud_work/progress.json" \
  --resume

uv run --frozen python scripts/colab_provision_zip.py \
  --maps-zip "$RAW_ROOT/map-keyframes-aic25-b1.zip" \
  "${KEYFRAME_ARGS[@]}" \
  "${VIDEO_ARGS[@]}" \
  --output-dir "$OUTPUT_ROOT" \
  --stages asr \
  --asr-profile production \
  --device cuda \
  --ffmpeg-path ffmpeg \
  --scratch-dir /content/aic2026-scratch \
  --progress-file "$OUTPUT_ROOT/cloud_work/progress.json" \
  --resume
```

SigLIP2 uses mixed-precision CUDA on a GPU runtime. The encoder halves a live
batch after CUDA OOM without skipping an image. Start at 128 on a T4; use 256
only after observing stable free VRAM. OCR keeps only detections with confidence
at least 0.50. ASR pipes each compressed video member to ffmpeg in 1 MiB chunks,
retains only one temporary 16 kHz mono WAV at a time, and enables Whisper VAD.

Rerunning either command validates the shard receipt and checksum for each
video. Valid shards are skipped; missing, partial, or mismatched shards are
rewritten atomically.

Every completed or resumed video prints one flushed JSON event, appends it to
`cloud_work/progress.jsonl`, and atomically replaces `cloud_work/progress.json`.
The heartbeat contains completed/total videos, throughput, ETA, and the latest
receipt, so progress remains observable after a browser reconnect.

## Finalize and verify

```bash
uv run --frozen python scripts/colab_provision_zip.py \
  --maps-zip "$RAW_ROOT/map-keyframes-aic25-b1.zip" \
  --keyframes-zip "$RAW_ROOT/Keyframes_L21.zip" \
  --output-dir "$OUTPUT_ROOT" \
  --asr-profile production \
  --finalize-only --finalize-channels siglip2,ocr,asr

uv run --frozen python scripts/import_cloud_artifacts.py \
  --artifact-dir "$OUTPUT_ROOT" \
  --batch-id aic25-b1-reference \
  --asr-profile production \
  --validate-only
```

Full finalization is strict by default: all three stages must contain the same 873
videos, SigLIP2 must contain 176,707 canonical frames, and OCR receipts must
report exactly 176,707 scanned canonical frames. Multi-line OCR detections are
also emitted as deterministic frame-level aggregate documents without rerunning
RapidOCR. Omit `--allow-partial` for the final publication. The flag is reserved
for an explicit cumulative checkpoint whose coverage metadata is internally
consistent; it never fabricates missing artifacts or relaxes hash/model/ID
validation.

The final notebook cell stores the four finalized files in one uncompressed
`b01_cloud_artifacts_<manifest-prefix>.zip` plus a `.sha256` sidecar below
`out/bundles/`. Download both files together. A missing, mixed, or edited
artifact causes local validation to fail before either service is mutated.

## Local import

Start the existing loopback-only Qdrant and Meilisearch services, then run:

```powershell
$Uv = Resolve-Path .tools/uv/0.11.16/uv.exe
& $Uv run --frozen python scripts/import_cloud_artifacts.py `
  --artifact-dir D:\AIC2026\data\cloud_artifacts\aic25-b1-reference\<manifest-hash> `
  --batch-id aic25-b1-reference `
  --channels siglip2,ocr,asr `
  --asr-profile production `
  --resume
```

The importer validates every requested completed channel before mutation. It memory-maps the FP16 NPY
and writes to the configured batch-qualified SigLIP2 collection in batches of
1,000. It writes OCR and ASR to the configured shared Meilisearch indexes in
batches of 2,000 and 1,000, always including `batch_id`. IDs are deterministic,
collections/indexes are never recreated, and reruns resume from completed
per-video ledger stages.

Partial channel imports write stable per-channel checkpoints but never write the Full Multi-Modal
completion receipt. A full import verifies exact batch-scoped Qdrant and
Meilisearch counts before atomically publishing that receipt; readiness also
verifies its manifest SHA-256 before advertising Full Multi-Modal.

The JSON result contains `elapsed_ms`, the 180-second target, and
`within_target`. The operational target must be measured against local
services; the default unit suite remains offline.

After import:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/v1/health/status
```

After SigLIP2-only import, exact coverage enables native SigLIP2 in KIS while
the HUD reports `DEGRADED · Multi-Modal Incomplete` and lists OCR/ASR. When
Organizer CLIP, native SigLIP2, OCR, and ASR have positive compatible counts,
all expected per-video stages are complete, and the full receipt matches its manifest, health reports
`HEALTHY · Full Multi-Modal`.

## Recovery

- CUDA OOM: rerun with `--siglip-batch-size 128`; completed video shards remain.
- Colab disconnect: remount Drive and rerun the same command with `--resume`.
- Interrupted finalization: rerun `--finalize-only`; final files are atomically
  replaced only after each temporary output is complete.
- Service interruption: restart the affected service and rerun local import with
  `--resume`. Deterministic IDs make repeated upserts safe.
- Hash/count/schema failure: restore all four files from the same cloud output;
  never bypass validation or recreate a live collection.
