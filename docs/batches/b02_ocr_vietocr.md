# B02 OCR — Colab staging

Status on 2026-09-25: implementation and offline checks passed; **inference has
not run**. Colab rejected GPU allocation because the current account has reached
its GPU usage limit. No CPU fallback was started. B01 and all production stores
are unchanged. This is P2 preparation, not P2 acceptance or a provisioned OCR channel.

## Frozen inputs and execution

- Canonical batch: `aic26-b2`, 614 videos, 220270 keyframes, 21 keyframe ZIPs.
- Drive: `MyDrive/AIC2026/B02/raw`, including `aic26-b2-map-keyframes.zip`.
- Validated locally against the P0 central-directory hashes and archive sizes;
  each Colab run repeats the check. Image reads verify individual ZIP CRCs.
- Plan fingerprint: `333aff5ca5998d1f14bf3db92061222fa1150b56696e2c10ea372db9dd2a39d0`.
- Worker 0: 306 videos, 110150 frames, 144 pilot frames.
- Worker 1: 308 videos, 110120 frames, 156 pilot frames.
- Pilot: deterministic 100 frames per M/N/S group, 300 total across both workers.
- Identity is `(batch_id, video_id, keyframe_ordinal)`. Duplicate PTS and advisory
  source frame indexes do not merge images. Timestamps come from the organizer map.

## Model and environment

Use the B01 PP-OCRv6 small ONNX detector (CPU, two inference threads) followed by
[VietOCR](https://github.com/pbcquoc/vietocr) `vgg_seq2seq` on CUDA, crop batches of
16, greedy decoding, confidence >= 0.5, at least two characters. Runtime detector
or recognizer errors abort the video; they do not produce zero-text receipts.

VietOCR is pinned to upstream commit `fe8c3a7fc714aec57ab81cec844eb3adf0c1636c`.
The PyPI 0.3.13 release pins an incompatible old einops; the upstream commit has
compatible dependency bounds. The `ocr-vietocr` uv group is separate from the
Launcher/default environment. Neither model is downloaded on the local machine.

The detector hash is verified against `checksums/rapidocr-reference.sha256`.
VietOCR weights come from the HTTPS URL in the pinned upstream config; their
first-download SHA-256 is recorded and reused, not claimed to be an independently
published checksum. The pilot must verify the actual model load, quality, speed,
GPU memory, and asset hashes before approving the full run. Disable pretrained
backbone downloads: the complete OCR checkpoint supplies the weights.

## Notebook and resume

Builder: `scripts/build_b02_ocr_notebooks.py`.
Local output: `outputs/colab/b02-ocr-vietocr-v1/` (two notebooks, about 0.9 MB each).
Embedded credential-free source SHA-256:
`df88e4589f197e97a3e49b705a60461dfe8f65db2718231230505c2f1ad128f0`.

Worker 0 (team-private Colab notebook)
is uploaded and open in external Chrome. Worker 1 is prepared locally, not uploaded
or connected yet. Start setup sequentially so workers do not prepare the shared
asset directory concurrently.

Each notebook has exactly three code cells: mount, setup/source validation, pilot.
The final cell is intentionally NOT a full-corpus run. When GPU is available:

1. Run mount and complete Google Drive authorization personally when requested.
2. Run setup, inspect the archive contract and CUDA checks.
3. Run the pilot on each worker (never duplicate a worker ID).
4. Review OCR output against sampled images, elapsed time and failures. A pilot
   approval must bind the exact source, model and plan; do not fabricate it merely
   to unblock `--full`.

Durable staging: `MyDrive/AIC2026/B02/out/ocr-vietocr-v1/`.
Model assets: `MyDrive/AIC2026/B02/source/ocr-vietocr-v1-assets/`.
Pilot and full receipts/shards are in separate subdirectories. Receipt is written
last; resume verifies identity, size, SHA-256, scanned count and document count.
Orphans, corrupt checkpoints, changed bindings and stale worker locks require an
audit; no force overwrite or automatic stale-lock deletion is implemented.
Heartbeat is written every 30 seconds; progress is also saved at video boundaries.

No finalizer or importer is invoked by these notebooks. All receipts explicitly
remain `import_ready=false`. Full OCR acceptance and a B02 ordinal-safe import
remain subsequent steps after the pilot and complete 614/220270 reconciliation.

## Verification

- Preflight and default bootstrap passed (28.6 GiB free on D at preflight).
- Local ZIP/map inventory matched 614/220270; no full archive hash or extraction.
- `scripts/check.ps1`: 1642 passed, 3 skipped; Ruff passed.
- `git diff --check`: passed.
- Six new offline tests cover checkpoint corruption/orphans, no-text versus
  inference failure, equal PTS/frame zero, deterministic pilot, frozen assets,
  and the three-cell credential-free notebook bundle.
- Colab installation, real CUDA inference and OCR quality remain **unverified**
  because GPU allocation is blocked by quota. Do not report production readiness.
