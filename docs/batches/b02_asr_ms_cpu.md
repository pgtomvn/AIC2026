# B02 ASR M/S — CPU pilot (2026-09-25)

The user switched from OCR to ASR after Colab rejected GPU allocation. Exclude
all 298 N-series traffic videos by explicit user instruction. Target 316 M/S
videos, checked against the frozen canonical P0 manifest. Exclusion is not an
inference result: do not create no-audio receipts for N or report 614 ASR receipts.

## Current state

### Selected notebook repair, 2026-09-25

The active user-selected notebook is
B02 three-cell worker (team-private Colab notebook).
It was reopened in external Chrome. Its old setup failed because
`Keyframes_M01.zip` was passed as an unrecognized positional CLI argument;
the run cell also retained a `NameError: json` from being executed without setup.
The notebook was repaired in place, with an explicit M/S manifest allowlist,
one-video pilot, visible subprocess output and independent receipt namespace:
`MyDrive/AIC2026/B02/out/asr-ms-cpu-seekable-20260925-v3`.

Source ZIP: `B02/source/b02-asr-ms-0b06ded92f68.zip` (695878 bytes), SHA-256
`0b06ded92f68e9845b865cca6c1ecdfa7005d8af46b404e7fc86498985a45768`.
The canonical inventory passed: 316 M/S, 298 N excluded, including all 31 M10
videos. Plan fingerprint:
`35f85174817039cb98081eca781c83f96c4def48fe933edf07f9f97d8f48e6b6`.
The runtime is CPU; no GPU availability is claimed.

`asr-colab` now explicitly includes OpenCV, because the public extraction package
imports the keyframe module eagerly. In this reused Colab environment, removing
other OpenCV distributions also removed the shared `cv2` files; a targeted
`uv sync --reinstall-package opencv-python-headless` restored the locked package.
The notebook confirmed `ASR imports OK`. No corpus, model weights, old receipts
or production indexes were deleted. Only the temporary runtime package set was
synced to the ASR group. Model preparation finished (`Prepared asr; froze 19 files`)
and the one-video pilot started on `S01-V001` at approximately 18:53 ICT.

Follow-up at 22:06 ICT: notebook output continued to reflect the durable progress
file, with heartbeat `1790348764.8766472`, state `transcribing`, completed videos
`0/316` and documents `0`. Colab's session manager showed one current active
notebook session with recent execution. The inspection tab itself remained
connecting, so CPU utilization and direct receipt-file inspection were unavailable;
heartbeat liveness does not establish transcript quality or inference progress.
The canonical P0 manifest's last keyframe for S01-V001 is at 9,925,333 ms (at least
2h45m of source video), explaining why this CPU pilot can take hours. No exception
or committed receipt was observed. The running worker was not restarted or
duplicated. Keep `MAX_NEW_VIDEOS=1` until the first transcript and timing can be
validated; no full-corpus launch or import has been performed.

Local verification after the repair: `scripts/check.ps1` passed Ruff and
1645 tests with 3 skips. Bootstrap/check required execution outside the Windows
sandbox due to temporary-directory access errors; bootstrap then used the uv
offline cache. `git diff --check` passed. The diagnostic v2 notebook was not
opened or run; only its hash-verified source ZIP was uploaded into the selected
notebook's runtime and persisted in Drive.

The previous three-cell Colab notebook (team-private)
was uploaded and Drive mounted. Cell 2 reported 316 M/S videos and 298 N excluded.
The one-video cell exited with `CalledProcessError` in 0.074 seconds. The supplied
traceback contains only the wrapper exception, not child stdout/stderr. No successful
video receipt is confirmed. The exact worker error therefore remains unknown.

Previous local notebook:
`outputs/colab/b02-asr-ms-cpu-20260925-v1/aic2026_b02_asr_ms_cpu_pilot.ipynb`.
Previous source SHA-256: `aa5c7d983894464d0463a638422956f6bbc9d6ee6174125fa22a3cb1769a4092`.
Builder: `scripts/build_b02_asr_ms_notebook.py`.
Diagnostic revision: `outputs/colab/b02-asr-ms-cpu-20260925-v2/aic2026_b02_asr_ms_cpu_diagnostic.ipynb`.
It writes to the separate namespace `MyDrive/AIC2026/B02/out/asr-ms-cpu-seekable-20260925-v2`
and captures stdout/stderr in `diagnostics/worker-0-pilot.log`. It does not touch the
previous v1 run, receipts or checkpoints.

## Frozen scope and safety

- Sources: `MyDrive/AIC2026/B02/raw/Video[s]_Mxx.zip` and `Video[s]_Sxx.zip` only.
- Diagnostic output: `MyDrive/AIC2026/B02/out/asr-ms-cpu-seekable-20260925-v2`.
- No reuse/deletion of previous 86 questionable receipts, no B01 mutation.
- Profile: production large-v3-turbo, same pinned model as B01, Vietnamese,
  CPU int8, VAD enabled. No automatic model substitution.
- Dedicated `asr-colab` uv group; no Torch/CUDA/OCR package installation is needed
  for this CPU notebook. Models download only into temporary Colab storage.
- One video is copied from ZIP to seekable Colab scratch before FFmpeg decoding;
  this preserves audio in MP4s whose moov atom is at the end. Decode errors must
  not be recorded as no-audio. Scratch is released per video by existing code.
- ZIP inventory must equal the 316 canonical M/S IDs exactly. Missing M10 or any
  unexpected video stops before inference. Keyframe/object/N ZIPs are not opened.
- Cells: mount → setup/inventory/model verification → one complete video pilot.
- Receipt records audio stream presence, segment count, duration, inference time,
  realtime factor, model/source/plan fingerprint and shard hash.
- Review the first video audio/transcript and CPU speed before removing
  `--max-videos 1`. Extending to the entire corpus has not been performed.
- No finalize/import or Full Multi-Modal claim in this stage. Future GPU resumes
  need a reviewed execution-profile transition, not mixed CPU/GPU receipts.

## Verification / rollback

19 focused tests passed for the M/S scope, existing ASR worker and new OCR worker.
Full `scripts/check.ps1`: 1645 passed, 3 skipped. Ruff and diff checks passed.
Original ASR source plus pyproject/lock checkpoint:
`outputs/checkpoints/b02-asr-ms-20260925/`.

Rollback only the M/S option or corresponding files; do not remove generated
state, notebooks, checkpoints, models or index data. OCR remains paused and its
prepared notebook/bundle is preserved unchanged.
