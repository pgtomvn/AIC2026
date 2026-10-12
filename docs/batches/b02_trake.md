# B02 TRAKE release — 2026-09-25

TRAKE uses the existing verified Organizer-CLIP query encoder and B02 vectors.
No extraction, re-encoding, index writes, corpus extraction or DRES submission
is required. Launcher is left off; B01 remains its saved default batch.

## Scope and identity

- Eligible: 303 M videos and 12 S videos (315 total).
- All 298 traffic N videos are excluded, as required by the organizer.
- M08_V001 remains quarantined because its mapping has equal-source/equal-PTS
  image collisions. It is still usable through the ordinal KIS/QA paths.
- Read 131,245 eligible vector payloads into a separate trajectory snapshot.
  Thirteen rounded source indices each identify two images with distinct PTS;
  the TRAKE candidate timeline retains the **earlier PTS image's actual vector**,
  not an average. The cache contains 131,232 strict source/time positions.
  Later-PTS images remain available for explicit ordinal-based neighbor edits.
- Every returned stage is resolved by exact `(batch, video, source_frame, PTS)`
  to a canonical ordinal. An unresolved/ambiguous mapping fails closed. A chain
  must have strictly increasing source frames, timestamps and ordinals in one video.
- B02 filtering happens before grouped candidate retrieval and vector scrolling.
  Traffic cannot crowd the candidate budget or enter exports via lexical rescue.

The cache policy `b02-ms-first-source-pts-v1` is included only in B02 cache
fingerprints. B01 cache fingerprint/alias-merge behavior is preserved.
The snapshot uses 271,912,704 bytes of arrays (~259 MiB), with no request-time rebuild.

## Launcher and export

Select B02 and open TRAKE. After warm-up, enter a global description and 2–20
ordered stages. Results display canonical stage images. Neighbor edits retain
ordinal identity (including source frame zero); pinning can append or update
one video per rank. CSV/ZIP export uses the B02 adapter, not the historical
L-series validator. Headerless TRAKE rows are `video_id,source_1,source_2,...`.
All rows must have the same number of stages; N and M08_V001 are rejected.

This validates **local CSV coordinates only**. A live DRES client, evaluation
selection, authentication and remote submission acceptance remain separate work.
It does not certify a DRES payload or claim B02 Full Multi-Modal. B02 OCR/ASR
are not provisioned by this release.

Real smoke also exposed S01 ZIP filenames such as `0030.jpg` while stored URIs
used `030.jpg`. The image reader now verifies the exact S-series archive,
video directory and ordinal and accepts the unique three-/four-digit member.
Ambiguous members are rejected. ZIPs, canonical rows and payloads are unchanged.

## Verification and evidence

Release checks: 1,619 offline tests passed, 3 environment/opt-in skips; Ruff,
Qdrant/Meilisearch health checks and full Streamlit AppTest passed. The B02
AppTest specifically exercises search, ordinal-neighbor edits, pin/update
and validated CSV staging without entering the historical L-only adapter.

`outputs/checkpoints/b02-trake-20260925/` contains source-before backups and
read-only/in-process runtime checks. `pre-release-smoke-v3.json` passed real
warm-up, all 131,232 source/PTS mappings, M/S search, images, CSV, prohibited-video
export rejection, KIS regression smoke and unchanged B01/B02 external counts.
`released-runtime.json` repeats these checks with the saved flag enabled;
`release.json` records the release, file-hash evidence reference and rollback.
Earlier failed smoke reports are retained: they revealed the S01 member error.
The offline suite also caught a null-ordinal compatibility regression in B01
export; unset optional ordinals are now omitted from serialization.

This is functional acceptance, **not a retrieval-quality benchmark**. The two
English smoke queries have no relevance labels. No model weights, fusion
coefficients or automatic ranking policy were tuned on them.

## Rollback

Set only `readiness.batches.aic26-b2.trake_enabled: false` in
`config/settings.yaml`, then restart Launcher when needed. B02 KIS/QA remain
enabled. Preserve the cache, checkpoints, indices, maps and artifacts. If a code
rollback is needed, compare and restore only affected files from `source-before/`;
do not reset/stash the dirty repository or restore whole databases.
