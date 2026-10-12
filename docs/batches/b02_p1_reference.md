# B02 P1: isolated organizer-reference ingestion

## Current release — 2026-09-25

**P1 reference retrieval is activated** for `aic26-b2`; Launcher remains off.
The older checkpoint notes below describe intermediate, inactive states.
They must not be used to relaunch the completed imports.

- 614 active videos; 220,270 ordinal canonical frames and Organizer-CLIP vectors.
- 210,404 searchable object documents from 220,270 input JSONs. Objects are not OCR.
- KIS supports English text, reference-image feedback, exact ordinal images,
  neighbor browsing, pinning and KIS/QA CSV/ZIP export through the B02 adapter.
- N exports use PTS-derived milliseconds; M/S use source frame indices.
  Equal-PTS images remain separate for browsing/QA, but duplicate submitted
  coordinates are rejected. Never submit an ordinal as a source frame.
- QA injection/manual answering retains the exact ordinal. Model-based VLM
  answer quality has not been evaluated in this release.
- TRAKE is now released separately for eligible M/S videos; see [B02 TRAKE](b02_trake.md)
  for excluded videos, canonical mapping, verification and rollback.
- OCR/ASR remain absent; source-frame refinement, B01 object-geometry
  reranking and KIS temporal/video-local paths remain disabled for B02 requests.
  No new corpus vectors or extraction were produced. This is **not Full Multi-Modal**.
- B01 remains the default Launcher batch. Select B02 when launching that batch;
  the current activation does not start Launcher or switch the saved default.

Acceptance: 1,606 offline tests passed (3 environment/opt-in skips), 18 final
focused checks passed, Ruff/services/Streamlit AppTest and `git diff --check`
passed. Real-model preactivation smoke and postactivation in-process HTTP smoke
passed; the latter used the real warm-up, search, images, evidence, QA manual
and export endpoints without opening ports 8080/8504. This is functional
acceptance, not a B02 retrieval-quality benchmark.

Key evidence under `outputs/checkpoints/b02-p1/`:

- `p1-reference-release-20260925.json`: final counts, source/file hashes and gates.
- `post-activation-api-20260925b.json`: successful HTTP/warm-up acceptance.
- `reference-activation-resume-20260925.json`: successful ledger activation.
- `vector-activation-audit-20260925.json`: fresh audit of all deterministic point
  IDs and canonical payloads before recording the missing vector ledger stages.
- `release-source-20260925/`: per-file source rollback copies.

The first activation attempt stopped before changing canonical readiness because
the repository object was not initialized. One audited vector stage had completed;
the idempotent resume preserved it and completed the remaining videos. Earlier
failed smoke reports are retained (one test harness expected `thumbnail_url`
instead of the established KIS `thumbnail_uri` field). Do not mistake those
reports for current running jobs. A temporary YAML indentation error was corrected
before the successful final checks; no Launcher was running during these edits.

B01 was checked against both exact external counts and the pre-import SQLite
fingerprints/readiness: Organizer 177,321; SigLIP2 176,707; OCR 1,167,264;
ASR 131,370; metadata 159,818. All remain unchanged.

Rollback: set only `readiness.batches.aic26-b2.retrieval_enabled: false` and
restart the API/UI if they are running. Keep the successful ingestion receipts,
ordinal mappings, vector/object stores, caches and ZIPs. Do not restore a whole
SQLite backup or reset/stash the dirty worktree. If code rollback is required,
restore only the affected phase files after reviewing intervening edits.

Next phase is separate OCR/ASR provisioning and acceptance; P1 does not certify
raw audio coverage, including user-reported silent N videos or the earlier M10
upload. Re-audit source availability before scheduling that work.

## Scope and release boundary

P1 uses existing organizer CLIP vectors, keyframe ZIP members, canonical map
rows, and object labels. It does not extract or re-encode the corpus, run
OCR/ASR, or change B01 indexes. Object labels are not OCR transcripts.

The user reports that M10 source-video upload is in progress and that N
traffic videos are silent. Neither blocks reference ingestion. These statements
are not substituted for a verified complete source inventory or ASR receipts.
Do not reuse the earlier faulty B02 zero-audio receipts.

The P0 manifest has 614 videos and 220,270 aligned keyframes/features/object
JSON records. Its nine documented equal-PTS image pairs in M08_V001 remain
distinct ordinal identities. N source-frame indices are advisory. Keep original
PTS exactly; derive submission milliseconds using Decimal half-even rounding.
Do not invent timestamps or merge these images.

## Model and storage

- Existing `ViT-B-32-quickgelu/openai` encoder, fingerprint
  `bb23f9883f7463e5`, matches six image/vector samples from M/N/S; minimum cosine
  `0.9999949932`. This verifies encoder compatibility, not retrieval quality.
- Receipt: `outputs/checkpoints/b02-p1/b02-clip-encoder-verification.json`.
- Dedicated Qdrant collection:
  `aic_native_aic26-b2_openclip_b32_quickgelu_openai_512d_v1`.
- Dedicated Meilisearch object index: `aic26_b2_objects_v1`.
- SQLite `canonical_ordinal_frames` primary key:
  `(batch_id, video_id, keyframe_ordinal)`. Legacy B01 canonical rows/mappings
  remain unchanged. ZIP image URIs avoid full extraction.
- N and M08_V001 payloads are ineligible for TRAKE. The B02 TRAKE runtime stays
  disabled until its temporal/cache contract is implemented and tested.

The production config deliberately keeps B02 `retrieval_enabled: false`, and
imported canonical rows are not marked index-ready. Collection discovery cannot
implicitly admit the new collection into B01 queries. B02 UI, QA evidence,
feedback identity, scrubber, and submission export must first be audited end to
end for ordinal identity and PTS-derived milliseconds. Import completion alone
does not authorize activating these paths or claiming Full Multi-Modal B02.

## Safe operations

`scripts/import_b02_reference.py` defaults to read-only validation. Writes
require `--apply` plus individually selected stages, a new `--report`, and
`--qdrant-host-disk`. It verifies the manifest, source inventories, and encoder
receipt before writes. `--video` restricts vector/canonical pilot ingestion;
object ingestion always processes the complete object sources.

The CLI records exact B01 counts before/after, verifies every imported vector
payload against its organizer map, and refuses to overwrite an earlier report.
Resuming a nonempty B02 collection requires `--resume` and matching manifest
provenance on all existing points. Upserts use deterministic IDs. No index is
recreated or deleted. Reports checkpoint completed stages; a report left in
`running` after process termination is not a completion receipt.

Disk guards run before each video and each object-document batch: preserve
30 GiB on D and 5 GiB on the Docker host disk C. Qdrant's Docker VHD is on C;
its virtual internal filesystem free space is not host free capacity.

SQLite online backups taken before the pilot:
`outputs/checkpoints/b02-p1/pre-import-db-20260924/`.
They are recovery artifacts, not files to copy over a live database. Normal
rollback is to keep B02 disabled; do not roll back the entire database and lose
unrelated subsequent work. Preserve source ZIPs, vectors, receipts, and sidecars.

## Runs and verification

- Pilot: `outputs/checkpoints/b02-p1/pilot-import-20260924.json`;
  M01_V001, N001-V001, S01-V001, 3,049 vectors and canonical ordinals verified.
- Full vector/canonical run:
  `outputs/checkpoints/b02-p1/full-reference-import-20260924.json`.
  The first process stopped with 183,206 points across 595 videos; the report
  remained `running`. No importer was alive before restarting. The durable
  resume run is `outputs/checkpoints/b02-p1/full-reference-resume-20260924.json`,
  with matching `.stdout.log` and `.stderr.log` files. Check its final `state`
  rather than inferring completion from this document.
- Resume completed: 614 videos, 220,270 vector points and 220,270 canonical
  ordinal rows. Independent read-only report:
  `outputs/checkpoints/b02-p1/reference-acceptance-20260924.json`. All payload
  identities/timestamps/image URIs match canonical metadata, all canonical PTS
  match organizer maps, nine collision pairs are retained, five sampled JPEGs
  decode, and five stored vector samples match their source rows.
- Deep acceptance: `outputs/checkpoints/b02-p1/reference-deep-acceptance-20260924.json`.
  The 176,707 B01 canonical row identities, content/processing fingerprints, and
  readiness flags are unchanged against the pre-import SQLite backup. Five
  stored-vector queries return correctly mapped B02 hits. This is a service
  smoke, not a text-query quality benchmark or an activation gate.
- Objects run: `outputs/checkpoints/b02-p1/objects-import-20260924.json`.
  A document is emitted only when there are readable object labels at confidence
  at least 0.50. The 220,270 input JSON count is not the expected document count;
  compare its final emitted count with the exact filtered Meilisearch count.
- The initial objects process stopped without a completion receipt at 153,000
  documents. Before resuming, an audit compared every stored deterministic UUID,
  original ID, and document field against the frozen source. All 153,000 matched.
  The full source emits **210,404** readable object-label documents (confidence
  >= 0.50), leaving **57,404** to import. Resume receipt/progress is in
  `outputs/checkpoints/b02-p1/objects-audited-resume-20260924.json`. Only documents
  that passed the content audit are skipped. Unknown/changed data rejects resume
  before create/settings/upsert; original reports remain preserved.
- Objects resume has now completed: **210,404 emitted documents = 210,404
  external documents**, including 57,404 newly written and 153,000 audited skips.
  Independent `outputs/checkpoints/b02-p1/full-data-acceptance-20260924.json`
  checks every object identity/source index/timestamp against canonical rows,
  all 220,270 vector payloads against maps, and the unchanged B01 counts and
  canonical fingerprints. B02 remains inactive. These are object labels, not OCR.
- At final acceptance D had 20,385,931,264 bytes free (about 19 GiB), below
  the 30 GiB write reserve; C had 6,559,100,928 bytes free (about 6.1 GiB).
  No additional large data job is authorized by this acceptance. The later
  reduction in free space has not been attributed to any specific process.
- B01 invariants: Organizer CLIP 177,321; SigLIP2 176,707; OCR 1,167,264;
  ASR 131,370; metadata 159,818.
- `scripts/check.ps1 -IncludeServices` passed: 1,574 tests, 3 environment/opt-in
  skips; Qdrant and Meilisearch healthy. No live retrieval benchmark was claimed.
- Offline Streamlit AppTest passed without starting Launcher. `git diff --check`
  passed. Preflight reported absent local raw B01 video archives; these are not
  inputs to the B02 reference import.

Launcher remains off. Further phase activation requires a separate verified
release; ASR/OCR scheduling must not be inferred from these reference counts.

## Ordinal presentation groundwork (not batch activation)

New read-only endpoints use an explicit keyframe coordinate:

- `/v1/keyframes/aic26-b2/{video_id}/{keyframe_ordinal}/image`
- `/v1/keyframes/aic26-b2/{video_id}/{keyframe_ordinal}/neighbors?radius=5`

Neighbors are bounded to eleven ordinals and retain equal-PTS images. Public
B02 hits require a positive ordinal and use its image URL; B01 routes retain
source-frame semantics. Response metadata includes explicit `submission_unit`
and `submission_value`: PTS-derived milliseconds for N, source frame for M/S.
This is coordinate preview, **not** a new CSV submission exporter or release.

Read-only HTTP smoke using the actual SQLite and ZIPs passed for M08 ordinals
72/73, N001-V001 ordinal 100, and S01-V001 ordinal 1000. M08 images have distinct
ETags despite identical source frame/time. Report:
`outputs/checkpoints/b02-p1/ordinal-api-smoke-20260924.json`.

UI pinning, feedback, QA, and CSV export still need an end-to-end ordinal-aware
integration; B02 remains disabled and no old source-frame handler is passed an
ordinal as if it were a decoded source frame. Source backups are in
`outputs/checkpoints/b02-p1/ordinal-presentation-source/`; preserve all unrelated
preexisting changes if a targeted rollback is needed.

The latest offline suite passes 1,586 tests with 3 environment/opt-in skips and
healthy storage services. An earlier run under simultaneous import load had
two timing-sensitive failures (event-inspection deadline and advanced reranker
rank delta); both passed isolated reruns and the subsequent complete suite.
No production deadline, candidate count, or fallback policy was changed to
obtain the passing result. Keep this timing sensitivity visible in later
performance acceptance; unit tests alone are not a latency benchmark.

The thin HTTP client now exposes exact ordinal lookup and bounded neighbors,
and the KIS image loader uses that path for B02. It validates returned batch,
video, ordinal, and coordinates before displaying an image. Missing ordinal
never falls back to a source-index image; B01 image URLs are unchanged.
This does not migrate pin/feedback/export identity or activate B02 retrieval.
Per-file rollback copies for this increment are under
`outputs/checkpoints/b02-p1/ordinal-client-source-20260924/`.

## Remaining P1 release checklist

Data ingestion and coordinate/image reads are complete; production activation is not.
Continue in this order, without rerunning the successful import:

1. Migrate UI pin/scrubber identity end to end. `result_identity()[2]` is still
   consumed as a source index in existing handlers; changing just that helper
   would silently misroute B02 images. Preserve B01 handler behavior.
2. Carry optional ordinal identity through feedback and QA requests and resolve
   metadata in that same coordinate system. Reject missing/ambiguous identities.
3. Add verified KIS/QA export using explicit coordinates (N: milliseconds;
   M/S: source frame), while retaining N and M08 TRAKE exclusions. Do not export
   an image ordinal as a source frame or infer milliseconds from nominal FPS.
4. Bind B02's isolated object index and verified visual collection only after
   these paths pass integration tests. Capture baseline and activate B02 with
   rollback independently from B01. Missing OCR/ASR must remain visible.

No OCR/ASR extraction or large cache construction is part of this increment.
Recheck actual C/D free capacity before any new material write; free capacity
has varied since the acceptance snapshot, whose values are historical observations.

Ordinal-client verification completed on 2026-09-25: Ruff passed, 1,596 offline
tests passed with 3 environment/opt-in skips, storage service checks passed,
Streamlit AppTest passed, and `git diff --check` passed. No live Launcher or
model-based retrieval benchmark was started. The later free-space observation
was D 32,362,356,736 bytes (about 30.1 GiB), only narrowly above its write guard.
