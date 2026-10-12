# Metadata-only retrieval optimization

`retrieval.metadata_resolution` accepts `scalar` (legacy rollback path),
`batched` (all KIS modes, experimental), or `asr_only_batched` (explicit ASR-only
requests; automatic/visual retain the exact legacy resolver). These resolve anchors
through the same artifact repository. Frame 0 is valid. Missing direct anchors
use the existing nearest-frame method, including its earlier-frame tie break.

The initial all-mode trial failed strict evidence invariance because the existing
20 ms advanced spatial/color deadline varies across requests. It was withdrawn;
do not activate all-mode batching from its speed numbers alone.

The batch implementation deduplicates identities, reads at most 100 per SQLite
query, and memoizes missing/nearest lookups within one thread's current KIS
request. It caches canonical metadata, never scores or evidence. Each new KIS
search resets its cache. It does not create vectors, modify indexes, or rebuild
artifacts. In ASR-only scope, automatic Visual Q&A and TRAKE remain unchanged.

## Controlled measurement

Use `scripts/benchmark_kis_operational.py` against a warmed canonical Launcher.
The PID must be the process listening on API port 8080, not a Python shim or
PowerShell parent. The benchmark does not launch services or load models.

```powershell
& .venv-runtime/Scripts/python.exe scripts/benchmark_kis_operational.py `
  --proxy-queries outputs/evaluation/user-qrels-20260905-v1/queries.jsonl `
  --split-file outputs/evaluation/user-qrels-20260905-v1/split.json `
  --partition development --batch-id aic25-b1-reference `
  --api-pid <LISTENER_PID> --output <NEW_BASELINE_PATH>
```

Restart with the candidate configuration, then repeat with a new output path
and `--baseline <NEW_BASELINE_PATH>`. Each capture warms every selected query
and mode before two measured passes with alternating query order. Raw-video
refinement is disabled by default to avoid extracting/writing frames; this is
a coarse-retrieval benchmark, not an end-to-end raw-refinement timing claim.
VI input follows the real translator; original VI remains available to ASR/OCR.

Acceptance compares the full returned hits (order, scores, evidence and metadata),
requires all repeated measurements, counts channel failures despite fallback,
and rejects p95 or process RSS exceeding 1.2 times baseline. Missing measurements
fail closed. Nonzero exit status denotes failure, not just an HTTP error.
Reports always set ranking activation eligibility to false. Positive-anchor
metrics are diagnostic; unannotated frames are not automatically negatives.

The 23-query pilot contains 14 train, 4 validation and 5 locked-test queries.
Do not select parameters or rewrite queries using locked-test outcomes. It does
not replace the existing production activation protocol or a larger qrels set.

## Rollback

Set only `retrieval.metadata_resolution: scalar` in `config/settings.yaml` and
restart the canonical Launcher. No index, database, bundle or cache deletion is
needed. Preserve all other worktree changes. If source rollback is necessary,
compare the per-file backups under
`outputs/checkpoints/retrieval-improvement/20260905-v1/source-backup/` first;
restore only this phase's changes, not the entire dirty repository.

Keep `semantic_views.fusion: global_only` and the existing feature modes; this
optimization is not authorization to activate semantic views, rescue, OCR
quality, video-local ranking or a new model.
