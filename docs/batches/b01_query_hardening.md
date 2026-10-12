# B01 query hardening — 2026-09-25

## Scope

Fix the B01 cold Tier-2 deadline/rank instability and raw-refinement resource
handling. No model, collection, index, artifact, receipt, corpus or configuration
change. This task did not start, stop or restart the existing Launcher process;
a new process is required to load this patch. All-batch routing and submission
coordinates stay unchanged. Runtime verification used an isolated in-process
API, not that older Launcher process.

## Tier 2

- Use `perf_counter` for the existing 20 ms verification budget. In the local
  Python 3.11 runtime, `monotonic` uses GetTickCount64 with 15.625 ms resolution;
  `perf_counter` uses QueryPerformanceCounter with 0.0001 ms resolution.
- Cache normalized fixed label aliases, reuse scoped candidate identities and
  initial ranks, avoid sorting twice when there is no color modification.
- Keep the complete candidate list and the configured spatial/color depths.
  Untouched tail candidates still participate in final sorting and rank deltas.
- Keep atomic deadline fallback, including expiry late in rank-delta generation.
  Partial scores or diagnostics must never leak into the baseline snapshot.

The 20 ms budget still starts after metadata/object prefetch, as before. Total
rerank latency therefore includes additional I/O; it is not a 20 ms end-to-end
SLA. Real overload can still trigger an honest deadline fallback.

## Raw refinement

`reference://batch/video` is an organizer identity, not a playable local file.
It must not be passed to PyAV or trigger GPU query/image encoding. Such hits keep
their original canonical frame and report `reference_only_source`, state
`not_applicable`, and `refined_count=0`.

Local/ZIP sources remain supported. Missing ZIP members and scratch permission
failures retain coarse results with specific resource diagnostics. Query encoding
is lazy: it runs once only after a usable window has been decoded. Invalid query
embeddings are not mislabeled as missing media. API sanitization preserves
unavailable/degraded states and the refined count.

The current B01 reference catalog cannot provide real raw-window refinement
without local video sources and verified exact anchors. This patch does **not**
claim to restore missing raw media or fabricate refined frames.

## Evidence and rollback

Evidence/checkpoint directory: `outputs/checkpoints/b01-query-hardening-20260925/`.
It contains file-level `source-before/` backups, diagnostic runners, failed
intermediate probes and successful probes. Earlier all-batch release evidence is
retained unchanged. These are functional regression checks, not recall benchmarks.

Final verification: 1,636 offline tests passed, 3 skipped; Ruff, read-only service
checks, Streamlit AppTest and `git diff --check` passed. Six B01 requests including
the cold request preserved top-6, without a Tier-2 timeout. Combined B01/B02 ranks
and counts were unchanged. See `final-runtime.json`, `handoff-check.log` and
`release.json`. One earlier run failed a QA concurrent-window test, then the
isolated group and final full suite passed; no QA code was changed to hide it.

Rollback only the files changed in this phase from its checkpoint; do not reset,
checkout or stash the entire dirty worktree. Source files in scope are
`src/retrieval/{color_filter,raw_refiner,spatial_rerank,textual_kis}.py`,
`src/api.py`, and `src/runtime_contract.py`, plus their focused tests. Check for
subsequent edits before restoring. Preserve stores, caches, outputs and receipts.
If Launcher is already running when rolling back, restart only Launcher and
recheck external counts. Do not stop or recreate Qdrant/Meilisearch.
