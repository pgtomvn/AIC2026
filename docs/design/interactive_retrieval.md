# Interactive retrieval local — production tools, not a paper reproduction

## Scope and provenance

This implementation borrows the **interaction goals** of
[OpenCubee_2](https://link.springer.com/chapter/10.1007/978-981-92-2590-3_18)
(reference image + text),
[OpenCubee_1](https://link.springer.com/chapter/10.1007/978-981-92-2587-3_38)
(multi-turn context), and
[LunchRetrieval/WESp](https://link.springer.com/chapter/10.1007/978-981-92-2590-3_44)
(ordered events). Full texts/exact author implementations were not verified.
No source from those projects was copied; no author-specific formula or optimal
coefficient is claimed. There are no new models, vector collections or extraction jobs.

Algorithms reuse the project's normalized
[Rocchio](https://nlp.stanford.edu/IR-book/html/htmledition/the-rocchio-algorithm-for-relevance-feedback-1.html),
[weighted RRF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), and strict TRAKE DP.

## How to use the Launcher

1. Search normally. On a result choose **Dùng làm ảnh mẫu**, **Gần đúng**, or
   **Không phù hợp**. Unselected results are never negatives. A reference is not a
   confirmed answer. Identity always includes batch, video and source frame (including 0).
2. In **Tìm tiếp**, edit the actual query preview. Explicitly add, replace or remove
   a selected condition; there is no autonomous interpretation of ambiguous edits.
   The preview is the text sent to retrieval, not a concatenation of every past turn.
3. Choose **Tìm tiếp**, **Hoàn tác**, or **Về kết quả ban đầu**. The expander displays
   **Tìm tiếp · Lượt n/20**, including while collapsed. Each session owns its
   own detached history, capped at 20 turns, plus the original snapshot. Undo/reset
   restore saved results without a new retrieval or translation. Starting a new normal
   search starts a new conversation. Pins/export remain under the operator's control.
4. For an ordered sequence, enable the manual sequence checkbox, enter 2–6 lines in
   chronological order, and select the step whose keyframe is wanted. Existing
   `current` is selected when available; otherwise the UI defaults visibly to the last
   step. Proof frames can be opened in the existing scrubber.
5. Open **±5 khung** and explicitly click **Tải OCR / transcript ±30 giây**.
   These reads happen on demand, not on every result card. Transcript is speech
   evidence, not visual verification. A neighboring frame can be pinned, used as a
   reference or sent to Visual Q&A.

## Backend contract

`POST /v1/kis/refine` retains old `positive_feedback_ids`/`negative_feedback_ids`
(`VIDEO:FRAME`, resolved within one selected batch). Optional new fields:

```json
{
  "query_id": "interactive-1",
  "original_text": "người đang đi ngoài trời",
  "batch_filters": ["aic25-b1-reference"],
  "reference_frame": {
    "batch_id": "aic25-b1-reference", "video_id": "L21_V001", "source_frame_idx": 0
  },
  "positive_frames": [],
  "negative_frames": [],
  "feedback_text": "người đang đứng ngoài trời",
  "image_only": false,
  "refine_raw": false
}
```

`feedback_text`, if supplied, is the **complete edited query**, not an instruction
to an LLM. Clear obsolete clause/entity overrides when replacing that text. OCR/ASR
keep its original language; existing content-keyed translation serves dense visual
spaces. Feedback cannot cross batches or enter `semantic_text`. Each valid visual
channel computes `norm(alpha*q + beta*mean(p) - gamma*mean(n) + rho*reference)`;
individual image observations are normalized, alpha=1, beta=.75, gamma=.25, rho=.75.
A reference already in positives is counted once. Image-only uses the reference
vector directly and omits lexical/semantic text branches. Invalid dimensions,
nonfinite/near-zero vectors fail that channel; successful channels still search.
All updated vectors are passed to one search/fusion call. Image/temporal tools are
incompatible with `asr_only`; the UI switches such explicit tools to automatic mode.

`POST /v1/kis/search` accepts `use_temporal_sequence`, `temporal_steps`, and
zero-based `temporal_target`, plus `temporal_mode: auto | manual | off`. `off`
overrides the legacy flag; an explicit legacy flag remains compatible. `auto`
cannot change ranking without an enabled switch **and** a verified release record.
The branch uses at most 12 candidate videos and the **same
Organizer-CLIP encoder and already-warmed TRAKE cache**. No request-time cache load
or build is permitted. Each proof step is revalidated against canonical metadata.
Frames/timestamps increase strictly within a video. Unknown shot IDs do not incur
shot-change cost. Penalties: .001/second beyond 120 seconds and .02 per known shot
change. Internal score is .25*global + .75*penalized sequence mean. The branch enters
RRF with seed weight .6 after baseline reranking; the old temporal bonus and window/
raw-frame replacement are skipped when this branch succeeds. The original channel
provenance survives. If unavailable, baseline ranking and a reason are returned.
This is explicit temporal retrieval, not a claim that vision confirms each event.

`GET /v1/frames/{batch}/{video}/{source_frame}/evidence?limit=200` is read-only.
It verifies the canonical anchor and returns OCR/ASR overlapping ±30 seconds, with
at most 200 segments total (budget reserved for both channels), text bounded per
segment and no filesystem paths. Neighboring frames continue to use canonical
ordinal adjacency, not arithmetic on source frame numbers.

## Flags, evaluation and rollback

Independent flags in `retrieval.interactive`:

- `feedback_enabled`
- `conversation_enabled`
- `temporal_enabled`
- `evidence_enabled`
- `event_inspection_enabled`

Feedback/conversation/temporal/evidence are enabled for **manual production use**.
`event_inspection_enabled` is **false**: its real-machine local Qwen smoke returned
`local_inference_failed`, so it remains off despite passing mocked/offline tests.
`automatic_activation`
is false. The router reads `temporal_release_record` at runtime creation and
verifies its source/configuration/cache fingerprints, parser version, parameters,
actual local label-file SHA-256 and user-confirmed ordered-event labels. It
recomputes the local gate instead of trusting `activation_gate.passed`. Setting
the YAML switch true alone never bypasses a gate. Keep semantic views and unrelated
rank policies unchanged. No automatic release record is supplied by this task.

### Explicit edit preview and interaction provenance

`POST /v1/kis/preview-edit` accepts `query` and `instruction`. Supported English
instructions are `add ...`, `remove ...`, and `replace ... with ...`. A replacement
must match exactly once; quote a phrase containing the word `with`. Unsupported,
ambiguous or destructive-empty edits return the original query with a reason.
The Launcher shows the draft/change and requires **Dùng bản nháp**, then **Tìm tiếp**;
preview alone never retrieves. Neither history nor translations are concatenated.

Optional query fields `interaction_session_id`, `interaction_turn` and
`event_source: user | benchmark | agent | unknown` preserve legacy clients.
Structured turn records include the actual query, canonical reference/feedback,
ranking, latency and retrieval fingerprint. Only the separate, explicit
`POST /v1/kis/confirm-answer` interaction with `event_source=user` is a human label.
Pinning/export and benchmark/agent actions do not confirm a correct answer.

### Manual local event inspection

The evidence panel exposes **Kiểm tra sự kiện**, independent of search ranking.
`POST /v1/kis/inspect-event` takes a question and 1–3 distinct canonical frame
identities from one video. Neighbor selection is explicit. It verifies the installed
Qwen model digest, uses the existing local provider and a labeled image contact
sheet, and returns `evidence_present`, `insufficient_evidence` or `contradiction`.
No cloud fallback, retry, corpus captioning or ranking bonus is permitted.

One inference is admitted at a time, with a total caller wait budget of 60 seconds.
Busy retrieval/warm-up declines inspection; an in-flight inspection rejects new
model requests instead of queuing them. A timed-out worker retains its lease until
it exits. Cache entries include image identity/content hash, model digest, provider
fingerprint, prompt version and question. A still image cannot establish motion
between frames, and OCR/transcripts are not visual confirmation. The rollout
report records whether the actual machine passed the manual inference smoke;
disable `event_inspection_enabled` if it cannot meet its resource budget.

`src/evaluation/interactive_pilot.py` defines a **separate** local gate, leaving the
120-qrels validator intact. It rejects absent/failed metrics, mismatched query/split
hashes, inappropriate labels, unlocked parameters, outside-scope rank changes,
quality regression, gain under .01, p95/RSS over 1.2x. Feedback requires actual
human multi-turn labels; ordinary KIS anchors are insufficient. No feedback is
generated from answers for evaluation. Exact anchor hit and video hit are separate;
unjudged frames are not declared incorrect. The existing 14/4/5 split is retained;
results are conservatively reported as a regression pilot, not independent evidence.
No author weights or query-specific tuning were selected in this release.

### Compact Launcher layout

Secondary controls default closed: search options, query diagnostics, **Tìm tiếp**,
and **Bài nộp · CSV / ZIP**. Selecting feedback/reference opens **Tìm tiếp**.
The query preview remains editable; indexed condition edits live under
**Sửa từng điều kiện**. Undo/reset do not perform retrieval. Result cards keep a
copyable `video_id,source_frame_idx`, timestamp, pin, reference, feedback, neighbors
and QA actions. Scores/channel ranks remain under **Bằng chứng**. Batch/health
counts, launch identity and warm-up details remain in the sidebar. Errors, required
channel failures, truncation warnings and external-transfer warnings are not hidden.
This is a presentation-only change: API payload defaults and ranking are unchanged.
UI rollback copies and hashes: `outputs/checkpoints/ui-simplify-20260922/`.

Baseline/checkpoint and local reports:
`outputs/checkpoints/interactive-retrieval-20260922/`. They include original dirty
source copies, SHA-256 manifest, before.patch, health/counts, rankings, latency/RSS,
test logs and a release record. Initial health was healthy; the strict pilot
nevertheless failed on pre-existing `advanced_rerank: deadline_exceeded`. Do not
use that failed run as a passing baseline or claim a speed/quality improvement.

Rollback each feature by setting its flag false and restarting only Launcher
API/Streamlit. Use the UI's **Về kết quả ban đầu** for a session-level rollback.
If disabling does not recover the baseline, compare current file hashes with the
release record before restoring only affected source files from `source/`.
Shared files contain other features: restoring one disables those too, so restore
the whole task's source set coherently only after checking no newer edits exist.
The backup includes pre-existing uncommitted changes. Never `git reset`, stash the
whole worktree, delete data/caches/receipts, or recreate Qdrant/Meilisearch.

New files may remain on disk when disabled; they perform no work at import time.
All reports and source backups are retained. No submissions or artifact bundles
are overwritten. Models/ranking production changes still require separate evidence.

## Earlier release observations (22 September 2026, before local v2)

The unchanged 23-query set was run in each mode, one warm-up and two recorded
passes. All 138 recorded rankings (100 hits each) matched baseline by canonical
identity and order. Anchor Hit@1/3/5 and MRR consequently did not change.
The existing train/validation/test membership remains 14/4/5; no coefficient was
selected using any of these answers, nor were answers injected as feedback.

| Mode | Baseline warm p95 | After warm p95 | Ratio |
| --- | ---: | ---: | ---: |
| visual_only | 1,871 ms | 1,898 ms | 1.014× |
| asr_only | 1,484 ms | 1,691 ms | 1.140× |
| automatic | 12,489 ms | 12,258 ms | 0.982× |

These are operational measurements on a shared local host, not an isolated speed
experiment: some offline verification overlapped the after capture. Working-set
RSS was below 1.2× baseline, but its decrease is **not** claimed as an optimization.
Baseline had 30 visual and 32 automatic requests with degraded channels; after
had 30 and 33 respectively (out of 46 requests per mode). ASR-only had none.
Most failures were the pre-existing advanced verification deadline. Therefore
zero-error/quality-gain/appropriate-label gates do **not** pass; automatic temporal
activation remains disabled. The new functionality is available as explicit tools,
not proven to improve held-out retrieval accuracy.

The live smoke selected its image from an ordinary result, not an answer file.
It verified multi-channel refinement, image-only search, edited text, actual warmed
temporal scoring, canonical step proofs, evidence reads, Vietnamese ASR routing,
QA candidate retrieval, and stable ordinary ranking after interaction. Full QA model
inference/external frame transfer is not part of this smoke.

Verification: `scripts/check.ps1 -IncludeServices -IncludeStreamlit` completed with
1,479 passed, 3 skipped, and the existing Starlette test-client deprecation warning.
The skipped tests require Windows symlink privilege or an explicit live-benchmark
opt-in. Real API smoke/health checks were run separately. `git diff --check` passes.
The runtime contract is version 7 so a stale UI/API pair is detected rather than
silently accepting unsupported interaction fields. User-owned dirty edits outside
this feature set remain unchanged and no commit/push is performed by this task.

## Local v2 continuation

The current HTTP/UI contract is version **8**. New scoped checkpoints, locked
English questions, request-level A/B captures, live smoke, source hashes and final
rollout decisions are under `outputs/checkpoints/local-retrieval-v2-20260922/`.
English questions are newly translated from the question text, not reconstructed
historical user queries. The same 14/4/5 split is a regression pilot: it has already
been inspected in earlier work. Repeated latency runs do not create new independent
queries or replace six human-confirmed temporal examples/six real feedback sessions.

Reranking now reports metadata/object-prefetch/geometry/color/fusion time and
reuses batch-aware metadata/evidence snapshots and bounded per-request decoded
images. The original 20 ms deadline, candidate count, score semantics and atomic
fallback are unchanged. Remaining deadline failures must be reported as failures;
HTTP 200 or healthy external stores alone do not satisfy the automatic ranking gate.

Final v2 checks: 1,509 passed, 3 skipped, Ruff/service/Streamlit checks passed.
The English A/B did not qualify: anchor Hit@5 47.83% → 39.13%, MRR .39196 → .36557,
56/69 rerank deadline failures per arm. The run was interrupted and recovered with
fresh warm-up; raw data and the caveat are retained. Parser v2 subsequently rejects
repetition idioms such as `one after another`; the v1 A/B does not license v2.
See `outputs/checkpoints/local-retrieval-v2-20260922/REPORT.md` for measurements,
manual/off decisions and file-scoped rollback. No automatic ranking was enabled.
