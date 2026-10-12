# Fine-grained KIS: architecture, validation, and activation

## Root cause trace

The reviewed seed query describes `L26_V056`, source frames `6362–6528`. The
pre-change trace showed Organizer CLIP placing these frames at ranks 1–3. Four
partial native SigLIP2 hits then displaced them, despite that collection containing
only `569/176707` canonical points. A Vietnamese phrase inside parentheses also
changed the language route, unconfirmed culinary hints were treated as
organizations, and the first object-archive access consumed the Tier-2 deadline.

The failure was therefore a composition problem rather than a missing Organizer
embedding: incomplete-channel fusion, dense-query contamination, direct metadata
or entity boosts, and non-transactional cold evidence obscured a strong visual hit.

The second reviewed seed is exactly `L21_V003:26220` (`1048.8 s`). Both visual
indexes contain it, but the original long caption missed Top-100 while the relaxed
description “red yellow and black vehicles in water under a stone bridge” ranked
first. The parser had interpreted “one after another” as a before/after clause,
attached “prepare one after” to the black-car subject, and collapsed the three cars
into one instance. The object archive also describes the amphibious vehicles as
`Tank`, `Boat`, and `Car`; a closed `car` label could not use that positive evidence.
Frames `26100` and `26310` are different shots and are not part of this regression.

## Runtime architecture

`QueryTextAdapter` detects the primary language from text outside parentheses.
Parenthetical bilingual cues and unconfirmed operator hints are independent
OCR/ASR/metadata aliases. They are never replaced by person/location/organization
hypernyms and never receive entity bonus or required-entity filtering.

Long structured queries expose at most four deterministic semantic views. The
legacy fine-grained views are:

1. `global` preserves the main scene and action.
2. `action` isolates the principal actor/action anchor.
3. `foreground_subject` lists the foreground objects.
4. `distinctive_attributes` binds coordinated colors to their subjects.

For repeated same-class instances and a prospective relation, the fourth slot can
instead be `relation_relaxed`. The vehicle seed becomes three stable nodes
`vehicle_1 {yellow}`, `vehicle_2 {red}`, and `vehicle_3 {black}` in one group. Its
independent candidate views are global, scene/action, the three bound attributes,
and “yellow, red and black vehicles in water at/by/near a bridge”. This view only
expands recall; it never asserts that “cross a bridge” means “under a bridge”.
Ordering idioms become positive-only `SequenceConstraint` evidence. Prospective
`prepare/plan/wait to …` relations never supply a current-frame negative penalty.

Each view is encoded and searched independently. The query embeddings are not
averaged before search. The supported within-encoder policies are `global_only`,
`weighted_rrf`, `max_rrf`, and `rrf_consensus`. Production remains `global_only`
until `outputs/evaluation/kis-semantic-view-tuning.json` has valid locked-qrels,
model, query-adapter, and ranking-cache fingerprints and a passing activation gate.

Native SigLIP2 participates only when its batch count exactly equals the configured
canonical count. Until B01 provisioning reaches `176707`, the complete Organizer
CLIP channel remains the visual fallback. Metadata is video-level calibrated
evidence; an unconfirmed culinary hint cannot add an entity bonus.

Tier-2 opens the object ZIP central directory during warmup. A detector mismatch is
negative evidence only when a compatible ROI exists. Missing boxes are
`unlocalized/inconclusive`. Full-frame and center 3×3 color fallback can add a
positive rank-calibrated delta but cannot demote a candidate. The 20 ms transaction
commits only a complete head; otherwise it returns the untouched Stage-1 snapshot.
Repeated instances use a one-to-one ROI assignment, so one car box cannot satisfy
three color constraints. The broad `vehicle` ontology accepts Car/Boat/Tank only as
positive evidence; strict class compatibility remains required for any penalty.
The 2026-08-24 local grid selected `(spatial_depth=64, color_depth=8)`: it was the
largest tested head whose five warm runs all committed, with p50 `8.555 ms` and
p95 `9.815 ms`. The previous `(100, 12)` configuration expired at object candidate
73 and returned Stage-1 after `31 ms`. This is a latency choice only; evidence
weights and color thresholds still require locked qrels.

Every request now records bounded rank traces for Tier 1, metadata, entity policy,
Tier 2, raw refinement, and diversity, plus per-result rank deltas. These diagnostics
make the point of displacement observable without exposing private store payloads.

## Human-reviewed qrels workflow

Prepare 120 query rows as JSONL with `query_id`, `text`, `intent`, `video_group`, and
optional cross-tags from `long_tail`, `fine_grained`, `bilingual_parenthetical`,
`oov`, `culinary`, `action`, `social`, `multi_instance_same_class`,
`attribute_binding`, `prospective_relation`, `temporal_idiom`, and
`caption_visual_mismatch`. The six primary intents each require 20
queries and the set must cover at least 60 video groups. An unreviewed seed interval
may be included under `seed_intervals`; it is explicitly not ground truth.

Build the review bundle from immutable cached per-channel rankings:

```powershell
$Uv = Resolve-Path .tools/uv/0.11.16/uv.exe
& $Uv run --frozen python scripts/build_kis_qrels_review.py `
  --queries outputs/evaluation/kis-review-queries.jsonl `
  --cached-rankings outputs/evaluation/kis-channel-rankings.json `
  --output-dir outputs/evaluation/kis-qrels-review `
  --candidate-depth 20
```

The bundle contains JSONL and CSV review rows, a contact sheet using the canonical
frame endpoint, and a SHA-256 manifest. Reviewers set only `relevant`,
`non_relevant`, or `unsure` and provide frame intervals. After joint review, lock the
video-grouped 60/20/20 split with `scripts/lock_kis_qrels.py`. The test fingerprint
must be written before any parameter selection.

Fusion candidates are evaluated deterministically on validation in this order:
R@3, MRR, R@1, Mean Top-K, then lower warm p95. Color thresholds come only from
midpoints at observed positive/negative train boundaries. The locked test is read
once for activation and must satisfy non-regression for R@1/R@3/R@5/MRR and every
intent, Mean Top-K gain of at least `0.01`, and p95 no greater than `1.20×` baseline.

Do not produce the production tuning record until SigLIP2, OCR, and ASR provisioning
is complete; otherwise its model and ranking-cache fingerprints are stale.

Cache semantic views for one encoder without changing the production
`global_only` policy, then search all `k=1..candidate_depth`. Weights are normalized
from standalone train MRR; policy/depth are selected on validation. The emitted
record has `activation_gate.passed=false` and runtime will reject it until the
locked test has passed.

```powershell
uv run --frozen python scripts/cache_kis_channel_rankings.py `
  --qrels-manifest outputs/evaluation/kis-qrels/manifest.json `
  --batch-id aic25-b1-reference --semantic-views `
  --dense-channel organizer_clip `
  --output outputs/evaluation/kis-organizer-semantic-rankings.json

uv run --frozen python scripts/tune_retrieval.py `
  --qrels-manifest outputs/evaluation/kis-qrels/manifest.json `
  --rankings outputs/evaluation/kis-organizer-semantic-rankings.json `
  --semantic-views --semantic-candidate-depth 100
```

## Verification commands

```powershell
uv run --frozen pytest -q tests/test_finegrained_retrieval_reasoning.py
$env:AIC_RUN_SERVICE_BENCHMARK = "1"
uv run --frozen pytest -q tests/test_finegrained_service_benchmark.py
uv run --frozen python scripts/benchmark_kis_advanced.py `
  --finegrained-seed-gate --iterations 3
uv run --frozen python scripts/benchmark_kis_advanced.py `
  --multi-vehicle-seed-gate --iterations 3
uv run --frozen python scripts/trace_kis_compositional_shadow.py `
  --fusion max_rrf --iterations 3
```

The service benchmark logs the seed rank, stage movement, cold/warm latency,
FastAPI RSS, object/color cache state, and fails if no frame in
`L26_V056:6362–6528` is in Top-3. Its compositional companion executes the original
vehicle query three times and requires exactly `L21_V003:26220` in Top-3 without
raw refinement, with stable cold/warm Top-10 and Tier-2 commit state.
The shadow command mutates only the in-memory policy and reports
`persistent_state_changed=false`; it requires the optional OpenCLIP and
PyTorch/Transformers competition profile already used by the Launcher.
