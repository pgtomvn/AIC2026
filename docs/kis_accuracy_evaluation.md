# KIS accuracy evaluation and activation

KIS tuning is label-gated. Do not generate or infer qrels from the current ranking: every row
must be reviewed against the source video by a human before it is marked ready for activation.

## Qrels contract

The production pack contains at least 120 queries and 60 independent video groups. It includes
at least 20 reviewed queries for each intent: `visual_dense`, `lexical_dominant`,
`absolute_spatial`, `relative_spatial`, `color`, and `negation`. Each JSONL row contains a stable
query ID, the reviewed query text plus its SHA-256, one intent, one video-group label, and one or
more canonical source-frame intervals.

Lock the grouped 60/20/20 split before collecting or tuning cached rankings:

```powershell
$Uv = Resolve-Path .tools/uv/0.11.16/uv.exe
& $Uv run --frozen python scripts/lock_kis_qrels.py `
  outputs/evaluation/kis-qrels/manifest.json --seed 2026 --write
```

The command updates only `split_metadata` atomically. `locked_test_sha256` binds the content hash,
seed, and test query IDs. Any later label or split change invalidates the activation contract.

## Per-intent policy tuning

Cache each retrieval channel once for the immutable pack. The cache may contain the locked test
queries, but the tuner consumes only train/validation IDs; test rows are retained solely for final
evaluation. Then tune the three autonomous runtime buckets without re-running encoders:

```powershell
& $Uv run --frozen python scripts/cache_kis_channel_rankings.py `
  --qrels-manifest outputs/evaluation/kis-qrels/manifest.json `
  --batch-id aic25-b1-reference

& $Uv run --frozen python scripts/tune_retrieval.py `
  --rankings outputs/evaluation/kis-channel-rankings.json `
  --qrels-manifest outputs/evaluation/kis-qrels/manifest.json `
  --per-intent
```

The fingerprinted result is written to the configured `retrieval.intent_tuning_record`. Legacy
API profiles remain accepted but do not select a routing branch. The CLI validates the production
qrels contract and excludes every locked-test query from parameter selection.

## Activation gate

Evaluate baseline and candidate on the locked test partition. Activation requires non-regression
for R@1, R@5, MRR, and every intent; mean top-k must improve by at least 0.01; warm p95 latency
must be no more than 1.20 times baseline. Build Meilisearch v2 shadow indexes with
`scripts/provision_meili_shadow.py`, verify batch count parity, and A/B them on the same locked
test split before changing any live index setting. The provisioner never promotes or deletes v1.
