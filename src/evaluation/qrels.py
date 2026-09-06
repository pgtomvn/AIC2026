"""Immutable qrels packs and activation gates for Advanced Textual KIS."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.evaluation.textual_kis import GroundTruthInterval, GroundTruthQuery

QRELS_SCHEMA_VERSION = 1
IMPORTANT_INTENTS = (
    "visual_dense",
    "lexical_dominant",
    "absolute_spatial",
    "relative_spatial",
    "color",
    "negation",
)
QRELS_TAGS = frozenset(
    {
        "long_tail",
        "fine_grained",
        "bilingual_parenthetical",
        "oov",
        "culinary",
        "action",
        "social",
        "multi_instance_same_class",
        "attribute_binding",
        "prospective_relation",
        "temporal_idiom",
        "caption_visual_mismatch",
        "ocr_heavy",
        "ocr_ticker_noise",
        "long_description",
        "multi_event",
        "speech_cue",
    }
)


@dataclass(frozen=True, slots=True)
class QrelsManifest:
    spec_id: str
    corpus_fingerprint: str
    index_fingerprint: str
    model_fingerprint: str
    content_sha256: str
    split_metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class QrelsPack:
    manifest: QrelsManifest
    queries: tuple[GroundTruthQuery, ...]
    video_groups: Mapping[str, str]
    query_tags: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LockedQrelsSplit:
    """Deterministic video-group partitions with an immutable test fingerprint."""

    train_query_ids: tuple[str, ...]
    validation_query_ids: tuple[str, ...]
    test_query_ids: tuple[str, ...]
    test_sha256: str


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def load_qrels_pack(manifest_path: str | Path) -> QrelsPack:
    """Load and verify one immutable JSON manifest plus JSONL qrels payload."""

    path = Path(manifest_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != QRELS_SCHEMA_VERSION:
        raise ValueError("unsupported qrels manifest schema")
    content_file = _required_text(raw.get("content_file"), "content_file")
    if Path(content_file).is_absolute() or ".." in Path(content_file).parts:
        raise ValueError("qrels content_file must be manifest-relative")
    content_path = path.parent / content_file
    content = content_path.read_bytes()
    expected = _required_text(raw.get("content_sha256"), "content_sha256").casefold()
    if len(expected) != 64 or hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("qrels content SHA-256 mismatch")
    split_metadata = raw.get("split_metadata")
    if not isinstance(split_metadata, Mapping):
        raise ValueError("split_metadata must be a mapping")
    manifest = QrelsManifest(
        spec_id=_required_text(raw.get("spec_id"), "spec_id"),
        corpus_fingerprint=_required_text(raw.get("corpus_fingerprint"), "corpus_fingerprint"),
        index_fingerprint=_required_text(raw.get("index_fingerprint"), "index_fingerprint"),
        model_fingerprint=_required_text(raw.get("model_fingerprint"), "model_fingerprint"),
        content_sha256=expected,
        split_metadata=dict(split_metadata),
    )
    queries, groups, query_tags = [], {}, {}
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"qrels line {line_number} must be an object")
        query_id = _required_text(value.get("query_id"), "query_id")
        if query_id in groups:
            raise ValueError("qrels query IDs must be unique")
        text = _required_text(value.get("text"), "text")
        text_sha = _required_text(value.get("text_sha256"), "text_sha256").casefold()
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != text_sha:
            raise ValueError(f"qrels text SHA-256 mismatch at line {line_number}")
        intent = _required_text(value.get("intent"), "intent")
        group = _required_text(value.get("video_group"), "video_group")
        raw_tags = value.get("tags", ())
        if not isinstance(raw_tags, (list, tuple)) or not all(
            isinstance(tag, str) and tag in QRELS_TAGS for tag in raw_tags
        ):
            raise ValueError(f"qrels tags are invalid at line {line_number}")
        intervals_raw = value.get("intervals")
        if not isinstance(intervals_raw, list) or not intervals_raw:
            raise ValueError("qrels intervals must be a non-empty list")
        intervals = tuple(
            GroundTruthInterval(
                _required_text(item.get("video_id"), "video_id"),
                int(item.get("start_frame_idx")),
                int(item.get("end_frame_idx")),
            )
            for item in intervals_raw
            if isinstance(item, Mapping)
        )
        if len(intervals) != len(intervals_raw):
            raise ValueError("qrels interval must be an object")
        queries.append(GroundTruthQuery(query_id, intervals, text=text, intent_class=intent))
        groups[query_id] = group
        query_tags[query_id] = tuple(dict.fromkeys(raw_tags))
    return QrelsPack(manifest, tuple(queries), groups, query_tags)


def validate_production_qrels(
    pack: QrelsPack, *, important_intents: tuple[str, ...] = IMPORTANT_INTENTS
) -> None:
    """Reject qrels packs too small or imbalanced for production activation."""

    if len(pack.queries) < 120:
        raise ValueError("production qrels require at least 120 labeled queries")
    if len(set(pack.video_groups.values())) < 60:
        raise ValueError("production qrels require at least 60 independent video groups")
    counts = Counter(query.intent_class for query in pack.queries)
    missing = [intent for intent in important_intents if counts[intent] < 20]
    if missing:
        raise ValueError("production qrels require at least 20 queries per important intent")
    metadata = pack.manifest.split_metadata
    ratios = metadata.get("ratios")
    expected_ratios = {"train": 0.6, "validation": 0.2, "test": 0.2}
    if metadata.get("method") != "video_group" or ratios != expected_ratios:
        raise ValueError("production qrels require a fixed video-grouped 60/20/20 split")
    seed = metadata.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("production qrels split seed must be an integer")
    split = build_locked_qrels_split(pack, seed=seed)
    if metadata.get("locked_test_sha256") != split.test_sha256:
        raise ValueError("production qrels locked test fingerprint mismatch")
    coverage = qrels_coverage_report(pack, split=split)
    if coverage["video_group_leakage"]:
        raise ValueError("production qrels split leaks a video group")
    missing_split_intents = [
        f"{partition}:{intent}"
        for partition, counts_by_intent in coverage["intent_counts_by_split"].items()
        for intent in important_intents
        if int(counts_by_intent.get(intent, 0)) < 1
    ]
    if missing_split_intents:
        raise ValueError("production qrels must represent every intent in every split")


def build_locked_qrels_split(pack: QrelsPack, *, seed: int = 2026) -> LockedQrelsSplit:
    """Build a deterministic 60/20/20 split using declared video groups."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("qrels split seed must be an integer")
    groups: dict[str, list[str]] = {}
    for query in pack.queries:
        group = pack.video_groups.get(query.query_id)
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"qrels query {query.query_id!r} is missing a video group")
        groups.setdefault(group, []).append(query.query_id)
    if len(groups) < 3:
        raise ValueError("at least three independent qrels video groups are required")
    ordered_groups = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest(),
    )
    train_end = max(1, round(len(ordered_groups) * 0.6))
    validation_end = max(train_end + 1, round(len(ordered_groups) * 0.8))
    validation_end = min(validation_end, len(ordered_groups) - 1)
    group_partitions = (
        ordered_groups[:train_end],
        ordered_groups[train_end:validation_end],
        ordered_groups[validation_end:],
    )
    query_partitions = tuple(
        tuple(sorted(query_id for group in partition for query_id in groups[group]))
        for partition in group_partitions
    )
    test_content = {
        "qrels_content_sha256": pack.manifest.content_sha256,
        "seed": seed,
        "test_query_ids": query_partitions[2],
    }
    test_sha256 = hashlib.sha256(
        json.dumps(test_content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return LockedQrelsSplit(*query_partitions, test_sha256)


def qrels_coverage_report(
    pack: QrelsPack,
    *,
    split: LockedQrelsSplit | None = None,
) -> dict[str, Any]:
    """Report intent/tag/group coverage and prove split isolation.

    This function never infers relevance. It only audits already human-reviewed
    qrels and is therefore safe to run before tuning or activation.
    """

    selected_split = split or build_locked_qrels_split(
        pack, seed=int(pack.manifest.split_metadata.get("seed", 2026))
    )
    query_by_id = {query.query_id: query for query in pack.queries}
    partitions = {
        "train": selected_split.train_query_ids,
        "validation": selected_split.validation_query_ids,
        "test": selected_split.test_query_ids,
    }
    intent_counts_by_split: dict[str, dict[str, int]] = {}
    groups_by_split: dict[str, set[str]] = {}
    for name, query_ids in partitions.items():
        intent_counts_by_split[name] = dict(
            sorted(Counter(query_by_id[query_id].intent_class for query_id in query_ids).items())
        )
        groups_by_split[name] = {pack.video_groups[query_id] for query_id in query_ids}
    leakage: list[dict[str, str]] = []
    names = tuple(partitions)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            leakage.extend(
                {"video_group": group, "splits": f"{left},{right}"}
                for group in sorted(groups_by_split[left] & groups_by_split[right])
            )
    return {
        "query_count": len(pack.queries),
        "video_group_count": len(set(pack.video_groups.values())),
        "intent_counts": dict(
            sorted(Counter(query.intent_class for query in pack.queries).items())
        ),
        "tag_counts": dict(
            sorted(Counter(tag for tags in pack.query_tags.values() for tag in tags).items())
        ),
        "query_counts_by_split": {
            name: len(query_ids) for name, query_ids in partitions.items()
        },
        "video_group_counts_by_split": {
            name: len(groups) for name, groups in groups_by_split.items()
        },
        "intent_counts_by_split": intent_counts_by_split,
        "video_group_leakage": leakage,
        "locked_test_sha256": selected_split.test_sha256,
    }


def _report(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return result
    raise TypeError("evaluation report must be a mapping or expose to_dict")


def advanced_activation_gate(baseline: Any, candidate: Any) -> dict[str, Any]:
    """Evaluate the frozen v3 non-regression and Mean Top-k improvement gate."""

    base, trial = _report(baseline), _report(candidate)
    base_recall, trial_recall = base.get("recall_at"), trial.get("recall_at")
    if not isinstance(base_recall, Mapping) or not isinstance(trial_recall, Mapping):
        raise ValueError("evaluation reports must contain recall_at")
    checks = {
        "r_at_1_non_regression": float(trial_recall.get("1", 0.0)) >= float(base_recall.get("1", 0.0)),
        "r_at_3_non_regression": float(trial_recall.get("3", 0.0)) >= float(base_recall.get("3", 0.0)),
        "r_at_5_non_regression": float(trial_recall.get("5", 0.0)) >= float(base_recall.get("5", 0.0)),
        "mean_top_k_gain": float(trial.get("mean_top_k_score", 0.0)) >= float(base.get("mean_top_k_score", 0.0)) + 0.01,
        "mrr_non_regression": float(trial.get("mrr", 0.0)) >= float(base.get("mrr", 0.0)),
    }
    base_latency = base.get("latency_ms", {})
    trial_latency = trial.get("latency_ms", {})
    if isinstance(base_latency, Mapping) and isinstance(trial_latency, Mapping):
        baseline_p95 = float(base_latency.get("p95") or 0.0)
        candidate_p95 = float(trial_latency.get("p95") or 0.0)
        checks["p95_latency_within_1_2x"] = (
            baseline_p95 > 0
            and candidate_p95 > 0
            and candidate_p95 <= baseline_p95 * 1.20
        )
    else:
        checks["p95_latency_within_1_2x"] = False
    base_intents, trial_intents = base.get("by_intent_class", {}), trial.get("by_intent_class", {})
    intent_non_regression = True
    if isinstance(base_intents, Mapping) and isinstance(trial_intents, Mapping):
        for intent, base_metrics in base_intents.items():
            trial_metrics = trial_intents.get(intent)
            if not isinstance(base_metrics, Mapping) or not isinstance(trial_metrics, Mapping):
                intent_non_regression = False
                break
            for metric in ("mean_top_k_score", "mrr", "r_at_1", "r_at_3", "r_at_5"):
                if float(trial_metrics.get(metric, 0.0)) < float(base_metrics.get(metric, 0.0)):
                    intent_non_regression = False
                    break
    checks["intent_non_regression"] = intent_non_regression
    return {"passed": all(checks.values()), "checks": checks}


__all__ = [
    "IMPORTANT_INTENTS",
    "LockedQrelsSplit",
    "QRELS_SCHEMA_VERSION",
    "QRELS_TAGS",
    "QrelsManifest",
    "QrelsPack",
    "advanced_activation_gate",
    "build_locked_qrels_split",
    "load_qrels_pack",
    "qrels_coverage_report",
    "validate_production_qrels",
]
