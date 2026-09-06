"""Tune Textual KIS RRF parameters from offline cached channel rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.qrels import (
    build_locked_qrels_split,
    load_qrels_pack,
    validate_production_qrels,
)
from src.evaluation.textual_kis import GroundTruthQuery, load_ground_truth
from src.evaluation.tuning import (
    load_cached_rankings,
    persist_tuning_report,
    tune_cached_rankings,
    tune_intent_policies,
    tune_semantic_view_policy,
)
from src.query.query_decomposer import QUERY_DECOMPOSER_VERSION
from src.utils.io_helpers import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune weighted RRF and report cumulative channel ablations"
    )
    parser.add_argument("--rankings", type=Path, help="Versioned cached-ranking JSON")
    parser.add_argument("--ground-truth", type=Path, help="Optional labeled interval CSV/JSON")
    parser.add_argument(
        "--qrels-manifest",
        type=Path,
        help="Production qrels pack; locked test queries are always excluded from tuning",
    )
    parser.add_argument("--rrf-k", type=int, action="append", help="Repeat to replace k grid")
    parser.add_argument(
        "--depth",
        action="append",
        default=[],
        metavar="CHANNEL=VALUE[,VALUE]",
        help="Candidate-depth grid override; repeat per channel",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="CHANNEL=VALUE[,VALUE]",
        help="Weight grid override; repeat per channel",
    )
    parser.add_argument("--max-candidates", type=int, default=50_000)
    parser.add_argument(
        "--per-intent",
        action="store_true",
        help="Tune the visual, lexical, and spatial/color autonomous policy buckets",
    )
    parser.add_argument(
        "--semantic-views",
        action="store_true",
        help=(
            "Tune semantic-view late fusion from role-named cached rankings; "
            "requires a locked qrels manifest"
        ),
    )
    parser.add_argument(
        "--semantic-candidate-depth",
        type=int,
        help="Maximum depth to search; defaults to the complete shared cache depth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser


def _grid(values: list[str], *, numeric: type[int] | type[float]) -> dict[str, tuple]:
    parsed: dict[str, tuple] = {}
    for value in values:
        channel, separator, raw_values = value.partition("=")
        if not separator or not channel or not raw_values:
            raise ValueError(f"invalid grid override: {value!r}")
        if channel in parsed:
            raise ValueError(f"duplicate grid override for channel: {channel}")
        parsed[channel] = tuple(numeric(item) for item in raw_values.split(","))
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Select parameters or explicitly preserve defaults when labels are absent."""

    args = _parser().parse_args(argv)
    if args.ground_truth and args.qrels_manifest:
        raise ValueError("use either --ground-truth or --qrels-manifest, not both")
    if args.per_intent and args.semantic_views:
        raise ValueError("--per-intent and --semantic-views are mutually exclusive")
    if (args.per_intent or args.semantic_views) and not args.qrels_manifest:
        raise ValueError("policy tuning requires a locked --qrels-manifest")
    if (args.ground_truth or args.qrels_manifest) and not args.rankings:
        raise ValueError("--rankings is required with labeled ground truth")
    config = load_config()["retrieval"]
    default_depths = {str(key): int(value) for key, value in config["candidate_depths"].items()}
    default_weights = {
        str(key): float(value) for key, value in config["channel_weights"].items()
    }
    depth_values = _grid(args.depth, numeric=int)
    weight_values = _grid(args.weight, numeric=float)
    unknown = (set(depth_values) | set(weight_values)) - set(default_weights)
    if unknown:
        raise ValueError(f"grid overrides contain unknown channels: {sorted(unknown)}")
    cached = load_cached_rankings(args.rankings) if args.rankings else None
    qrels_provenance: dict[str, object] = {}
    ground_truth: tuple[GroundTruthQuery, ...] | None
    if args.qrels_manifest:
        pack = load_qrels_pack(args.qrels_manifest)
        validate_production_qrels(pack)
        seed = int(pack.manifest.split_metadata["seed"])
        locked = build_locked_qrels_split(pack, seed=seed)
        test_ids = set(locked.test_query_ids)
        ground_truth = tuple(
            query for query in pack.queries if query.query_id not in test_ids
        )
        training_query_ids = tuple(sorted(query.query_id for query in ground_truth))
        qrels_provenance = {
            "content_sha256": pack.manifest.content_sha256,
            "locked_test_sha256": locked.test_sha256,
            "training_query_ids_sha256": hashlib.sha256(
                json.dumps(training_query_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "training_query_count": len(training_query_ids),
            "test_query_count": len(locked.test_query_ids),
        }
    else:
        ground_truth = load_ground_truth(args.ground_truth) if args.ground_truth else None
    if args.semantic_views:
        if cached is None or not ground_truth or not args.rankings:
            raise ValueError("--semantic-views requires labeled cached rankings")
        pack = load_qrels_pack(args.qrels_manifest)
        seed = int(pack.manifest.split_metadata["seed"])
        locked = build_locked_qrels_split(pack, seed=seed)
        by_id = {query.query_id: query for query in pack.queries}
        training = tuple(by_id[query_id] for query_id in locked.train_query_ids)
        validation = tuple(by_id[query_id] for query_id in locked.validation_query_ids)
        role_order = (
            "global",
            "action",
            "scene_action",
            "foreground_subject",
            "distinctive_attributes",
            "instance_attributes",
            "relation_relaxed",
        )
        available_roles = set().union(
            *(set(query_rankings) for query_rankings in cached.values())
        )
        roles = tuple(role for role in role_order if role in available_roles)
        if "global" not in roles:
            raise ValueError("semantic ranking cache must contain a global role")
        shared_depth = min(
            len(cached[query_id][role].results)
            for query_id in (*locked.train_query_ids, *locked.validation_query_ids)
            for role in roles
            if role in cached[query_id] and cached[query_id][role].results
        )
        candidate_depth = args.semantic_candidate_depth or shared_depth
        if candidate_depth > shared_depth:
            raise ValueError(
                "semantic candidate depth exceeds the complete shared cache depth"
            )
        report = tune_semantic_view_policy(
            cached,
            training,
            validation,
            roles=roles,
            candidate_depth=candidate_depth,
            qrels_provenance=qrels_provenance,
            fingerprints={
                "model": pack.manifest.model_fingerprint,
                "ranking_cache": hashlib.sha256(args.rankings.read_bytes()).hexdigest(),
                "query_adapter": QUERY_DECOMPOSER_VERSION,
            },
            max_candidates=args.max_candidates,
        )
        configured_output = Path(config["semantic_views"]["tuning_record"])
    elif args.per_intent:
        if cached is None or not ground_truth:
            raise ValueError("--per-intent requires labeled cached rankings")
        dense_channels = tuple(
            channel
            for channel in default_weights
            if channel not in {"ocr", "asr", "metadata"}
        )
        report = tune_intent_policies(
            cached,
            ground_truth,
            dense_channels=dense_channels,
            default_rrf_k=int(config["rrf_k"]),
            default_depths=default_depths,
            default_weights=default_weights,
            rrf_k_values=args.rrf_k or config["tune_rrf_k"],
            depth_values=depth_values,
            weight_values=weight_values,
            max_candidates=args.max_candidates,
            qrels_provenance=qrels_provenance,
        )
        configured_output = Path(config["intent_tuning_record"])
    else:
        report = tune_cached_rankings(
            cached,
            ground_truth,
            default_rrf_k=int(config["rrf_k"]),
            default_depths=default_depths,
            default_weights=default_weights,
            rrf_k_values=args.rrf_k or config["tune_rrf_k"],
            depth_values=depth_values,
            weight_values=weight_values,
            max_candidates=args.max_candidates,
        )
        configured_output = Path("outputs/evaluation/retrieval-tuning.json")
    selected_output = args.output or configured_output
    output = selected_output if selected_output.is_absolute() else PROJECT_ROOT / selected_output
    persist_tuning_report(report, output)
    payload = report.to_dict()
    payload["output"] = str(output.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
