from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.contracts import QueryRequest, TRAKEQueryRequest, TRAKEStage
from src.query.translator import (
    QueryTextAdapter,
    TranslationCache,
    TranslationResult,
    accent_fold,
    detect_query_language,
)
from src.retrieval.textual_kis import DenseChannel, LexicalChannel, TextualKISPipeline
from src.retrieval.trake_alignment import TRAKEPipeline


class _Translator:
    fingerprint = "translator-test-v1"

    def __init__(self, english: str | None = "a man wearing glasses") -> None:
        self.english = english
        self.calls: list[tuple[str, str, str | None]] = []

    def translate(
        self,
        text: str,
        *,
        language_hint: str = "auto",
        english_override: str | None = None,
    ) -> TranslationResult:
        self.calls.append((text, language_hint, english_override))
        detected = detect_query_language(text, language_hint)  # type: ignore[arg-type]
        normalized = " ".join(text.split())
        english = english_override or (normalized if detected == "en" else self.english)
        return TranslationResult(
            original_text=text,
            normalized_text=normalized,
            accent_folded_text=accent_fold(text),
            detected_language=detected,
            english_text=english,
            source="manual" if english_override else "ct2" if english else "unavailable",
            cache_status="bypass" if english_override else "miss",
            latency_ms=1.25,
            failure_code=None if english else "translator_not_provisioned",
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("một người đang đọc bản tin", "vi"),
        ("nguoi dan ong deo kinh", "vi"),
        ("người wearing glasses", "vi"),
        ("a man wearing glasses", "en"),
        ("雨の中を歩く人", "unknown"),
    ),
)
def test_language_detection_is_deterministic(text: str, expected: str) -> None:
    assert detect_query_language(text) == expected


def test_channel_routing_preserves_original_and_never_sends_raw_vi_to_openclip() -> None:
    adapter = QueryTextAdapter(_Translator())
    processing = adapter.prepare("  người đàn ông đeo kính  ")

    assert [item.text for item in adapter.dense_variants("organizer_clip", processing)] == [
        "a man wearing glasses"
    ]
    assert [item.text for item in adapter.dense_variants("accelerated_visual", processing)] == [
        "a man wearing glasses"
    ]
    assert [item.text for item in adapter.dense_variants("visual", processing)] == [
        "người đàn ông đeo kính",
        "a man wearing glasses",
    ]
    assert [item.text for item in adapter.dense_variants("semantic_text", processing)] == [
        "người đàn ông đeo kính"
    ]
    assert [item.text for item in adapter.lexical_variants("ocr", processing)] == [
        "người đàn ông đeo kính"
    ]
    assert [item.text for item in adapter.lexical_variants("asr", processing)] == [
        "người đàn ông đeo kính"
    ]
    assert [item.text for item in adapter.lexical_variants("metadata", processing)] == [
        "người đàn ông đeo kính",
        "a man wearing glasses",
    ]


def test_translation_failure_isolated_to_english_only_channels() -> None:
    adapter = QueryTextAdapter(_Translator(None))
    processing = adapter.prepare("người đàn ông đeo kính")

    assert adapter.dense_variants("organizer_clip", processing) == ()
    assert len(adapter.dense_variants("visual", processing)) == 1
    assert len(adapter.dense_variants("semantic_text", processing)) == 1
    assert len(adapter.lexical_variants("ocr", processing)) == 1


def test_manual_override_and_english_identity_preserve_exact_operator_text() -> None:
    translator = _Translator()
    adapter = QueryTextAdapter(translator)
    manual = adapter.prepare(
        "người đàn ông",
        language_hint="vi",
        english_override="the male presenter",
    )
    assert adapter.dense_variants("organizer_clip", manual)[0].text == "the male presenter"

    english = adapter.prepare("  a man   wearing glasses  ")
    assert [value.text for value in adapter.dense_variants("organizer_clip", english)] == [
        "a man wearing glasses"
    ]
    assert [value.text for value in adapter.dense_variants("visual", english)] == [
        "a man wearing glasses"
    ]


def test_translation_cache_is_revision_namespaced_and_bounded(tmp_path: Path) -> None:
    cache = TranslationCache(tmp_path / "query_translation.sqlite3", max_entries=2)
    cache.put("a", "one", "fp-1")
    cache.put("b", "two", "fp-1")
    assert cache.get("a", "fp-1") == "one"
    assert cache.get("a", "fp-2") is None
    cache.put("c", "three", "fp-1")
    assert cache.get("b", "fp-1") is None
    assert cache.get("a", "fp-1") == "one"
    assert cache.get("c", "fp-1") == "three"


def test_adapter_fingerprint_changes_with_translation_model() -> None:
    first = QueryTextAdapter(_Translator())
    second_translator = _Translator()
    second_translator.fingerprint = "translator-test-v2"
    second = QueryTextAdapter(second_translator)
    assert first.fingerprint != second.fingerprint


class _Encoder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode_text(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return np.asarray([1.0, 0.0], dtype=np.float32)


class _DenseStore:
    def search(self, _vector: np.ndarray, **_kwargs: object) -> list[dict[str, object]]:
        return [
            {
                "score": 0.9,
                "payload": {
                    "batch_id": "batch",
                    "video_id": "video",
                    "source_frame_idx": 10,
                    "timestamp_ms": 100,
                },
            }
        ]


class _LexicalStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...] | None]] = []

    def search(
        self,
        text: str,
        *,
        locales: tuple[str, ...] | None = None,
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        self.calls.append((text, locales))
        return [
            {
                "score": 1.0,
                "payload": {
                    "batch_id": "batch",
                    "video_id": "video",
                    "source_frame_idx": 10,
                    "timestamp_ms": 100,
                },
            }
        ]


def test_kis_fuses_bilingual_variants_inside_each_logical_channel() -> None:
    visual, organizer, semantic = _Encoder(), _Encoder(), _Encoder()
    ocr, metadata = _LexicalStore(), _LexicalStore()
    pipeline = TextualKISPipeline(
        dense_channels=(
            DenseChannel("visual", visual, _DenseStore(), 1.0, 10),
            DenseChannel("organizer_clip", organizer, _DenseStore(), 0.8, 10),
            DenseChannel("semantic_text", semantic, _DenseStore(), 0.7, 10),
        ),
        lexical_channels=(
            LexicalChannel("ocr", ocr, 0.8, 10),
            LexicalChannel("metadata", metadata, 0.4, 10),
        ),
        query_adapter=QueryTextAdapter(_Translator("a man reading news")),
        temporal_windows_ms=(),
    )
    request = QueryRequest(
        query_id="vi-routing",
        original_text="người đàn ông đọc bản tin",
        batch_filters=("batch",),
    )
    plan = pipeline._plan(request)
    processing = pipeline._prepare_query(request, plan)

    rankings = pipeline._collect_rankings(plan, request, processing=processing)

    assert [ranking.name for ranking in rankings] == [
        "visual",
        "organizer_clip",
        "semantic_text",
        "ocr",
    ]
    assert visual.calls == ["người đàn ông đọc bản tin", "a man reading news"]
    assert organizer.calls == ["a man reading news"]
    assert semantic.calls == ["người đàn ông đọc bản tin"]
    assert ocr.calls == [("người đàn ông đọc bản tin", ("vie",))]
    assert metadata.calls == [
        ("người đàn ông đọc bản tin", ("vie",)),
        ("a man reading news", ("eng",)),
    ]
    assert pipeline.last_channel_statuses["metadata"]["result_count"] == 0
    assert pipeline.last_channel_statuses["metadata"]["metadata_video_count"] == 1


class _BatchEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode_text(self, texts: str | list[str]) -> np.ndarray:
        values = [texts] if isinstance(texts, str) else list(texts)
        self.calls.append(values)
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(values), 1))


class _VectorShape:
    vector_size = 2


def test_trake_keeps_n_plus_one_prompts_while_scoring_vi_and_en_variants() -> None:
    translator = _Translator("translated")
    encoder = _BatchEncoder()
    request = TRAKEQueryRequest(
        query_id="trake-vi",
        global_description="người đi vào cửa hàng",
        stages=(
            TRAKEStage(stage_index=1, description="người lấy sản phẩm"),
            TRAKEStage(stage_index=2, description="người thanh toán"),
        ),
        batch_filters=("batch",),
        query_language="vi",
    )
    pipeline = TRAKEPipeline(
        encoder=encoder,
        vector_store=_VectorShape(),
        channel_name="visual",
        query_adapter=QueryTextAdapter(translator),
    )

    query_sets, processings = pipeline._query_embeddings(encoder, request)

    assert len(query_sets) == len(processings) == 3
    assert all(value.shape == (2, 2) for value in query_sets)
    assert encoder.calls == [
        [
            "người đi vào cửa hàng",
            "translated",
            "người lấy sản phẩm",
            "translated",
            "người thanh toán",
            "translated",
        ]
    ]


def test_trake_openclip_uses_only_one_english_variant_per_logical_prompt() -> None:
    encoder = _BatchEncoder()
    request = TRAKEQueryRequest(
        query_id="trake-openclip-vi",
        global_description="người đi vào cửa hàng",
        global_description_en="a person enters a store",
        stages=(
            TRAKEStage(
                stage_index=1,
                description="người lấy sản phẩm",
                description_en="a person takes a product",
            ),
            TRAKEStage(
                stage_index=2,
                description="người thanh toán",
                description_en="the person pays",
            ),
        ),
        batch_filters=("batch",),
        query_language="vi",
    )
    pipeline = TRAKEPipeline(
        encoder=encoder,
        vector_store=_VectorShape(),
        channel_name="organizer_clip",
        query_adapter=QueryTextAdapter(_Translator()),
    )

    query_sets, _ = pipeline._query_embeddings(encoder, request)

    assert len(query_sets) == 3
    assert all(value.shape == (1, 2) for value in query_sets)
    assert encoder.calls == [
        ["a person enters a store", "a person takes a product", "the person pays"]
    ]
