from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.contracts import QueryRequest, TRAKEQueryRequest
from src.query.entity_extractor import EntityConstraint, EntityExtractor
from src.query.query_decomposer import mask_entity_mentions, restore_entity_mentions
from src.query.translator import (
    OfflineVietnameseTranslator,
    QueryTextAdapter,
    TranslationResult,
)
from src.retrieval.textual_kis import LexicalChannel, TextualKISPipeline
from src.retrieval.trake_alignment import TRAKEPipeline

FANA_QUERY = (
    "Đoạn video về một chương trình từ thiện của một câu lạc bộ tên là FANA "
    "tại một xã thuộc tỉnh Khánh Hòa."
)
TEMPLE_QUERY = (
    "Trong đoạn video có 2 câu thơ của một nhà thơ ca ngợi anh hùng "
    "Nguyễn Trung Trực trong đình thần Nguyễn Trung Trực tại Kiên Giang."
)


def test_entity_extractor_finds_expected_phrases_and_legacy_locations() -> None:
    extractor = EntityExtractor()

    assert extractor.legacy_location_count == 63
    assert [value.text for value in extractor.extract(FANA_QUERY)] == ["FANA", "Khánh Hòa"]
    assert [value.text for value in extractor.extract(TEMPLE_QUERY)] == [
        "Nguyễn Trung Trực",
        "Kiên Giang",
    ]
    assert [value.text for value in extractor.extract("chuong trinh tai Khanh Hoa")] == [
        "Khánh Hòa"
    ]
    assert extractor.extract("Đoạn video mô tả một chiếc bàn màu đỏ") == ()


def test_entity_masking_is_exact_and_fail_closed() -> None:
    entities = EntityExtractor().extract(FANA_QUERY)
    masked, mentions = mask_entity_mentions(FANA_QUERY, entities)

    assert "FANA" not in masked and "Khánh Hòa" not in masked
    assert restore_entity_mentions(masked, mentions) == FANA_QUERY
    try:
        restore_entity_mentions("placeholder removed", mentions)
    except ValueError as error:
        assert str(error) == "entity_placeholder_invalid"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("missing entity placeholder must fail closed")


def _translator(tmp_path: Path) -> OfflineVietnameseTranslator:
    translator = OfflineVietnameseTranslator(
        model_name="fixture",
        revision="fixture",
        model_path=tmp_path / "model",
        tokenizer_path=tmp_path / "tokenizer",
        cache=None,
    )
    translator._translate_supervised = lambda value: (  # type: ignore[method-assign]
        value.replace("Đoạn video", "a video").replace("tại", "in"),
        "ct2",
    )
    return translator


def test_offline_translation_protects_entities_and_invalid_placeholder_isolated(
    tmp_path: Path,
) -> None:
    entities = EntityExtractor().extract(FANA_QUERY)
    translator = _translator(tmp_path)
    result = translator.translate(FANA_QUERY, language_hint="vi", protected_entities=entities)

    assert result.failure_code is None
    assert result.english_text is not None
    assert "FANA" in result.english_text and "Khánh Hòa" in result.english_text

    broken = _translator(tmp_path)
    broken._translate_supervised = lambda _value: ("translation lost markers", "ct2")  # type: ignore[method-assign]
    failed = broken.translate(FANA_QUERY, language_hint="vi", protected_entities=entities)
    assert failed.english_text is None
    assert failed.failure_code == "entity_placeholder_invalid"


class _TranslationFixture:
    fingerprint = "entity-translation-fixture-v1"

    def translate(
        self,
        text: str,
        *,
        language_hint: str = "auto",
        english_override: str | None = None,
        protected_entities: Any = (),
    ) -> TranslationResult:
        del protected_entities
        english = english_override or (
            "a charity club distributing gifts in a rural village in FANA Khánh Hòa"
        )
        return TranslationResult(
            original_text=text,
            normalized_text=" ".join(text.split()),
            accent_folded_text=" ".join(text.split()).casefold(),
            detected_language="vi" if language_hint != "en" else "en",
            english_text=english,
            source="ct2",
            cache_status="miss",
            latency_ms=1.0,
        )


class _EntityStore:
    def is_available(self, _filters: str | None = None) -> bool:
        return True

    @staticmethod
    def _rows() -> list[dict[str, Any]]:
        return [
            {
                "score": 0.99,
                "payload": {
                    "batch_id": "batch",
                    "video_id": "generic",
                    "source_frame_idx": 1,
                    "timestamp_ms": 100,
                    "text_raw": "một chương trình cộng đồng",
                },
            },
            {
                "score": 0.5,
                "payload": {
                    "batch_id": "batch",
                    "video_id": "entity",
                    "source_frame_idx": 2,
                    "timestamp_ms": 200,
                    "text_raw": "CLB FANA trao quà tại tỉnh Khánh Hòa",
                },
            },
        ]

    def search_many(self, queries: Any, **_kwargs: Any) -> list[list[dict[str, Any]]]:
        return [self._rows() for _query in queries]


def _entity_pipeline() -> TextualKISPipeline:
    return TextualKISPipeline(
        lexical_channels=(
            LexicalChannel("ocr", _EntityStore(), 0.8, 20),
            LexicalChannel("metadata", _EntityStore(), 0.4, 20),
        ),
        query_adapter=QueryTextAdapter(_TranslationFixture()),
        temporal_windows_ms=(),
        min_separation_ms=0,
        top_k=20,
    )


def test_entity_boost_is_internal_and_required_entities_filter_at_video_scope() -> None:
    optional = _entity_pipeline()
    optional_hits = optional.search(
        QueryRequest("q", FANA_QUERY, batch_filters=("batch",), refine_raw=False),
        refine_raw=False,
    )
    assert optional_hits[0]["video_id"] == "entity"
    assert optional_hits[0]["entity_bonus"] > 0
    assert set(optional_hits[0]["channel_ranks"]) == {"ocr"}
    assert optional_hits[0]["metadata_video_boost"] == 1.15

    required = _entity_pipeline()
    required_hits = required.search(
        QueryRequest(
            "q",
            FANA_QUERY,
            batch_filters=("batch",),
            entity_constraints=(EntityConstraint("FANA", "required"),),
            refine_raw=False,
        ),
        refine_raw=False,
    )
    assert {value["video_id"] for value in required_hits} == {"entity"}
    assert required.last_query_processing["required_entity_status"] == "required_entity_match"


class _BatchEncoder:
    def encode_text(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, float(index + 1)] for index, _text in enumerate(texts)], dtype=np.float32)


def test_trake_entity_constraints_preserve_n_plus_one_logical_prompts() -> None:
    pipeline = TRAKEPipeline(
        encoder=_BatchEncoder(),
        vector_store=None,
        channel_name="organizer_clip",
        query_adapter=QueryTextAdapter(_TranslationFixture()),
    )
    request = TRAKEQueryRequest.model_validate(
        {
            "query_id": "q",
            "global_description": FANA_QUERY,
            "global_entity_constraints": [{"text": "FANA", "mode": "required"}],
            "stages": [
                {"stage_index": 1, "description": "FANA trao quà"},
                {
                    "stage_index": 2,
                    "description": "sự kiện tại Kiên Giang",
                    "entity_constraints": [{"text": "Kiên Giang", "mode": "required"}],
                },
            ],
        }
    )

    query_sets, processings = pipeline._query_embeddings(pipeline.encoder, request)  # type: ignore[arg-type]
    assert len(query_sets) == len(processings) == 3
    assert [value.prompt_index for value in processings] == [0, 1, 2]
    assert pipeline.last_query_processing["required_entity_count"] == 2
