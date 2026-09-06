"""Offline regressions for Marian source EOS, chunk boundaries and cache isolation."""
import hashlib
import json
import sqlite3

import pytest

from src.query.translator import (
    ENTITY_EXTRACTOR_VERSION,
    OfflineVietnameseTranslator,
    TranslationCache,
    _source_token_batches,
    split_translation_text,
)


class Tokenizer:
    def num_special_tokens_to_add(self, pair=False):
        assert pair is False
        return 1

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()

    def build_inputs_with_special_tokens(self, ids):
        return [*ids, "</s>"]

    def convert_ids_to_tokens(self, ids):
        return ids


@pytest.mark.parametrize("length", [1, 3, 4, 6, 7])
def test_every_token_chunk_has_eos_and_retains_all_content(length):
    words = [f"token{i}" for i in range(length)]
    batches, groups = _source_token_batches(Tokenizer(), [" ".join(words), "last"], 4)
    assert all(len(batch) <= 4 and batch[-1] == "</s>" for batch in batches)
    assert [t for batch in batches for t in batch[:-1]] == [*words, "last"]
    assert groups == [0] * ((length + 2) // 3) + [1]


def test_empty_input_and_no_content_budget_rejected():
    with pytest.raises(ValueError):
        _source_token_batches(Tokenizer(), [""], 4)
    with pytest.raises(ValueError):
        _source_token_batches(Tokenizer(), ["hello"], 1)


def test_translation_keeps_sentence_boundaries_without_removing_repeated_source_words():
    text = "Một cô bé đeo bạch tuộc đỏ. Trên tay có túi giấy. Rất rất đẹp!"
    chunks = split_translation_text(text, pack_sentences=False)
    assert len(chunks) == 3
    assert " ".join(chunks) == text
    assert len(split_translation_text(text)) == 1  # public helper stays compatible


def translator(tmp_path, cache=None):
    return OfflineVietnameseTranslator(model_name="fixture", revision="rev",
        model_path=tmp_path / "model", tokenizer_path=tmp_path / "tokenizer", cache=cache)


def test_supervised_worker_receives_separate_sentences(tmp_path):
    instance = translator(tmp_path)
    received = []

    def worker(command, chunks, *, timeout):
        received.append((command, chunks, timeout))
        return {"text": "A girl. She holds a paper bag.", "source": "ct2"}

    instance._worker_request = worker
    result = instance.translate("Một cô bé. Cô cầm túi giấy.")
    assert result.failure_code is None
    assert received == [("translate", ("Một cô bé.", "Cô cầm túi giấy."), 3.0)]


def test_v3_cache_is_not_reused_or_deleted_and_v4_memory_disk_hits_work(tmp_path):
    cache = TranslationCache(tmp_path / "cache.sqlite3")
    instance = translator(tmp_path, cache)
    old_contract = {"version": "query-translation-v3", "model": "fixture", "revision": "rev",
        "compute_type": "int8", "beam_size": 1, "max_decoding_length": 128,
        "sentence_max_chars": 400, "source_token_limit": 448,
        "entity_mask_protocol": ENTITY_EXTRACTOR_VERSION}
    old_fingerprint = hashlib.sha256(json.dumps(old_contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    text = "Một cô bé cầm túi giấy."
    old_key = hashlib.sha256(f"vi\0en\0{old_fingerprint}\0{text}\0".encode()).hexdigest()
    cache.put(old_key, "paper bag " * 20, old_fingerprint)
    calls = []

    def translate(value):
        calls.append(value)
        return "A girl holding a paper bag.", "ct2"

    instance._translate_supervised = translate
    assert instance.fingerprint != old_fingerprint
    assert instance.translate(text).source == "ct2"
    assert instance.translate(text).source == "memory_cache"
    second = translator(tmp_path, cache)
    second._translate_supervised = lambda _: pytest.fail("disk hit should not infer")
    assert second.translate(text).source == "disk_cache"
    assert len(calls) == 1
    with sqlite3.connect(cache.path) as db:
        assert db.execute("SELECT COUNT(*) FROM query_translations").fetchone()[0] == 2
        assert db.execute("SELECT english_text FROM query_translations WHERE cache_key=?", (old_key,)).fetchone()[0] == "paper bag " * 20


def test_manual_english_is_not_rewritten_or_deduplicated(tmp_path):
    instance = translator(tmp_path)
    text = "Go go go!"
    assert instance.translate("một người", english_override=text).english_text == text
    assert instance.translate(text, language_hint="en").english_text == text
