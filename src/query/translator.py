"""Offline Vietnamese-to-English query translation and encoder-aware routing.

Construction and import are deliberately side-effect free. SQLite initialization,
model loading, and inference happen only through explicit method calls.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from src.assets import verify_asset_manifest

LanguageHint = Literal["auto", "vi", "en"]

_VIETNAMESE_MARKED = frozenset(
    "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị"
    "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)
_VIETNAMESE_WORDS = frozenset(
    {
        "một", "người", "đang", "được", "những", "các", "trong", "trên",
        "dưới", "với", "của", "không", "chiếc", "cảnh", "trước", "sau",
        "khi", "vào", "đi", "chạy", "cầm", "mặc", "nguoi", "dang", "duoc",
        "nhung", "cac", "trong", "tren", "duoi", "voi", "cua", "khong",
        "chiec", "canh", "truoc", "sau", "khi", "vao", "chay", "cam", "mac",
        "dan", "ong", "ba", "con", "cho", "meo", "xe", "duong", "thanh",
        "pho", "mau", "do", "xanh", "kinh", "mat", "ban", "tin",
    }
)
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_query_text(text: str) -> str:
    """Return a stable NFC query form while preserving its content."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("query text must be a non-empty string")
    return " ".join(unicodedata.normalize("NFC", text).split())


def accent_fold(text: str) -> str:
    """Return an accent-free Vietnamese auxiliary form, including đ→d."""

    value = unicodedata.normalize("NFD", normalize_query_text(text))
    value = "".join(character for character in value if unicodedata.category(character) != "Mn")
    return value.replace("đ", "d").replace("Đ", "D").casefold()


def detect_query_language(text: str, hint: LanguageHint = "auto") -> str:
    """Detect Vietnamese/English deterministically; return ``unknown`` otherwise."""

    if hint not in {"auto", "vi", "en"}:
        raise ValueError("query_language must be auto, vi, or en")
    if hint != "auto":
        return hint
    normalized = normalize_query_text(text)
    lowered = normalized.casefold()
    if any(character in _VIETNAMESE_MARKED for character in lowered):
        return "vi"
    tokens = tuple(token.casefold() for token in _TOKEN.findall(lowered))
    if sum(token in _VIETNAMESE_WORDS for token in tokens) >= 2:
        return "vi"
    if all(ord(character) < 128 for character in normalized):
        return "en"
    return "unknown"


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """One immutable and sanitized query-translation outcome."""

    original_text: str
    normalized_text: str
    accent_folded_text: str
    detected_language: str
    english_text: str | None
    source: str
    cache_status: str
    latency_ms: float
    failure_code: str | None = None

    @property
    def translation_applied(self) -> bool:
        return bool(
            self.detected_language == "vi"
            and self.english_text
            and self.english_text != self.normalized_text
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "detected_language": self.detected_language,
            "english_text": self.english_text,
            "translation_source": self.source,
            "translation_cache": self.cache_status,
            "translation_latency_ms": round(max(0.0, self.latency_ms), 3),
            "translation_applied": self.translation_applied,
            "failure_code": self.failure_code,
        }


class TranslationCache:
    """Bounded SQLite translation cache with explicit initialization."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path, *, max_entries: int = 10_000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.path = Path(path)
        self.max_entries = int(max_entries)
        self._lock = threading.RLock()
        self._initialized = False

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.path) as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA foreign_keys=ON")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS query_translations (
                        cache_key TEXT PRIMARY KEY,
                        english_text TEXT NOT NULL,
                        model_fingerprint TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        last_accessed_at REAL NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_query_translations_access "
                    "ON query_translations(last_accessed_at)"
                )
                connection.commit()
            self._initialized = True

    def get(self, cache_key: str, model_fingerprint: str) -> str | None:
        self.initialize()
        now = time.time()
        with self._lock, sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT english_text FROM query_translations "
                "WHERE cache_key = ? AND model_fingerprint = ?",
                (cache_key, model_fingerprint),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE query_translations SET last_accessed_at = ? WHERE cache_key = ?",
                (now, cache_key),
            )
            connection.commit()
            return str(row[0])

    def put(self, cache_key: str, english_text: str, model_fingerprint: str) -> None:
        self.initialize()
        now = time.time()
        with self._lock, sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO query_translations(
                    cache_key, english_text, model_fingerprint, created_at, last_accessed_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    english_text = excluded.english_text,
                    model_fingerprint = excluded.model_fingerprint,
                    last_accessed_at = excluded.last_accessed_at
                """,
                (cache_key, english_text, model_fingerprint, now, now),
            )
            connection.execute(
                """
                DELETE FROM query_translations WHERE cache_key IN (
                    SELECT cache_key FROM query_translations
                    ORDER BY last_accessed_at DESC LIMIT -1 OFFSET ?
                )
                """,
                (self.max_entries,),
            )
            connection.commit()


class OfflineVietnameseTranslator:
    """Lazy Vietnamese→English translator with memory and SQLite caching."""

    def __init__(
        self,
        *,
        model_name: str,
        revision: str,
        model_path: str | Path,
        tokenizer_path: str | Path,
        cache: TranslationCache | None = None,
        source_model_path: str | Path | None = None,
        memory_entries: int = 2_048,
        compute_type: str = "int8",
        intra_threads: int = 4,
        max_decoding_length: int = 128,
        allow_transformers_fallback: bool = True,
        monotonic: Callable[[], float] = time.monotonic,
        asset_manifest: str | Path | None = None,
        asset_root: str | Path | None = None,
    ) -> None:
        if memory_entries < 1:
            raise ValueError("memory_entries must be positive")
        self.model_name = str(model_name)
        self.revision = str(revision)
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.source_model_path = Path(source_model_path) if source_model_path else None
        self.cache = cache
        self.memory_entries = int(memory_entries)
        self.compute_type = str(compute_type)
        self.intra_threads = int(intra_threads)
        if not 16 <= int(max_decoding_length) <= 512:
            raise ValueError("max_decoding_length must be between 16 and 512")
        self.max_decoding_length = int(max_decoding_length)
        self.allow_transformers_fallback = bool(allow_transformers_fallback)
        self._monotonic = monotonic
        self.asset_manifest = Path(asset_manifest) if asset_manifest is not None else None
        self.asset_root = Path(asset_root) if asset_root is not None else Path.cwd()
        self._assets_verified = False
        self._memory: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.RLock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._translator: Any = None
        self._tokenizer: Any = None
        self._transformers_model: Any = None

    @property
    def fingerprint(self) -> str:
        value = json.dumps(
            {
                "version": "query-translation-v1",
                "model": self.model_name,
                "revision": self.revision,
                "compute_type": self.compute_type,
                "beam_size": 1,
                "max_decoding_length": self.max_decoding_length,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _cache_key(self, text: str) -> str:
        payload = f"vi\0en\0{self.fingerprint}\0{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _memory_get(self, key: str) -> str | None:
        with self._lock:
            value = self._memory.get(key)
            if value is not None:
                self._memory.move_to_end(key)
            return value

    def _memory_put(self, key: str, value: str) -> None:
        with self._lock:
            self._memory[key] = value
            self._memory.move_to_end(key)
            while len(self._memory) > self.memory_entries:
                self._memory.popitem(last=False)

    def _key_lock(self, key: str) -> threading.Lock:
        with self._lock:
            return self._key_locks.setdefault(key, threading.Lock())

    def _load_ct2(self) -> None:
        if self._translator is not None and self._tokenizer is not None:
            return
        if not self.model_path.is_dir() or not self.tokenizer_path.is_dir():
            raise FileNotFoundError("offline translation model is not provisioned")
        if self.asset_manifest is not None and not self._assets_verified:
            checked, errors = verify_asset_manifest(
                self.asset_manifest,
                root=self.asset_root,
                required_paths=(self.model_path, self.tokenizer_path),
            )
            if checked < 1 or errors:
                raise ValueError("offline translation model fingerprint is invalid")
            self._assets_verified = True
        import ctranslate2
        from transformers import AutoTokenizer

        if not ctranslate2.contains_model(str(self.model_path)):
            raise ValueError("offline translation model is invalid")
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_path), local_files_only=True
        )
        self._translator = ctranslate2.Translator(
            str(self.model_path),
            device="cpu",
            compute_type=self.compute_type,
            inter_threads=1,
            intra_threads=self.intra_threads,
            max_queued_batches=1,
        )

    def _translate_ct2(self, text: str) -> str:
        with self._lock:
            self._load_ct2()
            ids = self._tokenizer.encode(text)
            tokens = self._tokenizer.convert_ids_to_tokens(ids)
            result = self._translator.translate_batch(
                [tokens],
                beam_size=1,
                max_decoding_length=self.max_decoding_length,
                return_scores=False,
            )[0]
            target = result.hypotheses[0]
            output = self._tokenizer.decode(
                self._tokenizer.convert_tokens_to_ids(target),
                skip_special_tokens=True,
            )
        return normalize_query_text(output)

    def warm(self) -> None:
        """Load and validate the CPU INT8 translator without issuing a query."""

        with self._lock:
            self._load_ct2()

    def _translate_transformers(self, text: str) -> str:
        if not self.allow_transformers_fallback or self.source_model_path is None:
            raise FileNotFoundError("transformers translation fallback is unavailable")
        if not self.source_model_path.is_dir():
            raise FileNotFoundError("translation source checkpoint is not provisioned")
        with self._lock:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(self.source_model_path), local_files_only=True
                )
            if self._transformers_model is None:
                from transformers import AutoModelForSeq2SeqLM

                self._transformers_model = AutoModelForSeq2SeqLM.from_pretrained(
                    str(self.source_model_path), local_files_only=True
                ).eval()
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True)
            output_ids = self._transformers_model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_decoding_length,
            )
            output = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return normalize_query_text(output)

    def translate(
        self,
        text: str,
        *,
        language_hint: LanguageHint = "auto",
        english_override: str | None = None,
    ) -> TranslationResult:
        started = self._monotonic()
        normalized = normalize_query_text(text)
        detected = detect_query_language(normalized, language_hint)
        folded = accent_fold(normalized)
        if english_override is not None:
            english = normalize_query_text(english_override)
            return TranslationResult(
                text, normalized, folded, detected, english, "manual", "bypass",
                (self._monotonic() - started) * 1000,
            )
        if detected == "en":
            return TranslationResult(
                text, normalized, folded, detected, normalized, "identity", "bypass",
                (self._monotonic() - started) * 1000,
            )
        if detected != "vi":
            return TranslationResult(
                text, normalized, folded, detected, None, "unavailable", "bypass",
                (self._monotonic() - started) * 1000, "unsupported_language",
            )
        key = self._cache_key(normalized)
        memory = self._memory_get(key)
        if memory is not None:
            return TranslationResult(
                text, normalized, folded, detected, memory, "memory_cache", "hit",
                (self._monotonic() - started) * 1000,
            )
        with self._key_lock(key):
            memory = self._memory_get(key)
            if memory is not None:
                source, cache_status, english = "memory_cache", "hit", memory
            else:
                try:
                    disk = (
                        self.cache.get(key, self.fingerprint)
                        if self.cache is not None
                        else None
                    )
                except (OSError, sqlite3.DatabaseError):
                    disk = None
                if disk is not None:
                    source, cache_status, english = "disk_cache", "hit", disk
                else:
                    try:
                        english = self._translate_ct2(normalized)
                        source, cache_status = "ct2", "miss"
                    except (ImportError, FileNotFoundError, RuntimeError, ValueError):
                        try:
                            english = self._translate_transformers(normalized)
                            source, cache_status = "transformers", "miss"
                        except (ImportError, FileNotFoundError, RuntimeError, ValueError, OSError):
                            return TranslationResult(
                                text, normalized, folded, detected, None, "unavailable", "miss",
                                (self._monotonic() - started) * 1000,
                                "translator_not_provisioned",
                            )
                    if self.cache is not None:
                        try:
                            self.cache.put(key, english, self.fingerprint)
                        except (OSError, sqlite3.DatabaseError):
                            # Translation remains usable in memory even when a
                            # user-owned disk cache is unavailable or corrupt.
                            pass
                self._memory_put(key, english)
        return TranslationResult(
            text, normalized, folded, detected, english, source, cache_status,
            (self._monotonic() - started) * 1000,
        )


@dataclass(frozen=True, slots=True)
class QueryVariant:
    name: str
    text: str
    locale: str | None = None


@dataclass(frozen=True, slots=True)
class QueryProcessing:
    translation: TranslationResult
    policies: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        value = self.translation.to_dict()
        value["channel_policies"] = {
            name: list(variants) for name, variants in sorted(self.policies.items())
        }
        return value


class QueryTextAdapter:
    """Prepare one query once and expose variants appropriate to each channel."""

    def __init__(self, translator: OfflineVietnameseTranslator | None = None) -> None:
        self.translator = translator

    @property
    def fingerprint(self) -> str:
        source = self.translator.fingerprint if self.translator is not None else "identity"
        return hashlib.sha256(f"query-adapter-v1:{source}".encode()).hexdigest()

    def prepare(
        self,
        text: str,
        *,
        language_hint: LanguageHint = "auto",
        english_override: str | None = None,
    ) -> QueryProcessing:
        if self.translator is not None:
            result = self.translator.translate(
                text,
                language_hint=language_hint,
                english_override=english_override,
            )
        else:
            started = time.monotonic()
            normalized = normalize_query_text(text)
            detected = detect_query_language(normalized, language_hint)
            english = (
                normalize_query_text(english_override)
                if english_override is not None
                else normalized if detected == "en" else None
            )
            source = "manual" if english_override is not None else (
                "identity" if detected == "en" else "unavailable"
            )
            failure = None if english is not None else (
                "unsupported_language" if detected == "unknown" else "translator_not_provisioned"
            )
            result = TranslationResult(
                text, normalized, accent_fold(normalized), detected, english, source, "bypass",
                (time.monotonic() - started) * 1000, failure,
            )
        policies: dict[str, tuple[str, ...]] = {}
        for channel in (
            "visual", "organizer_clip", "accelerated_visual", "semantic_text",
            "ocr", "asr", "metadata",
        ):
            policies[channel] = tuple(value.name for value in self.variants(channel, result))
        return QueryProcessing(result, policies)

    @staticmethod
    def variants(channel: str, result: TranslationResult) -> tuple[QueryVariant, ...]:
        original = QueryVariant("original", result.normalized_text, "vie" if result.detected_language == "vi" else "eng")
        english = (
            QueryVariant("english", result.english_text, "eng")
            if result.english_text
            else None
        )
        if channel in {"organizer_clip", "accelerated_visual"}:
            return (english,) if english is not None else ()
        if channel == "visual" and result.detected_language == "vi":
            values = (original, english) if english is not None else (original,)
            return tuple(dict.fromkeys(value for value in values if value is not None))
        if channel == "metadata" and result.detected_language == "vi":
            values = (original, english) if english is not None else (original,)
            return tuple(dict.fromkeys(value for value in values if value is not None))
        return (original,)

    def dense_variants(self, channel: str, processing: QueryProcessing) -> tuple[QueryVariant, ...]:
        variants = self.variants(channel, processing.translation)
        # Preserve the exact operator text on every original dense path.  The
        # lexical path intentionally uses NFC/whitespace normalization, while
        # the English identity path must keep its historical embedding/rank.
        return tuple(
            QueryVariant(value.name, processing.translation.original_text, value.locale)
            if value.name == "original"
            else value
            for value in variants
        )

    def lexical_variants(self, channel: str, processing: QueryProcessing) -> tuple[QueryVariant, ...]:
        return self.variants(channel, processing.translation)
