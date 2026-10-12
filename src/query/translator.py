"""Offline Vietnamese-to-English query translation and encoder-aware routing.

Construction and import are deliberately side-effect free. SQLite initialization,
model loading, and inference happen only through explicit method calls.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import re
import sqlite3
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from src.assets import verify_asset_manifest
from src.query.entity_extractor import (
    ENTITY_EXTRACTOR_VERSION,
    EntityConstraint,
    EntityExtractor,
    NamedEntity,
    fold_entity_text,
    merge_entity_constraints,
    normalize_entity_constraints,
)
from src.query.query_decomposer import (
    QUERY_DECOMPOSER_VERSION,
    DecomposedQuery,
    SemanticView,
    decompose_query,
    mask_entity_mentions,
    primary_query_text,
    restore_entity_mentions,
)

LanguageHint = Literal["auto", "vi", "en"]

_VIETNAMESE_MARKED = frozenset(
    "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)
_VIETNAMESE_WORDS = frozenset(
    {
        "một",
        "người",
        "đang",
        "được",
        "những",
        "các",
        "trong",
        "trên",
        "dưới",
        "với",
        "của",
        "không",
        "chiếc",
        "cảnh",
        "trước",
        "sau",
        "khi",
        "vào",
        "đi",
        "chạy",
        "cầm",
        "mặc",
        "nguoi",
        "dang",
        "duoc",
        "nhung",
        "cac",
        "trong",
        "tren",
        "duoi",
        "voi",
        "cua",
        "khong",
        "chiec",
        "canh",
        "truoc",
        "sau",
        "khi",
        "vao",
        "chay",
        "cam",
        "mac",
        "dan",
        "ong",
        "ba",
        "con",
        "cho",
        "meo",
        "xe",
        "duong",
        "thanh",
        "pho",
        "mau",
        "do",
        "xanh",
        "kinh",
        "mat",
        "ban",
        "tin",
    }
)
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…;:])\s+|\n+")
_REPEATED_PUNCTUATION = re.compile(r"([!?.,;:…])\1{2,}")
_PRESERVED_CONTROLS = frozenset({"\n", "\r", "\t"})


def _strip_unsafe_controls(text: str) -> str:
    return "".join(
        character
        for character in text
        if character in _PRESERVED_CONTROLS
        or unicodedata.category(character) not in {"Cc", "Cf", "Cs"}
    )


def normalize_query_text(text: str) -> str:
    """Return a stable NFC query form while preserving its content."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("query text must be a non-empty string")
    normalized = unicodedata.normalize("NFC", _strip_unsafe_controls(text))
    normalized = _REPEATED_PUNCTUATION.sub(r"\1", normalized)
    normalized = " ".join(normalized.split())
    if not normalized:
        raise ValueError("query text contains no searchable content")
    return normalized


def split_translation_text(
    text: str, *, maximum_chars: int = 400, pack_sentences: bool = True
) -> tuple[str, ...]:
    """Split normalized prose into bounded sentence chunks without dropping content."""

    if maximum_chars < 32:
        raise ValueError("maximum_chars must be at least 32")
    normalized = normalize_query_text(text)
    sentences = [value.strip() for value in _SENTENCE_BREAK.split(normalized) if value.strip()]
    chunks: list[str] = []
    for sentence in sentences or [normalized]:
        remaining = sentence
        while len(remaining) > maximum_chars:
            boundary = remaining.rfind(" ", 0, maximum_chars + 1)
            if boundary < maximum_chars // 2:
                boundary = maximum_chars
            chunks.append(remaining[:boundary].strip())
            remaining = remaining[boundary:].strip()
        if remaining:
            if pack_sentences and chunks and len(chunks[-1]) + 1 + len(remaining) <= maximum_chars:
                chunks[-1] = f"{chunks[-1]} {remaining}"
            else:
                chunks.append(remaining)
    return tuple(chunks)


def _source_token_batches(
    tokenizer: Any, texts: Sequence[str], token_limit: int
) -> tuple[list[list[str]], list[int]]:
    """Frame every source chunk using model-required tokens, including Marian EOS.

    Splitting already-framed tokens would leave all but the final chunk without
    EOS. Reserve the special-token budget first, then frame each chunk separately.
    """
    content_limit = token_limit - tokenizer.num_special_tokens_to_add(pair=False)
    if content_limit < 1:
        raise ValueError("source token limit leaves no room for content")
    batches: list[list[str]] = []
    groups: list[int] = []
    for group, value in enumerate(texts):
        ids = tokenizer.encode(value, add_special_tokens=False)
        for start in range(0, len(ids), content_limit):
            framed = tokenizer.build_inputs_with_special_tokens(ids[start : start + content_limit])
            if len(framed) > token_limit:
                raise ValueError("framed source chunk exceeds token limit")
            batches.append(tokenizer.convert_ids_to_tokens(framed))
            groups.append(group)
    if not batches:
        raise ValueError("translation input produced no tokens")
    return batches, groups


def _translation_worker_main(connection: Any, config: dict[str, Any]) -> None:
    """Serve bounded translation requests inside a disposable native worker."""

    translator = None
    tokenizer = None
    transformers_model = None
    try:
        while True:
            command, payload = connection.recv()
            if command == "shutdown":
                break
            try:
                if tokenizer is None:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        config["tokenizer_path"], local_files_only=True
                    )
                if translator is None:
                    import ctranslate2

                    if not ctranslate2.contains_model(config["model_path"]):
                        raise ValueError("offline translation model is invalid")
                    translator = ctranslate2.Translator(
                        config["model_path"],
                        device="cpu",
                        compute_type=config["compute_type"],
                        inter_threads=1,
                        intra_threads=config["intra_threads"],
                        max_queued_batches=1,
                    )
                if command == "warm":
                    connection.send({"ok": True, "source": "ct2"})
                    continue
                texts = [str(value) for value in payload]
                token_batches, output_groups = _source_token_batches(
                    tokenizer, texts, int(config["source_token_limit"])
                )
                source = "ct2"
                try:
                    results = translator.translate_batch(
                        token_batches,
                        beam_size=1,
                        max_decoding_length=int(config["max_decoding_length"]),
                        return_scores=False,
                    )
                    decoded = [
                        tokenizer.decode(
                            tokenizer.convert_tokens_to_ids(result.hypotheses[0]),
                            skip_special_tokens=True,
                        )
                        for result in results
                    ]
                except Exception:
                    source_path = config.get("source_model_path")
                    if not config.get("allow_transformers_fallback") or not source_path:
                        raise
                    if transformers_model is None:
                        from transformers import AutoModelForSeq2SeqLM

                        transformers_model = AutoModelForSeq2SeqLM.from_pretrained(
                            source_path, local_files_only=True
                        ).eval()
                    raw_texts = [
                        tokenizer.decode(
                            tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True
                        )
                        for tokens in token_batches
                    ]
                    inputs = tokenizer(
                        raw_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=int(config["source_token_limit"]),
                    )
                    output_ids = transformers_model.generate(
                        **inputs,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=int(config["max_decoding_length"]),
                    )
                    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    source = "transformers"
                grouped: list[list[str]] = [[] for _ in texts]
                for group, value in zip(output_groups, decoded, strict=True):
                    if str(value).strip():
                        grouped[group].append(str(value).strip())
                output = " ".join(" ".join(values) for values in grouped if values).strip()
                if not output:
                    raise ValueError("translator produced an empty result")
                connection.send({"ok": True, "source": source, "text": output})
            except Exception as error:
                connection.send({"ok": False, "code": type(error).__name__})
    except (EOFError, OSError):
        pass
    finally:
        try:
            connection.close()
        except OSError:
            pass


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
    # Parenthetical aliases frequently use a second language.  Detect the
    # primary sentence instead of letting one Vietnamese cue route an English
    # visual query through translation.
    primary = primary_query_text(normalized) or normalized
    lowered = primary.casefold()
    if any(character in _VIETNAMESE_MARKED for character in lowered):
        return "vi"
    tokens = tuple(token.casefold() for token in _TOKEN.findall(lowered))
    if sum(token in _VIETNAMESE_WORDS for token in tokens) >= 2:
        return "vi"
    if all(ord(character) < 128 for character in primary):
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
            with sqlite3.connect(self.path, timeout=1.0) as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA foreign_keys=ON")
                connection.execute("PRAGMA busy_timeout=1000")
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
        with self._lock, sqlite3.connect(self.path, timeout=1.0) as connection:
            connection.execute("PRAGMA busy_timeout=1000")
            row = connection.execute(
                "SELECT english_text FROM query_translations "
                "WHERE cache_key = ? AND model_fingerprint = ?",
                (cache_key, model_fingerprint),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE query_translations SET last_accessed_at = ? WHERE cache_key = ?",
                (self._access_stamp(connection), cache_key),
            )
            connection.commit()
            return str(row[0])

    def put(self, cache_key: str, english_text: str, model_fingerprint: str) -> None:
        self.initialize()
        with self._lock, sqlite3.connect(self.path, timeout=1.0) as connection:
            connection.execute("PRAGMA busy_timeout=1000")
            now = self._access_stamp(connection)
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

    @staticmethod
    def _access_stamp(connection: sqlite3.Connection) -> float:
        """Return a strictly increasing access time so LRU order survives coarse clocks."""
        sql = "SELECT MAX(last_accessed_at) FROM query_translations"
        (latest,) = connection.execute(sql).fetchone()
        now = time.time()
        return now if latest is None or now > latest else latest + 1e-6


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
        request_timeout_seconds: float = 3.0,
        startup_timeout_seconds: float = 30.0,
        sentence_max_chars: int = 400,
        source_token_limit: int = 448,
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
        if request_timeout_seconds <= 0 or startup_timeout_seconds <= 0:
            raise ValueError("translation timeouts must be positive")
        if sentence_max_chars < 32 or not 32 <= source_token_limit <= 510:
            raise ValueError("invalid translation chunk limits")
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.sentence_max_chars = int(sentence_max_chars)
        self.source_token_limit = int(source_token_limit)
        self._assets_verified = False
        self._memory: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.RLock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._translator: Any = None
        self._tokenizer: Any = None
        self._transformers_model: Any = None
        self._worker_lock = threading.RLock()
        self._worker_process: Any = None
        self._worker_connection: Any = None

    @property
    def fingerprint(self) -> str:
        value = json.dumps(
            {
                "version": "query-translation-v4",
                "source_framing": "tokenizer-special-tokens-per-chunk-v1",
                "sentence_packing": False,
                "model": self.model_name,
                "revision": self.revision,
                "compute_type": self.compute_type,
                "beam_size": 1,
                "max_decoding_length": self.max_decoding_length,
                "sentence_max_chars": self.sentence_max_chars,
                "source_token_limit": self.source_token_limit,
                "entity_mask_protocol": ENTITY_EXTRACTOR_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _cache_key(self, text: str, protected_entities: Sequence[NamedEntity] = ()) -> str:
        identities = "\0".join(entity.identity for entity in protected_entities)
        payload = f"vi\0en\0{self.fingerprint}\0{text}\0{identities}".encode("utf-8")
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

    def _release_key_lock(self, key: str, lock: threading.Lock) -> None:
        with self._lock:
            if self._key_locks.get(key) is lock and not lock.locked():
                self._key_locks.pop(key, None)

    def _verify_worker_assets(self) -> None:
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

    def _worker_config(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "tokenizer_path": str(self.tokenizer_path),
            "source_model_path": str(self.source_model_path) if self.source_model_path else None,
            "compute_type": self.compute_type,
            "intra_threads": self.intra_threads,
            "max_decoding_length": self.max_decoding_length,
            "source_token_limit": self.source_token_limit,
            "allow_transformers_fallback": self.allow_transformers_fallback,
        }

    def _stop_worker_unlocked(self) -> None:
        connection, process = self._worker_connection, self._worker_process
        self._worker_connection = None
        self._worker_process = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1.0)

    def _start_worker_unlocked(self) -> bool:
        self._verify_worker_assets()
        if self._worker_process is not None and self._worker_process.is_alive():
            return False
        self._stop_worker_unlocked()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_translation_worker_main,
            args=(child, self._worker_config()),
            name="aic-query-translator",
            daemon=True,
        )
        process.start()
        child.close()
        self._worker_connection = parent
        self._worker_process = process
        return True

    def _worker_request(
        self, command: str, payload: tuple[str, ...], *, timeout: float
    ) -> dict[str, Any]:
        with self._worker_lock:
            cold_start = self._start_worker_unlocked()
            effective_timeout = float(timeout)
            if cold_start:
                effective_timeout += self.startup_timeout_seconds
            try:
                self._worker_connection.send((command, payload))
                if not self._worker_connection.poll(effective_timeout):
                    self._stop_worker_unlocked()
                    raise TimeoutError("offline translation timed out")
                result = self._worker_connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                self._stop_worker_unlocked()
                raise RuntimeError("offline translation worker failed") from None
            if not isinstance(result, dict) or not result.get("ok"):
                code = result.get("code") if isinstance(result, dict) else None
                raise RuntimeError(f"offline translation failed: {code or 'worker_error'}")
            return result

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
            chunks = split_translation_text(text, maximum_chars=self.sentence_max_chars, pack_sentences=False)
            batches, _ = _source_token_batches(self._tokenizer, chunks, self.source_token_limit)
            results = self._translator.translate_batch(
                batches,
                beam_size=1,
                max_decoding_length=self.max_decoding_length,
                return_scores=False,
            )
            output = " ".join(self._tokenizer.decode(
                self._tokenizer.convert_tokens_to_ids(result.hypotheses[0]),
                skip_special_tokens=True,
            ) for result in results)
        return normalize_query_text(output)

    def warm(self) -> None:
        """Load and validate the CPU INT8 translator without issuing a query."""

        self._worker_request("warm", (), timeout=self.startup_timeout_seconds)

    def close(self) -> None:
        """Release the disposable translator process without touching its caches."""

        with self._worker_lock:
            if self._worker_connection is not None:
                try:
                    self._worker_connection.send(("shutdown", ()))
                except (BrokenPipeError, OSError):
                    pass
            process = self._worker_process
            if process is not None:
                process.join(timeout=1.0)
            self._stop_worker_unlocked()

    def _translate_supervised(self, text: str) -> tuple[str, str]:
        chunks = split_translation_text(
            text, maximum_chars=self.sentence_max_chars, pack_sentences=False
        )
        result = self._worker_request("translate", chunks, timeout=self.request_timeout_seconds)
        return normalize_query_text(str(result["text"])), str(result.get("source", "ct2"))

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
            chunks = split_translation_text(text, maximum_chars=self.sentence_max_chars, pack_sentences=False)
            batches, _ = _source_token_batches(self._tokenizer, chunks, self.source_token_limit)
            raw_chunks = [self._tokenizer.decode(
                self._tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True
            ) for tokens in batches]
            inputs = self._tokenizer(
                raw_chunks, return_tensors="pt", padding=True, truncation=True,
                max_length=self.source_token_limit,
            )
            output_ids = self._transformers_model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_decoding_length,
            )
            output = " ".join(self._tokenizer.batch_decode(output_ids, skip_special_tokens=True))
        return normalize_query_text(output)

    def translate(
        self,
        text: str,
        *,
        language_hint: LanguageHint = "auto",
        english_override: str | None = None,
        protected_entities: Sequence[NamedEntity] = (),
    ) -> TranslationResult:
        started = self._monotonic()
        normalized = normalize_query_text(text)
        detected = detect_query_language(normalized, language_hint)
        folded = accent_fold(normalized)
        if english_override is not None:
            english = normalize_query_text(english_override)
            return TranslationResult(
                text,
                normalized,
                folded,
                detected,
                english,
                "manual",
                "bypass",
                (self._monotonic() - started) * 1000,
            )
        if detected == "en":
            return TranslationResult(
                text,
                normalized,
                folded,
                detected,
                normalized,
                "identity",
                "bypass",
                (self._monotonic() - started) * 1000,
            )
        if detected != "vi":
            return TranslationResult(
                text,
                normalized,
                folded,
                detected,
                None,
                "unavailable",
                "bypass",
                (self._monotonic() - started) * 1000,
                "unsupported_language",
            )
        masked_text, protected_mentions = mask_entity_mentions(normalized, protected_entities)
        key = self._cache_key(normalized, protected_entities)
        memory = self._memory_get(key)
        if memory is not None:
            return TranslationResult(
                text,
                normalized,
                folded,
                detected,
                memory,
                "memory_cache",
                "hit",
                (self._monotonic() - started) * 1000,
            )
        key_lock = self._key_lock(key)
        try:
            if not key_lock.acquire(
                timeout=self.request_timeout_seconds + self.startup_timeout_seconds
            ):
                return TranslationResult(
                    text,
                    normalized,
                    folded,
                    detected,
                    None,
                    "unavailable",
                    "miss",
                    (self._monotonic() - started) * 1000,
                    "translator_busy",
                )
            try:
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
                            english, source = self._translate_supervised(masked_text)
                            try:
                                english = restore_entity_mentions(english, protected_mentions)
                            except ValueError:
                                return TranslationResult(
                                    text,
                                    normalized,
                                    folded,
                                    detected,
                                    None,
                                    "unavailable",
                                    "miss",
                                    (self._monotonic() - started) * 1000,
                                    "entity_placeholder_invalid",
                                )
                            cache_status = "miss"
                        except TimeoutError:
                            return TranslationResult(
                                text,
                                normalized,
                                folded,
                                detected,
                                None,
                                "unavailable",
                                "miss",
                                (self._monotonic() - started) * 1000,
                                "translator_timeout",
                            )
                        except (ImportError, FileNotFoundError, RuntimeError, ValueError, OSError):
                            return TranslationResult(
                                text,
                                normalized,
                                folded,
                                detected,
                                None,
                                "unavailable",
                                "miss",
                                (self._monotonic() - started) * 1000,
                                "translator_not_provisioned",
                            )
                        if self.cache is not None:
                            try:
                                self.cache.put(key, english, self.fingerprint)
                            except (OSError, sqlite3.DatabaseError):
                                pass
                    self._memory_put(key, english)
            finally:
                key_lock.release()
        finally:
            self._release_key_lock(key, key_lock)
        return TranslationResult(
            text,
            normalized,
            folded,
            detected,
            english,
            source,
            cache_status,
            (self._monotonic() - started) * 1000,
        )


@dataclass(frozen=True, slots=True)
class QueryVariant:
    name: str
    text: str
    locale: str | None = None
    entity_identities: tuple[str, ...] = ()
    required_entity_identity: str | None = None
    semantic_role: str | None = None
    prior: float = 1.0


@dataclass(frozen=True, slots=True)
class QueryProcessing:
    translation: TranslationResult
    policies: dict[str, tuple[str, ...]]
    decomposition: DecomposedQuery | None = None
    prompt_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = self.translation.to_dict()
        value["channel_policies"] = {
            name: list(variants) for name, variants in sorted(self.policies.items())
        }
        if self.decomposition is not None:
            value["visual_description_en"] = self.decomposition.visual_description_en
            value["intent"] = self.decomposition.intent
            value["spatial_constraints"] = [
                {
                    "subject": constraint.subject,
                    "zone": constraint.zone,
                    "bound": constraint.bound,
                    "perspective_proxy": constraint.perspective_proxy,
                }
                for constraint in self.decomposition.spatial_constraints
            ]
            value["color_constraints"] = [
                {
                    "subject": constraint.subject,
                    "color": constraint.color,
                    "bound": constraint.bound,
                    "node_id": constraint.node_id,
                }
                for constraint in self.decomposition.color_constraints
            ]
            value["object_instances"] = [
                {
                    "instance_id": instance.instance_id,
                    "label": instance.label,
                    "canonical_label": instance.canonical_label,
                    "attributes": list(instance.attributes),
                    "group_id": instance.group_id,
                    "ordinal": instance.ordinal,
                }
                for instance in self.decomposition.object_instances
            ]
            value["sequence_constraints"] = [
                {
                    "kind": constraint.kind,
                    "source_text": constraint.source_text,
                    "member_label": constraint.member_label,
                    "prospective_action": constraint.prospective_action,
                    "confidence": constraint.confidence,
                }
                for constraint in self.decomposition.sequence_constraints
            ]
            value["prospective_relations"] = [
                {
                    "action": relation.action,
                    "anchor": relation.anchor,
                    "source_text": relation.source_text,
                    "confidence": relation.confidence,
                    "verification_policy": "positive_only",
                }
                for relation in self.decomposition.prospective_relations
            ]
            value["relative_constraints"] = [
                {
                    "subject": constraint.subject,
                    "relation": constraint.relation,
                    "object_subject": constraint.object_subject,
                    "second_object_subject": constraint.second_object_subject,
                    "arity": constraint.arity,
                    "verification_type": constraint.verification_type,
                    "bound": constraint.bound,
                    "subject_node_id": constraint.subject_node_id,
                    "object_node_id": constraint.object_node_id,
                    "second_object_node_id": constraint.second_object_node_id,
                }
                for constraint in self.decomposition.relative_constraints
            ]
            value["negation_constraints"] = [
                {
                    "subject": constraint.subject,
                    "color": constraint.color,
                    "bound": constraint.bound,
                }
                for constraint in self.decomposition.negation_constraints
            ]
            value["unsupported_relations"] = [
                {
                    "relation": constraint.relation,
                    "source_text": constraint.source_text,
                }
                for constraint in self.decomposition.unsupported_relations
            ]
            value["ambiguous_relations"] = [
                {
                    "relation": constraint.relation,
                    "reason": constraint.reason,
                }
                for constraint in self.decomposition.ambiguous_relations
            ]
            value["object_nodes"] = [
                {"node_id": node.node_id, "label": node.label}
                for node in self.decomposition.object_nodes
            ]
            value["spatial_constraint_count"] = len(
                self.decomposition.spatial_constraints
            )
            value["color_constraint_count"] = len(
                self.decomposition.color_constraints
            )
            value["relative_constraint_count"] = len(
                self.decomposition.relative_constraints
            )
            value["negation_constraint_count"] = len(
                self.decomposition.negation_constraints
            )
            value["unsupported_relation_count"] = len(
                self.decomposition.unsupported_relations
            )
            value["ambiguous_relation_count"] = len(
                self.decomposition.ambiguous_relations
            )
            value["bound_constraint_count"] = self.decomposition.bound_constraint_count
            value["lexical_must_match"] = list(self.decomposition.lexical_must_match)
            value["parenthetical_cues"] = [
                {"text": cue.text, "language": cue.language}
                for cue in self.decomposition.parenthetical_cues
            ]
            value["lexical_aliases"] = list(self.decomposition.lexical_aliases)
            value["semantic_views"] = {
                "vi": [
                    {
                        "role": view.role,
                        "text": view.text,
                        "alternatives": list(view.alternatives),
                    }
                    for view in self.decomposition.semantic_views_vi
                ],
                "en": [
                    {
                        "role": view.role,
                        "text": view.text,
                        "alternatives": list(view.alternatives),
                    }
                    for view in self.decomposition.semantic_views_en
                ],
            }
            value["semantic_view_count"] = max(
                len(self.decomposition.semantic_views_vi),
                len(self.decomposition.semantic_views_en),
            )
            value["entities"] = [
                {
                    "text": entity.text,
                    "category": entity.category,
                    "mode": entity.mode,
                    "source": entity.source,
                    "prompt_index": self.prompt_index,
                    "boost_applied": False,
                }
                for entity in self.decomposition.named_entities
            ]
            eligible_entities = tuple(
                entity
                for entity in self.decomposition.named_entities
                if not entity.source.startswith("operator_")
            )
            value["entity_boost_enabled"] = bool(eligible_entities)
            value["required_entity_count"] = sum(
                entity.mode == "required" for entity in eligible_entities
            )
        return value


class QueryTextAdapter:
    """Prepare one query once and expose variants appropriate to each channel."""

    def __init__(
        self,
        translator: OfflineVietnameseTranslator | None = None,
        *,
        entity_extractor: EntityExtractor | None = None,
    ) -> None:
        self.translator = translator
        self.entity_extractor = entity_extractor or EntityExtractor()

    @property
    def fingerprint(self) -> str:
        source = self.translator.fingerprint if self.translator is not None else "identity"
        return hashlib.sha256(
            f"query-adapter-v4:{source}:{self.entity_extractor.version}:{QUERY_DECOMPOSER_VERSION}".encode()
        ).hexdigest()

    def prepare(
        self,
        text: str,
        *,
        language_hint: LanguageHint = "auto",
        english_override: str | None = None,
        entity_constraints: Sequence[EntityConstraint | Mapping[str, Any]] = (),
        legacy_entities: Iterable[str] = (),
        prompt_index: int = 0,
    ) -> QueryProcessing:
        explicit = normalize_entity_constraints(entity_constraints)
        extracted = self.entity_extractor.extract(text)
        entities_with_modes = merge_entity_constraints(extracted, explicit, legacy_entities)
        protected_entities = tuple(entity for entity, _mode in entities_with_modes)
        if self.translator is not None:
            translate_kwargs: dict[str, Any] = {
                "language_hint": language_hint,
                "english_override": english_override,
            }
            try:
                parameters = tuple(signature(self.translator.translate).parameters.values())
                names = {value.name for value in parameters}
                supports_kwargs = any(value.kind is Parameter.VAR_KEYWORD for value in parameters)
            except (TypeError, ValueError):
                names, supports_kwargs = set(), False
            if protected_entities and (supports_kwargs or "protected_entities" in names):
                translate_kwargs["protected_entities"] = protected_entities
            result = self.translator.translate(text, **translate_kwargs)
        else:
            started = time.monotonic()
            normalized = normalize_query_text(text)
            detected = detect_query_language(normalized, language_hint)
            english = (
                normalize_query_text(english_override)
                if english_override is not None
                else normalized
                if detected == "en"
                else None
            )
            source = (
                "manual"
                if english_override is not None
                else ("identity" if detected == "en" else "unavailable")
            )
            failure = (
                None
                if english is not None
                else (
                    "unsupported_language"
                    if detected == "unknown"
                    else "translator_not_provisioned"
                )
            )
            result = TranslationResult(
                text,
                normalized,
                accent_fold(normalized),
                detected,
                english,
                source,
                "bypass",
                (time.monotonic() - started) * 1000,
                failure,
            )
        decomposition = decompose_query(
            result.normalized_text,
            entities_with_modes,
            english_text=result.english_text,
            manual_english_override=english_override is not None,
        )
        provisional = QueryProcessing(result, {}, decomposition, prompt_index)
        policies: dict[str, tuple[str, ...]] = {}
        for channel in (
            "visual",
            "organizer_clip",
            "accelerated_visual",
            "semantic_text",
            "ocr",
            "asr",
            "metadata",
        ):
            variants = (
                self.lexical_variants(channel, provisional)
                if channel in {"ocr", "asr", "metadata"}
                else self.dense_variants(channel, provisional)
            )
            policies[channel] = tuple(value.name for value in variants)
        return QueryProcessing(result, policies, decomposition, prompt_index)

    @staticmethod
    def variants(
        channel: str,
        result: TranslationResult,
        decomposition: DecomposedQuery | None = None,
    ) -> tuple[QueryVariant, ...]:
        original = QueryVariant(
            "original", result.normalized_text, "vie" if result.detected_language == "vi" else "eng"
        )
        english = (
            QueryVariant("english", result.english_text, "eng") if result.english_text else None
        )
        def view_variants(
            prefix: str,
            locale: str,
            views: Sequence[SemanticView],
        ) -> tuple[QueryVariant, ...]:
            values: list[QueryVariant] = []
            for view in views:
                texts = (view.text, *view.alternatives)
                for index, text in enumerate(texts, start=1):
                    suffix = "" if len(texts) == 1 else f"_{index}"
                    values.append(
                        QueryVariant(
                            f"{prefix}_view_{view.role}{suffix}",
                            text,
                            locale,
                            semantic_role=view.role,
                        )
                    )
            return tuple(values)

        english_views = view_variants(
            "en",
            "eng",
            decomposition.semantic_views_en if decomposition else (),
        )
        original_views = view_variants(
            "vi" if result.detected_language == "vi" else "en",
            "vie" if result.detected_language == "vi" else "eng",
            decomposition.semantic_views_vi if decomposition else (),
        )
        visual_english = english_views or ((english,) if english is not None else ())
        visual_original = original_views or (original,)
        if channel in {"organizer_clip", "accelerated_visual"}:
            return visual_english
        if channel == "visual" and result.detected_language == "vi":
            bilingual = [*visual_original[:1], *visual_english]
            if len(bilingual) < 4:
                bilingual.extend(visual_original[1 : 4 - len(bilingual)])
            return tuple(dict.fromkeys(bilingual[:4]))
        if channel == "visual":
            return visual_english or visual_original
        if channel == "metadata" and result.detected_language == "vi":
            values = (original, english) if english is not None else (original,)
            return tuple(dict.fromkeys(value for value in values if value is not None))
        return (original,)

    def dense_variants(self, channel: str, processing: QueryProcessing) -> tuple[QueryVariant, ...]:
        variants = self.variants(channel, processing.translation, processing.decomposition)
        return variants

    def lexical_variants(
        self, channel: str, processing: QueryProcessing
    ) -> tuple[QueryVariant, ...]:
        variants = list(self.variants(channel, processing.translation))
        decomposition = processing.decomposition
        if decomposition is None:
            return tuple(variants)
        for index, alias in enumerate(decomposition.lexical_aliases, start=1):
            language = next(
                (
                    cue.language
                    for cue in decomposition.parenthetical_cues
                    if cue.text == alias
                ),
                "vi",
            )
            variants.append(
                QueryVariant(
                    f"lexical_alias_{index}",
                    alias,
                    "eng" if language == "en" else "vie" if language == "vi" else None,
                )
            )
        if not decomposition.named_entities:
            return tuple(dict.fromkeys(variants))
        optional_terms: list[str] = []
        optional_ids: list[str] = []
        for entity in decomposition.named_entities:
            if entity.mode == "required":
                variants.append(
                    QueryVariant(
                        f"required_entity_{len(variants)}",
                        " ".join(entity.lexical_terms),
                        "vie",
                        (fold_entity_text(entity.text),),
                        fold_entity_text(entity.text),
                    )
                )
            else:
                optional_terms.extend(entity.lexical_terms)
                optional_ids.append(fold_entity_text(entity.text))
        if optional_terms:
            variants.append(
                QueryVariant(
                    "optional_entities",
                    " ".join(dict.fromkeys(optional_terms)),
                    "vie",
                    tuple(optional_ids),
                )
            )
        return tuple(dict.fromkeys(variants))
