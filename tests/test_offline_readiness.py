from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import scripts.competition_drill as drill_module
import scripts.prepare_models as prepare_module
from src.embedding.clip_encoder import CLIPEncoder
from src.embedding.siglip_encoder import SigLIP2Encoder
from src.embedding.text_encoder import VietnameseTextEncoder
from src.extraction.asr_engine import ASREngine
from src.extraction.ocr_engine import OCREngine


class _FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            projection_dim=768,
            hidden_size=768,
            text_config=SimpleNamespace(max_position_embeddings=64),
        )

    def to(self, _device: str) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        return self


class _Loader:
    calls: list[dict[str, Any]] = []
    value: Any = _FakeModel()

    @classmethod
    def from_pretrained(cls, _name: str, **kwargs: Any) -> Any:
        cls.calls.append(kwargs)
        return cls.value


def _fake_torch() -> ModuleType:
    module = ModuleType("torch")
    module.float16 = object()
    module.float32 = object()
    module.cuda = SimpleNamespace(is_available=lambda: False)
    return module


@pytest.mark.parametrize("variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_transformer_adapters_force_offline_before_backend_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variable: str
) -> None:
    monkeypatch.setenv(variable, "0")
    transformers = ModuleType("transformers")
    processor = type("Processor", (_Loader,), {"calls": [], "value": object()})
    auto_model = type("AutoModel", (_Loader,), {"calls": [], "value": _FakeModel()})
    tokenizer = type("Tokenizer", (_Loader,), {"calls": [], "value": object()})
    transformers.AutoProcessor = processor
    transformers.AutoModel = auto_model
    transformers.AutoTokenizer = tokenizer
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    SigLIP2Encoder(device="cpu", cache_dir=tmp_path, local_files_only=True)._load_model()
    VietnameseTextEncoder(device="cpu", cache_dir=tmp_path, local_files_only=True)._load_model()

    assert os.environ[variable] == "1"
    assert processor.calls[-1]["local_files_only"] is True
    assert tokenizer.calls[-1]["local_files_only"] is True
    assert all(call["local_files_only"] is True for call in auto_model.calls)


def test_openclip_local_only_forces_hub_cache_and_rejects_url_only_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    open_clip = ModuleType("open_clip")
    open_clip.get_pretrained_cfg = lambda _model, _tag: {"hf_hub": "org/repo/"}

    def create(*_args: Any, **kwargs: Any) -> tuple[Any, None, object]:
        calls.append(kwargs)
        return _FakeModel(), None, object()

    open_clip.create_model_and_transforms = create
    open_clip.get_tokenizer = lambda _model: object()
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    encoder = CLIPEncoder(device="cpu", cache_dir=tmp_path, local_files_only=True)
    encoder._load_model()

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert calls == [{"pretrained": "laion2b_s32b_b79k", "cache_dir": str(tmp_path)}]

    open_clip.get_pretrained_cfg = lambda _model, _tag: {"url": "https://example.invalid/x"}
    with pytest.raises(RuntimeError, match="no cache-only pretrained source"):
        CLIPEncoder(device="cpu", cache_dir=tmp_path, local_files_only=True)._load_model()


def test_asr_and_ocr_local_only_override_conflicting_offline_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    whisper_calls: list[dict[str, Any]] = []
    faster_whisper = ModuleType("faster_whisper")

    class FakeWhisper:
        def __init__(self, _reference: str, **kwargs: Any) -> None:
            whisper_calls.append(kwargs)

    faster_whisper.WhisperModel = FakeWhisper
    paddleocr = ModuleType("paddleocr")
    paddleocr.PaddleOCR = lambda **_kwargs: object()
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)
    monkeypatch.setitem(sys.modules, "paddleocr", paddleocr)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "False")
    whisper_path = tmp_path / "whisper"
    ocr_path = tmp_path / "ocr"
    whisper_path.mkdir()
    ocr_path.mkdir()

    ASREngine(device="cpu", model_path=whisper_path, local_files_only=True)._load_model()
    OCREngine(cache_dir=ocr_path, local_files_only=True)._load_model()

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] == "True"
    assert whisper_calls[-1]["local_files_only"] is True


def _drill_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "profile": "accelerated",
        "tools": {"ffmpeg": {"manifest": ".tools/ffmpeg/active.json"}},
        "models": {
            "visual": {"cache_dir": "models/huggingface"},
            "text": {"cache_dir": "models/huggingface"},
            "whisper": {"local_path": "models/whisper/model"},
            "ocr": {"cache_dir": "models/paddleocr"},
            "organizer_clip": {"enabled": False},
            "accelerated_visual": {
                "enabled": True,
                "cache_dir": "models/accelerated-openclip",
            },
            "qwen_vl": {
                "enabled": True,
                "enabled_profiles": ["accelerated"],
                "cache_dir": "models/qwen",
            },
        },
        "data": {"free_space_reserve_gib": 0},
    }


def test_competition_drill_checks_ffmpeg_and_accelerated_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(drill_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(drill_module, "load_config", lambda: _drill_config(tmp_path))
    monkeypatch.setattr(
        drill_module,
        "verify_asset_manifest",
        lambda manifest, *, root, required_paths: (
            captured.update(manifest=manifest, root=root, required=set(required_paths)) or (7, [])
        ),
    )
    monkeypatch.setattr(
        drill_module,
        "verify_ffmpeg_manifest",
        lambda path: captured.update(ffmpeg=path) or SimpleNamespace(version="pinned-v1"),
    )
    monkeypatch.setattr(
        drill_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=100_000),
    )
    monkeypatch.setattr(drill_module, "build_runtime", lambda _config: SimpleNamespace())

    assert drill_module.main(["--skip-services"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert tmp_path / "models/accelerated-openclip" in captured["required"]
    assert tmp_path / "models/qwen" in captured["required"]
    assert captured["ffmpeg"] == tmp_path / ".tools/ffmpeg/active.json"
    assert report["checks"]["ffmpeg"] == "pinned-v1"


def test_operational_drill_requires_full_inputs() -> None:
    with pytest.raises(SystemExit):
        drill_module.main(["--operational"])


def test_operational_drill_restarts_both_persisted_services(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    def run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(drill_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(drill_module.subprocess, "run", run)
    monkeypatch.setattr(
        drill_module,
        "_wait_for_services",
        lambda config: calls.update(waited=config),
    )
    config = {"services": {}}

    drill_module._physical_service_restart(config)

    assert calls["command"] == [
        "docker",
        "compose",
        "restart",
        "qdrant",
        "meilisearch",
    ]
    assert calls["kwargs"]["cwd"] == tmp_path
    assert calls["waited"] is config


def test_prepare_models_supports_accelerated_openclip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created: list[dict[str, Any]] = []
    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **_kwargs: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(prepare_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        prepare_module,
        "load_config",
        lambda: {
            "models": {
                "accelerated_visual": {
                    "model_name": "ViT-L-14-quickgelu",
                    "pretrained": "dfn2b",
                    "cache_dir": "models/open_clip",
                }
            }
        },
    )

    class FakeEncoder:
        def __init__(self, model_name: str, pretrained: str, **kwargs: Any) -> None:
            created.append({"model_name": model_name, "pretrained": pretrained, **kwargs})

        def encode_text(self, _text: str) -> None:
            return None

    import src.embedding.clip_encoder as clip_module

    monkeypatch.setattr(clip_module, "CLIPEncoder", FakeEncoder)
    monkeypatch.setattr(prepare_module, "freeze_assets", lambda *_args, **_kwargs: 1)

    assert prepare_module.main(["--only", "accelerated"]) == 0
    assert created == [
        {
            "model_name": "ViT-L-14-quickgelu",
            "pretrained": "dfn2b",
            "device": "cpu",
            "batch_size": 1,
            "cache_dir": tmp_path / "models/open_clip",
            "local_files_only": False,
        }
    ]


def test_prepare_models_supports_pinned_qwen_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    downloads: list[dict[str, Any]] = []
    frozen: dict[str, Any] = {}
    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **kwargs: downloads.append(kwargs)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(prepare_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        prepare_module,
        "load_config",
        lambda: {
            "models": {
                "qwen_vl": {
                    "model_name": "Qwen/Qwen3-VL-4B-Instruct",
                    "revision": "pinned-revision",
                    "cache_dir": "models/qwen",
                }
            }
        },
    )
    monkeypatch.setattr(
        prepare_module,
        "freeze_assets",
        lambda paths, **kwargs: frozen.update(paths=set(paths), **kwargs) or 9,
    )

    assert prepare_module.main(["--only", "qwen"]) == 0
    assert downloads[0]["repo_id"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert downloads[0]["revision"] == "pinned-revision"
    assert downloads[0]["cache_dir"] == tmp_path / "models/qwen"
    assert frozen["paths"] == {tmp_path / "models/qwen"}
