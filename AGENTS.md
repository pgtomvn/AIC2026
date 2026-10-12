# Agent guide

## Mission and current state

This repository is the completed AIC 2026 multimodal video-retrieval system, published as a
public MIT-licensed project. Preserve its three workflows: Textual KIS, Visual Q&A, and TRAKE.
Changes are maintenance: keep the public README, `docs/`, and behavior consistent with each
other. Models, datasets, and search indexes are prepared locally and are never part of Git.
`research/` is a local-only workspace (gitignored); do not commit or publish it.

## Safe working contract

- Work from the repository root and inspect `git status` before and after changes. Existing
  edits may belong to another agent; do not discard or overwrite unrelated work.
- Treat `Keyframes_L21.zip` and `Videos_L21_a.zip` as immutable. Their expected digests are in
  `checksums/archives.sha256`. Never rewrite, rename, recompress, or delete them.
- Treat `data/`, `docker_data/`, `models/`, `outputs/`, and `logs/` as user-owned generated
  state. Do not remove them during code or environment cleanup.
- Never commit `.env`, credentials, downloaded tools, virtual environments, model weights, or
  derived data. Use `.env.example` for names and safe placeholders only.
- Use the uv lockfile and project-local environment. Do not install packages globally or add a
  dependency without updating and reviewing both `pyproject.toml` and `uv.lock`.
- Keep import and constructor paths side-effect free. In particular, an import must not create
  logs, data directories, download models, or connect to services.
- Default tests must stay CPU-only and offline. Mark tests requiring Docker, a model, GPU,
  archive extraction, or network access as explicit integration tests.
- Make the smallest scoped change and preserve public `src.*` imports unless a migration is
  part of the task.

## Architecture

- `app/`: Streamlit UI and components; an HTTP client of the FastAPI runtime (`src/api.py`).
- `config/settings.yaml`: central data/model/service/retrieval configuration.
- `scripts/`: command-line orchestration. Extraction must remain validated, idempotent, and
  opt-in for the full corpus.
- `src/extraction` and `src/embedding`: media/OCR/ASR and feature-generation boundaries.
- `src/indexing`: Qdrant vector and Meilisearch text adapters.
- `src/retrieval`: Textual KIS, Visual Q&A, TRAKE, and reranking.
- `src/vlm`: optional visual-language model boundary; heavy local inference is not in the
  default environment.

## Required workflow

```powershell
.\scripts\preflight.ps1 -SkipArchiveHash  # omit the flag when archive integrity is relevant
.\scripts\bootstrap.ps1
.\scripts\check.ps1
```

Before handing off a normal Python change, run `.\scripts\check.ps1`. Use
`-IncludeCompose`, `-IncludeServices`, `-IncludeStreamlit`, and/or `-IncludeArchives` only when
the change touches those boundaries. Archive inspection reads the complete 4.5 GiB corpus;
the service check observes running services but does not start or delete them.

For data inspection, prefer the smallest operation:

```powershell
$Uv = Resolve-Path .tools/uv/0.11.16/uv.exe
& $Uv run --frozen python scripts/extract_data.py --inspect
& $Uv run --frozen python scripts/extract_data.py --sample L21_V001
```

Use `--full` only when the task explicitly requires the whole corpus. Existing targets are
skipped unless `--force` is deliberately supplied.

## Implementation expectations

- Python is 3.11 and source files are UTF-8. Add type hints to public functions and concise
  docstrings where behavior is not obvious.
- Configuration and secrets have distinct roles: commit non-secret defaults to YAML; load
  credentials and local endpoint overrides from `.env`/the process environment.
- Service endpoints default to loopback. Qdrant uses REST 6333 and gRPC 6334; Meilisearch uses
  7700. Do not expose them beyond localhost in the development Compose file.
- Fail clearly at unavailable stub or optional-backend boundaries. Do not hide missing models,
  services, or credentials behind fabricated results.
- Keep large outputs out of Git and report any command that creates material derived data.

Windows PowerShell 5.1 may render UTF-8 incorrectly. The project scripts set process encoding;
for manual work set `$env:PYTHONUTF8 = "1"` and use a UTF-8 console.
