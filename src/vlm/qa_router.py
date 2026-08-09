"""Request-scoped Visual Q&A provider routing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from src.contracts import QACompetitionSpec, QAQueryRequest, QAReasoningResult
from src.vlm.qa_reasoner import (
    QAImage,
    QAProviderRequestError,
    QAProviderResponseError,
    QAProviderUnavailableError,
    QAReasoner,
    QAReasonerError,
    _fingerprint,
    _raise_clean,
    _with_error_context,
)

_GEMINI_FALLBACK_CATEGORIES = frozenset(
    {
        "not_configured",
        "client_unavailable",
        "network_unreachable",
        "server_unavailable",
        "resource_exhausted",
        "gemini_timeout",
        "gemini_response_invalid",
        "rate_limited",
        "response_invalid",
    }
)


class QAReasonerRouter:
    """Route one-frame QA through Gemini and then the local Qwen fallback."""

    provider_name = "qa-reasoner-router"

    def __init__(self, gemini: QAReasoner, local: QAReasoner) -> None:
        self.gemini = gemini
        self.local = local

    @property
    def providers(self) -> tuple[QAReasoner, ...]:
        return (self.gemini, self.local)

    @property
    def fingerprint(self) -> str:
        values = [str(getattr(provider, "fingerprint")) for provider in self.providers]
        return _fingerprint(self.provider_name, {"providers": values})

    @property
    def max_attempts(self) -> int:
        """Each routed provider owns exactly one physical attempt."""

        return max(int(getattr(provider, "max_attempts", 1)) for provider in self.providers)

    def providers_for(self, request: QAQueryRequest) -> tuple[QAReasoner, ...]:
        policy = request.reasoning_policy
        if policy == "local_only":
            return (self.local,)
        if not request.allow_external_frame_transfer:
            raise ValueError("Gemini reasoning requires external frame transfer consent")
        if policy == "gemini_then_local":
            return (self.gemini, self.local)
        raise ValueError("unknown Visual Q&A reasoning policy")

    @staticmethod
    def _can_fallback(error: QAReasonerError) -> bool:
        if isinstance(error, QAProviderResponseError):
            return True
        return (
            isinstance(error, (QAProviderRequestError, QAProviderUnavailableError))
            and error.category in _GEMINI_FALLBACK_CATEGORIES
        )

    def reason(
        self,
        request: QAQueryRequest,
        images: Sequence[QAImage],
        spec: QACompetitionSpec,
    ) -> QAReasoningResult:
        providers = self.providers_for(request)
        last_error: QAReasonerError | None = None
        for index, provider in enumerate(providers):
            provider_name = str(getattr(provider, "provider_name", type(provider).__name__))
            model_name = str(getattr(provider, "model_name", "unknown"))
            try:
                result = provider.reason(request, images, spec)
                return replace(
                    result,
                    requested_policy=request.reasoning_policy,
                    fallback_used=index > 0,
                    fallback_reason_code=(last_error.category if index > 0 and last_error else None),
                    attempts=index + 1,
                )
            except QAReasonerError as error:
                last_error = _with_error_context(
                    error,
                    provider=provider_name,
                    model=model_name,
                    attempts=error.attempts,
                    category=error.category,
                    status_code=error.status_code,
                )
                if index == 0 and len(providers) > 1 and self._can_fallback(last_error):
                    continue
                if index > 0 and len(providers) > 1:
                    break
                _raise_clean(last_error)
            except Exception:
                _raise_clean(
                    QAProviderRequestError(
                        "the Visual Q&A provider failed unexpectedly",
                        provider=provider_name,
                        model=model_name,
                        attempts=1,
                        category="provider_request_failed",
                    )
                )
        if last_error is not None:
            _raise_clean(
                QAProviderUnavailableError(
                    "all configured Visual Q&A providers failed",
                    provider=self.provider_name,
                    model="gemini-local-chain",
                    attempts=len(providers),
                    category="provider_chain_exhausted",
                )
            )
        raise QAProviderUnavailableError(
            "no Visual Q&A provider is configured",
            provider=self.provider_name,
            category="not_configured",
        )

    def unload(self) -> None:
        """Release only the local model; the cloud provider holds no model state."""

        try:
            self.local.unload()
        except Exception:
            return


__all__ = ["QAReasonerRouter"]
