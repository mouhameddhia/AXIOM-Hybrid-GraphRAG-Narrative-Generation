"""
LLM client for text-to-Cypher generation.

A single-method :class:`LLMClient` protocol -- ``complete(prompt) -> text`` --
is what the engine depends on, the same injectable-model pattern already used
for the dense encoder and the cross-encoder reranker. Production wires
:class:`OllamaClient`; tests inject a small fake that returns canned text, so
the self-healing loop is fully testable with no network and no model running.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

from src.config.settings import LLMConfig

__all__ = ["LLMClient", "LLMRequestError", "OllamaClient"]

logger = logging.getLogger(__name__)


class LLMRequestError(RuntimeError):
    """Raised when the LLM backend cannot be reached or returns an error."""


@runtime_checkable
class LLMClient(Protocol):
    """Minimal contract: a prompt in, the model's raw text response out."""

    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        """
        Return the model's completion for ``prompt``.

        ``temperature`` overrides the configured default for this one call.
        The self-healing loop uses it to escalate on retries: at temperature
        0 a model re-reads a corrected prompt and greedily regenerates the
        same tokens, so a retry that does not vary sampling often repeats the
        rejected answer verbatim.
        """


class OllamaClient:
    """
    Production client for a local Ollama server.

    Uses the standard-library ``urllib`` rather than an HTTP dependency: this
    is a single POST-and-parse-JSON call, which does not need a full library
    to do correctly.

    Usage::

        client = OllamaClient(settings.llm)
        cypher = client.complete("Write a Cypher query that ...")
    """

    __slots__ = ("_config",)

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        """
        Send ``prompt`` to Ollama's ``/api/generate`` endpoint and return the
        model's text response.

        Raises:
            LLMRequestError: the server is unreachable, times out, returns a
                non-2xx status, or the response is not the expected JSON
                shape. The message is safe to surface to a caller deciding
                whether to fall back to a different path.
        """
        payload = json.dumps(
            {
                "model": self._config.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": (
                        self._config.temperature if temperature is None else temperature
                    )
                },
            }
        ).encode("utf-8")

        url = f"{self._config.base_url.rstrip('/')}/api/generate"
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self._config.request_timeout_s
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMRequestError(
                f"Ollama returned HTTP {exc.code} from {url}: {detail[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMRequestError(
                f"Could not reach Ollama at {url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise LLMRequestError(
                f"Ollama did not respond within {self._config.request_timeout_s}s"
            ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMRequestError(
                f"Ollama response was not valid JSON: {body[:500]}"
            ) from exc

        text = parsed.get("response")
        if not isinstance(text, str):
            raise LLMRequestError(
                f"Ollama response had no 'response' field: {body[:500]}"
            )
        return text
