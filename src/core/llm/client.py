"""LLM client abstraction.

Analyzers depend only on the `LLMClient` Protocol. The provider is selected
via the `LLM_PROVIDER` env var; the default is the deterministic mock, so the
whole repo runs without any API key or network access.

Add your own provider by implementing the Protocol and extending
`get_llm_client()`.
"""
from __future__ import annotations

import os
from typing import Protocol

import httpx


class LLMClient(Protocol):
    usage_log: list[dict]
    """Implementations append {"model", "input_tokens", "output_tokens"} per call; run.py drains this into the cost_log (the assignment requires measured cost)."""

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        """Return the model's raw text response (expected to be JSON when a schema is given)."""
        ...


class OpenAICompatibleClient:
    """Thin client for any OpenAI-compatible /chat/completions endpoint (no SDK)."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        # Drained by run.py into the proposals.json cost_log.
        self.usage_log: list[dict] = []

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            }

        response = self._post(payload)
        # Some providers/models reject response_format → retry ONCE without it.
        if response.is_client_error and "response_format" in payload:
            payload.pop("response_format")
            response = self._post(payload)
        response.raise_for_status()
        data = response.json()

        usage = data.get("usage", {})
        self.usage_log.append(
            {
                "model": self.model,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
        )
        return data["choices"][0]["message"]["content"]

    def _post(self, payload: dict) -> httpx.Response:
        return httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=60,
        )


class AnthropicClient:
    """Native Anthropic Messages API client (no SDK — plain httpx), for
    people who'd rather point this at a Claude API key directly instead of
    going through the generic OpenAI-compatible path. Deliberately does NOT
    use forced tool-call schemas: it goes through the exact same
    prompt -> validate -> repair loop as every other provider, so swapping
    providers is a fair comparison (same failure modes, not "this provider
    gets a safety net the others don't").
    """

    def __init__(self, model: str, api_key: str, max_tokens: int = 4096):
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.usage_log: list[dict] = []

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        usage = data.get("usage", {})
        self.usage_log.append(
            {
                "model": self.model,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }
        )
        return "".join(block["text"] for block in data.get("content", []) if block.get("type") == "text")


def get_llm_client() -> LLMClient:
    """Build the configured LLM client. Default: mock (no env vars needed)."""
    provider = os.environ.get("LLM_PROVIDER", "mock")

    if provider == "mock":
        from .mock import MockLLMClient

        return MockLLMClient()

    if provider == "mock_chaos":
        from .chaos import ChaosLLMClient

        return ChaosLLMClient()

    if provider == "openai_compatible":
        try:
            return OpenAICompatibleClient(
                model=os.environ["LLM_MODEL"],
                api_key=os.environ["LLM_API_KEY"],
                base_url=os.environ["LLM_BASE_URL"],
            )
        except KeyError as e:
            raise RuntimeError(
                f"LLM_PROVIDER=openai_compatible requires env var {e} (see .env.example)"
            ) from e

    if provider == "anthropic":
        try:
            return AnthropicClient(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                api_key=os.environ["ANTHROPIC_API_KEY"],
                max_tokens=int(os.environ.get("ANTHROPIC_MAX_TOKENS", "4096")),
            )
        except KeyError as e:
            raise RuntimeError(
                f"LLM_PROVIDER=anthropic requires env var {e} (see .env.example)"
            ) from e

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r} (expected 'mock', 'mock_chaos', "
        "'openai_compatible' or 'anthropic')"
    )
