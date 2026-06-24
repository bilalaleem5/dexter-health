"""Deterministic chaos LLM client — mock responses with injected failures.

Wraps `MockLLMClient` but replaces roughly every 3rd response with a disturbance
(cycling: broken JSON → out-of-enum JSON → empty string) to exercise your
validation/repair/fallback path; we run your pipeline with this provider during
grading. Disturbances are a pure function of the seed (`LLM_CHAOS_SEED`, default
42) and a call counter — same seed and call order yield the exact same sequence.
"""
from __future__ import annotations

import hashlib
import os

from .mock import _BUILTIN_CANNED, MockLLMClient

# Cycled in order; each payload must be rejected by the validation gate.
_DISTURBANCES: tuple[str, ...] = (
    _BUILTIN_CANNED["BROKEN_JSON_CASE"],  # (a) syntactically broken JSON
    _BUILTIN_CANNED["BAD_ENUM_CASE"],  # (b) valid JSON, out-of-enum value
    "",  # (c) empty string
)


def _is_disturbed(seed: int, call_index: int) -> bool:
    """Roughly 1 in 3 calls; pure function of (seed, call index) → reproducible."""
    digest = hashlib.sha256(f"{seed}:{call_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 3 == 0


class ChaosLLMClient:
    """Implements the LLMClient Protocol: MockLLMClient answers, deterministic chaos."""

    model = "mock-chaos"

    def __init__(self, canned: dict[str, str] | None = None, seed: int | None = None):
        self._mock = MockLLMClient(canned)
        self._seed = int(os.environ.get("LLM_CHAOS_SEED", "42")) if seed is None else seed
        self._calls = 0
        self._disturbance_count = 0
        # Shared with the wrapped mock, so run.py drains every call from one list.
        self.usage_log: list[dict] = self._mock.usage_log

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        call_index = self._calls
        self._calls += 1
        if _is_disturbed(self._seed, call_index):
            disturbance = _DISTURBANCES[self._disturbance_count % len(_DISTURBANCES)]
            self._disturbance_count += 1
            self.usage_log.append({"model": self.model, "input_tokens": 0, "output_tokens": 0})
            return disturbance
        return self._mock.complete(system, user, schema)
