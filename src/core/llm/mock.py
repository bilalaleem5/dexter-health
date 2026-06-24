"""Deterministic mock LLM client — the default provider.

Returns canned JSON selected by substring match on the user prompt. Two
special cases let tests (and you) exercise the error paths every real LLM
integration needs:

- "BROKEN_JSON_CASE" in the prompt → syntactically broken JSON (parse error)
- "BAD_ENUM_CASE" in the prompt → valid JSON with an out-of-enum value
  (pydantic validation error)

Pass your own `canned` dict to simulate responses for your analyzers in tests.
"""
from __future__ import annotations

import json

_DEFAULT_RESPONSE = json.dumps(
    {
        "admission_date": "2026-05-12",
        "discharge_date": "2026-05-26",
        "department": "Kardiologie",
    }
)

_BUILTIN_CANNED: dict[str, str] = {
    # Truncated mid-object: json.loads raises.
    "BROKEN_JSON_CASE": '{"admission_date": "2026-05-12", "discharge_date": ',
    # Parses, but "anhang" is not a valid LetterSection.
    "BAD_ENUM_CASE": json.dumps(
        {
            "admission_date": "2026-05-12",
            "discharge_date": "2026-05-26",
            "department": "Kardiologie",
            "source_section": "anhang",
        }
    ),
}


class MockLLMClient:
    """Implements the LLMClient Protocol with canned, deterministic responses."""

    model = "mock"

    def __init__(self, canned: dict[str, str] | None = None):
        # Custom triggers take precedence over the built-in ones (matched first).
        self.canned = dict(canned or {})
        for trigger, response in _BUILTIN_CANNED.items():
            self.canned.setdefault(trigger, response)
        # Drained by run.py into the proposals.json cost_log.
        self.usage_log: list[dict] = []

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        self.usage_log.append({"model": self.model, "input_tokens": 0, "output_tokens": 0})
        for trigger, response in self.canned.items():
            if trigger in user:
                return response
        return _DEFAULT_RESPONSE
