"""Tests for the deterministic MockLLMClient (the default, key-free provider)
and the ChaosLLMClient (mock + deterministic disturbance injection)."""
import json

import pytest

from src.core.domain.proposal import (
    LetterSection,
    ProposalAction,
    ProposalsOutput,
    Routing,
)
from src.core.llm.chaos import ChaosLLMClient
from src.core.llm.client import get_llm_client
from src.core.llm.mock import MockLLMClient


def test_default_provider_is_mock(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert isinstance(get_llm_client(), MockLLMClient)


def test_default_response_is_valid_stay_metadata_json():
    llm = MockLLMClient()
    raw = llm.complete(system="x", user="Extract stay metadata from this letter ...")
    parsed = json.loads(raw)
    assert parsed["admission_date"] == "2026-05-12"
    assert parsed["discharge_date"] == "2026-05-26"
    assert parsed["department"] == "Kardiologie"


def test_responses_are_deterministic():
    llm = MockLLMClient()
    prompt = "Extract stay metadata ..."
    assert llm.complete("s", prompt) == llm.complete("s", prompt)


def test_broken_json_case_raises_on_parse():
    # Documents the repair-loop hook: a real model can return syntactically
    # broken JSON — callers must catch the parse error and retry/fallback.
    llm = MockLLMClient()
    raw = llm.complete(system="x", user="... BROKEN_JSON_CASE ...")
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)


def test_bad_enum_case_parses_but_violates_enum():
    # Parses fine, but `source_section` is not a valid LetterSection —
    # pydantic validation (not json.loads) must catch this.
    llm = MockLLMClient()
    raw = llm.complete(system="x", user="... BAD_ENUM_CASE ...")
    parsed = json.loads(raw)
    valid_sections = {s.value for s in LetterSection}
    assert parsed["source_section"] not in valid_sections


def test_usage_log_records_zero_token_entries():
    llm = MockLLMClient()
    llm.complete("s", "first")
    llm.complete("s", "second")
    assert llm.usage_log == [
        {"model": "mock", "input_tokens": 0, "output_tokens": 0},
        {"model": "mock", "input_tokens": 0, "output_tokens": 0},
    ]


def test_chaos_provider_is_registered(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock_chaos")
    monkeypatch.delenv("LLM_CHAOS_SEED", raising=False)
    assert isinstance(get_llm_client(), ChaosLLMClient)


def test_chaos_same_seed_yields_identical_sequence():
    prompts = [f"Extract stay metadata, letter {i}" for i in range(10)]
    a, b = ChaosLLMClient(seed=7), ChaosLLMClient(seed=7)
    seq_a = [a.complete("s", p) for p in prompts]
    seq_b = [b.complete("s", p) for p in prompts]
    assert seq_a == seq_b
    # The chaos actually bit: the sequence differs from the undisturbed mock.
    assert seq_a != [MockLLMClient().complete("s", p) for p in prompts]


def test_chaos_all_disturbance_types_occur():
    llm = ChaosLLMClient(seed=42)
    responses = [llm.complete("s", "Extract stay metadata ...") for _ in range(30)]

    saw_empty = saw_broken_json = saw_bad_enum = False
    valid_sections = {s.value for s in LetterSection}
    for raw in responses:
        if raw == "":
            saw_empty = True
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            saw_broken_json = True
            continue
        if parsed.get("source_section", "other") not in valid_sections:
            saw_bad_enum = True

    assert saw_broken_json, "expected a syntactically broken JSON disturbance"
    assert saw_bad_enum, "expected a valid-JSON-but-out-of-enum disturbance"
    assert saw_empty, "expected an empty-string disturbance"


def test_chaos_usage_log_records_every_call():
    llm = ChaosLLMClient(seed=42)
    for _ in range(12):
        llm.complete("s", "Extract stay metadata ...")
    assert len(llm.usage_log) == 12
    assert all(entry["input_tokens"] == 0 and entry["output_tokens"] == 0 for entry in llm.usage_log)
    # Disturbed calls log under the chaos model name, clean calls under the mock's.
    assert {entry["model"] for entry in llm.usage_log} == {"mock", "mock-chaos"}


def test_run_with_chaos_provider_never_emits_invalid_proposals(tmp_path, monkeypatch):
    """run.py under the chaos provider must degrade gracefully: fallback
    INFO_ONLY flag proposals are fine, invalid proposals or exceptions are not."""
    monkeypatch.setenv("LLM_PROVIDER", "mock_chaos")
    monkeypatch.setenv("LLM_CHAOS_SEED", "42")
    from src.run import run_analysis

    letters_dir = tmp_path / "letters"
    letters_dir.mkdir()
    index = []
    for i in range(1, 7):
        letter_id = f"letter_{i:02d}"
        (letters_dir / f"{letter_id}.md").write_text(
            "Wir berichten über den stationären Aufenthalt vom 12.05.2026 bis "
            "26.05.2026 in unserer Klinik für Kardiologie.",
            encoding="utf-8",
        )
        index.append(
            {
                "letter_id": letter_id,
                "resident_id": "R001",
                "hospital": "St.-Marien-Hospital",
                "admission_date": "2026-05-12",
                "discharge_date": "2026-05-26",
                "file": f"letters/{letter_id}.md",
            }
        )
    (letters_dir / "index.json").write_text(json.dumps(index))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_path = tmp_path / "proposals.json"

    run_analysis(letters_dir, data_dir, out_path)  # must not raise

    # Every proposal must satisfy the contract (model_validate raises otherwise).
    output = ProposalsOutput.model_validate(json.loads(out_path.read_text()))
    # NOTE: with >1 analyzer registered, letters can yield more than one
    # proposal each, and not every proposal is an "extraction failed"
    # fallback (missing-attachment/task/care proposals are not) — so we no
    # longer assert an exact one-proposal-per-letter count. See DECISIONS.md §2.
    assert len(output.proposals) >= len(index)
    assert any(entry.model == "mock-chaos" for entry in output.cost_log), "chaos never triggered"

    # IF a system-reliability fallback occurred for any analyzer, it must be
    # well-formed. Whether one lands in this particular small sample depends
    # on where seed 42's disturbances fall across 3 analyzers x 6 letters of
    # calls — the deterministic guarantee that double-failure -> fallback is
    # tested directly per analyzer (see test_medication_reconciliation.py,
    # test_followup_care.py) with a forced double failure, not via chaos luck.
    fallbacks = [p for p in output.proposals if p.target_entity.endswith("_extraction_failed")]
    for proposal in fallbacks:
        assert proposal.routing is Routing.INFO_ONLY
        assert proposal.action is ProposalAction.FLAG
        assert proposal.confidence == 0.0
