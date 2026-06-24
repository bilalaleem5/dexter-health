"""Tests for the proposals.json contract (src/core/domain/proposal.py),
including the end-to-end contract of the file written by src/run.py."""
import json

import pytest
from pydantic import ValidationError

from src.core.domain.proposal import (
    LetterSection,
    Proposal,
    ProposalAction,
    ProposalCategory,
    ProposalsOutput,
    Provenance,
    Routing,
    Severity,
    make_proposal_id,
)


def _valid_proposal_dict() -> dict:
    # Fully fictional example — a fictional example.
    return {
        "proposal_id": "abc123abc123abc1",
        "resident_id": "R999",
        "letter_id": "letter_99",
        "category": "medication",
        "action": "verify",
        "target_entity": "Simvastatin",
        "routing": "human_confirm",
        "severity": "warn",
        "confidence": 0.85,
        "provenance": {
            "letter_quote": "Simvastatin 40 mg 0-0-1",
            "letter_section": "entlassmedikation",
            "db_reference": None,
        },
        "rationale": "Dose in letter differs from current plan.",
    }


def test_valid_proposal_roundtrip():
    proposal = Proposal.model_validate(_valid_proposal_dict())
    assert proposal.category is ProposalCategory.MEDICATION
    assert proposal.action is ProposalAction.VERIFY
    assert proposal.routing is Routing.HUMAN_CONFIRM
    assert proposal.severity is Severity.WARN
    assert proposal.provenance.letter_section is LetterSection.ENTLASSMEDIKATION

    dumped = proposal.model_dump(mode="json")
    assert Proposal.model_validate(dumped) == proposal


def test_invalid_routing_enum_rejected():
    bad = _valid_proposal_dict()
    bad["routing"] = "auto_yolo"
    with pytest.raises(ValidationError):
        Proposal.model_validate(bad)


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_confidence_out_of_bounds_rejected(confidence):
    bad = _valid_proposal_dict()
    bad["confidence"] = confidence
    with pytest.raises(ValidationError):
        Proposal.model_validate(bad)


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_confidence_bounds_inclusive(confidence):
    ok = _valid_proposal_dict()
    ok["confidence"] = confidence
    assert Proposal.model_validate(ok).confidence == confidence


def test_run_end_to_end_writes_valid_proposals_output(tmp_path, monkeypatch):
    """run.py against a tmp fixture produces a schema-valid proposals.json
    with at least one stay_metadata proposal and a populated cost_log."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from src.run import run_analysis

    letters_dir = tmp_path / "letters"
    letters_dir.mkdir()
    (letters_dir / "letter_01.md").write_text(
        "Wir berichten über den stationären Aufenthalt vom 12.05.2026 bis 26.05.2026 "
        "in unserer Klinik für Kardiologie.",
        encoding="utf-8",
    )
    (letters_dir / "index.json").write_text(
        json.dumps(
            [
                {
                    "letter_id": "letter_01",
                    "resident_id": "R001",
                    "hospital": "St.-Marien-Hospital",
                    "admission_date": "2026-05-12",
                    "discharge_date": "2026-05-26",
                    "file": "letters/letter_01.md",
                }
            ]
        )
    )
    data_dir = tmp_path / "data"
    (data_dir / "vitals").mkdir(parents=True)
    (data_dir / "vitals" / "R001.json").write_text(
        '{"resident_id": "R001", "weights": [{"created_at": "2026-05-01T08:00:00+00:00", "weight": 72.5}]}'
    )
    out_path = tmp_path / "proposals.json"

    run_analysis(letters_dir, data_dir, out_path)

    output = ProposalsOutput.model_validate(json.loads(out_path.read_text()))
    assert output.schema_version == "1.0"
    assert len(output.proposals) >= 1
    proposal = output.proposals[0]
    assert proposal.resident_id == "R001"
    assert proposal.letter_id == "letter_01"
    assert proposal.target_entity == "stay_metadata"
    assert output.cost_log, "mock client must still produce cost_log entries"
    assert output.cost_log[0].letter_id == "letter_01"
    assert output.cost_log[0].model == "mock"

    # Deterministic proposal ids → idempotent re-runs.
    first_ids = [p.proposal_id for p in output.proposals]
    run_analysis(letters_dir, data_dir, out_path)
    rerun = ProposalsOutput.model_validate(json.loads(out_path.read_text()))
    assert [p.proposal_id for p in rerun.proposals] == first_ids


def test_proposal_id_is_deterministic():
    a = make_proposal_id("R001", "letter_01", ProposalCategory.MEDICATION, ProposalAction.MODIFY, "Simvastatin")
    b = make_proposal_id("R001", "letter_01", ProposalCategory.MEDICATION, ProposalAction.MODIFY, "Simvastatin")
    c = make_proposal_id("R001", "letter_01", ProposalCategory.MEDICATION, ProposalAction.MODIFY, "Ibuprofen")
    assert a == b
    assert a != c
    assert len(a) == 16
    int(a, 16)  # hex digest prefix
