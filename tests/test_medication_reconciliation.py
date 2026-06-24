"""Unit tests for MedicationReconciliationAnalyzer.

Uses crafted canned/sequenced LLM responses rather than relying on chaos
alignment, so each behavior (success, repair, double-failure fallback,
allergy conflict, internal contradiction, true negatives) is deterministic
and directly attributable. See tests/test_clinical_followup_process.py and
test_followup_care.py for the same pattern applied to the other features.
"""
import json

import pytest

from src.core.domain.proposal import ProposalAction, ProposalCategory, Routing, Severity
from src.core.llm.mock import MockLLMClient
from src.features.medication_reconciliation.analyzer import MedicationReconciliationAnalyzer

LETTER_META = {"resident_id": "R004", "letter_id": "letter_04"}


class _SequenceLLM:
    """Tiny stub LLMClient that returns a fixed sequence of responses, one
    per call — lets us test "first call bad, repair call good" and
    "both calls bad" deterministically, which the substring-keyed
    MockLLMClient cannot express on its own."""

    model = "sequence-stub"

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.usage_log: list[dict] = []

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        self.usage_log.append({"model": self.model, "input_tokens": 0, "output_tokens": 0})
        return self._responses.pop(0)


def _resident_data(**overrides) -> dict:
    base = {
        "allergies": {"resident_id": "R004", "allergies": []},
        "medication_plan": {
            "resident_id": "R004",
            "medications_planned": [
                {"wirkstoff": "Ramipril", "strength": "5", "unit": "mg", "dosage": "1-0-0"},
            ],
            "medications_on_demand": [],
        },
    }
    base.update(overrides)
    return base


def _valid_response(medications: list[dict], missing_attachment_note=None) -> str:
    return json.dumps({"medications": medications, "missing_attachment_note": missing_attachment_note})


def test_new_medication_routes_human_confirm():
    med = {
        "raw_name": "Eliquis 5mg",
        "wirkstoff": "Apixaban",
        "strength_and_dosage": "5 mg 1-0-1",
        "status": "new",
        "internal_inconsistency_note": None,
        "letter_quote": "Wir setzen Apixaban 5 mg 1-0-1 neu an.",
        "letter_section": "entlassmedikation",
        "conflicts_with_allergy": False,
        "allergy_conflict_reasoning": None,
    }
    llm = MockLLMClient(canned={"APIXABAN_TRIGGER": _valid_response([med])})
    analyzer = MedicationReconciliationAnalyzer()

    proposals = analyzer.analyze("... APIXABAN_TRIGGER ...", LETTER_META, _resident_data(), llm)

    assert len(proposals) == 1
    p = proposals[0]
    assert p.action is ProposalAction.ADD
    assert p.category is ProposalCategory.MEDICATION
    assert p.target_entity == "apixaban"
    assert p.routing is Routing.HUMAN_CONFIRM  # medication changes never auto-apply
    assert p.severity is Severity.WARN
    assert 0.0 < p.confidence < 1.0


def test_allergy_conflict_forces_hard_stop_regardless_of_status():
    """Mirrors the real R004 case: Amoxicillin/Clavulansäure against a
    documented penicillin allergy must never be auto-applicable, even though
    the LLM reports it as a perfectly clear 'new' medication."""
    med = {
        "raw_name": "Amoxicillin/Clavulansäure 875/125mg",
        "wirkstoff": "Amoxicillin/Clavulansäure",
        "strength_and_dosage": "875/125 mg 1-0-1",
        "status": "new",
        "internal_inconsistency_note": None,
        "letter_quote": "Sequenztherapie: Amoxicillin/Clavulansäure 875/125 mg 1-0-1 p.o.",
        "letter_section": "procedere",
        "conflicts_with_allergy": True,
        "allergy_conflict_reasoning": "Amoxicillin ist ein Penicillin-Derivat; bekannte Penicillinallergie.",
    }
    llm = MockLLMClient(canned={"AMOX_TRIGGER": _valid_response([med])})
    analyzer = MedicationReconciliationAnalyzer()
    resident_data = _resident_data(
        allergies={
            "resident_id": "R004",
            "allergies": [{"substance": "Penicillin", "reaction": "generalisiertes Exanthem"}],
        }
    )

    proposals = analyzer.analyze("... AMOX_TRIGGER ...", LETTER_META, resident_data, llm)

    assert len(proposals) == 1
    p = proposals[0]
    assert p.routing is Routing.HARD_STOP_PHYSICIAN
    assert p.severity is Severity.CRITICAL
    assert p.confidence <= 0.5  # must not overstate confidence on a flagged conflict
    assert "Allergiekonflikt" in p.rationale


def test_internal_dose_contradiction_routes_hard_stop():
    """Mirrors the R001 insulin glargin case: 12 IE in the text vs 18 IE in
    the table, in the SAME letter. Never auto-applicable."""
    med = {
        "raw_name": "Insulin glargin (Lantus)",
        "wirkstoff": "Insulin glargin",
        "strength_and_dosage": "18 IE 1-0-0-0",
        "status": "unclear",
        "internal_inconsistency_note": "Text empfiehlt 12 IE, Tabelle nennt 18 IE.",
        "letter_quote": "Insulin glargin auf 12 IE reduziert / Insulin glargin (Lantus) 18 IE 1-0-0-0",
        "letter_section": "entlassmedikation",
        "conflicts_with_allergy": False,
        "allergy_conflict_reasoning": None,
    }
    llm = MockLLMClient(canned={"INSULIN_TRIGGER": _valid_response([med])})
    analyzer = MedicationReconciliationAnalyzer()

    proposals = analyzer.analyze("... INSULIN_TRIGGER ...", LETTER_META, _resident_data(), llm)

    assert len(proposals) == 1
    p = proposals[0]
    assert p.action is ProposalAction.FLAG
    assert p.routing is Routing.HARD_STOP_PHYSICIAN
    assert p.severity is Severity.CRITICAL


def test_general_safety_conflict_forces_hard_stop():
    """Mirrors the real R002 case: a newly started NSAID in a resident with a
    documented prior GI bleed/ulcer, combined with an anticoagulant, is a
    real drug-safety conflict even though no formal allergy is involved."""
    med = {
        "raw_name": "Ibuprofen 600mg",
        "wirkstoff": "Ibuprofen",
        "strength_and_dosage": "600 mg 1-1-1",
        "status": "new",
        "internal_inconsistency_note": None,
        "letter_quote": "Ibuprofen 600 mg 1-1-1 bei Schmerzen neu angesetzt.",
        "letter_section": "entlassmedikation",
        "conflicts_with_allergy": False,
        "allergy_conflict_reasoning": None,
        "conflicts_with_existing_condition": True,
        "condition_conflict_reasoning": (
            "NSAR bei Z.n. oberer GI-Blutung (Ulcus ventriculi) und gleichzeitiger "
            "Enoxaparin-Gabe — erhöhtes Blutungsrisiko."
        ),
    }
    llm = MockLLMClient(canned={"IBU_TRIGGER": _valid_response([med])})
    analyzer = MedicationReconciliationAnalyzer()

    proposals = analyzer.analyze("... IBU_TRIGGER ...", LETTER_META, _resident_data(), llm)

    assert len(proposals) == 1
    p = proposals[0]
    assert p.routing is Routing.HARD_STOP_PHYSICIAN
    assert p.severity is Severity.CRITICAL
    assert p.confidence <= 0.5
    assert "Sicherheitskonflikt" in p.rationale


@pytest.mark.parametrize("status", ["continued_unchanged", "deferred_not_started"])
def test_unchanged_or_deferred_medication_produces_no_proposal(status):
    """Negative cases: a medication explicitly continued unchanged, or
    explicitly deferred/not started, must not generate a proposal at all."""
    med = {
        "raw_name": "Candesartan",
        "wirkstoff": "Candesartan",
        "strength_and_dosage": "16 mg 1-0-0",
        "status": status,
        "internal_inconsistency_note": None,
        "letter_quote": "Candesartan unverändert fortgeführt." if status == "continued_unchanged" else "Betablocker zunächst zurückgestellt.",
        "letter_section": "entlassmedikation",
        "conflicts_with_allergy": False,
        "allergy_conflict_reasoning": None,
    }
    llm = MockLLMClient(canned={"NEG_TRIGGER": _valid_response([med])})
    analyzer = MedicationReconciliationAnalyzer()

    proposals = analyzer.analyze("... NEG_TRIGGER ...", LETTER_META, _resident_data(), llm)

    assert proposals == []


def test_missing_attachment_note_routes_hard_stop_with_low_confidence():
    llm = MockLLMClient(canned={"ANLAGE_TRIGGER": _valid_response([], missing_attachment_note="siehe Anlage 2 — nicht im Brief enthalten")})
    analyzer = MedicationReconciliationAnalyzer()

    proposals = analyzer.analyze("... ANLAGE_TRIGGER ...", LETTER_META, _resident_data(), llm)

    assert len(proposals) == 1
    p = proposals[0]
    assert p.action is ProposalAction.FLAG
    assert p.target_entity == "missing_medication_attachment"
    assert p.routing is Routing.HARD_STOP_PHYSICIAN
    assert p.confidence < 0.5


def test_repair_retry_recovers_from_one_bad_response():
    bad = "{not valid json"
    good = _valid_response(
        [
            {
                "raw_name": "Torasemid",
                "wirkstoff": "Torasemid",
                "strength_and_dosage": "20 mg 1-0-0",
                "status": "dose_changed",
                "internal_inconsistency_note": None,
                "letter_quote": "Torasemid auf 20 mg erhöht.",
                "letter_section": "entlassmedikation",
                "conflicts_with_allergy": False,
                "allergy_conflict_reasoning": None,
            }
        ]
    )
    llm = _SequenceLLM([bad, good])
    analyzer = MedicationReconciliationAnalyzer()

    proposals = analyzer.analyze("anything", LETTER_META, _resident_data(), llm)

    assert len(llm.usage_log) == 2  # one extract call + one repair call
    assert len(proposals) == 1
    assert proposals[0].action is ProposalAction.MODIFY
    assert proposals[0].routing is Routing.HUMAN_CONFIRM


def test_double_failure_yields_info_only_fallback_flag():
    llm = _SequenceLLM(["{not valid", "{still not valid"])
    analyzer = MedicationReconciliationAnalyzer()

    proposals = analyzer.analyze("anything", LETTER_META, _resident_data(), llm)

    assert len(proposals) == 1
    p = proposals[0]
    assert p.target_entity == "medication_extraction_failed"
    assert p.routing is Routing.INFO_ONLY
    assert p.action is ProposalAction.FLAG
    assert p.confidence == 0.0
