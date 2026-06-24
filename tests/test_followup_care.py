"""Unit tests for FollowUpCareAnalyzer — diagnoses, tasks, and fluid/wound/
behavior care instructions. Same crafted-response pattern as
test_medication_reconciliation.py."""
import json

from src.core.decision_policy import parse_due_offset
from src.core.domain.proposal import ProposalAction, ProposalCategory, Routing, Severity
from src.core.llm.mock import MockLLMClient
from src.features.followup_care.analyzer import FollowUpCareAnalyzer

LETTER_META = {"resident_id": "R001", "letter_id": "letter_01"}


class _SequenceLLM:
    model = "sequence-stub"

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.usage_log: list[dict] = []

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        self.usage_log.append({"model": self.model, "input_tokens": 0, "output_tokens": 0})
        return self._responses.pop(0)


def _resident_data(**overrides) -> dict:
    base = {
        "diagnoses": {
            "resident_id": "R001",
            "diagnoses": [{"diagnose": "Herzinsuffizienz NYHA III (HFrEF)", "icd10": "I50.13"}],
        },
        "drink_protocol": {"resident_id": "R001", "target_ml": 2000, "strategy": "encourage"},
        "wounds": {"resident_id": "R001", "wounds": []},
    }
    base.update(overrides)
    return base


def _valid_response(diagnoses=None, tasks=None, care_instructions=None) -> str:
    return json.dumps(
        {
            "diagnoses": diagnoses or [],
            "tasks": tasks or [],
            "care_instructions": care_instructions or [],
        }
    )


def test_already_tracked_diagnosis_produces_no_proposal():
    diag = {
        "raw_name": "Herzinsuffizienz NYHA III (HFrEF)",
        "icd10": "I50.13",
        "is_new_chronic_diagnosis": True,
        "letter_quote": "Bekannte Herzinsuffizienz NYHA III.",
        "letter_section": "diagnoses",
    }
    llm = MockLLMClient(canned={"TRIGGER_DUP": _valid_response(diagnoses=[diag])})
    analyzer = FollowUpCareAnalyzer()

    proposals = analyzer.analyze("... TRIGGER_DUP ...", LETTER_META, _resident_data(), llm)

    assert proposals == []


def test_acute_resolved_finding_is_filtered_by_is_new_chronic_flag():
    """The LLM itself sets is_new_chronic_diagnosis=False for a treated UTI;
    the analyzer must trust that and not propose anything."""
    diag = {
        "raw_name": "Harnwegsinfekt",
        "icd10": "N39.0",
        "is_new_chronic_diagnosis": False,
        "letter_quote": "Harnwegsinfekt, antibiotisch behandelt, klinisch ausgeheilt.",
        "letter_section": "diagnoses",
    }
    llm = MockLLMClient(canned={"TRIGGER_UTI": _valid_response(diagnoses=[diag])})
    analyzer = FollowUpCareAnalyzer()

    proposals = analyzer.analyze("... TRIGGER_UTI ...", LETTER_META, _resident_data(), llm)

    assert proposals == []


def test_new_chronic_diagnosis_routes_human_confirm():
    diag = {
        "raw_name": "Pertrochantäre Femurfraktur links, konservativ behandelt",
        "icd10": "S72.10",
        "is_new_chronic_diagnosis": True,
        "letter_quote": "Pertrochantäre Femurfraktur links, konservative Therapie.",
        "letter_section": "diagnoses",
    }
    llm = MockLLMClient(canned={"TRIGGER_NEW_DX": _valid_response(diagnoses=[diag])})
    analyzer = FollowUpCareAnalyzer()

    proposals = analyzer.analyze("... TRIGGER_NEW_DX ...", LETTER_META, _resident_data(), llm)

    assert len(proposals) == 1
    p = proposals[0]
    assert p.category is ProposalCategory.DIAGNOSIS
    assert p.action is ProposalAction.ADD
    assert p.routing is Routing.HUMAN_CONFIRM


def test_administrative_task_auto_applies_and_encodes_due_offset():
    task = {
        "slug": "cardiology_followup",
        "description": "Kardiologische Verlaufskontrolle in 3 Monaten vereinbaren.",
        "due_in_days": 90,
        "requires_clinical_judgment": False,
        "letter_quote": "Wir bitten um kardiologische Verlaufskontrolle in 3 Monaten.",
        "letter_section": "procedere",
    }
    llm = MockLLMClient(canned={"TRIGGER_TASK_AUTO": _valid_response(tasks=[task])})
    analyzer = FollowUpCareAnalyzer()

    proposals = analyzer.analyze("... TRIGGER_TASK_AUTO ...", LETTER_META, _resident_data(), llm)

    assert len(proposals) == 1
    p = proposals[0]
    assert p.category is ProposalCategory.TASK
    assert p.action is ProposalAction.CREATE_TASK
    assert p.routing is Routing.AUTO_APPLY
    assert parse_due_offset(p.target_entity) == ("cardiology_followup", 90)


def test_clinical_judgment_task_routes_hard_stop_not_auto():
    task = {
        "slug": "metformin_dose_review",
        "description": "Re-Evaluation der Metformin-Dosis bei eingeschränkter Nierenfunktion.",
        "due_in_days": 14,
        "requires_clinical_judgment": True,
        "letter_quote": "Re-Evaluation der Metformin-Dosis durch die Hausärztin empfohlen.",
        "letter_section": "procedere",
    }
    llm = MockLLMClient(canned={"TRIGGER_TASK_JUDGMENT": _valid_response(tasks=[task])})
    analyzer = FollowUpCareAnalyzer()

    proposals = analyzer.analyze("... TRIGGER_TASK_JUDGMENT ...", LETTER_META, _resident_data(), llm)

    assert len(proposals) == 1
    p = proposals[0]
    assert p.routing is Routing.HARD_STOP_PHYSICIAN
    assert p.severity is Severity.WARN


def test_fluid_restriction_conflicting_with_existing_encourage_strategy_flagged_warn():
    """Mirrors the real R001 case: existing strategy is 'encourage', the
    discharge letter restricts fluids — a direction reversal worth flagging
    even though both routings end up human_confirm."""
    instr = {
        "care_category": "fluid_management",
        "care_action": "modify",
        "target_entity": "fluid_restriction",
        "description": "Trinkmengenbeschränkung auf 1500 ml/Tag.",
        "target_ml": 1500,
        "strategy": "restrict",
        "wound_location": None,
        "dressing_interval_days": None,
        "letter_quote": "Trinkmengenbeschränkung auf maximal 1.500 ml/Tag.",
        "letter_section": "procedere",
    }
    llm = MockLLMClient(canned={"TRIGGER_FLUID": _valid_response(care_instructions=[instr])})
    analyzer = FollowUpCareAnalyzer()

    proposals = analyzer.analyze("... TRIGGER_FLUID ...", LETTER_META, _resident_data(), llm)

    assert len(proposals) == 1
    p = proposals[0]
    assert p.category is ProposalCategory.FLUID_MANAGEMENT
    assert p.action is ProposalAction.MODIFY
    assert p.routing is Routing.HUMAN_CONFIRM
    assert p.severity is Severity.WARN
    assert "encourage" in p.rationale


def test_repair_retry_recovers_from_one_bad_response():
    bad = "{broken"
    good = _valid_response(
        tasks=[
            {
                "slug": "lab_recheck",
                "description": "Laborkontrolle in 2 Wochen.",
                "due_in_days": 14,
                "requires_clinical_judgment": False,
                "letter_quote": "Laborkontrolle in 2 Wochen b. HA.",
                "letter_section": "procedere",
            }
        ]
    )
    llm = _SequenceLLM([bad, good])
    analyzer = FollowUpCareAnalyzer()

    proposals = analyzer.analyze("anything", LETTER_META, _resident_data(), llm)

    assert len(llm.usage_log) == 2
    assert len(proposals) == 1
    assert proposals[0].routing is Routing.AUTO_APPLY


def test_double_failure_yields_info_only_fallback_flag():
    llm = _SequenceLLM(["{nope", "{still nope"])
    analyzer = FollowUpCareAnalyzer()

    proposals = analyzer.analyze("anything", LETTER_META, _resident_data(), llm)

    assert len(proposals) == 1
    p = proposals[0]
    assert p.target_entity == "followup_care_extraction_failed"
    assert p.routing is Routing.INFO_ONLY
    assert p.action is ProposalAction.FLAG
    assert p.confidence == 0.0
