"""Medication reconciliation analyzer.

Compares the hospital discharge medication list against the resident's
pre-admission medication plan and allergy list, and emits ADD / STOP /
MODIFY / FLAG proposals.

LLM responsibility (domain knowledge + language): read the German letter,
normalize brand names to active substances, classify each discharge
medication's status relative to the letter's OWN narrative (new / continued
unchanged / dose changed / stopped / explicitly deferred / unclear), and
judge whether a discharge medication plausibly cross-reacts with a
documented allergy.

Deterministic code responsibility (this module): decide which statuses are
even worth a proposal (continued-unchanged and explicitly-deferred are true
negatives, not proposals), and — critically — decide ROUTING via
`src/core/decision_policy.py`. Routing is never taken from the LLM.

Follows the house pattern from `src/features/stay_metadata/analyzer.py`:
extract -> validate -> one repair retry -> fallback flag on double failure.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.core.decision_policy import route_medication_change, route_missing_critical_data
from src.core.domain.proposal import (
    LetterSection,
    Proposal,
    ProposalAction,
    ProposalCategory,
    Provenance,
    Routing,
    Severity,
    make_proposal_id,
)
from src.core.llm.client import LLMClient

_SYSTEM_PROMPT = (
    "You are a careful clinical pharmacist reviewing a German hospital discharge letter "
    "for a nursing-home resident, reconciling it against the resident's pre-admission "
    "medication plan. Read German fluently. Be conservative: only classify a medication's "
    "status as something definite if the letter is explicit about it. "
    "Answer with a single JSON object and nothing else."
)

_VALID_STATUSES = (
    "new",
    "continued_unchanged",
    "dose_changed",
    "stopped",
    "deferred_not_started",
    "unclear",
)

_EXTRACT_PROMPT = """Below is a German hospital discharge letter, the resident's known
allergies, the resident's known chronic diagnoses, and the resident's
medication plan from BEFORE this hospital stay.

Known allergies (substance: reaction):
{allergies}

Known chronic diagnoses:
{diagnoses}

Pre-admission medication plan (active substance, strength, dosage):
{existing_meds}

Extract every medication that is part of the DISCHARGE regimen ("Entlassmedikation",
including anything in the Procedere that changes it), and classify each one. For each:

- raw_name: exactly as written in the letter (brand or generic)
- wirkstoff: the active substance / generic name, normalized to standard German
  pharmacological naming, using your own clinical knowledge even if the letter only
  gives a brand name (e.g. "Eliquis" -> "Apixaban")
- strength_and_dosage: the discharge strength/dosing scheme as stated in the letter, or
  null if genuinely not stated
- status: exactly one of {statuses}
  - "new": not on the pre-admission plan, started during/at the end of this stay
  - "continued_unchanged": same substance AND same dose as the pre-admission plan
  - "dose_changed": same substance, different dose/strength than the pre-admission plan
  - "stopped": was on the pre-admission plan, the letter says it is discontinued or paused
  - "deferred_not_started": the letter explicitly says starting/considering this drug was
    POSTPONED or NOT done — do not use "new" for a drug that was only discussed but
    explicitly not started
  - "unclear": you cannot confidently determine the status/dose because the LETTER ITSELF
    contains conflicting information about this specific drug (e.g. two different doses
    for the same drug in different sections of the same letter)
- internal_inconsistency_note: required (non-null) if status is "unclear" for the reason
  above; quote both conflicting parts of the letter; null otherwise
- letter_quote: a short verbatim quote (<= 200 characters) from the letter, your primary
  evidence for this entry
- letter_section: one of {sections}
- conflicts_with_allergy: true only if this medication's substance (or class) plausibly
  cross-reacts with one of the resident's KNOWN ALLERGIES listed above (use real
  pharmacological knowledge, e.g. amoxicillin and a penicillin allergy); false otherwise
- allergy_conflict_reasoning: one sentence if conflicts_with_allergy is true, else null
- conflicts_with_existing_condition: true if, using real clinical knowledge, this
  medication poses a meaningful safety concern given the resident's KNOWN CHRONIC
  DIAGNOSES above or another medication on the discharge/pre-admission list (e.g. an
  NSAID newly started for a resident with a documented history of GI bleeding/ulcer,
  especially combined with an anticoagulant; or reduced renal function with a
  renally-cleared/nephrotoxic drug at an unadjusted dose). Be specific and conservative —
  only true for a real, describable mechanism, not a vague "could interact with anything"
- condition_conflict_reasoning: one sentence naming the specific diagnosis/medication and
  mechanism if conflicts_with_existing_condition is true, else null

Also set "missing_attachment_note" (top-level key, nullable): non-null only if the letter
explicitly references an attachment/appendix as containing (part of) the medication list,
and that attachment's content is NOT actually present in the letter text you were given
(e.g. "siehe Anlage 2" with no Anlage 2 content anywhere below). Quote the reference.

Return JSON exactly as:
{{"medications": [...], "missing_attachment_note": "..." or null}}

Letter:
{letter}"""

_REPAIR_PROMPT = """Your previous answer could not be used.

Previous answer: {previous}
Problem: {error}

Answer again with ONLY a valid JSON object, same instructions and letter as before.

Letter:
{letter}"""


class ExtractedMedication(BaseModel):
    raw_name: str
    wirkstoff: str
    strength_and_dosage: str | None = None
    status: str
    internal_inconsistency_note: str | None = None
    letter_quote: str
    letter_section: LetterSection = LetterSection.ENTLASSMEDIKATION
    conflicts_with_allergy: bool = False
    allergy_conflict_reasoning: str | None = None
    conflicts_with_existing_condition: bool = False
    condition_conflict_reasoning: str | None = None

    @field_validator("status")
    @classmethod
    def status_must_be_known(cls, v: str) -> str:
        if v not in _VALID_STATUSES:
            raise ValueError(f"status must be one of {_VALID_STATUSES}, got {v!r}")
        return v


class MedicationExtraction(BaseModel):
    medications: list[ExtractedMedication] = Field(default_factory=list)
    missing_attachment_note: str | None = None


class MedicationReconciliationAnalyzer:
    name = "medication_reconciliation"

    def analyze(
        self,
        letter_text: str,
        letter_meta: dict,
        resident_data: dict,
        llm: LLMClient,
    ) -> list[Proposal]:
        extracted, failure = self._extract_with_repair(letter_text, resident_data, llm)
        if extracted is None:
            return [self._extraction_failed_proposal(letter_meta, letter_text, failure)]

        proposals: list[Proposal] = []
        if extracted.missing_attachment_note:
            proposals.append(
                self._missing_attachment_proposal(letter_meta, extracted.missing_attachment_note)
            )

        for med in extracted.medications:
            proposal = self._proposal_for(letter_meta, med)
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    def _extract_with_repair(
        self, letter_text: str, resident_data: dict, llm: LLMClient
    ) -> tuple[MedicationExtraction | None, str | None]:
        schema = MedicationExtraction.model_json_schema()
        sections = ", ".join(s.value for s in LetterSection)
        allergy_lines = (
            "\n".join(
                f"- {a['substance']}: {a['reaction']}"
                for a in ((resident_data.get("allergies") or {}).get("allergies") or [])
            )
            or "(keine bekannt)"
        )
        diagnosis_lines = (
            "\n".join(
                f"- {d['diagnose']} ({d.get('icd10', 'kein ICD-10')})"
                for d in ((resident_data.get("diagnoses") or {}).get("diagnoses") or [])
            )
            or "(keine dokumentiert)"
        )
        plan = resident_data.get("medication_plan") or {}
        existing_lines = [
            f"- {m['wirkstoff']} {m['strength']}{m['unit']} ({m['dosage']})"
            for m in (plan.get("medications_planned") or [])
        ] + [
            f"- {m['wirkstoff']} {m['strength']}{m['unit']} "
            f"(bei Bedarf, max. {m.get('max_daily_dose', 'k.A.')})"
            for m in (plan.get("medications_on_demand") or [])
        ]
        existing_meds = "\n".join(existing_lines) or "(keine dokumentiert)"

        prompt = _EXTRACT_PROMPT.format(
            allergies=allergy_lines,
            diagnoses=diagnosis_lines,
            existing_meds=existing_meds,
            statuses=", ".join(_VALID_STATUSES),
            sections=sections,
            letter=letter_text,
        )
        raw = llm.complete(_SYSTEM_PROMPT, prompt, schema=schema)
        try:
            return MedicationExtraction.model_validate_json(raw), None
        except ValidationError as error:
            repair = _REPAIR_PROMPT.format(previous=raw, error=error, letter=letter_text)
            raw = llm.complete(_SYSTEM_PROMPT, repair, schema=schema)
            try:
                return MedicationExtraction.model_validate_json(raw), None
            except ValidationError as final_error:
                return None, str(final_error)

    def _proposal_for(self, letter_meta: dict, med: ExtractedMedication) -> Proposal | None:
        if med.status in ("continued_unchanged", "deferred_not_started"):
            return None  # true negative — letter confirms no actionable change

        action = {
            "new": ProposalAction.ADD,
            "dose_changed": ProposalAction.MODIFY,
            "stopped": ProposalAction.STOP,
            "unclear": ProposalAction.FLAG,
        }[med.status]
        target_entity = _slugify(med.wirkstoff)

        is_safety_conflict = med.conflicts_with_allergy or med.conflicts_with_existing_condition
        routing, severity = route_medication_change(
            is_safety_conflict=is_safety_conflict,
            is_ambiguous_or_conflicting=(med.status == "unclear"),
            change_kind=med.status,
        )

        confidence = 0.9 if med.status in ("new", "stopped") else 0.85
        if is_safety_conflict or med.status == "unclear":
            confidence = 0.45  # genuinely uncertain or dangerous — do not overstate

        rationale = f"{med.wirkstoff} ({med.strength_and_dosage or 'Dosierung siehe Zitat'}): {med.status}."
        if med.internal_inconsistency_note:
            rationale += f" Widerspruch im Brief: {med.internal_inconsistency_note}"
        if med.conflicts_with_allergy:
            rationale += f" Allergiekonflikt: {med.allergy_conflict_reasoning}"
        if med.conflicts_with_existing_condition:
            rationale += f" Sicherheitskonflikt: {med.condition_conflict_reasoning}"

        return Proposal(
            proposal_id=make_proposal_id(
                letter_meta["resident_id"], letter_meta["letter_id"], ProposalCategory.MEDICATION, action, target_entity
            ),
            resident_id=letter_meta["resident_id"],
            letter_id=letter_meta["letter_id"],
            category=ProposalCategory.MEDICATION,
            action=action,
            target_entity=target_entity,
            routing=routing,
            severity=severity,
            confidence=confidence,
            provenance=Provenance(letter_quote=med.letter_quote[:200], letter_section=med.letter_section),
            rationale=rationale,
        )

    def _missing_attachment_proposal(self, letter_meta: dict, note: str) -> Proposal:
        routing, severity = route_missing_critical_data()
        return Proposal(
            proposal_id=make_proposal_id(
                letter_meta["resident_id"],
                letter_meta["letter_id"],
                ProposalCategory.MEDICATION,
                ProposalAction.FLAG,
                "missing_medication_attachment",
            ),
            resident_id=letter_meta["resident_id"],
            letter_id=letter_meta["letter_id"],
            category=ProposalCategory.MEDICATION,
            action=ProposalAction.FLAG,
            target_entity="missing_medication_attachment",
            routing=routing,
            severity=severity,
            confidence=0.3,
            provenance=Provenance(letter_quote=note[:200], letter_section=LetterSection.ENTLASSMEDIKATION),
            rationale=(
                "Die Entlassmedikation kann nicht vollständig bestätigt werden — der Brief "
                f"verweist auf eine fehlende Anlage: {note}"
            ),
        )

    def _extraction_failed_proposal(
        self, letter_meta: dict, letter_text: str, failure: str | None
    ) -> Proposal:
        from src.core.decision_policy import route_extraction_failure

        routing, severity = route_extraction_failure()
        return Proposal(
            proposal_id=make_proposal_id(
                letter_meta["resident_id"],
                letter_meta["letter_id"],
                ProposalCategory.MEDICATION,
                ProposalAction.FLAG,
                "medication_extraction_failed",
            ),
            resident_id=letter_meta["resident_id"],
            letter_id=letter_meta["letter_id"],
            category=ProposalCategory.MEDICATION,
            action=ProposalAction.FLAG,
            target_entity="medication_extraction_failed",
            routing=routing,
            severity=severity,
            confidence=0.0,
            provenance=Provenance(letter_quote=_first_line(letter_text), letter_section=LetterSection.OTHER),
            rationale=(
                "Medikamentenabgleich nicht möglich: LLM-Antwort zweimal nicht verwertbar. "
                f"Ein Mensch sollte den Brief lesen. Letzter Fehler: {failure}"
            ),
        )


def _slugify(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in text.strip()]
    slug = "".join(keep)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


def _first_line(letter_text: str) -> str:
    return next((line.strip() for line in letter_text.splitlines() if line.strip()), "")[:200]
