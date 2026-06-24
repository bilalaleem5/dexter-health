"""Follow-up & care-instruction analyzer.

Extracts, from the discharge letter's Procedere/Epikrise/Diagnosen sections:
- new CHRONIC diagnoses (deliberately excludes transient/resolved findings
  from this stay, e.g. a treated UTI — see DECISIONS.md §1)
- follow-up tasks (GP/specialist appointments, lab/imaging checks,
  re-evaluations)
- care-plan instructions affecting fluid management, wound care, or
  monitoring/behavior precautions

... and reconciles each against the resident's existing diagnoses / drink
protocol / wound records before deciding whether a proposal is warranted at
all. Routing is decided by `src/core/decision_policy.py`, never by the LLM.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.core.decision_policy import (
    encode_due_offset,
    route_care_instruction,
    route_diagnosis_addition,
    route_extraction_failure,
    route_task,
)
from src.core.domain.proposal import (
    LetterSection,
    Proposal,
    ProposalAction,
    ProposalCategory,
    Provenance,
    make_proposal_id,
)
from src.core.llm.client import LLMClient

_SYSTEM_PROMPT = (
    "You are a careful clinical documentation assistant for a nursing home, reading a "
    "German hospital discharge letter. Read German fluently. Distinguish chronic, "
    "ongoing conditions from acute findings that were fully treated and resolved during "
    "this stay. Answer with a single JSON object and nothing else."
)

_VALID_CARE_CATEGORIES = ("fluid_management", "wound", "behavior")
_VALID_CARE_ACTIONS = ("add", "modify", "stop")

_EXTRACT_PROMPT = """Below is a German hospital discharge letter for a nursing-home
resident, plus the resident's currently documented chronic diagnoses, fluid/drink
protocol, and active wounds (all from BEFORE this hospital stay).

Existing chronic diagnoses:
{existing_diagnoses}

Existing drink/fluid protocol: {existing_drink_protocol}

Existing active wounds: {existing_wounds}

1) DIAGNOSES: list any diagnosis mentioned in the letter that is a NEW, ONGOING/CHRONIC
condition the nursing home should now track going forward (e.g. a new fracture with
lasting functional consequence, a newly diagnosed permanent arrhythmia). Do NOT include:
 - diagnoses already in the existing chronic diagnoses list above
 - acute conditions that were fully treated and resolved DURING this stay with no lasting
   care implication (e.g. a treated urinary tract infection, treated dehydration, a
   completed pneumonia treatment) — these belong in care notes, not the chronic list
For each: raw_name, icd10 (or null), is_new_chronic_diagnosis (true/false — set false and
omit nothing if borderline, we filter on this), letter_quote (<=200 chars), letter_section.

2) TASKS: list every follow-up action the letter asks the nursing home / GP to do after
discharge (lab checks, imaging, specialist appointments, re-evaluations, vaccination
status checks). For each: slug (short snake_case id, e.g. "renal_lab_recheck"),
description (one sentence, in German, what needs to happen), due_in_days (integer,
your best estimate from the letter's wording, relative to the discharge date),
requires_clinical_judgment (true if a clinician must actually decide/evaluate something
— e.g. "re-evaluate the dose", "judge whether to restart X"; false if it is purely
scheduling/administrative — e.g. "book a lab check", "book a follow-up appointment"),
letter_quote (<=200 chars), letter_section.

3) CARE_INSTRUCTIONS: list new or changed care-plan instructions for fluid management,
wound care, or behavioral/monitoring precautions (e.g. fluid restriction or target
change, a new wound dressing schedule, a new fall/aspiration/sedation precaution). For
each: care_category (one of {care_categories}), care_action (one of {care_actions}),
target_entity (short snake_case id), description (German, one sentence),
target_ml (int or null, only for fluid_management), strategy ("encourage"/"restrict"/null,
only for fluid_management), wound_location (string or null, only for wound),
dressing_interval_days (int or null, only for wound), letter_quote (<=200 chars),
letter_section.

Letter sections to choose from for letter_section: {sections}

Return JSON exactly as:
{{"diagnoses": [...], "tasks": [...], "care_instructions": [...]}}

Letter:
{letter}"""

_REPAIR_PROMPT = """Your previous answer could not be used.

Previous answer: {previous}
Problem: {error}

Answer again with ONLY a valid JSON object, same instructions and letter as before.

Letter:
{letter}"""


class ExtractedDiagnosis(BaseModel):
    raw_name: str
    icd10: str | None = None
    is_new_chronic_diagnosis: bool
    letter_quote: str
    letter_section: LetterSection = LetterSection.DIAGNOSES


class ExtractedTask(BaseModel):
    slug: str
    description: str
    due_in_days: int = Field(ge=0, le=3650)
    requires_clinical_judgment: bool
    letter_quote: str
    letter_section: LetterSection = LetterSection.PROCEDERE


class ExtractedCareInstruction(BaseModel):
    care_category: str
    care_action: str
    target_entity: str
    description: str
    target_ml: int | None = None
    strategy: str | None = None
    wound_location: str | None = None
    dressing_interval_days: int | None = None
    letter_quote: str
    letter_section: LetterSection = LetterSection.PROCEDERE

    @field_validator("care_category")
    @classmethod
    def category_must_be_known(cls, v: str) -> str:
        if v not in _VALID_CARE_CATEGORIES:
            raise ValueError(f"care_category must be one of {_VALID_CARE_CATEGORIES}, got {v!r}")
        return v

    @field_validator("care_action")
    @classmethod
    def action_must_be_known(cls, v: str) -> str:
        if v not in _VALID_CARE_ACTIONS:
            raise ValueError(f"care_action must be one of {_VALID_CARE_ACTIONS}, got {v!r}")
        return v


class FollowUpExtraction(BaseModel):
    diagnoses: list[ExtractedDiagnosis] = Field(default_factory=list)
    tasks: list[ExtractedTask] = Field(default_factory=list)
    care_instructions: list[ExtractedCareInstruction] = Field(default_factory=list)


_CATEGORY_MAP = {
    "fluid_management": ProposalCategory.FLUID_MANAGEMENT,
    "wound": ProposalCategory.WOUND,
    "behavior": ProposalCategory.BEHAVIOR,
}
_ACTION_MAP = {
    "add": ProposalAction.ADD,
    "modify": ProposalAction.MODIFY,
    "stop": ProposalAction.STOP,
}


class FollowUpCareAnalyzer:
    name = "followup_care"

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
        existing_diagnoses = self._existing_diagnosis_names(resident_data)
        for diag in extracted.diagnoses:
            proposal = self._diagnosis_proposal(letter_meta, diag, existing_diagnoses)
            if proposal is not None:
                proposals.append(proposal)

        for task in extracted.tasks:
            proposals.append(self._task_proposal(letter_meta, task))

        existing_strategy = ((resident_data.get("drink_protocol") or {}).get("strategy"))
        for instr in extracted.care_instructions:
            proposals.append(self._care_proposal(letter_meta, instr, existing_strategy))

        return proposals

    def _extract_with_repair(
        self, letter_text: str, resident_data: dict, llm: LLMClient
    ) -> tuple[FollowUpExtraction | None, str | None]:
        schema = FollowUpExtraction.model_json_schema()
        sections = ", ".join(s.value for s in LetterSection)

        diagnoses = resident_data.get("diagnoses") or {}
        existing_diagnoses = (
            "\n".join(f"- {d['diagnose']} ({d['icd10']})" for d in (diagnoses.get("diagnoses") or []))
            or "(keine dokumentiert)"
        )
        drink = resident_data.get("drink_protocol") or {}
        existing_drink_protocol = (
            f"Ziel {drink['target_ml']} ml/Tag, Strategie: {drink['strategy']}"
            if drink
            else "(kein Protokoll dokumentiert)"
        )
        wounds = resident_data.get("wounds") or {}
        active_wounds = [w for w in (wounds.get("wounds") or []) if w.get("status") == "active"]
        existing_wounds = (
            "\n".join(f"- {w['location']} ({w['type']}, Grad {w['grade']})" for w in active_wounds)
            or "(keine aktiven Wunden dokumentiert)"
        )

        prompt = _EXTRACT_PROMPT.format(
            existing_diagnoses=existing_diagnoses,
            existing_drink_protocol=existing_drink_protocol,
            existing_wounds=existing_wounds,
            care_categories=", ".join(_VALID_CARE_CATEGORIES),
            care_actions=", ".join(_VALID_CARE_ACTIONS),
            sections=sections,
            letter=letter_text,
        )
        raw = llm.complete(_SYSTEM_PROMPT, prompt, schema=schema)
        try:
            return FollowUpExtraction.model_validate_json(raw), None
        except ValidationError as error:
            repair = _REPAIR_PROMPT.format(previous=raw, error=error, letter=letter_text)
            raw = llm.complete(_SYSTEM_PROMPT, repair, schema=schema)
            try:
                return FollowUpExtraction.model_validate_json(raw), None
            except ValidationError as final_error:
                return None, str(final_error)

    def _existing_diagnosis_names(self, resident_data: dict) -> set[str]:
        diagnoses = resident_data.get("diagnoses") or {}
        return {d["diagnose"].strip().lower() for d in (diagnoses.get("diagnoses") or [])}

    def _diagnosis_proposal(
        self, letter_meta: dict, diag: ExtractedDiagnosis, existing: set[str]
    ) -> Proposal | None:
        if not diag.is_new_chronic_diagnosis:
            return None
        if diag.raw_name.strip().lower() in existing:
            return None  # already tracked — true negative, not a duplicate ADD

        target_entity = _slugify(diag.raw_name)
        routing, severity = route_diagnosis_addition(is_chronic=True)
        return Proposal(
            proposal_id=make_proposal_id(
                letter_meta["resident_id"], letter_meta["letter_id"], ProposalCategory.DIAGNOSIS, ProposalAction.ADD, target_entity
            ),
            resident_id=letter_meta["resident_id"],
            letter_id=letter_meta["letter_id"],
            category=ProposalCategory.DIAGNOSIS,
            action=ProposalAction.ADD,
            target_entity=target_entity,
            routing=routing,
            severity=severity,
            confidence=0.8,
            provenance=Provenance(letter_quote=diag.letter_quote[:200], letter_section=diag.letter_section),
            rationale=f"Neue chronische Diagnose laut Brief: {diag.raw_name} ({diag.icd10 or 'kein ICD-10 angegeben'}).",
        )

    def _task_proposal(self, letter_meta: dict, task: ExtractedTask) -> Proposal:
        target_entity = encode_due_offset(_slugify(task.slug), task.due_in_days)
        routing, severity = route_task(requires_clinical_judgment=task.requires_clinical_judgment)
        return Proposal(
            proposal_id=make_proposal_id(
                letter_meta["resident_id"],
                letter_meta["letter_id"],
                ProposalCategory.TASK,
                ProposalAction.CREATE_TASK,
                target_entity,
            ),
            resident_id=letter_meta["resident_id"],
            letter_id=letter_meta["letter_id"],
            category=ProposalCategory.TASK,
            action=ProposalAction.CREATE_TASK,
            target_entity=target_entity,
            routing=routing,
            severity=severity,
            confidence=0.85,
            provenance=Provenance(letter_quote=task.letter_quote[:200], letter_section=task.letter_section),
            rationale=task.description,
        )

    def _care_proposal(
        self, letter_meta: dict, instr: ExtractedCareInstruction, existing_strategy: str | None
    ) -> Proposal:
        category = _CATEGORY_MAP[instr.care_category]
        action = _ACTION_MAP[instr.care_action]
        direction_conflict = (
            instr.care_category == "fluid_management"
            and instr.strategy is not None
            and existing_strategy is not None
            and instr.strategy != existing_strategy
        )
        routing, severity = route_care_instruction(direction_conflict=direction_conflict)
        target_entity = _slugify(instr.target_entity)

        rationale = instr.description
        if direction_conflict:
            rationale += f" (Achtung: bestehende Strategie war '{existing_strategy}', neu: '{instr.strategy}'.)"

        return Proposal(
            proposal_id=make_proposal_id(letter_meta["resident_id"], letter_meta["letter_id"], category, action, target_entity),
            resident_id=letter_meta["resident_id"],
            letter_id=letter_meta["letter_id"],
            category=category,
            action=action,
            target_entity=target_entity,
            routing=routing,
            severity=severity,
            confidence=0.8,
            provenance=Provenance(letter_quote=instr.letter_quote[:200], letter_section=instr.letter_section),
            rationale=rationale,
        )

    def _extraction_failed_proposal(
        self, letter_meta: dict, letter_text: str, failure: str | None
    ) -> Proposal:
        routing, severity = route_extraction_failure()
        return Proposal(
            proposal_id=make_proposal_id(
                letter_meta["resident_id"],
                letter_meta["letter_id"],
                ProposalCategory.OTHER,
                ProposalAction.FLAG,
                "followup_care_extraction_failed",
            ),
            resident_id=letter_meta["resident_id"],
            letter_id=letter_meta["letter_id"],
            category=ProposalCategory.OTHER,
            action=ProposalAction.FLAG,
            target_entity="followup_care_extraction_failed",
            routing=routing,
            severity=severity,
            confidence=0.0,
            provenance=Provenance(letter_quote=_first_line(letter_text), letter_section=LetterSection.OTHER),
            rationale=(
                "Folgeaufgaben/Diagnosen-Abgleich nicht möglich: LLM-Antwort zweimal nicht "
                f"verwertbar. Ein Mensch sollte den Brief lesen. Letzter Fehler: {failure}"
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
