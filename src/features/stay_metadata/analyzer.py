"""Worked example: extract stay metadata (dates + department) from a letter.

This is the house pattern for LLM-backed analyzers — follow it in your own:

1. Focused prompt: one extraction concern per analyzer.
2. Validate, never trust: raw model output goes through a pydantic model.
3. Retry with repair: on parse/validation failure, ONE retry that shows the
   model its previous answer and the error.
4. Fallback, never crash/hallucinate: if validation fails twice, emit an
   INFO_ONLY flag proposal so the failure is visible downstream.
5. Provenance: every proposal quotes the letter verbatim.
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ValidationError, field_validator

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
    "You are a careful clinical documentation assistant. "
    "Answer with a single JSON object and nothing else."
)

_EXTRACT_PROMPT = """Extract the hospital stay metadata from the German discharge letter below.

Return JSON with exactly these keys:
- admission_date: ISO date (YYYY-MM-DD)
- discharge_date: ISO date (YYYY-MM-DD)
- department: the treating department as written in the letter
- source_section: the letter section the dates come from, one of: {sections}

Letter:
{letter}"""

_REPAIR_PROMPT = """Your previous answer could not be used.

Previous answer: {previous}
Problem: {error}

Answer again with ONLY a valid JSON object exactly as specified.

Letter:
{letter}"""

# Confidence should reflect validation strength (how much of the output was verified).
EXTRACTION_CONFIDENCE = 0.9


class StayMetadata(BaseModel):
    """Validation gate for the LLM output — bad JSON or bad values fail here."""

    admission_date: str
    discharge_date: str
    department: str
    source_section: LetterSection = LetterSection.OTHER

    @field_validator("admission_date", "discharge_date")
    @classmethod
    def must_be_iso_date(cls, v: str) -> str:
        date.fromisoformat(v)  # raises ValueError → ValidationError
        return v


class StayMetadataAnalyzer:
    name = "stay_metadata"

    def analyze(
        self,
        letter_text: str,
        letter_meta: dict,
        resident_data: dict,
        llm: LLMClient,
    ) -> list[Proposal]:
        extracted, failure = self._extract_with_repair(letter_text, llm)
        if extracted is None:
            return [self._fallback_proposal(letter_meta, letter_text, failure)]
        return [self._metadata_proposal(letter_meta, letter_text, extracted)]

    def _extract_with_repair(
        self, letter_text: str, llm: LLMClient
    ) -> tuple[StayMetadata | None, str | None]:
        """One extraction attempt + one repair attempt. (None, error) if both fail."""
        schema = StayMetadata.model_json_schema()
        sections = ", ".join(s.value for s in LetterSection)

        prompt = _EXTRACT_PROMPT.format(sections=sections, letter=letter_text)
        raw = llm.complete(_SYSTEM_PROMPT, prompt, schema=schema)
        try:
            # model_validate_json catches both broken JSON and out-of-enum values.
            return StayMetadata.model_validate_json(raw), None
        except ValidationError as error:
            repair = _REPAIR_PROMPT.format(previous=raw, error=error, letter=letter_text)
            raw = llm.complete(_SYSTEM_PROMPT, repair, schema=schema)
            try:
                return StayMetadata.model_validate_json(raw), None
            except ValidationError as final_error:
                return None, str(final_error)

    def _metadata_proposal(
        self, letter_meta: dict, letter_text: str, extracted: StayMetadata
    ) -> Proposal:
        return Proposal(
            proposal_id=make_proposal_id(
                letter_meta["resident_id"],
                letter_meta["letter_id"],
                ProposalCategory.OTHER,
                ProposalAction.FLAG,
                "stay_metadata",
            ),
            resident_id=letter_meta["resident_id"],
            letter_id=letter_meta["letter_id"],
            category=ProposalCategory.OTHER,
            action=ProposalAction.FLAG,
            target_entity="stay_metadata",
            routing=Routing.INFO_ONLY,
            severity=Severity.INFO,
            confidence=EXTRACTION_CONFIDENCE,
            provenance=Provenance(
                letter_quote=_quote_for(letter_text, extracted.admission_date),
                letter_section=extracted.source_section,
            ),
            rationale=(
                f"Stay metadata extracted: admitted {extracted.admission_date}, "
                f"discharged {extracted.discharge_date}, department {extracted.department}."
            ),
        )

    def _fallback_proposal(self, letter_meta: dict, letter_text: str, failure: str | None) -> Proposal:
        """Validation failed twice → visible flag instead of a crash or made-up data."""
        return Proposal(
            proposal_id=make_proposal_id(
                letter_meta["resident_id"],
                letter_meta["letter_id"],
                ProposalCategory.OTHER,
                ProposalAction.FLAG,
                "stay_metadata_extraction_failed",
            ),
            resident_id=letter_meta["resident_id"],
            letter_id=letter_meta["letter_id"],
            category=ProposalCategory.OTHER,
            action=ProposalAction.FLAG,
            target_entity="stay_metadata_extraction_failed",
            routing=Routing.INFO_ONLY,
            severity=Severity.WARN,
            confidence=0.0,
            provenance=Provenance(
                letter_quote=_first_line(letter_text),
                letter_section=LetterSection.OTHER,
            ),
            rationale=(
                "Stay metadata could not be extracted: the LLM output failed validation "
                f"twice. A human should read this letter. Last error: {failure}"
            ),
        )


def _quote_for(letter_text: str, admission_date: str) -> str:
    """Verbatim letter line containing the admission date, else the first line.

    Provenance quotes must come from the letter itself — never reconstructed.
    """
    needles = [admission_date]
    try:
        needles.append(date.fromisoformat(admission_date).strftime("%d.%m.%Y"))
    except ValueError:
        pass
    for line in letter_text.splitlines():
        if any(needle in line for needle in needles):
            return line.strip()[:200]
    return _first_line(letter_text)


def _first_line(letter_text: str) -> str:
    return next((line.strip() for line in letter_text.splitlines() if line.strip()), "")[:200]
