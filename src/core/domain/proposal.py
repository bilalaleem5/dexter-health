"""The proposals.json contract.

Every analyzer emits `Proposal` objects; `src/run.py` merges them into one
validated `ProposalsOutput`. Field names and enum values are part of the
assignment contract — do not change them (your eval depends on them).
"""
from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ProposalCategory(StrEnum):
    MEDICATION = "medication"
    DIAGNOSIS = "diagnosis"
    WOUND = "wound"
    BEHAVIOR = "behavior"
    FLUID_MANAGEMENT = "fluid_management"
    TASK = "task"
    OTHER = "other"


class ProposalAction(StrEnum):
    ADD = "add"
    STOP = "stop"
    MODIFY = "modify"
    VERIFY = "verify"
    CREATE_TASK = "create_task"
    FLAG = "flag"


class Routing(StrEnum):
    AUTO_APPLY = "auto_apply"
    HUMAN_CONFIRM = "human_confirm"
    HARD_STOP_PHYSICIAN = "hard_stop_physician"
    INFO_ONLY = "info_only"


class Severity(StrEnum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


class LetterSection(StrEnum):
    DIAGNOSES = "diagnoses"
    ANAMNESE = "anamnese"
    BEFUND = "befund"
    DIAGNOSTIK = "diagnostik"
    EPIKRISE = "epikrise"
    ENTLASSMEDIKATION = "entlassmedikation"
    PROCEDERE = "procedere"
    OTHER = "other"


class Provenance(BaseModel):
    """Where a proposal comes from: verbatim letter quote + optional DB reference."""

    letter_quote: str
    letter_section: LetterSection
    db_reference: str | None = None


class Proposal(BaseModel):
    proposal_id: str
    resident_id: str
    letter_id: str
    category: ProposalCategory
    action: ProposalAction
    target_entity: str
    routing: Routing
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    provenance: Provenance
    rationale: str


class CostLogEntry(BaseModel):
    letter_id: str
    model: str
    input_tokens: int
    output_tokens: int


class ProposalsOutput(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    proposals: list[Proposal]
    cost_log: list[CostLogEntry]


def make_proposal_id(
    resident_id: str,
    letter_id: str,
    category: ProposalCategory,
    action: ProposalAction,
    target_entity: str,
) -> str:
    """Deterministic proposal id — identical inputs yield the same id, so re-runs are idempotent."""
    key = f"{resident_id}|{letter_id}|{category}|{action}|{target_entity}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
