"""FastAPI service for the reconciliation system.

`GET /health` always worked; everything below is the implementation of the
five TODO stubs. The request/response models from the original skeleton are
unchanged — they are the API contract.

Run with: make api  (uvicorn src.api.app:app --reload)

Design notes (see DECISIONS.md §3 for the prose version):

- Idempotency key for `/letters/ingest` is `letter_id`. Re-ingesting an
  already-ingested letter is a true no-op: we don't just dedupe the
  resulting proposals (those already dedupe via deterministic proposal_id),
  we skip re-calling the LLM at all, recorded via
  `ProposalsRepository.is_letter_ingested`.
- ROUTING is operationalized, not just a label: AUTO_APPLY proposals are
  immediately self-decided ("accept", actor="system") right at ingest —
  no human ever has to touch them. HUMAN_CONFIRM / HARD_STOP_PHYSICIAN /
  INFO_ONLY proposals sit in the queue until a human calls
  `/proposals/{id}/decision`.
- A `Proposal` is immutable once created; a decision is a separate,
  append-only record referencing it (`DecisionsRepository`), so the
  original AI output is never silently rewritten.
- Process identification for `/processes/{id}/action`: we mint
  `"{resident_id}|{suggestion_id}|{process_name}"` ourselves when a
  follow-up process is created (in `_attach_clinical_followup`), and hand
  it back via the audit trail (`clinical_followup_created` event) — there
  was no existing identifier to reuse, so this is our own design decision,
  kept resident-prefixed so a lookup never needs to scan every resident's
  suggestion file.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.core.analyzers import get_registered_analyzers
from src.core.apply_effects import apply_accept
from src.core.decision_policy import parse_due_offset
from src.core.domain.process import ActionType, ProcessResult
from src.core.domain.proposal import Proposal, ProposalAction, ProposalCategory, Routing
from src.core.domain.suggestion import AISuggestion
from src.core.llm.client import get_llm_client
from src.core.repositories import (
    AuditRepository,
    DecisionsRepository,
    ProposalsRepository,
    ResidentDataRepository,
    SuggestionsRepository,
)
from src.features.clinical_followup.process import create_clinical_followup

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
LETTERS_DIR = Path(os.environ.get("LETTERS_DIR", "letters"))

app = FastAPI(title="dexter health — discharge letter reconciliation", version="0.1.0")


# ---------------------------------------------------------------------------
# Models (complete — the API contract)
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class IngestLetterRequest(BaseModel):
    letter_id: str


class IngestLetterResponse(BaseModel):
    letter_id: str
    status: Literal["accepted"] = "accepted"


class ResidentProposalsResponse(BaseModel):
    resident_id: str
    proposals: list[Proposal]


class ProposalDecisionRequest(BaseModel):
    decision: Literal["accept", "reject", "modify"]
    comment: str | None = None
    modified_payload: dict[str, Any] | None = None  # required when decision == "modify"


class ProposalDecisionResponse(BaseModel):
    proposal_id: str
    decision: Literal["accept", "reject", "modify"]
    decided_at: datetime


class ProcessActionRequest(BaseModel):
    action_id: str
    user_id: str
    input: dict[str, Any] | None = None


class ProcessActionResponse(BaseModel):
    process_id: str
    status: str
    message: str


class AuditEntry(BaseModel):
    created_at: datetime
    event: str
    actor: str | None = None  # None for system events
    data: dict[str, Any] | None = None


class ResidentAuditResponse(BaseModel):
    resident_id: str
    entries: list[AuditEntry]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _letters_index() -> list[dict]:
    path = LETTERS_DIR / "index.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _letter_lookup(letter_id: str) -> dict | None:
    return next((e for e in _letters_index() if e["letter_id"] == letter_id), None)


def _process_id(resident_id: str, suggestion_id: str, process_name: str) -> str:
    return f"{resident_id}|{suggestion_id}|{process_name}"


def _attach_clinical_followup(
    *, resident_id: str, proposal: Proposal, now: datetime, decided_by: str | None
) -> dict[str, Any] | None:
    """For an accepted CREATE_TASK proposal whose target_entity encodes a due
    offset, create + persist the guided ClinicalFollowUpProcess. Returns the
    audit payload to log, or None if this proposal doesn't encode a task."""
    parsed = parse_due_offset(proposal.target_entity)
    if parsed is None:
        return None
    slug, days = parsed

    letter_entry = _letter_lookup(proposal.letter_id)
    base_date = now
    if letter_entry and letter_entry.get("discharge_date"):
        try:
            base_date = datetime.fromisoformat(letter_entry["discharge_date"]).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    due_date = base_date + timedelta(days=days)

    process = create_clinical_followup(
        resident_id=resident_id,
        due_date=due_date,
        task_description=proposal.rationale,
        target_entity=proposal.target_entity,
        now=now,
        services={"resident_data_repo": ResidentDataRepository(DATA_DIR)},
    )
    suggestion_id = f"followup-{proposal.proposal_id[:16]}"
    suggestion = AISuggestion(
        suggestion_id=suggestion_id,
        resident_id=resident_id,
        alert_category="discharge_followup",
        alert_subcategory=slug,
        alert_title=proposal.rationale[:80],
        alert_level="medium",
        reason=f"Aus akzeptiertem Vorschlag {proposal.proposal_id} (Brief {proposal.letter_id}).",
        processes=[process],
    )
    SuggestionsRepository(DATA_DIR).save(suggestion)

    process_id = _process_id(resident_id, suggestion_id, process.name)
    return {
        "process_id": process_id,
        "due_date": due_date.isoformat(),
        "task_description": proposal.rationale,
        "decided_by": decided_by,
    }


def _auto_apply_if_needed(*, resident_id: str, proposal: Proposal, now: datetime) -> None:
    """AUTO_APPLY proposals never wait for a human: self-decide right away."""
    if proposal.routing is not Routing.AUTO_APPLY:
        return

    decisions_repo = DecisionsRepository(DATA_DIR)
    audit_repo = AuditRepository(DATA_DIR)

    effect = apply_accept(proposal, DATA_DIR)
    decisions_repo.append(
        resident_id,
        {
            "proposal_id": proposal.proposal_id,
            "resident_id": resident_id,
            "decision": "accept",
            "comment": "auto_applied — routing=auto_apply, no human review required",
            "modified_payload": None,
            "decided_at": now.isoformat(),
            "decided_by": "system",
        },
    )
    audit_repo.append(
        resident_id,
        event="proposal_auto_applied",
        actor="system",
        data={"proposal_id": proposal.proposal_id, "category": proposal.category.value, "effect": effect},
        created_at=now,
    )
    if proposal.category is ProposalCategory.TASK and proposal.action is ProposalAction.CREATE_TASK:
        followup = _attach_clinical_followup(resident_id=resident_id, proposal=proposal, now=now, decided_by="system")
        if followup:
            audit_repo.append(
                resident_id, event="clinical_followup_created", actor="system", data=followup, created_at=now
            )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/letters/ingest", response_model=IngestLetterResponse, status_code=202)
def ingest_letter(body: IngestLetterRequest) -> IngestLetterResponse:
    entry = _letter_lookup(body.letter_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown letter_id: {body.letter_id!r}")

    resident_id = entry["resident_id"]
    proposals_repo = ProposalsRepository(DATA_DIR)
    audit_repo = AuditRepository(DATA_DIR)
    now = datetime.now(timezone.utc)

    if proposals_repo.is_letter_ingested(resident_id, body.letter_id):
        audit_repo.append(
            resident_id,
            event="letter_ingest_skipped_duplicate",
            data={"letter_id": body.letter_id},
            created_at=now,
        )
        return IngestLetterResponse(letter_id=body.letter_id, status="accepted")

    letter_path = LETTERS_DIR / Path(entry["file"]).name
    if not letter_path.exists():
        raise HTTPException(status_code=500, detail=f"Letter file missing on disk: {letter_path}")
    try:
        letter_text = letter_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        letter_text = letter_path.read_text(encoding="latin-1")

    resident_data = ResidentDataRepository(DATA_DIR).load(resident_id)
    llm = get_llm_client()

    proposals: list[Proposal] = []
    for analyzer in get_registered_analyzers():
        proposals.extend(analyzer.analyze(letter_text, entry, resident_data, llm))

    proposals_repo.save_many(resident_id, proposals, body.letter_id, ingested_at=now)
    audit_repo.append(
        resident_id,
        event="letter_ingested",
        data={"letter_id": body.letter_id, "proposal_count": len(proposals)},
        created_at=now,
    )
    for proposal in proposals:
        _auto_apply_if_needed(resident_id=resident_id, proposal=proposal, now=now)

    return IngestLetterResponse(letter_id=body.letter_id, status="accepted")


@app.get("/residents/{resident_id}/proposals", response_model=ResidentProposalsResponse)
def get_resident_proposals(resident_id: str) -> ResidentProposalsResponse:
    proposals = ProposalsRepository(DATA_DIR).list_for_resident(resident_id)
    return ResidentProposalsResponse(resident_id=resident_id, proposals=proposals)


@app.post("/proposals/{proposal_id}/decision", response_model=ProposalDecisionResponse)
def decide_proposal(proposal_id: str, body: ProposalDecisionRequest) -> ProposalDecisionResponse:
    found = ProposalsRepository(DATA_DIR).find(proposal_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"Unknown proposal_id: {proposal_id!r}")
    resident_id, proposal = found

    if body.decision == "modify" and body.modified_payload is None:
        raise HTTPException(status_code=422, detail="modified_payload is required when decision == 'modify'")

    existing = DecisionsRepository(DATA_DIR).find(proposal_id)
    if existing is not None and existing.get("decided_by") != "system":
        # A human already decided this one — decisions are append-only audit
        # records, not something a second call should silently overwrite.
        raise HTTPException(
            status_code=409,
            detail=f"Proposal {proposal_id!r} was already decided ({existing['decision']!r}) at {existing['decided_at']}",
        )

    now = datetime.now(timezone.utc)
    decisions_repo = DecisionsRepository(DATA_DIR)
    audit_repo = AuditRepository(DATA_DIR)

    if body.decision == "accept":
        effect = apply_accept(proposal, DATA_DIR)
    elif body.decision == "reject":
        effect = "Vorschlag abgelehnt — keine Änderung vorgenommen."
    else:
        effect = (
            "Mit geänderten Werten akzeptiert — manuelle Übernahme erforderlich (siehe "
            "modified_payload); das Proposal-Schema speichert keinen strukturierten Wert."
        )

    decisions_repo.append(
        resident_id,
        {
            "proposal_id": proposal_id,
            "resident_id": resident_id,
            "decision": body.decision,
            "comment": body.comment,
            "modified_payload": body.modified_payload,
            "decided_at": now.isoformat(),
            "decided_by": "human",
        },
    )
    audit_repo.append(
        resident_id,
        event=f"proposal_{body.decision}",
        data={"proposal_id": proposal_id, "category": proposal.category.value, "effect": effect, "comment": body.comment},
        created_at=now,
    )

    if (
        body.decision in ("accept", "modify")
        and proposal.category is ProposalCategory.TASK
        and proposal.action is ProposalAction.CREATE_TASK
    ):
        followup = _attach_clinical_followup(resident_id=resident_id, proposal=proposal, now=now, decided_by="human")
        if followup:
            audit_repo.append(resident_id, event="clinical_followup_created", data=followup, created_at=now)

    return ProposalDecisionResponse(proposal_id=proposal_id, decision=body.decision, decided_at=now)


@app.post("/processes/{process_id}/action", response_model=ProcessActionResponse)
def act_on_process(process_id: str, body: ProcessActionRequest) -> ProcessActionResponse:
    parts = process_id.split("|", 2)
    if len(parts) != 3:
        raise HTTPException(status_code=404, detail=f"Unknown process_id format: {process_id!r}")
    resident_id, suggestion_id, process_name = parts

    suggestions_repo = SuggestionsRepository(DATA_DIR)
    suggestion = next(
        (s for s in suggestions_repo.list_for_resident(resident_id) if s.suggestion_id == suggestion_id), None
    )
    if suggestion is None:
        raise HTTPException(status_code=404, detail=f"Unknown suggestion for process_id: {process_id!r}")
    process = next((p for p in suggestion.processes if p.name == process_name), None)
    if process is None:
        raise HTTPException(status_code=404, detail=f"Unknown process for process_id: {process_id!r}")

    action = next((a for a in process.suggested_actions if a.action_id == body.action_id), None)
    if action is None:
        raise HTTPException(
            status_code=422, detail=f"Unknown action_id {body.action_id!r} for process {process_name!r}"
        )

    now = datetime.now(timezone.utc)
    next_status = action.update_status_to or process.status
    process.apply_result(
        ProcessResult(
            success=True,
            next_status=next_status,
            message=f"Aktion '{action.label}' ausgeführt.",
            data={"action_id": action.action_id, "input": body.input},
        ),
        now=now,
    )
    process.add_action_log(
        ActionType.USER_ACTION_TAKEN,
        description=f"{body.user_id}: {action.label}",
        user_id=body.user_id,
        data={"action_id": action.action_id, "input": body.input},
        now=now,
    )
    suggestions_repo.save(suggestion)

    AuditRepository(DATA_DIR).append(
        resident_id,
        event="process_action",
        actor=body.user_id,
        data={"process_id": process_id, "action_id": action.action_id},
        created_at=now,
    )

    return ProcessActionResponse(
        process_id=process_id, status=process.status.value, message=f"Aktion '{action.label}' ausgeführt."
    )


@app.get("/residents/{resident_id}/audit", response_model=ResidentAuditResponse)
def get_resident_audit(resident_id: str) -> ResidentAuditResponse:
    entries: list[AuditEntry] = []

    for e in AuditRepository(DATA_DIR).list_for_resident(resident_id):
        entries.append(AuditEntry(created_at=e["created_at"], event=e["event"], actor=e.get("actor"), data=e.get("data")))

    for d in DecisionsRepository(DATA_DIR).list_for_resident(resident_id):
        entries.append(
            AuditEntry(
                created_at=d["decided_at"],
                event=f"decision_{d['decision']}",
                actor=d.get("decided_by"),
                data={"proposal_id": d["proposal_id"], "comment": d.get("comment")},
            )
        )

    for suggestion in SuggestionsRepository(DATA_DIR).list_for_resident(resident_id):
        for process in suggestion.processes:
            for log in process.action_logs:
                action_type = log.action_type.value if hasattr(log.action_type, "value") else log.action_type
                entries.append(
                    AuditEntry(
                        created_at=log.created_at,
                        event=f"process:{process.name}:{action_type}",
                        actor=log.user_id,
                        data=log.data,
                    )
                )

    entries.sort(key=lambda e: e.created_at)
    return ResidentAuditResponse(resident_id=resident_id, entries=entries)
