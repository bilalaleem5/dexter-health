"""Clinical follow-up process — the guided process for CREATE_TASK proposals.

Generic on purpose (works for "cardiology review in 3 months", "Bisoprolol
re-evaluation in 4 weeks", "GP lab recheck in 2 weeks", ...): it does not
know or care what the task is about, only that something needs a
*documented* follow-up contact (GP or specialist) by a due date.

Lifecycle (matches the house pattern in `src/features/weights/processes.py`):
1. `initialize_state` (runs once, via direct `execute()` at creation time —
   NOT via tick): stores the due date + a human-readable description in
   `process_state`. Starts ACTIVE.
2. While ACTIVE, a care-staff SuggestedAction moves it to WAITING once a
   reminder/task has actually been created for someone to act on (or
   straight to CLOSED if it turns out it was already done).
3. `check_completion` (driven by `tick.py`, repeatedly): closes the process
   once a new `encounters` entry (type gp/specialist, dated after the
   process started) appears for the resident — i.e. it waits for an actual
   recorded event, not just for time to pass, mirroring
   `VerificationMeasurementProcess` waiting for a new weight entry rather
   than auto-closing on a timer. If the due date passes with nothing
   recorded yet, it flags itself overdue (once) without closing, so a
   missed follow-up stays visible instead of silently expiring.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from src.core.domain.process import (
    Process,
    ProcessResult,
    ProcessStatus,
    SuggestedAction,
    register_process_class,
)


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _iso(value: datetime) -> str:
    return value.isoformat()


_QUALIFYING_ENCOUNTER_TYPES = {"gp", "specialist"}


class ClinicalFollowUpProcess(Process):
    """Waits for a documented GP/specialist follow-up contact by a due date."""

    def __init__(self):
        super().__init__(
            name="clinical_followup",
            description=(
                "Wartet auf einen dokumentierten Folgekontakt (Hausarzt/Facharzt) zur "
                "Umsetzung einer Entlassungsbrief-Empfehlung."
            ),
            suggested_actions=[
                SuggestedAction(
                    action_id="schedule_followup",
                    label="Aufgabe/Termin angelegt",
                    description=(
                        "Eine Erinnerung bzw. ein Termin für den Folgekontakt wurde angelegt; "
                        "warte auf den dokumentierten Kontakt."
                    ),
                    icon="task",
                    confirmation_required=True,
                    update_status_to=ProcessStatus.WAITING,
                ),
                SuggestedAction(
                    action_id="mark_done_now",
                    label="Bereits erledigt",
                    description="Der Folgekontakt liegt bereits vor / wurde anderweitig bestätigt.",
                    icon="check",
                    confirmation_required=True,
                    update_status_to=ProcessStatus.CLOSED,
                ),
            ],
        )

    def initialize_state(self, context: dict[str, Any]) -> ProcessResult:
        due_date = context.get("due_date")
        if due_date is None:
            return ProcessResult(
                success=False,
                next_status=ProcessStatus.FAILED,
                message="due_date fehlt im Kontext",
                data={"error": "missing due_date"},
            )
        now = context.get("now") or datetime.now(timezone.utc)
        self.process_state = {
            "due_date": _iso(due_date) if hasattr(due_date, "isoformat") else due_date,
            "task_description": context.get("task_description", ""),
            "proposal_target_entity": context.get("proposal_target_entity"),
            "started_at": _iso(now),
            "overdue_flagged": False,
        }
        return ProcessResult(
            success=True,
            next_status=ProcessStatus.ACTIVE,
            message=f"Folgeaufgabe angelegt: {self.process_state['task_description']}",
            data={"due_date": self.process_state["due_date"]},
        )

    def check_completion(self, context: dict[str, Any]) -> Optional[ProcessResult]:
        if self.status not in (ProcessStatus.ACTIVE, ProcessStatus.WAITING):
            return None

        resident_data_repo = context.get("services", {}).get("resident_data_repo")
        now = context.get("now")
        if resident_data_repo is None or now is None:
            return None  # invalid state — keep waiting rather than crash the tick

        started_at = _parse_dt(self.process_state["started_at"])
        due_date = _parse_dt(self.process_state["due_date"])

        resident_data = resident_data_repo.load(context["resident_id"]) or {}
        encounters_doc = resident_data.get("encounters") or {}
        encounters = encounters_doc.get("encounters") or []

        qualifying = [
            e
            for e in encounters
            if e.get("type") in _QUALIFYING_ENCOUNTER_TYPES and self._encounter_is_after(e, started_at)
        ]
        if qualifying:
            return ProcessResult(
                success=True,
                next_status=ProcessStatus.CLOSED,
                message=f"Folgekontakt dokumentiert: {qualifying[0].get('summary', qualifying[0].get('reason', ''))}",
                data={"outcome": "followup_documented", "encounter": qualifying[0]},
            )

        if now >= due_date and not self.process_state.get("overdue_flagged"):
            self.process_state["overdue_flagged"] = True
            return ProcessResult(
                success=True,
                next_status=self.status,  # stays open — overdue, not abandoned
                message=f"Fällig und noch nicht dokumentiert: {self.process_state['task_description']}",
                data={"outcome": "overdue", "escalate": True, "due_date": self.process_state["due_date"]},
            )

        return None  # not due yet, or already flagged overdue — no change

    @staticmethod
    def _encounter_is_after(encounter: dict, started_at: datetime) -> bool:
        date_str = encounter.get("date_from") or encounter.get("date")
        if not date_str:
            return False
        try:
            return _parse_dt(date_str) >= started_at
        except ValueError:
            return False


def create_clinical_followup(
    *, resident_id: str, due_date: datetime, task_description: str, target_entity: str, now: datetime, services: dict
) -> ClinicalFollowUpProcess:
    """Convenience constructor: builds + runs `initialize_state` once, the
    way `Pipeline.initialize_all` would for a single-stage pipeline."""
    process = ClinicalFollowUpProcess()
    context = {
        "services": services,
        "resident_id": resident_id,
        "now": now,
        "due_date": due_date,
        "task_description": task_description,
        "proposal_target_entity": target_entity,
    }
    process.execute(context)
    return process


# Make the process loadable from persisted suggestions (save/load roundtrip).
register_process_class("clinical_followup", ClinicalFollowUpProcess)
