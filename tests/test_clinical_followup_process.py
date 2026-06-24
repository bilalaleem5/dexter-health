"""Lifecycle tests for ClinicalFollowUpProcess — the guided process for
CREATE_TASK proposals (e.g. "GP re-evaluation in 4 weeks").

Mirrors the pattern in test_process_lifecycle.py: self-contained tmp data
dir, never the real data/ directory.
"""
import json
from datetime import datetime, timezone

import pytest

from src.core.domain.process import ProcessStatus
from src.core.domain.suggestion import AISuggestion
from src.core.repositories import ResidentDataRepository, SuggestionsRepository
from src.features.clinical_followup.process import (
    ClinicalFollowUpProcess,
    create_clinical_followup,
)
from src.tick import run_tick

DUE_DATE = datetime(2026, 6, 23, tzinfo=timezone.utc)
STARTED_AT = datetime(2026, 5, 26, tzinfo=timezone.utc)


def _write_encounters(data_dir, encounters):
    enc_dir = data_dir / "encounters"
    enc_dir.mkdir(parents=True, exist_ok=True)
    (enc_dir / "R003.json").write_text(json.dumps({"resident_id": "R003", "encounters": encounters}))


@pytest.fixture
def data_dir(tmp_path):
    data_dir = tmp_path / "data"
    _write_encounters(data_dir, [])
    return data_dir


def _context(data_dir, now=None):
    return {
        "services": {"resident_data_repo": ResidentDataRepository(data_dir)},
        "resident_id": "R003",
        "now": now or STARTED_AT,
        "due_date": DUE_DATE,
        "task_description": "Bisoprolol-Reevaluation durch die Hausärztin in 4 Wochen",
        "proposal_target_entity": "bisoprolol_reevaluation__due+28d",
    }


def test_initialize_state_stores_due_date_and_goes_active(data_dir):
    process = ClinicalFollowUpProcess()
    result = process.execute(_context(data_dir))

    assert result.success
    assert process.status is ProcessStatus.ACTIVE
    assert process.process_state["due_date"] == DUE_DATE.isoformat()
    assert "Bisoprolol" in process.process_state["task_description"]


def test_initialize_state_fails_without_due_date(data_dir):
    process = ClinicalFollowUpProcess()
    context = _context(data_dir)
    del context["due_date"]
    result = process.execute(context)

    assert not result.success
    assert process.status is ProcessStatus.FAILED


def test_check_completion_closes_when_qualifying_encounter_appears(data_dir):
    process = create_clinical_followup(
        resident_id="R003",
        due_date=DUE_DATE,
        task_description="Bisoprolol-Reevaluation",
        target_entity="bisoprolol_reevaluation__due+28d",
        now=STARTED_AT,
        services={"resident_data_repo": ResidentDataRepository(data_dir)},
    )
    assert process.status is ProcessStatus.ACTIVE

    _write_encounters(
        data_dir,
        [
            {
                "date_from": "2026-06-10",
                "date_to": "2026-06-10",
                "type": "gp",
                "facility": "Hausarztpraxis",
                "reason": "Bisoprolol-Reevaluation",
                "summary": "Bisoprolol nicht wieder angesetzt, weiter pausiert.",
            }
        ],
    )

    result = process.check_completion(_context(data_dir, now=datetime(2026, 6, 12, tzinfo=timezone.utc)))
    assert result is not None
    assert result.next_status is ProcessStatus.CLOSED
    assert result.data["outcome"] == "followup_documented"


def test_check_completion_ignores_encounter_before_process_started(data_dir):
    """An old GP visit that predates the discharge letter must not satisfy the follow-up."""
    process = create_clinical_followup(
        resident_id="R003",
        due_date=DUE_DATE,
        task_description="Bisoprolol-Reevaluation",
        target_entity="bisoprolol_reevaluation__due+28d",
        now=STARTED_AT,
        services={"resident_data_repo": ResidentDataRepository(data_dir)},
    )
    _write_encounters(
        data_dir,
        [{"date_from": "2021-09-13", "type": "specialist", "summary": "Alter Termin, irrelevant."}],
    )

    result = process.check_completion(_context(data_dir, now=datetime(2026, 6, 1, tzinfo=timezone.utc)))
    assert result is None
    assert process.status is ProcessStatus.ACTIVE


def test_check_completion_flags_overdue_once_without_closing(data_dir):
    process = create_clinical_followup(
        resident_id="R003",
        due_date=DUE_DATE,
        task_description="Bisoprolol-Reevaluation",
        target_entity="bisoprolol_reevaluation__due+28d",
        now=STARTED_AT,
        services={"resident_data_repo": ResidentDataRepository(data_dir)},
    )
    process._update_status(ProcessStatus.WAITING)

    past_due = datetime(2026, 7, 1, tzinfo=timezone.utc)
    result = process.check_completion(_context(data_dir, now=past_due))
    assert result is not None
    assert result.next_status is ProcessStatus.WAITING  # stays open, flagged overdue
    assert result.data["outcome"] == "overdue"
    process.apply_result(result, now=past_due)
    assert process.process_state["overdue_flagged"] is True

    # Ticking again must not re-flag (no duplicate noise in the audit trail).
    result_again = process.check_completion(_context(data_dir, now=past_due))
    assert result_again is None


def test_full_tick_roundtrip_closes_on_documented_followup(data_dir):
    process = create_clinical_followup(
        resident_id="R003",
        due_date=DUE_DATE,
        task_description="Bisoprolol-Reevaluation",
        target_entity="bisoprolol_reevaluation__due+28d",
        now=STARTED_AT,
        services={"resident_data_repo": ResidentDataRepository(data_dir)},
    )
    process._update_status(ProcessStatus.WAITING)

    suggestion = AISuggestion(
        suggestion_id="S900",
        resident_id="R003",
        alert_category="discharge_followup",
        alert_subcategory="clinical_followup",
        alert_title="Bisoprolol-Reevaluation",
        alert_level="medium",
        reason="Entlassungsbrief verweist auf Reevaluation durch die Hausärztin in 4 Wochen.",
        processes=[process],
    )
    SuggestionsRepository(data_dir).save(suggestion)

    # Latest timestamp in data is the encounter we add here (2026-06-10) + 3 days
    # advance must land after the due date for the closing test below; for this
    # roundtrip we just need a documented gp encounter newer than STARTED_AT.
    _write_encounters(
        data_dir,
        [
            {
                "date_from": "2026-06-10",
                "type": "gp",
                "summary": "Bisoprolol weiterhin pausiert, HA einverstanden.",
            }
        ],
    )

    changes = run_tick(data_dir, advance_days=1)
    assert changes == 1

    loaded = SuggestionsRepository(data_dir).list_for_resident("R003")[0]
    assert loaded.processes[0].status is ProcessStatus.CLOSED
    assert loaded.processes[0].action_logs[-1].data["outcome"] == "followup_documented"
