"""End-to-end lifecycle test: process initialization → tick → CLOSED.

Uses a self-contained tmp data dir (same layout as data/), never the real
data/ directory.
"""
import json

import pytest

from src.core.domain.enums import AlertStatus
from src.core.domain.process import Pipeline, ProcessNode, ProcessStatus
from src.core.domain.suggestion import AISuggestion
from src.core.repositories import SuggestionsRepository, VitalsRepository
from src.features.weights.processes import VerificationMeasurementProcess
from src.tick import run_tick

BASELINE = {"created_at": "2026-05-01T08:00:00+00:00", "weight": 78.0}
SUSPICIOUS = {"created_at": "2026-05-20T08:00:00+00:00", "weight": 72.0}


def _write_vitals(data_dir, weights):
    vitals_dir = data_dir / "vitals"
    vitals_dir.mkdir(parents=True, exist_ok=True)
    (vitals_dir / "R001.json").write_text(
        json.dumps({"resident_id": "R001", "weights": weights})
    )


def _context(data_dir):
    return {
        "services": {"vitals_repo": VitalsRepository(data_dir)},
        "resident_id": "R001",
        "now": None,
    }


@pytest.fixture
def data_dir(tmp_path):
    data_dir = tmp_path / "data"
    _write_vitals(data_dir, [BASELINE, SUSPICIOUS])
    return data_dir


def _saved_waiting_suggestion(data_dir) -> AISuggestion:
    """Create a suggestion whose process WAITs for a verification measurement."""
    process = VerificationMeasurementProcess()
    process.execute(_context(data_dir))
    assert process.status is ProcessStatus.ACTIVE

    # Care staff took the "create measurement task" action → WAITING.
    process._update_status(ProcessStatus.WAITING)

    suggestion = AISuggestion(
        suggestion_id="S001",
        resident_id="R001",
        alert_category="weight_analysis",
        alert_subcategory="loss_3m_red",
        alert_title="Gewichtsverlust",
        alert_level="high",
        reason="Auffälliger Gewichtsverlust",
        processes=[process],
    )
    SuggestionsRepository(data_dir).save(suggestion)
    return suggestion


def test_initialize_state_stores_measurements_and_goes_active(data_dir):
    process = VerificationMeasurementProcess()
    result = process.execute(_context(data_dir))

    assert result.success
    assert process.status is ProcessStatus.ACTIVE
    assert process.process_state["suspicious_measurement"] == SUSPICIOUS
    assert process.process_state["baseline_measurement"] == BASELINE


def test_initialize_state_fails_without_enough_weights(data_dir):
    _write_vitals(data_dir, [BASELINE])
    process = VerificationMeasurementProcess()
    result = process.execute(_context(data_dir))

    assert not result.success
    assert process.status is ProcessStatus.FAILED


def test_tick_closes_waiting_process_when_verification_confirms(data_dir):
    _saved_waiting_suggestion(data_dir)

    # A newer measurement arrives, close to the suspicious one → confirmed.
    verification = {"created_at": "2026-05-25T08:00:00+00:00", "weight": 72.4}
    _write_vitals(data_dir, [BASELINE, SUSPICIOUS, verification])

    changes = run_tick(data_dir, advance_days=3)
    assert changes == 1

    loaded = SuggestionsRepository(data_dir).list_for_resident("R001")[0]
    process = loaded.processes[0]
    assert process.status is ProcessStatus.CLOSED
    assert process.action_logs[-1].data["outcome"] == "confirmed"
    assert loaded.status is AlertStatus.ACTIVE  # confirmed loss keeps the suggestion open


def test_tick_closes_suggestion_on_measurement_error(data_dir):
    _saved_waiting_suggestion(data_dir)

    # The verification lands close to the baseline → suspicious value was wrong.
    verification = {"created_at": "2026-05-25T08:00:00+00:00", "weight": 77.6}
    _write_vitals(data_dir, [BASELINE, SUSPICIOUS, verification])

    run_tick(data_dir, advance_days=3)

    loaded = SuggestionsRepository(data_dir).list_for_resident("R001")[0]
    assert loaded.processes[0].status is ProcessStatus.CLOSED
    assert loaded.processes[0].action_logs[-1].data["outcome"] == "measurement_error"
    assert loaded.status is AlertStatus.CLOSED


def test_tick_promotes_pending_stage_when_dependency_closes(data_dir):
    """Two-stage pipeline: closing stage 1 lets the tick promote PENDING stage 2 to ACTIVE."""
    pipeline = Pipeline(
        name="two_stage_verification",
        description="Verifiziere, dann erneut verifizieren",
        process_graph={
            "verification_measurement": ProcessNode(
                process=VerificationMeasurementProcess(), depends_on=[]
            ),
            "second_stage": ProcessNode(
                process=VerificationMeasurementProcess(),
                depends_on=["verification_measurement"],
            ),
        },
    )
    stage1, stage2 = pipeline.initialize_all(_context(data_dir))
    assert stage1.status is ProcessStatus.ACTIVE
    assert stage2.status is ProcessStatus.PENDING

    suggestion = AISuggestion(
        suggestion_id="S002",
        resident_id="R001",
        alert_category="weight_analysis",
        alert_subcategory="loss_3m_red",
        alert_title="Gewichtsverlust",
        alert_level="high",
        reason="Auffälliger Gewichtsverlust",
        processes=[stage1, stage2],
    )
    SuggestionsRepository(data_dir).save(suggestion)

    # A verification measurement closes stage 1; the same tick promotes stage 2.
    verification = {"created_at": "2026-05-25T08:00:00+00:00", "weight": 72.4}
    _write_vitals(data_dir, [BASELINE, SUSPICIOUS, verification])
    run_tick(data_dir, advance_days=3)

    loaded = SuggestionsRepository(data_dir).list_for_resident("R001")[0]
    assert loaded.processes[0].status is ProcessStatus.CLOSED
    assert loaded.processes[1].status is ProcessStatus.ACTIVE
    assert any("dependencies met" in (log.description or "") for log in loaded.processes[1].action_logs)


def test_tick_without_new_measurement_changes_nothing(data_dir):
    _saved_waiting_suggestion(data_dir)

    changes = run_tick(data_dir, advance_days=3)
    assert changes == 0

    loaded = SuggestionsRepository(data_dir).list_for_resident("R001")[0]
    assert loaded.processes[0].status is ProcessStatus.WAITING
