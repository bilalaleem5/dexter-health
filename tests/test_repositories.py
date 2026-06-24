"""Tests for the JSON-file repositories."""
import pytest

from src.core.domain.process import (
    Process,
    ProcessResult,
    ProcessStatus,
    register_process_class,
)
from src.core.domain.suggestion import AISuggestion
from src.core.repositories import SuggestionsRepository, VitalsRepository


class DummyProcess(Process):
    """Minimal process so the roundtrip test doesn't depend on a feature module."""

    def __init__(self):
        super().__init__(name="dummy_process", description="test only")

    def initialize_state(self, context) -> ProcessResult:
        return ProcessResult(success=True, next_status=ProcessStatus.ACTIVE, message="ok")


register_process_class("dummy_process", DummyProcess)


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


def _make_suggestion() -> AISuggestion:
    process = DummyProcess()
    process.execute({"now": None})
    process.process_state = {"some": "state", "threshold_kg": 2.5}
    return AISuggestion(
        suggestion_id="S001",
        resident_id="R001",
        alert_category="weight_analysis",
        alert_subcategory="loss_3m_red",
        alert_title="Gewichtsverlust",
        alert_level="high",
        reason="8% Verlust in 3 Monaten",
        processes=[process],
    )


def test_save_load_roundtrip_including_process(data_dir):
    repo = SuggestionsRepository(data_dir)
    repo.save(_make_suggestion())

    loaded = repo.list_for_resident("R001")
    assert len(loaded) == 1
    suggestion = loaded[0]
    assert suggestion.suggestion_id == "S001"
    assert suggestion.alert_level == "high"

    # The process came back as a real Process instance via the registry,
    # with status and state restored.
    assert len(suggestion.processes) == 1
    process = suggestion.processes[0]
    assert isinstance(process, DummyProcess)
    assert process.status is ProcessStatus.ACTIVE
    assert process.process_state == {"some": "state", "threshold_kg": 2.5}


def test_double_save_same_id_yields_one_record(data_dir):
    repo = SuggestionsRepository(data_dir)
    first = _make_suggestion()
    repo.save(first)

    updated = _make_suggestion()
    updated.alert_level = "medium"
    repo.save(updated)

    loaded = repo.list_for_resident("R001")
    assert len(loaded) == 1
    assert loaded[0].alert_level == "medium"


def test_save_requires_ids(data_dir):
    repo = SuggestionsRepository(data_dir)
    suggestion = _make_suggestion()
    suggestion.suggestion_id = None
    with pytest.raises(ValueError):
        repo.save(suggestion)


def test_vitals_repository_reads_weights(data_dir):
    vitals_dir = data_dir / "vitals"
    vitals_dir.mkdir(parents=True)
    (vitals_dir / "R001.json").write_text(
        '{"resident_id": "R001", "weights": [{"created_at": "2026-05-01T08:00:00+00:00", "weight": 72.5}]}'
    )

    repo = VitalsRepository(data_dir)
    assert repo.get_weights("R001") == [{"created_at": "2026-05-01T08:00:00+00:00", "weight": 72.5}]
    assert repo.get_weights("R999") == []
