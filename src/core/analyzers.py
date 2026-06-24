"""Single source of truth for which analyzers run over a letter.

Both `src/run.py` (batch) and `src/api/app.py` (single-letter ingest) import
this instead of keeping their own lists, so the two entrypoints can never
drift out of sync.
"""
from __future__ import annotations

from src.features.followup_care.analyzer import FollowUpCareAnalyzer
from src.features.medication_reconciliation.analyzer import MedicationReconciliationAnalyzer
from src.features.stay_metadata.analyzer import StayMetadataAnalyzer


def get_registered_analyzers() -> list:
    """New instance per call — analyzers are stateless, but this avoids any
    accidental shared mutable state across a long-lived API process."""
    return [
        StayMetadataAnalyzer(),
        MedicationReconciliationAnalyzer(),
        FollowUpCareAnalyzer(),
    ]
