"""Weight verification process — reference feature for the process framework.

Lifecycle under tick.py: created via the pipeline factory (initialize_state
stores the suspicious + baseline measurement), then check_completion runs on
every tick and closes the process once a newer verification measurement
appears in `data/vitals/`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from src.core.config import get_effective_config
from src.core.domain.process import (
    Pipeline,
    Process,
    ProcessNode,
    ProcessResult,
    ProcessStatus,
    SuggestedAction,
    register_process_class,
)


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _sorted_weights(raw_weights: list[dict]) -> list[dict]:
    weights = [w for w in raw_weights if "created_at" in w and "weight" in w]
    return sorted(weights, key=lambda w: _parse_dt(w["created_at"]))


class VerificationMeasurementProcess(Process):
    """Request and evaluate a verification measurement after a suspicious weight."""

    def __init__(self):
        super().__init__(
            name="verification_measurement",
            description="Führe eine erneute Gewichtsmessung durch, um eine Fehlmessung auszuschließen.",
            suggested_actions=[
                SuggestedAction(
                    action_id="create_weight_measurement_task",
                    label="Erstelle eine Maßnahme zur Gewichtsmessung",
                    description="Das Gewicht soll erneut gemessen werden, um eine Fehlmessung auszuschließen.",
                    icon="nursing_intervention",
                    confirmation_required=True,
                    update_status_to=ProcessStatus.WAITING,
                ),
                SuggestedAction(
                    action_id="skip_verification",
                    label="Verifizierung überspringen",
                    description="Direkt zum nächsten Schritt gehen ohne neue Messung",
                    icon="skip",
                    confirmation_required=True,
                    update_status_to=ProcessStatus.IGNORED,
                ),
            ],
        )

    def initialize_state(self, context: dict[str, Any]) -> ProcessResult:
        """Store the suspicious (newest) and baseline (previous) measurement. Runs once."""
        weights = _sorted_weights(
            context["services"]["vitals_repo"].get_weights(context["resident_id"])
        )
        if len(weights) < 2:
            return ProcessResult(
                success=False,
                next_status=ProcessStatus.FAILED,
                message="Nicht genügend Gewichtsdaten für Vergleich",
                data={"error": "insufficient weight data"},
            )

        threshold_kg = get_effective_config()["weight"]["verification_threshold_kg"]
        self.process_state = {
            "suspicious_measurement": weights[-1],
            "baseline_measurement": weights[-2],
            "threshold_kg": threshold_kg,
        }
        return ProcessResult(
            success=True,
            next_status=ProcessStatus.ACTIVE,
            message="Warte auf Verifizierungsmessung",
            data={
                "suspicious_weight": weights[-1]["weight"],
                "baseline_weight": weights[-2]["weight"],
            },
        )

    def check_completion(self, context: dict[str, Any]) -> Optional[ProcessResult]:
        """Close the process once a measurement newer than the suspicious one exists."""
        if self.status not in (ProcessStatus.ACTIVE, ProcessStatus.WAITING):
            return None

        suspicious = self.process_state.get("suspicious_measurement", {})
        baseline = self.process_state.get("baseline_measurement", {})
        threshold_kg = self.process_state.get("threshold_kg", 2.5)
        if not suspicious.get("created_at") or baseline.get("weight") is None:
            return None  # invalid state — keep waiting rather than crash the tick

        weights = _sorted_weights(
            context["services"]["vitals_repo"].get_weights(context["resident_id"])
        )
        suspicious_at = _parse_dt(suspicious["created_at"])
        newer = [w for w in weights if _parse_dt(w["created_at"]) > suspicious_at]
        if not newer:
            return None  # no verification measurement yet

        verification_weight = newer[0]["weight"]
        distance_to_suspicious = abs(verification_weight - suspicious["weight"])
        distance_to_baseline = abs(verification_weight - baseline["weight"])
        distances = {
            "to_suspicious": distance_to_suspicious,
            "to_baseline": distance_to_baseline,
        }

        # Edge case: the verification shows the trend got even worse → escalate.
        if verification_weight < suspicious["weight"] - threshold_kg:
            return ProcessResult(
                success=True,
                next_status=ProcessStatus.CLOSED,
                message=f"Verschlechterung des Trends erkannt: {verification_weight} kg",
                data={
                    "verification_weight": verification_weight,
                    "outcome": "trend_worsening",
                    "escalate": True,
                    "distances": distances,
                },
            )

        if distance_to_suspicious < distance_to_baseline:
            # Verification CONFIRMS the suspicious measurement.
            return ProcessResult(
                success=True,
                next_status=ProcessStatus.CLOSED,
                message=f"Verifizierung bestätigt: {verification_weight} kg validiert "
                f"{suspicious['weight']} kg",
                data={
                    "verification_weight": verification_weight,
                    "outcome": "confirmed",
                    "distances": distances,
                },
            )

        # Verification is closer to the baseline → the suspicious value was a
        # measurement error, the whole suggestion can be dismissed.
        return ProcessResult(
            success=True,
            next_status=ProcessStatus.CLOSED,
            message=f"Messfehler erkannt: {verification_weight} kg näher an "
            f"{baseline['weight']} kg",
            data={
                "verification_weight": verification_weight,
                "outcome": "measurement_error",
                "distances": distances,
            },
            should_close_suggestion=True,
            suggestion_close_reason="Ursprüngliche Gewichtsmessung war fehlerhaft",
        )


def create_weight_verification_pipeline() -> Pipeline:
    """Single-stage pipeline; add dependent stages via ProcessNode.depends_on."""
    return Pipeline(
        name="weight_verification",
        description="Verifiziere eine auffällige Gewichtsmessung",
        process_graph={
            "verification_measurement": ProcessNode(
                process=VerificationMeasurementProcess(), depends_on=[]
            ),
        },
    )


# Make the process loadable from persisted suggestions (save/load roundtrip).
register_process_class("verification_measurement", VerificationMeasurementProcess)
