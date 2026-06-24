"""JSON-file repositories — the persistence layer of the assignment.

All resident data lives in per-resident JSON files under `data/` (schemas in
DATA_SCHEMA.md). Repositories are the only code that touches those files;
analyzers and processes receive repositories via the `context["services"]`
dict and stay free of I/O details.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .domain.process import deserialize_processes
from .domain.suggestion import AISuggestion

# resident_data key (as analyzers see it) → subdirectory under data/
RESIDENT_DATA_DIRS: dict[str, str] = {
    "medication_plan": "medication_plans",
    "allergies": "allergies",
    "diagnoses": "diagnoses",
    "care_notes": "care_notes",
    "drink_protocol": "drink_protocols",
    "vitals": "vitals",
    "wounds": "wounds",
    "medication_history": "medication_history",
    "labs": "labs",
    "assessments": "assessments",
    "encounters": "encounters",
    "fall_log": "fall_log",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class VitalsRepository:
    """Read access to `data/vitals/Rnnn.json`."""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)

    def get_weights(self, resident_id: str) -> list[dict]:
        """All weight entries ({created_at, weight}) for a resident, [] if none."""
        path = self.data_dir / "vitals" / f"{resident_id}.json"
        if not path.exists():
            return []
        return _read_json(path).get("weights", [])


class ResidentDataRepository:
    """Loads the full per-resident data dict that analyzers receive."""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)

    def list_residents(self) -> list[dict]:
        """All residents from `data/residents.json`."""
        path = self.data_dir / "residents.json"
        if not path.exists():
            return []
        return _read_json(path)

    def load(self, resident_id: str) -> dict[str, Any]:
        """Resident data keyed by type ("medication_plan", "allergies", ...).

        Keys are always present; the value is None when the file is missing.
        """
        resident_data: dict[str, Any] = {}
        for key, subdir in RESIDENT_DATA_DIRS.items():
            path = self.data_dir / subdir / f"{resident_id}.json"
            resident_data[key] = _read_json(path) if path.exists() else None
        return resident_data


class SuggestionsRepository:
    """Persistence for AISuggestions in `data/ai_suggestions/Rnnn.json`.

    File schema: {"resident_id": ..., "suggestions": [AISuggestion dicts]}.
    Saving upserts by suggestion_id, so re-saving the same suggestion never
    duplicates it. Attached processes survive the roundtrip via the process
    registry (see core/domain/process.py).
    """

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.suggestions_dir = self.data_dir / "ai_suggestions"

    def _path(self, resident_id: str) -> Path:
        return self.suggestions_dir / f"{resident_id}.json"

    def list_resident_ids(self) -> list[str]:
        """Resident ids that have a suggestions file."""
        if not self.suggestions_dir.exists():
            return []
        return sorted(p.stem for p in self.suggestions_dir.glob("*.json"))

    def list_for_resident(self, resident_id: str) -> list[AISuggestion]:
        path = self._path(resident_id)
        if not path.exists():
            return []

        suggestions = []
        for raw in _read_json(path).get("suggestions", []):
            process_dicts = raw.pop("processes", [])
            suggestion = AISuggestion.model_validate(raw)
            suggestion.processes = deserialize_processes(process_dicts)
            suggestions.append(suggestion)
        return suggestions

    def save(self, suggestion: AISuggestion) -> None:
        """Upsert one suggestion into its resident's file."""
        if not suggestion.suggestion_id or not suggestion.resident_id:
            raise ValueError("Suggestion needs suggestion_id and resident_id before saving")

        path = self._path(suggestion.resident_id)
        existing = _read_json(path).get("suggestions", []) if path.exists() else []

        replaced = False
        for i, raw in enumerate(existing):
            if raw.get("suggestion_id") == suggestion.suggestion_id:
                existing[i] = suggestion.to_dict()
                replaced = True
                break
        if not replaced:
            existing.append(suggestion.to_dict())

        _write_json(path, {"resident_id": suggestion.resident_id, "suggestions": existing})


class ProposalsRepository:
    """Persistence for `Proposal`s created by the API's single-letter
    `/letters/ingest` path — at `data/proposals/Rnnn.json`.

    This is deliberately separate from `proposals.json`, the flat artifact
    `run.py` writes for the batch deliverable: that file is a point-in-time
    export, not something the API should read back from (re-running the
    batch job shouldn't silently resurrect/duplicate API state, and vice
    versa). Idempotency key = letter_id: ingesting the same letter twice is
    a no-op (we also skip re-calling the LLM, not just re-saving — see
    DECISIONS.md §3).
    """

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.proposals_dir = self.data_dir / "proposals"

    def _path(self, resident_id: str) -> Path:
        return self.proposals_dir / f"{resident_id}.json"

    def _load_raw(self, resident_id: str) -> dict[str, Any]:
        path = self._path(resident_id)
        if not path.exists():
            return {"resident_id": resident_id, "proposals": [], "ingested_letters": {}}
        raw = _read_json(path)
        raw.setdefault("ingested_letters", {})
        return raw

    def is_letter_ingested(self, resident_id: str, letter_id: str) -> bool:
        return letter_id in self._load_raw(resident_id)["ingested_letters"]

    def save_many(self, resident_id: str, proposals: list, letter_id: str, ingested_at) -> None:
        raw = self._load_raw(resident_id)
        by_id = {p["proposal_id"]: p for p in raw["proposals"]}
        for proposal in proposals:
            by_id[proposal.proposal_id] = proposal.model_dump(mode="json")
        raw["proposals"] = list(by_id.values())
        raw["ingested_letters"][letter_id] = ingested_at.isoformat()
        _write_json(self._path(resident_id), raw)

    def list_for_resident(self, resident_id: str):
        from .domain.proposal import Proposal

        raw = self._load_raw(resident_id)
        return [Proposal.model_validate(p) for p in raw["proposals"]]

    def find(self, proposal_id: str):
        """Scan all resident proposal files for one proposal_id.

        Fine at this dataset's scale (a handful of residents); a real
        deployment would index proposal_id -> resident_id directly.
        """
        from .domain.proposal import Proposal

        if not self.proposals_dir.exists():
            return None
        for path in sorted(self.proposals_dir.glob("*.json")):
            raw = _read_json(path)
            for p in raw.get("proposals", []):
                if p.get("proposal_id") == proposal_id:
                    return raw["resident_id"], Proposal.model_validate(p)
        return None


class DecisionsRepository:
    """Append-only human/physician decisions per resident, at
    `data/decisions/Rnnn.json`. Proposals are immutable once created
    (see `Proposal` in core/domain/proposal.py) — a decision is a separate
    record referencing a proposal_id, never a mutation of the proposal
    itself, so the original AI output stays intact for audit purposes."""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.decisions_dir = self.data_dir / "decisions"

    def _path(self, resident_id: str) -> Path:
        return self.decisions_dir / f"{resident_id}.json"

    def append(self, resident_id: str, decision: dict[str, Any]) -> None:
        path = self._path(resident_id)
        existing = _read_json(path).get("decisions", []) if path.exists() else []
        existing.append(decision)
        _write_json(path, {"resident_id": resident_id, "decisions": existing})

    def list_for_resident(self, resident_id: str) -> list[dict[str, Any]]:
        path = self._path(resident_id)
        if not path.exists():
            return []
        return _read_json(path).get("decisions", [])

    def find(self, proposal_id: str) -> dict[str, Any] | None:
        if not self.decisions_dir.exists():
            return None
        for path in self.decisions_dir.glob("*.json"):
            for d in _read_json(path).get("decisions", []):
                if d.get("proposal_id") == proposal_id:
                    return d
        return None


class AuditRepository:
    """Append-only, human-readable audit trail per resident, at
    `data/audit/Rnnn.json`. Complements (does not replace) the structured
    decisions log and each guided process's own `action_logs` — the API's
    `/residents/{id}/audit` endpoint merges all three by timestamp."""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.audit_dir = self.data_dir / "audit"

    def _path(self, resident_id: str) -> Path:
        return self.audit_dir / f"{resident_id}.json"

    def append(
        self,
        resident_id: str,
        *,
        event: str,
        actor: str | None = None,
        data: dict[str, Any] | None = None,
        created_at=None,
    ) -> None:
        from datetime import datetime, timezone

        path = self._path(resident_id)
        existing = _read_json(path).get("entries", []) if path.exists() else []
        existing.append(
            {
                "created_at": (created_at or datetime.now(timezone.utc)).isoformat(),
                "event": event,
                "actor": actor,
                "data": data,
            }
        )
        _write_json(path, {"resident_id": resident_id, "entries": existing})

    def list_for_resident(self, resident_id: str) -> list[dict[str, Any]]:
        path = self._path(resident_id)
        if not path.exists():
            return []
        return _read_json(path).get("entries", [])
