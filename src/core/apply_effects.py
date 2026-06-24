"""Best-effort application of an *accepted* proposal to the resident's JSON
records.

Honest limitation, documented here rather than hidden: `Proposal` (the
fixed contract) carries no structured "new value" payload — only
category/action/target_entity plus free-text rationale/provenance. A
deterministic, schema-safe mutation is only possible where the existing
contract already gives us everything we need:

- STOP (any category): remove the matching entry by `target_entity` — no
  new value required, just "this no longer applies".
- DIAGNOSIS / ADD: the diagnosis name is recoverable from our OWN
  rationale template (we author both sides of that string), so it's safe
  to parse back out deterministically — unlike parsing a model's free text.

ADD / MODIFY for medications and care instructions are NOT auto-mutated:
doing so would mean silently reconstructing a dose/value from unstructured
text, which is exactly the kind of silent-guess this system is designed to
avoid. Those are acknowledged in the audit trail with a pointer to the
provenance quote, for a human to enter the actual value. See DECISIONS.md §1.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.core.domain.proposal import Proposal, ProposalAction, ProposalCategory
from src.core.repositories import _read_json, _write_json  # noqa: F401 (intentional reuse)


def _slugify(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in text.strip()]
    slug = "".join(keep)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


_DIAGNOSIS_NAME_RE = re.compile(r"Neue chronische Diagnose laut Brief: (.+?) \(")


def apply_accept(proposal: Proposal, data_dir: Path) -> str:
    """Apply an accepted proposal's effect, return a short description for
    the audit log. Never raises — a failed mutation is logged, not fatal."""
    try:
        if proposal.category == ProposalCategory.MEDICATION and proposal.action == ProposalAction.STOP:
            return _stop_medication(proposal, data_dir)
        if proposal.category == ProposalCategory.DIAGNOSIS and proposal.action == ProposalAction.ADD:
            return _add_diagnosis(proposal, data_dir)
        if proposal.category == ProposalCategory.TASK:
            return "Aufgabe akzeptiert — Folgeprozess wird separat angelegt (siehe Audit-Eintrag 'clinical_followup_created')."
        if proposal.action == ProposalAction.FLAG:
            return "Hinweis zur Kenntnis genommen — keine automatische Datenänderung für FLAG-Vorschläge."
    except Exception as e:  # best-effort — never let an apply failure break the decision endpoint
        return f"Anwendung fehlgeschlagen ({type(e).__name__}: {e}) — bitte manuell prüfen."

    return (
        f"Akzeptiert, aber nicht automatisch in die Akte übernommen: das Proposal-Schema "
        f"enthält keinen strukturierten Wert für '{proposal.action.value}' auf "
        f"'{proposal.target_entity}'. Bitte den Beleg-Zitat ({proposal.provenance.letter_quote!r}) "
        f"manuell in die Pflegedokumentation übertragen."
    )


def _stop_medication(proposal: Proposal, data_dir: Path) -> str:
    path = data_dir / "medication_plans" / f"{proposal.resident_id}.json"
    if not path.exists():
        return "Kein Medikationsplan vorhanden — nichts zu entfernen."
    plan = _read_json(path)
    removed = []
    for list_key in ("medications_planned", "medications_on_demand"):
        kept = []
        for m in plan.get(list_key, []):
            if _slugify(m["wirkstoff"]) == proposal.target_entity:
                removed.append(m["wirkstoff"])
            else:
                kept.append(m)
        plan[list_key] = kept
    if not removed:
        return f"Kein passender Eintrag für '{proposal.target_entity}' im Medikationsplan gefunden."
    _write_json(path, plan)
    return f"Aus dem Medikationsplan entfernt: {', '.join(removed)}."


def _add_diagnosis(proposal: Proposal, data_dir: Path) -> str:
    match = _DIAGNOSIS_NAME_RE.search(proposal.rationale)
    diagnose_name = match.group(1) if match else proposal.target_entity
    icd10_match = re.search(r"\(([^()]*)\)\.$", proposal.rationale)
    icd10 = icd10_match.group(1) if icd10_match else None
    if icd10 == "kein ICD-10 angegeben":
        icd10 = None

    path = data_dir / "diagnoses" / f"{proposal.resident_id}.json"
    doc: dict[str, Any] = _read_json(path) if path.exists() else {
        "resident_id": proposal.resident_id,
        "diagnoses": [],
    }
    if any(d["diagnose"].strip().lower() == diagnose_name.strip().lower() for d in doc["diagnoses"]):
        return f"Diagnose '{diagnose_name}' war bereits dokumentiert."
    doc["diagnoses"].append(
        {
            "diagnose": diagnose_name,
            "icd10": icd10,
            "internal_diagnose": proposal.target_entity,
            "source_letter_id": proposal.letter_id,
        }
    )
    _write_json(path, doc)
    return f"Neue Diagnose hinzugefügt: {diagnose_name}."
