"""Integration tests for the FastAPI service.

Uses a scripted LLM stub that branches on each analyzer's (letter-text-free)
system prompt, so a single fake letter can deterministically exercise all
three registered analyzers without cross-contamination — see the per-analyzer
unit tests for why a shared substring-keyed MockLLMClient can't do this
safely across multiple analyzers in one call.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import src.api.app as app_module
from src.core.llm.mock import MockLLMClient

RESIDENT_ID = "R900"
LETTER_ID = "letter_900"

_STOP_MED = {
    "raw_name": "Ramipril 5mg",
    "wirkstoff": "Ramipril",
    "strength_and_dosage": None,
    "status": "stopped",
    "internal_inconsistency_note": None,
    "letter_quote": "Ramipril wird abgesetzt.",
    "letter_section": "entlassmedikation",
    "conflicts_with_allergy": False,
    "allergy_conflict_reasoning": None,
}
_NEW_MED = {
    "raw_name": "Eliquis 5mg",
    "wirkstoff": "Apixaban",
    "strength_and_dosage": "5 mg 1-0-1",
    "status": "new",
    "internal_inconsistency_note": None,
    "letter_quote": "Apixaban 5 mg 1-0-1 neu angesetzt.",
    "letter_section": "entlassmedikation",
    "conflicts_with_allergy": False,
    "allergy_conflict_reasoning": None,
}
_ADMIN_TASK = {
    "slug": "gp_lab_recheck",
    "description": "Kreatinin-Kontrolle in 2 Wochen beim Hausarzt.",
    "due_in_days": 14,
    "requires_clinical_judgment": False,
    "letter_quote": "Laborkontrolle in 2 Wochen b. HA.",
    "letter_section": "procedere",
}


class _ScriptedLLM:
    model = "scripted-test"

    def __init__(self):
        self.usage_log: list[dict] = []
        self._mock = MockLLMClient()

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        self.usage_log.append({"model": self.model, "input_tokens": 0, "output_tokens": 0})
        if "clinical pharmacist" in system:
            return json.dumps({"medications": [_STOP_MED, _NEW_MED], "missing_attachment_note": None})
        if "for a nursing home" in system:
            return json.dumps({"diagnoses": [], "tasks": [_ADMIN_TASK], "care_instructions": []})
        return self._mock.complete(system, user, schema)  # stay_metadata gets the real default


@pytest.fixture
def client(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    letters_dir = tmp_path / "letters"
    (data_dir / "medication_plans").mkdir(parents=True)
    letters_dir.mkdir()

    (data_dir / "medication_plans" / f"{RESIDENT_ID}.json").write_text(
        json.dumps(
            {
                "resident_id": RESIDENT_ID,
                "medications_planned": [
                    {"wirkstoff": "Ramipril", "strength": "5", "unit": "mg", "dosage": "1-0-0"}
                ],
                "medications_on_demand": [],
            }
        )
    )
    (letters_dir / f"{LETTER_ID}.md").write_text(
        "Wir berichten über den Aufenthalt vom 01.06.2026 bis 08.06.2026. "
        "Ramipril wird abgesetzt. Apixaban 5 mg 1-0-1 neu angesetzt. "
        "Laborkontrolle in 2 Wochen b. HA."
    )
    (letters_dir / "index.json").write_text(
        json.dumps(
            [
                {
                    "letter_id": LETTER_ID,
                    "resident_id": RESIDENT_ID,
                    "hospital": "Test-Klinik",
                    "admission_date": "2026-06-01",
                    "discharge_date": "2026-06-08",
                    "file": f"letters/{LETTER_ID}.md",
                }
            ]
        )
    )

    monkeypatch.setattr(app_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(app_module, "LETTERS_DIR", letters_dir)
    monkeypatch.setattr(app_module, "get_llm_client", lambda: _ScriptedLLM())

    return TestClient(app_module.app), data_dir


def _ingest(client) -> dict:
    resp = client.post("/letters/ingest", json={"letter_id": LETTER_ID})
    assert resp.status_code == 202
    return resp.json()


def test_health():
    client = TestClient(app_module.app)
    assert client.get("/health").json() == {"status": "ok"}


def test_ingest_unknown_letter_is_404(client):
    c, _ = client
    resp = c.post("/letters/ingest", json={"letter_id": "does_not_exist"})
    assert resp.status_code == 404


def test_ingest_then_list_proposals(client):
    c, _ = client
    _ingest(c)

    resp = c.get(f"/residents/{RESIDENT_ID}/proposals")
    assert resp.status_code == 200
    body = resp.json()
    assert body["resident_id"] == RESIDENT_ID
    target_entities = {p["target_entity"] for p in body["proposals"]}
    assert "ramipril" in target_entities  # stop
    assert "apixaban" in target_entities  # new
    assert any(p["category"] == "task" for p in body["proposals"])


def test_ingest_is_idempotent(client):
    c, _ = client
    _ingest(c)
    first_count = len(c.get(f"/residents/{RESIDENT_ID}/proposals").json()["proposals"])

    _ingest(c)  # re-ingest the same letter
    second_count = len(c.get(f"/residents/{RESIDENT_ID}/proposals").json()["proposals"])
    assert first_count == second_count

    audit = c.get(f"/residents/{RESIDENT_ID}/audit").json()["entries"]
    assert any(e["event"] == "letter_ingest_skipped_duplicate" for e in audit)


def test_task_auto_applies_without_human_decision(client):
    c, _ = client
    _ingest(c)

    proposals = c.get(f"/residents/{RESIDENT_ID}/proposals").json()["proposals"]
    task = next(p for p in proposals if p["category"] == "task")
    assert task["routing"] == "auto_apply"

    audit = c.get(f"/residents/{RESIDENT_ID}/audit").json()["entries"]
    assert any(e["event"] == "proposal_auto_applied" and e["actor"] == "system" for e in audit)
    assert any(e["event"] == "clinical_followup_created" for e in audit)

    # A human trying to decide an already auto-applied proposal a second
    # time as "system" is allowed once (idempotent re-run), but a real human
    # decision after that should not be silently clobbered by another
    # system call — covered by test_cannot_redecide_a_human_decision.


def test_accept_stop_medication_actually_mutates_plan(client):
    c, data_dir = client
    _ingest(c)

    proposals = c.get(f"/residents/{RESIDENT_ID}/proposals").json()["proposals"]
    stop_proposal = next(p for p in proposals if p["target_entity"] == "ramipril")
    assert stop_proposal["routing"] == "human_confirm"

    resp = c.post(f"/proposals/{stop_proposal['proposal_id']}/decision", json={"decision": "accept"})
    assert resp.status_code == 200

    plan = json.loads((data_dir / "medication_plans" / f"{RESIDENT_ID}.json").read_text())
    assert all(m["wirkstoff"] != "Ramipril" for m in plan["medications_planned"])

    audit = c.get(f"/residents/{RESIDENT_ID}/audit").json()["entries"]
    assert any(e["event"] == "decision_accept" for e in audit)


def test_reject_does_not_mutate_plan(client):
    c, data_dir = client
    _ingest(c)
    proposals = c.get(f"/residents/{RESIDENT_ID}/proposals").json()["proposals"]
    new_med = next(p for p in proposals if p["target_entity"] == "apixaban")

    resp = c.post(f"/proposals/{new_med['proposal_id']}/decision", json={"decision": "reject", "comment": "GP will decide at next visit"})
    assert resp.status_code == 200
    assert resp.json()["decision"] == "reject"

    plan = json.loads((data_dir / "medication_plans" / f"{RESIDENT_ID}.json").read_text())
    assert all(m["wirkstoff"] != "Apixaban" for m in plan["medications_planned"])  # never added — contract has no structured value to add


def test_decide_unknown_proposal_is_404(client):
    c, _ = client
    resp = c.post("/proposals/does-not-exist/decision", json={"decision": "accept"})
    assert resp.status_code == 404


def test_cannot_redecide_a_human_decision(client):
    c, _ = client
    _ingest(c)
    proposals = c.get(f"/residents/{RESIDENT_ID}/proposals").json()["proposals"]
    new_med = next(p for p in proposals if p["target_entity"] == "apixaban")

    first = c.post(f"/proposals/{new_med['proposal_id']}/decision", json={"decision": "accept"})
    assert first.status_code == 200
    second = c.post(f"/proposals/{new_med['proposal_id']}/decision", json={"decision": "reject"})
    assert second.status_code == 409


def test_modify_requires_modified_payload(client):
    c, _ = client
    _ingest(c)
    proposals = c.get(f"/residents/{RESIDENT_ID}/proposals").json()["proposals"]
    new_med = next(p for p in proposals if p["target_entity"] == "apixaban")

    resp = c.post(f"/proposals/{new_med['proposal_id']}/decision", json={"decision": "modify"})
    assert resp.status_code == 422


def test_process_action_lifecycle_and_audit_ordering(client):
    c, _ = client
    _ingest(c)

    audit = c.get(f"/residents/{RESIDENT_ID}/audit").json()["entries"]
    followup_entry = next(e for e in audit if e["event"] == "clinical_followup_created")
    process_id = followup_entry["data"]["process_id"]

    resp = c.post(f"/processes/{process_id}/action", json={"action_id": "schedule_followup", "user_id": "nurse_anna"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "waiting"

    resp2 = c.post(f"/processes/{process_id}/action", json={"action_id": "mark_done_now", "user_id": "nurse_anna"})
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "closed"

    bad_action = c.post(f"/processes/{process_id}/action", json={"action_id": "nope", "user_id": "x"})
    assert bad_action.status_code == 422

    bad_process = c.post("/processes/R900|nope|clinical_followup/action", json={"action_id": "mark_done_now", "user_id": "x"})
    assert bad_process.status_code == 404

    audit_after = c.get(f"/residents/{RESIDENT_ID}/audit").json()["entries"]
    timestamps = [e["created_at"] for e in audit_after]
    assert timestamps == sorted(timestamps)
    assert any(e["event"].startswith("process:clinical_followup:") for e in audit_after)
