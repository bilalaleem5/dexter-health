"""Eval harness: compares proposals.json against eval/labels.json.

Run with: make eval
  (equivalent to: python eval/run_eval.py --proposals proposals.json --labels eval/labels.json)

Matching rule (documented, not just implemented):
  A proposal matches a labeled case if (resident_id, letter_id, category)
  match exactly AND (action matches exactly, when the case specifies one)
  AND `expected_target_contains` is a case-insensitive substring of the
  proposal's target_entity.

  Substring matching on target_entity (not exact-match) is deliberate for
  TASK proposals: `slug` is free text the LLM authors itself (see
  src/features/followup_care/analyzer.py) — an exact match would really be
  testing our own guess at the LLM's phrasing, not the underlying clinical
  finding. For MEDICATION/DIAGNOSIS/WOUND/FLUID_MANAGEMENT proposals,
  target_entity is deterministically slugified from a name the LLM also
  free-texts (a drug or diagnosis name) — same reasoning applies.

  For NEGATIVE cases, a "match" is exactly what must NOT happen: any
  matching proposal is a false positive.

Routing correctness is reported as a second, independent dimension: a
positive case can be "found" (a matching proposal exists) but with the
wrong `routing` — we count that separately rather than silently folding it
into pass/fail, because getting the existence right but the safety routing
wrong is its own kind of failure for a system whose whole point is routing.

What this script does NOT do (see EVAL.md "Honest notes"):
  it cannot tell you about proposals that have no basis in the letter at
  all ("hallucinated", in the strict sense) unless they happen to collide
  with a negative-case target_entity we thought to write down. A full
  hallucination audit needs a human to read every unlabeled proposal
  against its source letter — we did that for a spot-check sample (see
  EVAL.md), not exhaustively. This script reports the *count* of unlabeled
  proposals plus a sample, so that audit is at least easy to do by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _matches(proposal: dict, case: dict) -> bool:
    if proposal["resident_id"] != case["resident_id"]:
        return False
    if proposal["letter_id"] != case["letter_id"]:
        return False
    if proposal["category"] != case["expected_category"]:
        return False
    if case.get("expected_action") and proposal["action"] != case["expected_action"]:
        return False
    needle = case["expected_target_contains"].lower()
    return needle in proposal["target_entity"].lower()


def evaluate(proposals: list[dict], cases: list[dict]) -> dict:
    found: list[str] = []
    missed: list[str] = []
    wrong_routing: list[dict] = []
    false_positives: list[dict] = []

    for case in cases:
        matches = [p for p in proposals if _matches(p, case)]

        if case["is_negative"]:
            if matches:
                false_positives.append({"case_id": case["case_id"], "matched_proposal_ids": [m["proposal_id"] for m in matches]})
            continue

        if not matches:
            missed.append(case["case_id"])
            continue

        found.append(case["case_id"])
        expected_routing = case.get("expected_routing")
        if expected_routing and not any(m["routing"] == expected_routing for m in matches):
            wrong_routing.append(
                {
                    "case_id": case["case_id"],
                    "expected_routing": expected_routing,
                    "actual_routing": sorted({m["routing"] for m in matches}),
                }
            )

    positive_cases = [c for c in cases if not c["is_negative"]]
    negative_cases = [c for c in cases if c["is_negative"]]

    # Rough hallucination proxy: proposals for a (resident, letter, category)
    # combination we never even labeled a case for. NOT a hallucination
    # count on its own — see module docstring — just a worklist for manual
    # spot-checking.
    labeled_buckets = {(c["resident_id"], c["letter_id"], c["expected_category"]) for c in cases}
    unlabeled_proposals = [
        p for p in proposals if (p["resident_id"], p["letter_id"], p["category"]) not in labeled_buckets
    ]

    return {
        "labeled_positive": len(positive_cases),
        "labeled_negative": len(negative_cases),
        "found": found,
        "missed": missed,
        "wrong_routing": wrong_routing,
        "false_positives_on_negative_cases": false_positives,
        "unlabeled_proposals_count": len(unlabeled_proposals),
        "unlabeled_proposals_sample": [
            {
                "proposal_id": p["proposal_id"],
                "resident_id": p["resident_id"],
                "letter_id": p["letter_id"],
                "category": p["category"],
                "target_entity": p["target_entity"],
            }
            for p in unlabeled_proposals[:10]
        ],
    }


def print_report(result: dict) -> None:
    print("=" * 70)
    print("EVAL SUMMARY")
    print("=" * 70)
    print(f"Labeled positive cases:              {result['labeled_positive']}")
    print(f"Labeled negative cases:               {result['labeled_negative']}")
    print(f"Found (true positives):               {len(result['found'])}/{result['labeled_positive']}")
    print(f"Missed (false negatives):             {len(result['missed'])}/{result['labeled_positive']}")
    if result["missed"]:
        print(f"  -> {result['missed']}")
    print(f"Wrong routing on an otherwise-found case: {len(result['wrong_routing'])}")
    if result["wrong_routing"]:
        for w in result["wrong_routing"]:
            print(f"  -> {w['case_id']}: expected {w['expected_routing']!r}, got {w['actual_routing']}")
    print(f"False positives on negative cases:    {len(result['false_positives_on_negative_cases'])}/{result['labeled_negative']}")
    if result["false_positives_on_negative_cases"]:
        for fp in result["false_positives_on_negative_cases"]:
            print(f"  -> {fp['case_id']}: {fp['matched_proposal_ids']}")
    print(f"Unlabeled proposals (not hallucination-checked, see docstring): {result['unlabeled_proposals_count']}")
    for p in result["unlabeled_proposals_sample"]:
        print(f"  - {p['proposal_id'][:16]}... {p['resident_id']}/{p['letter_id']} {p['category']}/{p['target_entity']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python eval/run_eval.py")
    parser.add_argument("--proposals", default="proposals.json")
    parser.add_argument("--labels", default="eval/labels.json")
    parser.add_argument(
        "--max-missed", type=int, default=2, help="CI gate: fail if more labeled positives than this are missed"
    )
    parser.add_argument(
        "--max-false-positives",
        type=int,
        default=0,
        help="CI gate: fail if more negative cases than this fire a proposal (default: zero-tolerance)",
    )
    args = parser.parse_args(argv)

    proposals_path, labels_path = Path(args.proposals), Path(args.labels)
    if not proposals_path.exists():
        print(f"error: {proposals_path} not found — run `make run` first.", file=sys.stderr)
        return 1
    if not labels_path.exists():
        print(f"error: {labels_path} not found.", file=sys.stderr)
        return 1

    proposals = _load(proposals_path).get("proposals", [])
    cases = _load(labels_path)["cases"]

    result = evaluate(proposals, cases)
    print_report(result)

    gate_failed = len(result["missed"]) > args.max_missed or len(result["false_positives_on_negative_cases"]) > args.max_false_positives
    if gate_failed:
        print("\nFAILED: eval gate exceeded (see --max-missed / --max-false-positives).", file=sys.stderr)
        return 1
    print("\nOK: within eval gate thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
