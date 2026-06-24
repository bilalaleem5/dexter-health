"""Deterministic routing policy.

The LLM extracts clinical facts and a few self-reported risk signals (e.g.
"this drug plausibly conflicts with a documented allergy"); this module is
the ONLY place that turns those signals into a `Routing` decision.

Why routing is not something we ask the LLM to output directly:
- it must be stable across model swaps (the assignment explicitly re-runs
  our pipeline against a different model),
- it must be auditable by reading one short policy table instead of a
  prompt buried in a string,
- a miscalibrated or prompt-injected model should not be able to talk its
  way into AUTO_APPLY for something safety-relevant.

See DECISIONS.md §4 for the policy reasoning in prose.
"""
from __future__ import annotations

from src.core.domain.proposal import Routing, Severity

# ---------------------------------------------------------------------------
# Medication changes never auto-apply. They touch resident safety directly
# and the source (a discharge letter of varying quality) is not a fully
# trusted machine-readable feed. Default: a human confirms. Escalate to a
# physician whenever the LLM flags a safety conflict (allergy, or a
# drug-disease / drug-drug concern against existing diagnoses/medications)
# or the letter itself is internally inconsistent about the change.
# ---------------------------------------------------------------------------


def route_medication_change(
    *, is_safety_conflict: bool, is_ambiguous_or_conflicting: bool, change_kind: str
) -> tuple[Routing, Severity]:
    if is_safety_conflict:
        return Routing.HARD_STOP_PHYSICIAN, Severity.CRITICAL
    if is_ambiguous_or_conflicting:
        return Routing.HARD_STOP_PHYSICIAN, Severity.CRITICAL
    if change_kind in ("stopped", "dose_changed"):
        return Routing.HUMAN_CONFIRM, Severity.WARN
    return Routing.HUMAN_CONFIRM, Severity.WARN  # "new" — still WARN, it's a med plan change


def route_diagnosis_addition(*, is_chronic: bool) -> tuple[Routing, Severity]:
    """Caller is expected to suppress transient/resolved findings before this is reached."""
    if not is_chronic:
        return Routing.INFO_ONLY, Severity.INFO
    return Routing.HUMAN_CONFIRM, Severity.INFO


def route_care_instruction(*, direction_conflict: bool) -> tuple[Routing, Severity]:
    """Fluid/wound/behavior care-plan updates. direction_conflict = the new
    instruction reverses an existing strategy (e.g. encourage -> restrict
    fluids) — still human_confirm either way, but worth a higher severity so
    it doesn't get lost in a long queue."""
    if direction_conflict:
        return Routing.HUMAN_CONFIRM, Severity.WARN
    return Routing.HUMAN_CONFIRM, Severity.INFO


def route_task(*, requires_clinical_judgment: bool) -> tuple[Routing, Severity]:
    """Pure scheduling/reminder tasks (book a lab check, a GP appointment) are
    low-risk and administrative -> auto_apply. Anything that asks a
    clinician to actually decide something (re-evaluate a dose, judge an
    indication) goes back to the physician, never auto_apply."""
    if requires_clinical_judgment:
        return Routing.HARD_STOP_PHYSICIAN, Severity.WARN
    return Routing.AUTO_APPLY, Severity.INFO


def route_missing_critical_data() -> tuple[Routing, Severity]:
    """A clinically relevant attachment/section is referenced in the letter
    but absent from what we were given (e.g. "see Anlage 2", no Anlage 2).
    This is a gap in the source document, not a system extraction failure —
    routed back to a physician rather than treated as background noise."""
    return Routing.HARD_STOP_PHYSICIAN, Severity.WARN


def route_extraction_failure() -> tuple[Routing, Severity]:
    """The LLM output failed validation twice — a system reliability issue,
    not a clinical-content gap. Surface it, but as info, not a physician
    escalation (there's no clinical claim being made at all)."""
    return Routing.INFO_ONLY, Severity.WARN


# ---------------------------------------------------------------------------
# Follow-up due-date encoding.
#
# `Proposal.target_entity` is a plain string in the fixed contract — there is
# no structured due-date field. We encode the offset into target_entity
# itself (`"<slug>__due+<N>d"`) rather than adding an out-of-band store, so a
# CREATE_TASK proposal is still a single, idempotent, self-contained record.
# See DECISIONS.md §1.
# ---------------------------------------------------------------------------

_DUE_MARKER = "__due+"


def encode_due_offset(slug: str, days: int) -> str:
    return f"{slug}{_DUE_MARKER}{days}d"


def parse_due_offset(target_entity: str) -> tuple[str, int] | None:
    if _DUE_MARKER not in target_entity or not target_entity.endswith("d"):
        return None
    slug, rest = target_entity.split(_DUE_MARKER, 1)
    digits = rest[:-1]
    if not digits.isdigit():
        return None
    return slug, int(digits)
