"""Analysis configuration: defaults + one level of facility overrides.

Usage:
    cfg = get_effective_config({"weight": {"no_recent_weight_days": 14}})
    cfg["weight"]["no_recent_weight_days"]  # → 14, everything else default
"""
from __future__ import annotations

DEFAULTS: dict[str, dict] = {
    "weight": {
        # Recency: alert when the newest weight entry is older than this.
        "no_recent_weight_days": 30,
        # 3-month relative change thresholds (fractions, not percent).
        "change_3m_yellow_pct": 0.05,
        "change_3m_red_pct": 0.075,
        # Verification measurement: max distance (kg) to count as "close".
        "verification_threshold_kg": 2.5,
    },
}


def get_effective_config(facility_overrides: dict | None = None) -> dict:
    """Deep-merge facility overrides onto DEFAULTS (one override level)."""
    config = {section: dict(values) for section, values in DEFAULTS.items()}
    for section, overrides in (facility_overrides or {}).items():
        config.setdefault(section, {}).update(overrides)
    return config
