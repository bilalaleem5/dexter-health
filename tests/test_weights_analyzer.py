"""Tests for the reference weight analyzer (pure, injected clock)."""
from datetime import datetime, timedelta, timezone

from src.core.domain.enums import AlertLevel
from src.features.weights.analyzer import WeightAnalyzer

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def _entry(days_ago: int, weight: float) -> dict:
    return {"created_at": (NOW - timedelta(days=days_ago)).isoformat(), "weight": weight}


def _subcategories(alerts) -> set[str]:
    return {a.subcategory for a in alerts}


def test_recency_alert_fires_at_31_days():
    alerts = WeightAnalyzer(now=NOW).analyze([_entry(31, 72.0)])
    recency = [a for a in alerts if a.subcategory == "no_recent_weight"]
    assert len(recency) == 1
    assert recency[0].level is AlertLevel.MEDIUM


def test_recency_alert_does_not_fire_at_29_days():
    alerts = WeightAnalyzer(now=NOW).analyze([_entry(29, 72.0)])
    assert "no_recent_weight" not in _subcategories(alerts)


def test_no_entries_at_all_fires_recency_alert():
    alerts = WeightAnalyzer(now=NOW).analyze([])
    assert "no_recent_weight" in _subcategories(alerts)


def test_3m_loss_of_8_percent_is_red():
    alerts = WeightAnalyzer(now=NOW).analyze([_entry(80, 80.0), _entry(1, 73.6)])  # -8%
    assert "loss_3m_red" in _subcategories(alerts)
    red = next(a for a in alerts if a.subcategory == "loss_3m_red")
    assert red.level is AlertLevel.HIGH


def test_3m_loss_of_6_percent_is_yellow():
    alerts = WeightAnalyzer(now=NOW).analyze([_entry(80, 80.0), _entry(1, 75.2)])  # -6%
    assert "loss_3m_yellow" in _subcategories(alerts)


def test_small_change_is_quiet():
    alerts = WeightAnalyzer(now=NOW).analyze([_entry(80, 80.0), _entry(1, 79.0)])
    assert alerts == []


def test_invalid_entries_are_dropped():
    entries = [
        {"created_at": "not-a-date", "weight": 80.0},
        {"weight": 75.0},
        _entry(1, 72.0),
    ]
    alerts = WeightAnalyzer(now=NOW).analyze(entries)
    # Only one valid entry remains → no change alert possible, no recency alert (1 day old).
    assert alerts == []
