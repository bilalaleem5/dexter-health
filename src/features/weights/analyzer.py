"""Reference weight analyzer.

Shows the house pattern for rule-based analyzers: pure logic over prepared
data, thresholds from config, and an injected clock (`now` is a parameter —
never datetime.now() inside analysis code, the simulated clock in tick.py
depends on this).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.core.config import get_effective_config
from src.core.domain.alert import Alert
from src.core.domain.enums import AlertCategory, AlertLevel


class WeightAnalyzer:
    """Detects missing recent measurements and significant 3-month changes."""

    def __init__(self, now: datetime, config: dict | None = None):
        self.now = now
        self.cfg = (config or get_effective_config())["weight"]

    def analyze(self, raw_entries: list[dict]) -> list[Alert]:
        entries = self._prepare_data(raw_entries)
        alerts = []

        recency_alert = self._check_recency(entries)
        if recency_alert:
            alerts.append(recency_alert)

        change_alert = self._check_3m_change(entries)
        if change_alert:
            alerts.append(change_alert)

        return alerts

    def _prepare_data(self, raw_entries: list[dict]) -> list[dict]:
        """Drop invalid entries, normalize timezones, sort by date."""
        valid = []
        for entry in raw_entries:
            try:
                created_at = datetime.fromisoformat(entry["created_at"])
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                valid.append({"weight": float(entry["weight"]), "created_at": created_at})
            except (ValueError, KeyError, TypeError):
                continue
        valid.sort(key=lambda e: e["created_at"])
        return valid

    def _check_recency(self, entries: list[dict]) -> Alert | None:
        max_age = timedelta(days=self.cfg["no_recent_weight_days"])

        if not entries:
            return self._alert(
                subcategory="no_recent_weight",
                level=AlertLevel.MEDIUM,
                title="Keine Gewichtsmessung vorhanden",
                reason="Keine Gewichtseinträge vorhanden.",
            )

        last = entries[-1]
        if last["created_at"] < self.now - max_age:
            days = (self.now - last["created_at"]).days
            return self._alert(
                subcategory="no_recent_weight",
                level=AlertLevel.MEDIUM,
                title="Keine aktuelle Gewichtsmessung",
                reason=f"Letzte Gewichtsmessung liegt {days} Tage zurück "
                f"(Limit: {self.cfg['no_recent_weight_days']} Tage).",
            )
        return None

    def _check_3m_change(self, entries: list[dict]) -> Alert | None:
        cutoff = self.now - timedelta(days=90)
        window = [e for e in entries if e["created_at"] >= cutoff]
        if len(window) < 2:
            return None

        oldest, newest = window[0], window[-1]
        if oldest["weight"] <= 0:
            return None

        change_pct = (newest["weight"] - oldest["weight"]) / oldest["weight"]
        if abs(change_pct) < self.cfg["change_3m_yellow_pct"]:
            return None

        is_red = abs(change_pct) >= self.cfg["change_3m_red_pct"]
        direction = "loss" if change_pct < 0 else "gain"
        direction_de = "Gewichtsverlust" if change_pct < 0 else "Gewichtszunahme"

        return self._alert(
            subcategory=f"{direction}_3m_{'red' if is_red else 'yellow'}",
            level=AlertLevel.HIGH if is_red else AlertLevel.MEDIUM,
            title=f"{direction_de} in 3 Monaten",
            reason=f"{direction_de} von {abs(change_pct) * 100:.1f}% in 3 Monaten "
            f"({oldest['weight']:.1f} kg → {newest['weight']:.1f} kg).",
        )

    def _alert(self, subcategory: str, level: AlertLevel, title: str, reason: str) -> Alert:
        return Alert(
            category=AlertCategory.WEIGHT,
            subcategory=subcategory,
            level=level,
            title=title,
            reason=reason,
            suggested_action=None,
            created_at=self.now,
        )
