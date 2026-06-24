"""Shared enums used across all features.

These enums are used by multiple analysis features and should be imported
from here rather than duplicated.
"""
from __future__ import annotations

from enum import StrEnum


class AlertCategory(StrEnum):
    """High-level category for alerts. Add new categories as you add features."""

    WEIGHT = "weight_analysis"
    DISCHARGE_LETTER = "discharge_letter"


class AlertLevel(StrEnum):
    """Severity level for alerts.

    - LOW: informational, no immediate action required
    - MEDIUM: should be reviewed soon
    - HIGH: requires immediate attention
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlertStatus(StrEnum):
    """Lifecycle status of an alert or suggestion.

    - ACTIVE: current and actionable
    - CLOSED: resolved or superseded
    - HANDLED: care staff took action
    - IGNORED: care staff explicitly dismissed
    """

    ACTIVE = "active"
    CLOSED = "closed"
    HANDLED = "handled"
    IGNORED = "ignored"


class LogCategory(StrEnum):
    """Categories for suggestion log entries (shared across features).

    Feature-specific categories may be passed as plain strings where needed.
    """

    SUGGESTION_CREATED = "suggestion_created"
    ACTION_CREATED = "action_created"
    IGNORED = "ignored"
    CLOSED = "closed"

    @classmethod
    def get_default_description(cls, category: LogCategory) -> str:
        """Default German description for a log category."""
        descriptions = {
            cls.SUGGESTION_CREATED: "Der Vorschlag wurde erstellt.",
            cls.ACTION_CREATED: "Die vorgeschlagene Aktion wurde ausgeführt.",
            cls.IGNORED: "Der Vorschlag wurde ignoriert.",
            cls.CLOSED: "Der Vorschlag ist abgeschlossen.",
        }
        return descriptions.get(category, "Aktion durchgeführt.")
