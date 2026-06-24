"""Alert domain model — feature-independent.

This model represents a detected problem/anomaly from any analysis feature.
It is the output of analyzers before being converted to an AISuggestion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Union

from .enums import AlertCategory, AlertLevel, AlertStatus


@dataclass(frozen=True)
class Alert:
    """A detected problem/anomaly from analysis.

    Attributes:
        category: High-level category (e.g. weight_analysis)
        subcategory: Feature-specific subcategory (enum or string)
        level: Severity level (low, medium, high)
        title: Human-readable title
        reason: Explanation of why this alert was generated
        suggested_action: Recommended action structure (optional)
        created_at: When the alert was detected
        status: Alert status (default: ACTIVE)
    """

    category: AlertCategory
    subcategory: Union[str, object]  # feature-specific enum or string
    level: AlertLevel
    title: str
    reason: str
    suggested_action: dict | None
    created_at: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    relevant_info: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Dictionary format with alert_-prefixed keys, consistent with AISuggestion."""
        alert_subcategory = getattr(self.subcategory, "value", self.subcategory)
        return {
            "alert_category": self.category.value,
            "alert_subcategory": alert_subcategory,
            "alert_title": self.title,
            "alert_level": self.level.value,
            "reason": self.reason,
            "suggested_action": self.suggested_action,
            "relevant_info": self.relevant_info,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
        }
