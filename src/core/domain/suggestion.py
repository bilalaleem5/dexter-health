"""AI Suggestion domain models.

The core AISuggestion model used across all analysis features, plus related
models for logs and data references.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from .enums import AlertStatus, LogCategory


class SuggestionLog(BaseModel):
    """A single log entry in an AI suggestion's lifecycle.

    Attributes:
        created_at: Timestamp when this log entry was created
        category: Log category (LogCategory enum or string value)
        description: Human-readable description (auto-filled from category if not provided)
        user_id: Optional user ID who triggered this action (None for system actions)
        data: Optional metadata for this log entry
    """

    created_at: datetime
    category: Union[LogCategory, str]
    description: Optional[str] = None
    user_id: Optional[str] = None
    data: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def set_default_description(self) -> "SuggestionLog":
        """Set default description from category if not provided."""
        if self.description is None:
            try:
                category = LogCategory(self.category)
                self.description = LogCategory.get_default_description(category)
            except ValueError:
                self.description = "Aktion durchgeführt."
        return self

    model_config = {"use_enum_values": True}


class AISuggestion(BaseModel):
    """An AI-generated suggestion for care staff.

    Used across ALL analysis features and persisted as JSON under
    `data/ai_suggestions/`.

    Lifecycle:
    1. Created by an analysis feature with status='active'
    2. Its attached processes advance via `src/tick.py` (check_completion)
    3. Closed automatically or marked 'handled'/'ignored' by care staff
    """

    suggestion_id: Optional[str] = Field(
        default=None, description="Stable identifier used by the repository for upserts"
    )
    resident_id: Optional[str] = Field(
        default=None, description="Resident this suggestion is for"
    )

    # Alert information
    alert_category: str = Field(..., min_length=1)
    alert_subcategory: str = Field(..., min_length=1)
    alert_title: str = Field(..., min_length=1, description="Title displayed to care staff")
    alert_level: Literal["low", "medium", "high"]

    # Core content
    reason: str = Field(..., min_length=1, description="Why this suggestion was created")
    relevant_info: list[dict[str, Any]] = Field(default_factory=list)
    potential_reasons: list[dict[str, Any]] = Field(
        default_factory=list, description="Potential causal factors identified by analysis"
    )
    analysis_steps: list[dict[str, Any]] = Field(
        default_factory=list, description="Transparency log of analysis steps taken"
    )
    suggested_action: Optional[dict[str, Any]] = None

    # Lifecycle
    status: Union[AlertStatus, str] = AlertStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Attached process pipeline (list of Process objects, see core/domain/process.py)
    pipeline_name: Optional[str] = None
    processes: list[Any] = Field(default_factory=list)

    # Audit trail
    logs: list[SuggestionLog] = Field(default_factory=list)

    # Data references (e.g. the measurements/diagnoses that triggered this)
    data: dict | None = None

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: Union[AlertStatus, str]) -> AlertStatus:
        if isinstance(v, AlertStatus):
            return v
        if isinstance(v, str):
            try:
                return AlertStatus(v)
            except ValueError:
                raise ValueError(
                    f"Invalid status: {v}. Must be one of: "
                    f"{', '.join(s.value for s in AlertStatus)}"
                )
        raise ValueError(f"Status must be AlertStatus or str, got {type(v)}")

    def add_log(
        self,
        category: Union[LogCategory, str],
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> None:
        """Add a log entry and update the timestamp.

        `now` is injected by callers running under a simulated clock (tick.py).
        """
        timestamp = now or datetime.now(timezone.utc)
        self.logs.append(
            SuggestionLog(
                created_at=timestamp,
                category=category,
                description=description,
                user_id=user_id,
                data=data,
            )
        )
        self.updated_at = timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict.

        Process objects are serialized at this persistence boundary via their
        own `to_dict()`; loading them back goes through the process registry
        (see core/domain/process.py).
        """
        data = self.model_dump(mode="json", exclude_none=False, exclude={"processes"})
        data["processes"] = [p.to_dict() for p in self.processes]
        return data

    model_config = {
        "validate_assignment": True,
        "arbitrary_types_allowed": True,  # allow Process objects (non-pydantic)
    }
