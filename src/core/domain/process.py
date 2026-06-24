"""Process and Pipeline base classes with dependency-based execution.

Canonical lifecycle (there is exactly one):
1. `initialize_state(context)` runs ONCE when the process is created
   (via `Process.execute()` or `Pipeline.initialize_all()`).
2. `check_completion(context)` runs REPEATEDLY (driven by `src/tick.py`)
   until it returns a ProcessResult that closes the process.

Persistence roundtrip: `Process.to_dict()` serializes; `deserialize_process()`
restores via the process registry (`register_process_class`). Feature modules
register their process classes at import time.

`context` always contains:
    - services: dict of repositories (e.g. {"vitals_repo": ...})
    - resident_id: resident identifier
    - now: the current (possibly simulated) datetime — never call
      datetime.now() inside process logic, always use context["now"]

Pipelines support dependency graphs; processes in the same stage are
conceptually parallel:

    A ─┐
       ├→ C → E
    B ─┘
       └→ D
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable, Optional


class ProcessStatus(StrEnum):
    """Lifecycle status of a process."""

    ACTIVE = "active"  # ready / in progress
    PENDING = "pending"  # not ready yet, waiting for parent process

    CLOSED = "closed"  # successfully done
    FAILED = "failed"  # error occurred
    WAITING = "waiting"  # waiting for an action to be finished

    SKIPPED = "skipped"  # system logic bypassed
    IGNORED = "ignored"  # user manually bypassed


class ActionType(StrEnum):
    """Types of actions that can occur in a process."""

    PROCESS_STARTED = "process_started"
    PROCESS_COMPLETED = "process_completed"
    PROCESS_FAILED = "process_failed"
    PROCESS_SKIPPED = "process_skipped"
    STATUS_CHANGED = "status_changed"
    USER_ACTION_TAKEN = "user_action_taken"
    VERIFICATION_DONE = "verification_done"
    NOTIFICATION_SENT = "notification_sent"


@dataclass
class ActionLog:
    """Audit trail entry for process execution."""

    created_at: datetime
    action_type: ActionType
    description: str
    user_id: Optional[str] = None
    data: Optional[dict[str, Any]] = None


@dataclass
class SuggestedAction:
    """User-facing action that can be performed on a process.

    Attributes:
        action_id: Unique identifier for this action
        label: Short label shown to user (e.g. "Verifizierungsmessung durchführen")
        description: Detailed description of what this action does
        icon: Optional icon name for UI display
        requires_input: Whether this action requires user input (e.g. a measurement value)
        confirmation_required: Whether to show a confirmation dialog before executing
        update_status_to: If set, the process status is updated to this value when taken
    """

    action_id: str
    label: str
    description: str
    icon: Optional[str] = None
    requires_input: bool = False
    confirmation_required: bool = False
    update_status_to: Optional[ProcessStatus] = None


@dataclass
class ProcessResult:
    """Result of a process lifecycle step."""

    success: bool
    next_status: ProcessStatus
    message: str
    data: Optional[dict[str, Any]] = None
    should_close_suggestion: bool = False
    suggestion_close_reason: Optional[str] = None


@dataclass
class ProcessNode:
    """A node in the process dependency graph.

    Attributes:
        process: The Process instance
        depends_on: List of process IDs that must complete before this one
    """

    process: Process
    depends_on: list[str] = field(default_factory=list)


class Process(ABC):
    """Abstract base class for all process types."""

    def __init__(
        self,
        name: str,
        description: str,
        suggested_actions: Optional[list[SuggestedAction]] = None,
    ):
        self.name = name
        self.description = description
        self.suggested_actions: list[SuggestedAction] = suggested_actions or []

        self.status = ProcessStatus.PENDING
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.action_logs: list[ActionLog] = []
        self.process_state: dict[str, Any] = {}
        # Process names this one waits for (set when flattened from a Pipeline).
        self.depends_on: list[str] = []

    @abstractmethod
    def initialize_state(self, context: dict[str, Any]) -> ProcessResult:
        """Initialize process state (runs ONCE at creation).

        Use this to set up initial state in self.process_state.
        """

    def check_completion(self, context: dict[str, Any]) -> Optional[ProcessResult]:
        """Check if this process can be completed (re-evaluation).

        Runs REPEATEDLY (via tick.py) to check whether the process conditions
        have been met (e.g. a verification measurement was added).

        Returns:
            ProcessResult if status should change, None if no change needed.
        """
        return None

    def execute(self, context: dict[str, Any]) -> ProcessResult:
        """Run initialize_state once and record the outcome in the audit trail."""
        now = context.get("now")
        self.add_action_log(
            action_type=ActionType.PROCESS_STARTED,
            description=f"Process '{self.name}' started",
            now=now,
        )
        self._update_status(ProcessStatus.ACTIVE, now=now)

        try:
            result = self.initialize_state(context)
            self.apply_result(result, now=now)
            return result

        except Exception as e:
            self._update_status(ProcessStatus.FAILED, now=now)
            self.add_action_log(
                action_type=ActionType.PROCESS_FAILED,
                description=f"Process failed with error: {e}",
                data={"error": str(e), "error_type": type(e).__name__},
                now=now,
            )
            return ProcessResult(
                success=False,
                next_status=ProcessStatus.FAILED,
                message=f"Process failed: {e}",
            )

    def apply_result(self, result: ProcessResult, now: Optional[datetime] = None) -> None:
        """Apply a ProcessResult: status transition plus audit log entry.

        Used by execute() and by tick.py when check_completion() returns a result.
        """
        self._update_status(result.next_status, now=now)
        action_type = {
            ProcessStatus.CLOSED: ActionType.PROCESS_COMPLETED,
            ProcessStatus.FAILED: ActionType.PROCESS_FAILED,
            ProcessStatus.SKIPPED: ActionType.PROCESS_SKIPPED,
        }.get(result.next_status, ActionType.STATUS_CHANGED)
        self.add_action_log(
            action_type=action_type,
            description=result.message,
            data=result.data,
            now=now,
        )

    def add_action_log(
        self,
        action_type: ActionType,
        description: str,
        user_id: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> None:
        """Add an action to the audit trail.

        `now` is injected by callers running under a simulated clock (tick.py).
        """
        timestamp = now or datetime.now(timezone.utc)
        self.action_logs.append(
            ActionLog(
                created_at=timestamp,
                action_type=action_type,
                description=description,
                user_id=user_id,
                data=data,
            )
        )
        self.updated_at = timestamp

    def _update_status(self, new_status: ProcessStatus, now: Optional[datetime] = None) -> None:
        """Update process status and timestamp."""
        old_status = self.status
        self.status = new_status
        self.updated_at = now or datetime.now(timezone.utc)

        if old_status != new_status:
            self.add_action_log(
                action_type=ActionType.STATUS_CHANGED,
                description=f"Status changed: {old_status} → {new_status}",
                data={"old_status": old_status, "new_status": new_status},
                now=now,
            )

    def get_active_process(self) -> Optional[Process]:
        """Get this process if it's active (ACTIVE or WAITING)."""
        if self.status in (ProcessStatus.ACTIVE, ProcessStatus.WAITING):
            return self
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize process to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "process_state": self.process_state,
            "depends_on": self.depends_on,
            "action_logs": [
                {
                    "created_at": a.created_at.isoformat(),
                    "action_type": a.action_type,
                    "description": a.description,
                    "user_id": a.user_id,
                    "data": a.data,
                }
                for a in self.action_logs
            ],
            "suggested_actions": [
                {
                    "action_id": sa.action_id,
                    "label": sa.label,
                    "description": sa.description,
                    "icon": sa.icon,
                    "requires_input": sa.requires_input,
                    "confirmation_required": sa.confirmation_required,
                    "update_status_to": sa.update_status_to.value if sa.update_status_to else None,
                }
                for sa in self.suggested_actions
            ],
        }


class Pipeline(Process):
    """A Pipeline orchestrates child processes with dependency management.

    Example:
        pipeline = Pipeline(
            name="weight_loss",
            description="Investigate weight loss",
            process_graph={
                "verify_measurement": ProcessNode(
                    process=VerificationMeasurementProcess(),
                    depends_on=[],
                ),
                "notify_physician": ProcessNode(
                    process=PhysicianNotificationProcess(),
                    depends_on=["verify_measurement"],
                ),
            },
        )
        processes = pipeline.initialize_all(context)  # → attach to AISuggestion
    """

    def __init__(
        self,
        name: str,
        description: str,
        process_graph: dict[str, ProcessNode],
        suggested_actions: Optional[list[SuggestedAction]] = None,
    ):
        super().__init__(name, description, suggested_actions)
        self.process_graph = process_graph
        self._execution_stages: list[list[str]] = []
        self._compute_execution_stages()

    def _compute_execution_stages(self) -> None:
        """Compute execution stages using topological sort.

        Processes in the same stage can run in parallel; each stage must
        complete before the next stage starts.
        """
        in_degree = {pid: 0 for pid in self.process_graph}
        adjacency: dict[str, list[str]] = {pid: [] for pid in self.process_graph}

        for pid, node in self.process_graph.items():
            for dep in node.depends_on:
                if dep not in self.process_graph:
                    raise ValueError(f"Process '{pid}' depends on unknown process '{dep}'")
                adjacency[dep].append(pid)
                in_degree[pid] += 1

        self._execution_stages = []
        remaining = set(self.process_graph.keys())

        while remaining:
            current_stage = [pid for pid in remaining if in_degree[pid] == 0]
            if not current_stage:
                raise ValueError("Circular dependency detected in process graph")

            self._execution_stages.append(current_stage)
            for pid in current_stage:
                remaining.remove(pid)
                for neighbor in adjacency[pid]:
                    in_degree[neighbor] -= 1

    def initialize_state(self, context: dict[str, Any]) -> ProcessResult:
        """Pipelines need no own state — child processes are initialized individually."""
        return ProcessResult(
            success=True,
            next_status=ProcessStatus.ACTIVE,
            message="Pipeline initialized",
        )

    def get_active_process(self) -> Optional[Process]:
        """Get the currently active process in the graph."""
        if self.status in (ProcessStatus.ACTIVE, ProcessStatus.WAITING):
            for node in self.process_graph.values():
                if node.process.status in (ProcessStatus.ACTIVE, ProcessStatus.WAITING):
                    return node.process
        return None

    def get_process(self, process_id: str) -> Optional[Process]:
        """Get a specific process by ID."""
        node = self.process_graph.get(process_id)
        return node.process if node else None

    def initialize_all(self, context: dict[str, Any]) -> list[Process]:
        """Initialize all child processes in execution order."""
        return serialize_pipeline_to_processes(self, context)

    def to_dict(self) -> dict[str, Any]:
        """Serialize pipeline to dictionary."""
        base_dict = super().to_dict()
        base_dict["processes"] = {
            pid: {"process": node.process.to_dict(), "depends_on": node.depends_on}
            for pid, node in self.process_graph.items()
        }
        base_dict["execution_stages"] = self._execution_stages
        return base_dict


# ---------------------------------------------------------------------------
# Serialization helpers + process registry
#
# This is the ONE canonical save/load roundtrip:
#   save: AISuggestion.to_dict() → Process.to_dict() per process
#   load: deserialize_processes() → registry factory + restored state
# Feature modules call register_process_class(...) at import time.
# ---------------------------------------------------------------------------

_PROCESS_CLASS_REGISTRY: dict[str, Callable[[], Process]] = {}


def register_process_class(process_name: str, factory: Callable[[], Process]) -> None:
    """Register a Process class for deserialization.

    Example:
        register_process_class("verification_measurement", VerificationMeasurementProcess)
    """
    _PROCESS_CLASS_REGISTRY[process_name] = factory


def deserialize_process(process_dict: dict[str, Any]) -> Optional[Process]:
    """Deserialize a single process from dict format.

    Returns the Process instance with restored state, or None if the process
    class is not registered.
    """
    factory = _PROCESS_CLASS_REGISTRY.get(process_dict.get("name"))
    if not factory:
        return None

    process = factory()
    process.status = ProcessStatus(process_dict.get("status", ProcessStatus.PENDING))
    process.created_at = datetime.fromisoformat(process_dict["created_at"])
    process.updated_at = datetime.fromisoformat(process_dict["updated_at"])
    process.process_state = process_dict.get("process_state", {})
    process.depends_on = process_dict.get("depends_on", [])
    process.action_logs = [
        ActionLog(
            created_at=datetime.fromisoformat(a["created_at"]),
            action_type=ActionType(a["action_type"]),
            description=a["description"],
            user_id=a.get("user_id"),
            data=a.get("data"),
        )
        for a in process_dict.get("action_logs", [])
    ]
    # suggested_actions are already set by the factory's __init__
    return process


def deserialize_processes(process_dicts: list[dict[str, Any]]) -> list[Process]:
    """Deserialize multiple processes, skipping unregistered process classes."""
    processes = []
    for process_dict in process_dicts:
        process = deserialize_process(process_dict)
        if process:
            processes.append(process)
    return processes


def serialize_pipeline_to_processes(pipeline: Pipeline, context: dict[str, Any]) -> list[Process]:
    """Flatten a Pipeline into initialized processes, in execution order.

    Each process gets its initialize_state() called once; processes with
    dependencies are set back to PENDING until their parents close.
    """
    processes: list[Process] = []

    for stage in pipeline._execution_stages:
        for process_name in stage:
            node = pipeline.process_graph[process_name]
            process = node.process

            process.execute(context)
            process.depends_on = list(node.depends_on)

            if node.depends_on and process.status != ProcessStatus.FAILED:
                # Has dependencies → not ready yet
                process._update_status(ProcessStatus.PENDING, now=context.get("now"))

            processes.append(process)

    return processes
