"""Simulated-clock driver for the process lifecycle.

    python -m src.tick --data data --advance-days 3

The simulated `now` is the latest timestamp found anywhere in the data dir
plus N days. Every ACTIVE/WAITING process attached to a stored suggestion gets
its `check_completion(context)` called; returned results are applied and
saved back. This only works because processes never call datetime.now() —
they read `context["now"]`.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from src.core.domain.enums import AlertStatus, LogCategory
from src.core.domain.process import ActionType, ProcessStatus
from src.core.repositories import (
    ResidentDataRepository,
    SuggestionsRepository,
    VitalsRepository,
)

# Importing feature process modules registers their classes for deserialization.
from src.features.weights import processes as _weights_processes  # noqa: F401
from src.features.clinical_followup import process as _clinical_followup_process  # noqa: F401

_TIMESTAMP_KEYS = {"created_at", "updated_at", "timestamp", "noted_at", "healed_at"}


def _iter_timestamps(node: Any) -> Iterator[datetime]:
    """Recursively yield parseable timestamps from known keys in a JSON structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _TIMESTAMP_KEYS and isinstance(value, str):
                try:
                    parsed = datetime.fromisoformat(value)
                except ValueError:
                    continue
                yield parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
            else:
                yield from _iter_timestamps(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_timestamps(item)


def find_simulated_now(data_dir: Path, advance_days: int) -> datetime:
    """max(timestamp in data) + advance_days."""
    timestamps = []
    for path in sorted(data_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        timestamps.extend(_iter_timestamps(payload))

    if not timestamps:
        raise FileNotFoundError(f"No timestamps found in any JSON file under {data_dir}")
    return max(timestamps) + timedelta(days=advance_days)


def run_tick(data_dir: Path | str, advance_days: int) -> int:
    """Run one tick over all stored suggestions. Returns the number of state changes."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    now = find_simulated_now(data_dir, advance_days)
    print(f"Simulated clock: {now.isoformat()} (latest data timestamp + {advance_days} day(s))")

    suggestions_repo = SuggestionsRepository(data_dir)
    vitals_repo = VitalsRepository(data_dir)
    resident_data_repo = ResidentDataRepository(data_dir)
    changes = 0
    open_processes = 0

    for resident_id in suggestions_repo.list_resident_ids():
        for suggestion in suggestions_repo.list_for_resident(resident_id):
            context = {
                # Your check_completion() can use any of these services.
                "services": {
                    "vitals_repo": vitals_repo,
                    "resident_data_repo": resident_data_repo,
                    "suggestions_repo": suggestions_repo,
                },
                "resident_id": resident_id,
                "now": now,
            }
            suggestion_changed = False

            for process in suggestion.processes:
                if process.status in (
                    ProcessStatus.ACTIVE,
                    ProcessStatus.WAITING,
                    ProcessStatus.PENDING,
                ):
                    open_processes += 1
                if process.status not in (ProcessStatus.ACTIVE, ProcessStatus.WAITING):
                    continue

                result = process.check_completion(context)
                if result is None:
                    print(f"[{resident_id}] {process.name}: {process.status} (no change)")
                    continue

                old_status = process.status
                process.apply_result(result, now=now)
                if result.should_close_suggestion:
                    suggestion.status = AlertStatus.CLOSED
                    suggestion.add_log(
                        LogCategory.CLOSED,
                        description=result.suggestion_close_reason,
                        now=now,
                    )
                suggestion_changed = True
                changes += 1
                print(
                    f"[{resident_id}] {process.name}: {old_status} → {process.status}"
                    f" — {result.message}"
                )

            # Promote PENDING processes whose dependencies are all closed/skipped.
            done = {
                p.name
                for p in suggestion.processes
                if p.status in (ProcessStatus.CLOSED, ProcessStatus.SKIPPED)
            }
            for process in suggestion.processes:
                deps = getattr(process, "depends_on", [])
                if process.status is ProcessStatus.PENDING and deps and set(deps) <= done:
                    process._update_status(ProcessStatus.ACTIVE, now=now)
                    process.add_action_log(
                        ActionType.STATUS_CHANGED, "dependencies met — activated", now=now
                    )
                    suggestion_changed = True
                    changes += 1
                    print(f"[{resident_id}] {process.name}: pending → active — dependencies met")

            if suggestion_changed:
                suggestions_repo.save(suggestion)

    if changes == 0:
        print("No process changed state.")
    if open_processes == 0:
        print(
            "No open processes. (Candidates: attach your process to an AISuggestion "
            "saved in data/ai_suggestions/ — see tests/test_process_lifecycle.py.)"
        )
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.tick",
        description="Advance the simulated clock and re-check all open processes.",
    )
    parser.add_argument("--data", default="data", help="Path to the data directory")
    parser.add_argument(
        "--advance-days", type=int, default=1, help="Days to advance past the latest data timestamp"
    )
    args = parser.parse_args(argv)

    try:
        run_tick(Path(args.data), args.advance_days)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
