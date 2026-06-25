"""Batch entrypoint: run all registered analyzers over all letters.

    python -m src.run --letters letters --data data --out proposals.json

Letters come from `letters/index.json`; per-resident data from `data/`.
Proposal ids are deterministic (sha256 of the identity fields), so re-runs
are idempotent.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.core.analyzers import get_registered_analyzers
from src.core.domain.proposal import CostLogEntry, Proposal, ProposalsOutput
from src.core.llm.client import get_llm_client
from src.core.repositories import ResidentDataRepository

ANALYZERS = get_registered_analyzers()


def load_letter_index(letters_dir: Path) -> list[dict]:
    index_path = letters_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Letter index not found: {index_path}\n"
            "Expected an index.json next to the letter files "
            "(see DATA_SCHEMA.md for the schema)."
        )
    return json.loads(index_path.read_text(encoding="utf-8"))


def run_analysis(letters_dir: Path, data_dir: Path, out_path: Path) -> ProposalsOutput:
    letter_entries = load_letter_index(letters_dir)
    resident_repo = ResidentDataRepository(data_dir)
    llm = get_llm_client()
    if not hasattr(llm, "usage_log"):
        print(
            "warning: LLM client has no usage_log — cost_log will be empty; "
            "see LLMClient protocol",
            file=sys.stderr,
        )

    proposals: dict[str, Proposal] = {}  # keyed by deterministic id → no duplicates
    cost_log: list[CostLogEntry] = []

    for letter_meta in letter_entries:
        # Index "file" may be a bare filename ("letter_01.md") or a path
        # ("letters/letter_01.md"); the files live flat inside the letters dir.
        letter_path = letters_dir / Path(letter_meta["file"]).name
        letter_text = letter_path.read_text(encoding="utf-8")
        resident_data = resident_repo.load(letter_meta["resident_id"])

        for analyzer in ANALYZERS:
            for proposal in analyzer.analyze(letter_text, letter_meta, resident_data, llm):
                proposals.setdefault(proposal.proposal_id, proposal)

        # Attribute all LLM usage since the last drain to this letter.
        usage_log = getattr(llm, "usage_log", None)
        if usage_log:
            cost_log.extend(
                CostLogEntry(letter_id=letter_meta["letter_id"], **usage) for usage in usage_log
            )
            usage_log.clear()

    output = ProposalsOutput(
        run_id=f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
        proposals=list(proposals.values()),
        cost_log=cost_log,
    )
    out_path.write_text(output.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(output.proposals)} proposal(s) from {len(letter_entries)} letter(s) "
        f"to {out_path}"
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.run",
        description="Analyze discharge letters and write proposals.json.",
    )
    parser.add_argument("--letters", default="letters", help="Path to the letters directory")
    parser.add_argument("--data", default="data", help="Path to the data directory")
    parser.add_argument("--out", default="proposals.json", help="Output file path")
    args = parser.parse_args(argv)

    try:
        run_analysis(Path(args.letters), Path(args.data), Path(args.out))
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
