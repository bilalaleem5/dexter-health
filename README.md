# dexter health — AI Engineer Take-Home: Discharge Letter Reconciliation

Welcome! This assignment mirrors a real problem we are building towards: **a nursing-home resident returns from the hospital with a discharge letter (Arztbrief), and the care documentation now has to be updated in several places.** Your job is to build the system that reads the letter, reconciles it against the resident's existing documentation, and proposes the right updates and follow-up actions.

## Context

dexter health builds AI-powered software for nursing homes. Our QM platform analyzes care documentation and produces **alerts** and **AI suggestions** with guided follow-up **processes** that care staff interact with. This starter repo contains a simplified but faithful slice of our production architecture:

- `src/core/domain/process.py` — our process framework (guided tasks: `SuggestedAction`s, status lifecycle, audit log). Read the status contract: **WAITING is only exited by `check_completion()` (driven by `tick.py`); user actions act on ACTIVE processes.**
- `src/features/weights/` — a complete reference feature (analyzer + a verification process) showing our house style end to end.
- `src/features/stay_metadata/` — a fully worked example of an **LLM analyzer** (prompt → schema → validation → retry → fallback → proposal). Use it as your pattern.
- `data/` — a fictional nursing-home database for 5 residents, modelled as a realistic **multi-year record** (diagnoses, current and historical medication, labs, assessments, prior encounters, vitals trends, care notes, falls; see `DATA_SCHEMA.md`). It is deliberately broader than any one analyzer needs — reconcile against what's relevant and cut the rest.
- `letters/` — 5 fictional German discharge letters (one per resident, varying quality, as in real life).

Everything is fictional. No real patient data.

## Your task (~4-6 hours of focused work)

**Fair warning: the scope below is deliberately larger than the time budget.** We do not expect everything to be finished. What we evaluate is what you prioritize, how deep you go on what matters, and whether your cut list (DECISIONS.md §7) shows real judgment. Cleanly cut scope costs you nothing; everything half-done does.

1. **Build discharge-letter analyzers** that read each letter, reconcile it against that resident's existing data, and emit **update proposals** — written to `proposals.json` via `make run` (schema: `src/core/domain/proposal.py`, mandatory). Every clinical proposal needs **provenance** (quote + letter section) and **confidence**. Register your analyzers in `src/run.py`.
2. **Decide the routing** per proposal: what may be applied automatically, what needs human confirmation, what must go back to a physician, what is information only. This is the heart of the task.
3. **Design and implement the API logic** in `src/api/app.py` (stubs provided): letter ingest (letter content is loaded from `letters/` by `letter_id`; think about idempotency — define and defend your key), fetching proposals, accept/reject decisions with an audit trail, and partial application (= some of a letter's proposals get accepted while others are rejected or modified).
4. **Build one guided follow-up process** on our framework (`process.py`) for a proposal type of your choice, runnable via `make tick`. Note: `tick.py` only sees processes attached to an `AISuggestion` saved in `data/ai_suggestions/` (via `SuggestionsRepository`) — see `src/tick.py` and `tests/test_process_lifecycle.py` for the pattern.
5. **Write a small eval** (`make eval`): label ≥10 expected proposals yourself from the letters — **including ≥2 negative cases** (things that must NOT fire) — and report found/missed/hallucinated plus the false-positive rate. See `EVAL_TEMPLATE.md`.
6. **Write `DECISIONS.md`** (max 1.5 pages, template provided): your assumptions where the spec has gaps, problems you found in the data/letters, your LLM-vs-deterministic-code boundary, model choice with measured token counts per letter (the cost log must be produced by your code; converting tokens to money via a price table is fine), how you tested, what you deliberately cut, and a short **self-report** (§9): a residual-risk register of 2–3 things you think your system still gets wrong, plus an honest read of your own eval. **Every claimed finding or gap needs a coordinate (letter N, section X / file:line).**

## Rules

- **You do not need any medical knowledge.** Use your LLM as the domain expert (e.g. to normalize drug names to active substances, or to judge whether two medications conflict). We evaluate whether your *system* finds and safely handles the issues — not whether you know them yourself.
- **AI coding tools (Claude Code, Cursor, Copilot, …) are explicitly encouraged.** Tell us in DECISIONS.md how you used them and where you overrode them.
- **LLM access:** bring your own API key, any provider. Your code must work behind the provided `LLMClient` interface (`src/core/llm/client.py`) with provider/model set via env vars (`.env.example`) — **we will re-run your pipeline with our own key and a different model.** If you use a provider-specific structured-output feature, implement a prompt+parse+validate fallback. All tests must pass with **no API key** (use the `MockLLMClient`).
- **Heads-up:** at least one letter contains a finding that **cannot be cleanly resolved** from the available information. We grade your reasoning about it, not your answer.
- **Hold-out test:** we will additionally run your unchanged pipeline on **a discharge letter you have never seen** (same data schemas, new resident). Build generic mechanisms, not letter-specific handling — hardcoding the five known letters scores zero on the hold-out.
- **Robustness target:** `LLM_PROVIDER=mock_chaos make run` must degrade gracefully (the chaos provider deterministically returns broken JSON, invalid enums, and empty answers — your validation/repair/fallback path has to hold). We run this during grading.
- The letters vary in quality on purpose (clean clinic letter, telegram style with abbreviations and typos, incomplete). `GLOSSARY.md` explains the German care/medical terms you'll encounter.
- **The mock LLM is for tests only.** It returns canned answers (the same stay-metadata JSON for every letter), so `make run` without a real key produces placeholder output. Your committed `proposals.json` must come from a run against a real model.
- **A small, correct system with sharp reasoning beats a large generated one.** Please don't gold-plate; cut scope deliberately and say why.

## Out of scope (spending time here earns nothing)

UI/frontend, real databases (the JSON files + repositories are the persistence layer), auth, deployment, OCR/PDF parsing, exhaustive test coverage, multi-language support.

## Getting started

Requires Python 3.11+.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# when using a real LLM provider: cp .env.example .env, fill it in, then: set -a; source .env; set +a
make test     # must stay green without any API key
make run      # produces proposals.json (initially: stay_metadata example only)
make tick     # advances guided processes (simulated clock)
make api      # serves the API skeleton
make eval     # your eval (stub until you implement it)
```

## Deliverables & submission

A zip or private repo link containing your code, your final `proposals.json` output (please commit it explicitly), `DECISIONS.md`, `EVAL.md`, and your git history (please commit as you go; one giant commit tells us nothing). Designed for **~4 focused hours of core work if you cut deliberately** (budget up to 6 if you want to polish); submit within **one week**. Questions are welcome and count positively — email us; where a question touches an ambiguity we built in on purpose, the answer will be "your call — document it."

Good luck — we're looking forward to reading your reasoning.
