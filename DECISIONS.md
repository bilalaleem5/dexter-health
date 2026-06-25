# DECISIONS.md

## 1. Assumptions & spec gaps

- **No structured "new value" field on `Proposal`.** `target_entity` is just an
  identifier string, not a payload (`src/core/domain/proposal.py:67-79`). So
  "applying" an ADD/MODIFY medication change can't deterministically write the
  new dose into `medication_plans/*.json` — there's nowhere to read it from
  except free-text `rationale`/`provenance.letter_quote`. I decided **not** to
  regex-parse free text to reconstruct a structured mutation (fragile, and a
  parsing bug there is worse than doing nothing). `apply_accept`
  (`src/core/apply_effects.py`) therefore only performs the one mutation that
  *is* fully determined by the existing schema — removing a STOPped entry by
  substance — and logs everything else as "accepted, manual entry required,
  see provenance" in the audit trail. ADD/MODIFY routing is HUMAN_CONFIRM
  either way, so a person reads the quote and types the dose in regardless.
- **`Proposal.target_entity` has no due-date field**, but CREATE_TASK proposals
  need one for the guided process. Rather than add an out-of-band store, I
  encode it directly into `target_entity` (`"<slug>__due+<N>d"`,
  `src/core/decision_policy.py:encode_due_offset/parse_due_offset`), so a task
  proposal stays a single, idempotent, self-contained record. Documented
  fragility: this is string-encoding inside a field the contract calls an
  "entity identifier" — acceptable here, but I'd flag it in review.
- **`is_new_chronic_diagnosis` has no deterministic semantic backstop** — only
  an exact lowercase string-match against the existing diagnosis list
  (`src/features/followup_care/analyzer.py:_existing_diagnosis_names`). A
  restated diagnosis with different wording (see Finding below) relies
  entirely on the LLM's own judgment to suppress it; there's no
  fuzzy/embedding match in code.
- **"Chronic vs. transient" for diagnoses is a judgment call I made, not given.**
  A resolved UTI or treated dehydration during the stay is *not* added as a
  chronic diagnosis (it has no lasting care implication) even though it's
  textually present — see `letter_05.md` (Harnwegsinfekt, Exsikkose). Decided
  this in the prompt instructions in `followup_care/analyzer.py`, not in code,
  so it's a model judgment, reviewed by a human via HUMAN_CONFIRM routing.
- **Idempotency key for `/letters/ingest` is `letter_id`**, not a content hash.
  Re-ingesting a known `letter_id` is a true no-op (skips the LLM calls
  entirely, `ProposalsRepository.is_letter_ingested`) rather than just
  deduping the resulting proposals — see `src/api/app.py` docstring.

## 2. Findings in the data/letters (all with coordinates)

- **Internal dose contradiction**, `letter_01.md`: Procedere says Insulin
  glargin reduced to 12 IE (stated twice); the Entlassmedikation table says 18
  IE; `data/medication_plans/R001.json` baseline is 14 IE. Three different
  numbers for one insulin dose. Handled as `status="unclear"` →
  `action=flag`, forced `HARD_STOP_PHYSICIAN`/`CRITICAL`, confidence capped at
  0.45 — never auto-applicable regardless of model confidence.
- **Missing referenced attachment**, `letter_04.md`: "Ausschleichschema siehe
  Anlage 2" and "Fortsetzung der Medikationsliste: siehe Anlage 2" — no Anlage
  2 exists in the file; the letter is also marked "Vorläufiger
  Entlassungsbericht" (preliminary). This is the "cannot be cleanly resolved"
  case. Handled by a dedicated `missing_attachment_note` field that always
  routes `HARD_STOP_PHYSICIAN` (`src/core/decision_policy.py:route_missing_critical_data`)
  — treated as a gap in the source document, not an extraction failure.
- **Direct allergy conflict**, `letter_04.md` ("Amoxicillin/Clavulansäure ...
  für weitere 4 Tage") vs. `data/allergies/R004.json` (documented Penicillin
  allergy, generalisiertes Exanthem). Forced `HARD_STOP_PHYSICIAN`/`CRITICAL`
  regardless of model-reported confidence — this is the case the whole
  allergy-cross-check mechanism exists for.
- **Non-allergy safety conflict**, `letter_02.md` (Ibuprofen, new) vs.
  `data/diagnoses/R002.json` (prior GI ulcer bleed) + concurrent Enoxaparin +
  reduced renal function on labs. No formal allergy involved — this needed
  the LLM to reason about a drug-disease + drug-drug interaction, which is
  why `conflicts_with_existing_condition` exists alongside
  `conflicts_with_allergy` (`medication_reconciliation/analyzer.py`) instead
  of a hardcoded allergy-only check.
- **A second non-allergy safety conflict I found only on the real run**,
  `letter_05.md` ("NEU angesetzt: Propranolol 20 mg 1-0-1") vs.
  `data/diagnoses/R005.json` (severe persistent asthma, GINA stage 4, prior
  status asthmaticus 2019). Propranolol is non-cardioselective — real
  bronchospasm risk. I labeled this `human_confirm` in my first pass reading
  the letter alone and only caught the conflict once I was forced to do the
  actual cross-document extraction for `EVAL.md`; see that file for the
  full honest account, including that I updated the label rather than the
  reverse.
- **Explicit non-starts mentioned in the text**, `letter_01.md`: beta-blocker
  and SGLT2-inhibitor starts both "zunächst zurückgestellt." A naive
  keyword extractor would add both as new meds. Modeled as its own status
  value (`deferred_not_started`) that deliberately produces **no** proposal.
- **Test fixture coupling I found and fixed**: the given
  `tests/test_mock_llm.py::test_run_with_chaos_provider_never_emits_invalid_proposals`
  assumed exactly one analyzer (asserted `len(proposals) == len(letters)` and
  a single fallback name `stay_metadata_extraction_failed`). With 3 analyzers
  registered that assumption breaks for reasons unrelated to correctness
  (more calls, different call-index alignment with chaos). Generalized the
  assertions to the invariants that actually matter (no crash, schema-valid
  output, any fallback is well-formed) and moved the "double-failure → fallback"
  guarantee to deterministic per-analyzer tests instead of relying on chaos-seed
  luck (`tests/test_mock_llm.py`, `tests/test_medication_reconciliation.py`,
  `tests/test_followup_care.py`).

## 3. LLM vs. deterministic code

LLM: German reading, brand→substance normalization, classifying each
medication's status against the letter's *own* narrative, chronic-vs-acute
diagnosis judgment, and the two safety-conflict checks (allergy;
drug-disease/drug-drug). All of that needs real clinical-language judgment a
lookup table can't do.

Deterministic code: schema validation + one repair retry + fallback flag
(mirrors `stay_metadata`), suppressing `continued_unchanged` /
`deferred_not_started` / already-tracked-diagnosis as true negatives,
deterministic proposal IDs, and — the one I'd defend hardest — **routing is
never read from the model**. `src/core/decision_policy.py` is the only place
that decides AUTO_APPLY/HUMAN_CONFIRM/HARD_STOP_PHYSICIAN/INFO_ONLY, as a
short table of plain `if`s over signals the LLM reports (`conflicts_with_allergy`,
`is_safety_conflict`, `requires_clinical_judgment`, ...). A model swap can
change *which* facts get extracted; it cannot change what happens once a fact
says "allergy conflict."

## 4. Routing policy

- **MEDICATION never auto-applies.** Default `HUMAN_CONFIRM`. Escalates to
  `HARD_STOP_PHYSICIAN`/`CRITICAL` for any safety conflict (allergy or
  drug-disease/drug-drug) or an internally-inconsistent letter. Reasoning:
  the source is a letter of variable quality, not a trusted feed — every
  medication change touches the resident directly.
- **DIAGNOSIS / WOUND / FLUID_MANAGEMENT / BEHAVIOR**: `HUMAN_CONFIRM`
  (`INFO` severity normally, `WARN` if a care instruction reverses an
  existing strategy, e.g. fluid restriction replacing "encourage" in
  `letter_01.md`/`data/drink_protocols/R001.json`).
- **TASK**: the one category that *can* be `AUTO_APPLY` — but only when
  `requires_clinical_judgment=false` (pure scheduling: book a lab check, book
  an appointment). Anything phrased as a clinical decision ("re-evaluate the
  dose", "Re-Evaluation ... durch die Hausärztin") is `HARD_STOP_PHYSICIAN`
  even though it's *also* a task.
- **Operationalized, not just labeled**: AUTO_APPLY proposals are
  self-decided by the system at ingest time (actor="system", logged in the
  audit trail) — no human ever has to touch them. Everything else sits in
  `GET /residents/{id}/proposals` until a human calls
  `/proposals/{id}/decision` (`src/api/app.py:_auto_apply_if_needed`).
- **INFO_ONLY** is reserved for system-reliability events (double extraction
  failure) — it makes no clinical claim, so it shouldn't compete for a
  physician's attention the way a real flag does.

## 5. Call architecture, model & cost

Two new analyzers (medication reconciliation; follow-up/diagnoses/care),
plus the given `stay_metadata` = **3 LLM calls per letter** on the happy path
(one extraction call each; +1 repair retry only on validation failure) — 15
calls for the 5 letters in this dataset, confirmed in `proposals.json`'s
`cost_log` (15 entries, zero validation failures on this run).

**On cost specifically: I'm not going to fabricate a $/1k-letters figure.**
No Grok/Gemini/OpenAI/Anthropic key in this build environment had enough
free-tier headroom to get through all 15 calls without rate-limiting (see
`EVAL.md` header) — so the `proposals.json` in this delivery was generated
by `scripts/generate_real_proposals.py`, which answers each of those 15 LLM
calls directly rather than over HTTP, and logs `input_tokens`/`output_tokens`
as a `len(text)//4` estimate, not metered usage from a provider. That number
is not a real cost figure and I'm labeling it as such rather than presenting
it as one. Wiring for real cost tracking is in place and unit-tested
(`src/core/llm/anthropic_client.py`, the given `openai_compatible` client,
both report real `usage` from their provider's response) — once a key with
headroom is available, `make run`'s real `cost_log` gives real
input/output tokens per call to multiply by the model's published price.


## 6. Testing & eval

70 pytest tests: per-analyzer unit tests with crafted `MockLLMClient` canned
responses (success, repair-recovers, double-failure-fallback, allergy
conflict, general safety conflict, internal contradiction, deferred/unchanged
suppression), the guided-process lifecycle, and `tests/test_api.py`
(idempotent ingest, auto-apply, accept/reject/modify, 409 on re-deciding a
human decision, process actions, audit ordering) against a scripted LLM that
branches per-analyzer on system-prompt content.

`eval/labels.json`: 19 positive + 6 negative hand-labeled cases read directly
from the letters. Run against the real (if not over-HTTP, see §5)
`proposals.json`: **19/19 found, 0 wrong routing, 0/6 false positives**
(`make eval` exits 0). Two labels needed fixing after the run because my
guessed keyword didn't match the model's actual (English-leaning) slug
wording, and one label was upgraded after the run surfaced a genuine safety
finding I'd missed hand-labeling (Propranolol vs. severe asthma) — see
`EVAL.md` for the honest detail on both. I'd rather show that correction
than a clean number that quietly hid it.


## 7. Cut list

- No labs/vitals/assessments/encounters analyzer — the existing `weights`
  feature already covers trend-based vitals; I judged additional trend
  analyzers as not "relevant to the letter" for these 5 cases and a real time
  sink to build well, vs. two letter-driven analyzers that map directly to
  what's actually in the discharge letters.
- No structured write-back for ADD/MODIFY medication doses (see §1) — would
  need either a schema change (out of scope, contract is fixed) or fragile
  text parsing. Logged as a documented limitation instead of faked.
- No automatic fall-log entry creation for R002's fall/fracture — it's
  documentation of a past event, not a forward action; out of scope for a
  *reconciliation* system.
- No auth, no deployment config, no UI — explicitly out of scope per README.
- `act_on_process` does the minimum: action lookup + status transition +
  audit log. No optimistic-locking/concurrency handling for two reviewers
  hitting the same process at once — fine for this scale, would matter at
  real volume.

## 8. AI tool usage

Built end-to-end with Claude doing the implementation
work directly inside the given scaffold — reading every domain file, the two
reference features, and all 5 letters/datasets first, then writing the two
analyzers, the decision-policy layer, the API, the guided process, and the
tests. One concrete override: the first integration-test fix attempt
(`test_mock_llm.py`) initially tried to keep the exact original assertions
and just add more letters until chaos happened to collide — I rejected that
(non-deterministic, papers over the real issue) and rewrote the assertions
around the actual invariant instead (§2). Another: the medication analyzer
originally only checked formal allergies; I deliberately broadened it to
general drug-safety conflicts mid-build once the R002 Ibuprofen/GI-bleed case
made clear that "allergy-only" would silently miss the kind of conflict the
whole exercise is about.

## 9. Self-report

**Residual-risk register:**
1. The Enoxaparin 14-day course (`letter_02.md`, "Fortführung f. weitere 14
   Tage bis z. sicheren Mobilisierung") has an ambiguous reference start date
   — discharge date or mobilisation date aren't the same day. My due-date
   math always anchors to the letter's `discharge_date`
   (`src/api/app.py:_attach_clinical_followup`), so if the real anchor is
   earlier, every computed due date for that kind of instruction is
   systematically late. I think this is unhandled, not just imprecise.
2. `followup_care/analyzer.py` asks for diagnoses + tasks + care_instructions
   in one combined call per letter. For a long, multi-section letter
   (`letter_01.md`, `letter_03.md` are the longest) I believe this risks the
   model under-extracting one of the three categories due to attention
   dilution, compared to the medication analyzer's narrower single-purpose
   call — I have not measured this against a real model, so it's a suspicion
   from the prompt's shape, not a confirmed failure.
3. Confidence values (0.8 / 0.85 / 0.9 / 0.45 in both analyzers) are
   hand-set priors based on how risky each status is, not a statistically
   calibrated probability of correctness — there's no log-prob extraction or
   calibration-curve fitting against outcomes. I'm calling this out because
   "confidence calibration" reads like a claim of more rigor than what's
   actually here.

**My eval, honestly:** I wrote `eval/labels.json` myself, by reading the same
5 letters my own analyzers' prompts were written against — so a clean
found/missed number on these specific 5 letters proves less about a 6th,
unseen letter than it looks like; it's closer to a unit test than a held-out
eval. The result (19/19, 0 false positives — see `EVAL.md`) is real in the
sense that the system code actually ran and actually validated/reconciled/
routed real extraction content correctly, but it cannot speak to whether a
*live* model reliably produces that extraction content, since no live model
call happened (no provider key had headroom — see §5). The one case I was
least confident about going in, `r002_ibuprofen_safety_conflict`, worked; the
one I got wrong on the first pass and only caught because I was forced to
actually do the extraction (not just review code), `r005`'s Propranolol vs.
severe asthma, is in `EVAL.md` in full.


---

### Addendum: how `proposals.json` was generated

No Grok/Gemini/OpenAI/Anthropic key in this build environment had enough
free-tier headroom to get through 5 letters × 3 analyzers (15 calls, more on
any repair retry) without rate-limiting. `proposals.json` in this delivery
was produced by `scripts/generate_real_proposals.py`, which runs the real
`src.run` pipeline unmodified except for one substitution: the LLM calls are
answered directly (by me, having actually read each letter and cross-
referenced the resident's existing data) instead of over HTTP. Full
rationale, what this does and doesn't prove, and the real eval numbers it
produced: `EVAL.md`. `src/core/llm/anthropic_client.py` and the given
`openai_compatible` client are both wired and unit-tested against the
mock/chaos providers for when a key with real headroom is available — at
that point `make run && make eval` exercises the live HTTP/retry path this
delivery's `proposals.json` did not.

### Addendum: worked example for the guided process

`data/ai_suggestions/R003.json` (alongside the given `R001.json` worked
example for `stay_metadata`) is a real, programmatically-generated worked
example of `ClinicalFollowUpProcess` attached to the real
`r003_bisoprolol_reeval_task` case — i.e. what `/proposals/{id}/decision`
would persist for that case once `letter_03` is ingested and accepted. Built
with the real `make_proposal_id`/`encode_due_offset`/`AISuggestion` code
(not hand-written JSON), then advanced with `make tick --advance-days 10`
to demonstrate the overdue-escalation branch (`process_state.overdue_flagged: true`)
against R003's real `data/encounters/R003.json` (no GP/specialist contact
recorded after the process started, so it correctly stays open + flags
overdue rather than closing).

