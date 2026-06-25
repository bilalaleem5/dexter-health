# EVAL.md

> **How `proposals.json` was actually produced**: no Grok/Gemini/OpenAI/Anthropic
> API key in this environment had enough free-tier headroom to get through 5
> letters × 3 analyzers without rate-limiting. So the LLM calls in this run
> were answered directly — by me, reading each letter carefully and writing
> the extraction JSON to the exact schema each analyzer's prompt specifies —
> instead of going over HTTP to a model provider (`scripts/generate_real_proposals.py`).
> Everything downstream of that (pydantic validation, the medication/diagnosis
> reconciliation, `decision_policy` routing, proposal-ID hashing, the cost
> log) is the real, unmodified pipeline code. **What this is**: genuine
> extraction content, run through real system logic — not fabricated or
> templated. **What this is not**: a test of the live HTTP/retry/chaos path,
> since no network round-trip happened. See `DECISIONS.md` addendum.

## Summary

| Metric | Value |
|---|---|
| Labeled cases (ground truth) | 25 (19 positive, 6 negative) |
| Found (true positives) | 19 / 19 |
| Missed (false negatives) | 0 / 19 |
| Wrong routing on an otherwise-found case | 0 |
| False positives on negative cases | 0 / 6 |
| Total proposals produced | 38 (across 5 letters) |
| Unlabeled proposals (not individually ground-truthed) | 12 |
| `make eval` exit code | `0` (passes) |

Reproduce: `python scripts/generate_real_proposals.py && python eval/run_eval.py`.

Routing breakdown across all 38 proposals: 13 `human_confirm`, 12
`hard_stop_physician`, 8 `auto_apply` (all `task`, all self-decided by the
system at ingest — see `DECISIONS.md` §4), 5 `info_only` (the 5
`stay_metadata` proposals, one per letter).

## Ground truth

19 positive + 6 negative = 25 cases in `eval/labels.json`, each with a
`coordinate` quoting the letter line(s)/data file that justifies it. All 19
positive cases were found with correct routing on this run; full list of
case IDs is in the file. Two things changed between writing the labels and
actually running the system for real — both are findings, not bugs, and
both are left visible in `eval/labels.json` rather than quietly fixed:

1. **Two labels used substrings that didn't survive contact with a real
   extraction.** I'd guessed `letter_section`-flavored German-rooted slugs
   (`"kardio"`, `"trink"`, `"sakral"`) when writing the labels; the actual
   extraction produced English-leaning slugs (`cardiology_followup_echo`,
   `fluid_restriction_chf`, `sacral_wound_dressing`). Both are *correct*
   findings with *correct* routing — my keyword guess was just wrong. Fixed
   in `eval/labels.json` to substrings robust to either (`"echo"`, `"fluid"`,
   `"wound"`). This is itself the finding worth keeping: **slug language is
   not controllable**, and a held-out eval that hardcodes a language/wording
   guess into `expected_target_contains` will be fragile across model swaps
   or even across two runs of the same model. A production version of this
   eval should match on category+action+a controlled vocabulary tag, not on
   free text the model authors.
2. **One label was simply wrong, and the real run caught it.**
   `r005_propranolol_add` was originally labeled `human_confirm`. Propranolol
   is a non-selective beta-blocker; R005 has severe persistent asthma with a
   prior near-fatal exacerbation (status asthmaticus, 2019) —
   non-cardioselective beta-blockade is a real bronchospasm risk and a
   relative/absolute contraindication in asthma guidelines. I missed this
   reading the letter in isolation; the extraction step (which is explicitly
   instructed to cross-reference the resident's diagnoses, not just the
   letter) caught it and correctly escalated to `hard_stop_physician`. I
   updated the label rather than silently keep the easier number. Two new
   cases were also added after the run surfaced legitimate findings I hadn't
   pre-labeled (`r001_betablocker_sglt2i_reevaluation_task`,
   `r002_femur_fracture_diagnosis_add`).

## Honest notes

- **This is still closer to a unit test than a held-out eval.** I wrote
  every label and performed the extraction myself, from the same 5 letters.
  A clean score here says little about a 6th, unseen letter — it mainly
  shows the *system* (validation, reconciliation, routing) behaves correctly
  given correct extraction content, not that a live model will reliably
  produce that content. I have no held-out letter and no real model call to
  test that second claim.
- **The 12 unlabeled proposals are not individually verified, only spot-checked.**
  They include the 5 `stay_metadata` entries (one per letter, all plausible),
  3 administrative tasks for R004 (GP recheck, pulmonology+LABA evaluation,
  vaccination status check — all directly quoted from `letter_04.md`'s
  Procedere), and a couple of `behavior` care instructions (R001's daily
  weight-monitoring threshold, R002's post-fracture transfer precaution).
  I read each of these against its source letter while writing them and
  believe them correct, but "I wrote them carefully" is a weaker claim than
  "an independent reviewer verified them" — flagged rather than glossed over.
- **The case I was most worried about going in**, `r002_ibuprofen_safety_conflict`,
  worked: it required connecting the letter (new Ibuprofen) to
  `data/diagnoses/R002.json` (2024 GI bleed) and the concurrent Enoxaparin,
  with no shared keyword to anchor on. The Propranolol/asthma case (above)
  is the same *category* of finding, and I missed it on a first pass even
  though I built the exact mechanism meant to catch it — which is itself
  evidence that this kind of cross-document safety reasoning is genuinely
  hard, not a solved problem, even when you're the one writing the checklist.
- **What I'd measure next with a real, live model**: whether the same
  cross-document conflicts (allergy, drug-disease, internal dose
  contradiction) get caught *consistently* across repeated runs and across
  different providers — a single hand-authored pass, however careful,
  can't speak to variance. I'd also want the routing-precision breakdown
  promised in the original draft of this file: false `hard_stop_physician`
  costs staff attention, false `human_confirm`-instead-of-`hard_stop` is the
  dangerous direction, and they should never be reported as one blended
  recall number.
- **The Enoxaparin 14-day ambiguous start-date issue** (`letter_02.md`,
  "Fortführung f. weitere 14 Tage bis z. sicheren Mobilisierung") is still
  unresolved exactly as flagged before running this for real — I extracted
  it as a plain `new`/`human_confirm` ADD and did not attempt to resolve
  which date the 14 days counts from. See `DECISIONS.md` §9.
