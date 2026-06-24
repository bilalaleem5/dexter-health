# EVAL.md

> **No LLM API key was available in the environment this was built in.**
> The table below is from `LLM_PROVIDER=mock` (`make run && make eval`),
> which is a harness smoke test, not a real evaluation — the mock client
> returns a fixed `stay_metadata`-shaped response to every call regardless
> of schema, so the medication/follow-up analyzers correctly see "no
> matching findings" and produce nothing. That's expected mock behavior
> (see `README.md`), not a bug. **To get a real number: add a key to `.env`
> (see `.env.example`, both `openai_compatible` and a native `anthropic`
> provider are wired and tested), then run `make run && make eval`.**

## Summary

| Metric | Value (mock smoke test — NOT a real result) | Real result |
|---|---|---|
| Labeled expected proposals (ground truth) | 23 (17 positive, 6 negative) | — |
| Found (true positives) | 0 / 17 | *(run with a real key)* |
| Missed (false negatives) | 17 / 17 | *(run with a real key)* |
| Wrong routing on an otherwise-found case | 0 | *(run with a real key)* |
| Hallucinated (proposals with no basis in letter/data) | not measurable from mock — see "Honest notes" | *(needs manual spot-check after a real run)* |
| False positives on negative cases | 0 / 6 | *(run with a real key)* |
| Eval gate (`make eval` exit code) | `1` (fails — correctly, since 17 cases were missed) | — |

The mock run does demonstrate the harness mechanics work: it ran 5 letters
through 3 analyzers (15 LLM calls), produced 5 schema-valid `stay_metadata`
proposals, matched 0 of them against the labels (correct — none of the 23
labels are stay_metadata cases), and the eval gate correctly failed rather
than silently passing. Full console output is reproducible with
`LLM_PROVIDER=mock make run && python eval/run_eval.py`.

## Ground truth

All 23 cases are in `eval/labels.json`, each with a `coordinate` field
quoting the letter line(s) (and/or the existing data file) that justifies
it. Summary:

| case_id | letter | expect | routing |
|---|---|---|---|
| r001_insulin_dose_contradiction | letter_01 | medication/flag, "insulin" | hard_stop_physician |
| r001_torasemid_dose_increase | letter_01 | medication/modify, "torasemid" | human_confirm |
| r001_metformin_gp_reeval_task | letter_01 | task/create_task, "metformin" | hard_stop_physician |
| r001_fluid_restriction_conflict | letter_01 | fluid_management/modify, "trink" | human_confirm |
| r001_cardiology_followup_task | letter_01 | task/create_task, "kardio" | auto_apply |
| r002_ibuprofen_safety_conflict | letter_02 | medication/add, "ibuprofen" | hard_stop_physician |
| r002_enoxaparin_add | letter_02 | medication/add, "enoxaparin" | human_confirm |
| r002_dxa_osteoporosis_task | letter_02 | task/create_task, "osteoporos" | hard_stop_physician |
| r003_marcumar_stop | letter_03 | medication/stop, "phenprocoumon" | human_confirm |
| r003_apixaban_add | letter_03 | medication/add, "apixaban" | human_confirm |
| r003_bisoprolol_stop | letter_03 | medication/stop, "bisoprolol" | human_confirm |
| r003_bisoprolol_reeval_task | letter_03 | task/create_task, "bisoprolol" | hard_stop_physician |
| r004_amoxicillin_allergy_conflict | letter_04 | medication/add, "amoxicillin" | hard_stop_physician |
| r004_missing_anlage2 | letter_04 | medication/flag, "missing_medication_attachment" | hard_stop_physician |
| r004_ramipril_dose_increase | letter_04 | medication/modify, "ramipril" | human_confirm |
| r005_propranolol_add | letter_05 | medication/add, "propranolol" | human_confirm |
| r005_sacral_wound_add | letter_05 | wound/add, "sakral" | human_confirm |
| **negative:** neg_r001_betablocker_deferred | letter_01 | must NOT fire medication/add "betablocker" | — |
| **negative:** neg_r001_sglt2i_deferred | letter_01 | must NOT fire medication/add "sglt2" | — |
| **negative:** neg_r001_heart_failure_dx_already_tracked | letter_01 | must NOT fire diagnosis/add "herzinsuffizienz" (already tracked, different wording) | — |
| **negative:** neg_r002_colecalciferol_unchanged | letter_02 | must NOT fire any medication proposal "colecalciferol" | — |
| **negative:** neg_r003_pantoprazol_unchanged | letter_03 | must NOT fire any medication proposal "pantoprazol" | — |
| **negative:** neg_r005_inpatient_antibiotic_not_discharge_med | letter_05 | must NOT fire medication/add "cefuroxim" (course completed entirely in-stay) | — |

Matching logic (substring on `target_entity`, exact on category/action/IDs)
is documented in `eval/run_eval.py`'s module docstring.

## Honest notes

- **This is closer to a unit test than a held-out eval.** I wrote every
  label myself from the same 5 letters my own prompts were written against
  and tuned against. A clean score here would say very little about a 6th,
  unseen letter. I have no held-out letter to test that claim with.
- **I genuinely could not decide on one case while writing the labels**:
  `r002_enoxaparin_add`. The letter gives a 14-day continuation window
  ("Fortführung f. weitere 14 Tage bis z. sicheren Mobilisierung") but the
  reference start date for that window is ambiguous — discharge date, or
  the date of "sichere Mobilisierung" (which isn't itself dated in the
  letter)? I labeled it as a plain `human_confirm` ADD and did **not** add a
  separate "ambiguous duration" sub-case, because I couldn't define a
  precise, falsifiable expected answer for it myself. This is the same
  ambiguity flagged as residual risk #1 in `DECISIONS.md`.
- **The case I'd bet against my own system on, if run for real**:
  `r002_ibuprofen_safety_conflict`. Every other hard_stop case in the labels
  has one strong textual anchor (an explicit dose number, an explicit
  "abgesetzt", an allergy record with the word "Penicillin" sitting right
  next to a drug literally named "Amoxicillin"). This one requires
  synthesizing three separate documents (letter + diagnoses.json showing a
  2024 GI bleed + the absence of any allergy record) into one safety
  judgment, with no shared keyword to anchor on. I added
  `conflicts_with_existing_condition` specifically to make this catchable
  (see DECISIONS.md §1/§8), but I have not been able to verify it actually
  fires against a real model.
- **Hallucination rate is not really measured here**, only proxied by "did a
  negative case fire" (0/6) plus the unlabeled-proposal list `run_eval.py`
  prints for manual spot-checking. A true hallucination rate needs a human
  to read every produced proposal against its source letter, including
  ones I didn't think to write a negative case for — out of scope for the
  time available; flagged rather than glossed over.
- **What I'd measure next with more time / a real model**: routing
  precision specifically on the `hard_stop_physician` bucket (false
  hard-stops cost staff time and erode trust in the alerts that matter;
  false `human_confirm`-instead-of-`hard_stop` is the dangerous direction) —
  I'd want that broken out separately from overall recall, not blended into
  one number.
