# Data Schema

One JSON file per resident in each subdirectory (`R001.json` … `R005.json`). The **canonical match key** for medications is `internal_name`; `wirkstoff` is the active substance (use it for substance-level reasoning). Residents are linked to letters via `letters/index.json`.

| File | Shape | Notes |
|---|---|---|
| `data/residents.json` | list of `{resident_id, name, birth_year, sex, room, care_level, hausarzt: {name, phone}, legal_guardian}` | `care_level` = Pflegegrad 1–5. `hausarzt` is the addressee for GP follow-up tasks. |
| `data/allergies/R*.json` | `{resident_id, allergies: [{substance, reaction, noted_at}]}` | May be empty. |
| `data/diagnoses/R*.json` | `{resident_id, diagnoses: [{diagnose, internal_diagnose, icd10, noted_at}]}` | `internal_diagnose` is the canonical key. |
| `data/medication_plans/R*.json` | `{resident_id, updated_at, medications_planned: [...], medications_on_demand: [...]}` | Planned: `{internal_name, name, wirkstoff, strength, unit, dosage, indication, prescriber}`. `dosage` is the German scheme `"1-0-1-0"` = morning-noon-evening-night. On-demand (Bedarfsmedikation) additionally has `max_daily_dose` instead of `dosage`. **State as of BEFORE the hospital stay.** |
| `data/care_notes/R*.json` | `{resident_id, notes: [{timestamp, shift, author, text}]}` | Progress documentation (Verlaufsdoku), German free text. Notes stop at hospital admission; one return-day note exists. |
| `data/drink_protocols/R*.json` | `{resident_id, target_ml, strategy, entries: [{timestamp, amount_ml}]}` | `strategy`: `encourage` or `restrict`. |
| `data/vitals/R*.json` | `{resident_id, weights: [{created_at, weight}], measurements: [{created_at, type, value, unit}]}` | `weights` feeds the weights reference feature. `measurements.type` ∈ `bp_sys`/`bp_dia`/`pulse`/`temp`/`spo2`/`blood_glucose`, a sparse multi-month series. |
| `data/wounds/R*.json` | `{resident_id, wounds: [{wound_id, location, type, grade, status, noted_at, healed_at}]}` | `status`: `active` or `healed`. |
| `data/medication_history/R*.json` | `{resident_id, events: [{date, action, internal_name, wirkstoff, from_dosage?, to_dosage?, reason, prescriber}]}` | Longitudinal med log. `action` ∈ `started`/`stopped`/`changed`/`dose_changed`. Lets you tell **chronic vs. newly-introduced** apart from the data, not just the letter. The current `medication_plans` is the resulting state. |
| `data/labs/R*.json` | `{resident_id, panels: [{collected_at, source, values: [{analyte, value, unit, ref_range, flag}]}]}` | Periodic lab panels over ~2 years. `flag` ∈ `H`/`L`/`""`. Mostly chronic-stable trends. |
| `data/assessments/R*.json` | `{resident_id, assessments: [{date, instrument, score, max?, note}]}` | Standardized assessments over time (`Barthel`, `MMST`, `Braden`, `Esslinger`, `Timed-up-and-go`). A functional/cognitive baseline. |
| `data/encounters/R*.json` | `{resident_id, encounters: [{date_from, date_to, type, facility, reason, key_findings, med_changes, summary}]}` | Structured summaries of prior hospital/specialist contacts. `type` ∈ `hospital`/`gp`/`specialist`/`er`. Historical context; their `med_changes` are already reflected in the current plan. |
| `data/fall_log/R*.json` | `{resident_id, falls: [{date, time, location, circumstances, injury, measures, witnessed}]}` | Fall protocol (Sturzprotokoll). |
| `data/ai_suggestions/R*.json` | `{resident_id, suggestions: [AISuggestion]}` | Existing suggestions in the production format (see `src/core/domain/suggestion.py`); R001 contains a worked example. |
| `letters/index.json` | list of `{letter_id, resident_id, hospital, admission_date, discharge_date, file}` | Letter ↔ resident mapping. |

In analyzers you receive the loaded per-resident data as a dict keyed by type: `medication_plan`, `allergies`, `diagnoses`, `care_notes`, `drink_protocol`, `vitals`, `wounds`, `medication_history`, `labs`, `assessments`, `encounters`, `fall_log` (see `RESIDENT_DATA_DIRS` in `src/core/repositories.py`).

> **Scope note.** This is a realistic multi-year record and is deliberately broader than any single analyzer needs. Not every data type warrants its own analyzer — reconcile against what is relevant to the letter and **cut the rest deliberately** (say so in DECISIONS.md). Most of the long-term data is chronic-stable context; the signal is in what *changed* and what the letter introduces.
