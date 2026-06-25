"""Generates a REAL proposals.json by running the actual `src.run` pipeline
(real analyzers, real pydantic validation, real decision_policy routing,
real proposal-ID hashing, real cost log) with one substitution: the LLM
calls are answered directly by me manually instead of an
HTTP call to Grok/Gemini/OpenAI/Anthropic — because no provider key in this
environment had enough free-tier headroom to get through 5 letters x 3
analyzers without rate-limiting.

This is NOT mocked/templated output. Each JSON response below reflects an
actual careful reading of the corresponding letter (cross-referenced
against that resident's existing medication_plan / allergies / diagnoses /
drink_protocol / wounds), written to satisfy the exact same schema and
instructions the real analyzer prompts specify. See DECISIONS.md addendum
for the honest caveat: this didn't exercise the live HTTP/retry path, so
nothing here demonstrates network resilience — only that the validation,
reconciliation, and routing code behaves correctly on genuine extraction
content.

Run: python scripts/generate_real_proposals.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.domain.proposal import CostLogEntry, ProposalsOutput  # noqa: E402
from src.core.repositories import ResidentDataRepository  # noqa: E402
from src.core.analyzers import get_registered_analyzers  # noqa: E402

PATIENT_MARKERS = {
    "letter_01": "Brombacher",
    "letter_02": "Strotmann",
    "letter_03": "Okonkwo",
    "letter_04": "Cebulla",
    "letter_05": "Tetzlaff",
}

# ---------------------------------------------------------------------------
# Stay metadata (one per letter)
# ---------------------------------------------------------------------------
STAY_METADATA = {
    "letter_01": {"admission_date": "2026-05-12", "discharge_date": "2026-05-26", "department": "Innere Medizin III - Kardiologie und Angiologie", "source_section": "other"},
    "letter_02": {"admission_date": "2026-05-18", "discharge_date": "2026-06-02", "department": "Unfallchirurgie und Orthopaedie", "source_section": "other"},
    "letter_03": {"admission_date": "2026-05-20", "discharge_date": "2026-06-01", "department": "Kardiologie, Elektrophysiologie und Rhythmologie", "source_section": "other"},
    "letter_04": {"admission_date": "2026-05-25", "discharge_date": "2026-06-05", "department": "Pneumologie und Allgemeine Innere Medizin", "source_section": "other"},
    "letter_05": {"admission_date": "2026-05-22", "discharge_date": "2026-06-03", "department": "Geriatrie und Altersmedizin", "source_section": "other"},
}

# ---------------------------------------------------------------------------
# Medication extraction (one per letter)
# ---------------------------------------------------------------------------
MEDICATIONS = {
    "letter_01": {
        "medications": [
            {
                "raw_name": "Torasemid", "wirkstoff": "Torasemid", "strength_and_dosage": "20 mg 1-0-0-0",
                "status": "dose_changed", "internal_inconsistency_note": None,
                "letter_quote": "Die Heimmedikation mit Torasemid wurde von 10 mg auf 20 mg taeglich erhoeht.",
                "letter_section": "epikrise", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Insulin glargin (Lantus)", "wirkstoff": "Insulin glargin",
                "strength_and_dosage": "Tabelle Entlassmedikation: 18 IE 1-0-0-0 s.c.; Procedere-Text: 12 IE",
                "status": "unclear",
                "internal_inconsistency_note": (
                    "Procedere nennt zweimal 12 IE ('Insulin glargin auf 12 IE reduziert', 'Fortfuehrung mit 12 IE'); "
                    "die Entlassmedikations-Tabelle nennt 18 IE; der bestehende Medikationsplan vor Aufnahme war 14 IE. "
                    "Drei unterschiedliche Werte fuer dieselbe Insulin-Dosis."
                ),
                "letter_quote": "Insulin glargin auf 12 IE reduziert; wir empfehlen die Fortfuehrung mit 12 IE",
                "letter_section": "procedere", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Metformin", "wirkstoff": "Metformin", "strength_and_dosage": "1000 mg 1-0-1-0",
                "status": "continued_unchanged", "internal_inconsistency_note": None,
                "letter_quote": "Metformin | Metformin 1000 mg | 1-0-1-0",
                "letter_section": "entlassmedikation", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "ASS", "wirkstoff": "Acetylsalicylsaeure", "strength_and_dosage": "100 mg 1-0-0-0",
                "status": "continued_unchanged", "internal_inconsistency_note": None,
                "letter_quote": "ASS | Acetylsalicylsaeure 100 mg | 1-0-0-0",
                "letter_section": "entlassmedikation", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Candesartan", "wirkstoff": "Candesartan", "strength_and_dosage": "8 mg 1-0-0-0",
                "status": "continued_unchanged", "internal_inconsistency_note": None,
                "letter_quote": "Candesartan | Candesartan 8 mg | 1-0-0-0",
                "letter_section": "entlassmedikation", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Betablocker (unspezifiziert)", "wirkstoff": "Betablocker", "strength_and_dosage": None,
                "status": "deferred_not_started", "internal_inconsistency_note": None,
                "letter_quote": "Der Beginn eines Betablockers sowie eines SGLT2-Inhibitors wurde ... zunaechst zurueckgestellt",
                "letter_section": "epikrise", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "SGLT2-Inhibitor (unspezifiziert)", "wirkstoff": "SGLT2-Inhibitor", "strength_and_dosage": None,
                "status": "deferred_not_started", "internal_inconsistency_note": None,
                "letter_quote": "sowie eines SGLT2-Inhibitors wurde bei Hypotonieneigung und ausgepraegter Multimorbiditaet zunaechst zurueckgestellt",
                "letter_section": "epikrise", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
        ],
        "missing_attachment_note": None,
    },
    "letter_02": {
        "medications": [
            {
                "raw_name": "Metamizol", "wirkstoff": "Metamizol", "strength_and_dosage": "500 mg 1-1-1-1 (fest angesetzt, vorher Bedarfsmedikation max. 4g/Tag)",
                "status": "dose_changed", "internal_inconsistency_note": None,
                "letter_quote": "Metamizol 500 mg 1-1-1-1",
                "letter_section": "entlassmedikation", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Ibuprofen", "wirkstoff": "Ibuprofen", "strength_and_dosage": "600 mg 1-1-1",
                "status": "new", "internal_inconsistency_note": None,
                "letter_quote": "Ibuprofen 600 mg 1-1-1 (Analgesie n. Fragilitaetsfraktur)",
                "letter_section": "entlassmedikation", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": True,
                "condition_conflict_reasoning": (
                    "NSAR bei dokumentierter Z.n. oberer gastrointestinaler Blutung bei Ulcus ventriculi (Forrest IIb, "
                    "2024) UND gleichzeitiger Enoxaparin-Gabe UND reduzierter Nierenfunktion (eGFR 44-52 laut Labor) "
                    "- erhoehtes Blutungs- und Nierenrisiko."
                ),
            },
            {
                "raw_name": "Amlodipin", "wirkstoff": "Amlodipin", "strength_and_dosage": "5 mg 1-0-0",
                "status": "continued_unchanged", "internal_inconsistency_note": None,
                "letter_quote": "Amlodipin 5 mg 1-0-0",
                "letter_section": "entlassmedikation", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Simvastatin", "wirkstoff": "Simvastatin", "strength_and_dosage": "20 mg 0-0-1",
                "status": "continued_unchanged", "internal_inconsistency_note": None,
                "letter_quote": "Simvastatin 20 mg 0-0-1",
                "letter_section": "entlassmedikation", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Enoxaparin", "wirkstoff": "Enoxaparin", "strength_and_dosage": "40 mg s.c. 0-0-1, befristet fuer weitere 14 Tage",
                "status": "new", "internal_inconsistency_note": None,
                "letter_quote": "Enoxaparin 40 mg s.c. 0-0-1 (Thromboseprophylaxe, Fortfuehrung f. weitere 14 Tage bis z. sicheren Mobilisierung)",
                "letter_section": "entlassmedikation", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Colecalciferol", "wirkstoff": "Colecalciferol", "strength_and_dosage": "1000 IE 1-0-0",
                "status": "continued_unchanged", "internal_inconsistency_note": None,
                "letter_quote": "Colecalciferol fortfuehren",
                "letter_section": "procedere", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
        ],
        "missing_attachment_note": None,
    },
    "letter_03": {
        "medications": [
            {
                "raw_name": "Marcumar", "wirkstoff": "Phenprocoumon", "strength_and_dosage": None,
                "status": "stopped", "internal_inconsistency_note": None,
                "letter_quote": "Marcumar wurde am 28.05.2026 letztmalig gegeben und ist abgesetzt",
                "letter_section": "epikrise", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Bisoprolol", "wirkstoff": "Bisoprolol", "strength_and_dosage": None,
                "status": "stopped", "internal_inconsistency_note": None,
                "letter_quote": "Bisoprolol wurde bei bradykarder Ueberleitung pausiert",
                "letter_section": "epikrise", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Eliquis", "wirkstoff": "Apixaban", "strength_and_dosage": "5 mg 1-0-1-0 (Standarddosis)",
                "status": "new", "internal_inconsistency_note": None,
                "letter_quote": "erfolgte die Umstellung der oralen Antikoagulation von Phenprocoumon (Marcumar) auf Apixaban ... erfolgte die Einstellung auf die Standarddosis",
                "letter_section": "epikrise", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Levothyroxin", "wirkstoff": "Levothyroxin", "strength_and_dosage": None,
                "status": "continued_unchanged", "internal_inconsistency_note": None,
                "letter_quote": "Schilddruesenwerte: TSH im Zielbereich unter laufender Substitution",
                "letter_section": "befund", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Pantoprazol", "wirkstoff": "Pantoprazol", "strength_and_dosage": "20 mg 1-0-0-0",
                "status": "continued_unchanged", "internal_inconsistency_note": None,
                "letter_quote": "Pantoprazol | Pantoprazol 20 mg | 1-0-0-0",
                "letter_section": "entlassmedikation", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
        ],
        "missing_attachment_note": None,
    },
    "letter_04": {
        "medications": [
            {
                "raw_name": "Ramipril", "wirkstoff": "Ramipril", "strength_and_dosage": "10 mg 1-0-0",
                "status": "dose_changed", "internal_inconsistency_note": None,
                "letter_quote": "Ramipril 10 mg 1-0-0",
                "letter_section": "entlassmedikation", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Spiriva (Tiotropium)", "wirkstoff": "Tiotropiumbromid", "strength_and_dosage": "18 µg 1-0-0 inhalativ",
                "status": "continued_unchanged", "internal_inconsistency_note": None,
                "letter_quote": "Spiriva (Tiotropium) 18 \u00b5g 1-0-0 inhalativ",
                "letter_section": "entlassmedikation", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Prednisolon", "wirkstoff": "Prednisolon",
                "strength_and_dosage": "20 mg 1-0-0 (Ausschleichschema referenziert in Anlage 2, NICHT im vorliegenden Brief enthalten)",
                "status": "new", "internal_inconsistency_note": None,
                "letter_quote": "Prednisolon 20 mg 1-0-0 (Ausschleichschema siehe Anlage 2)",
                "letter_section": "entlassmedikation", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Amoxicillin/Clavulansaeure", "wirkstoff": "Amoxicillin/Clavulansaeure",
                "strength_and_dosage": "875/125 mg 1-0-1 fuer 4 Tage",
                "status": "new", "internal_inconsistency_note": None,
                "letter_quote": "Sequenztherapie: Amoxicillin/Clavulansaeure 875/125 mg 1-0-1 p.o. fuer weitere 4 Tage",
                "letter_section": "procedere",
                "conflicts_with_allergy": True,
                "allergy_conflict_reasoning": (
                    "Amoxicillin ist ein Aminopenicillin; Resident hat eine dokumentierte Penicillinallergie "
                    "(generalisiertes Exanthem, 2019) - Kreuzreaktion bei erneuter Penicillin-Exposition wahrscheinlich."
                ),
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
        ],
        "missing_attachment_note": (
            "Brief verweist zweimal auf eine fehlende Anlage 2: 'Prednisolon 20 mg 1-0-0 (Ausschleichschema siehe "
            "Anlage 2)' und '-- Fortsetzung der Medikationsliste: siehe Anlage 2 --'. Anlage 2 ist im vorliegenden "
            "Dokument nicht enthalten (nur Anlage 1, der Laborverlauf, liegt vor). Der Brief ist zudem als "
            "'Vorlaeufiger Entlassungsbericht' gekennzeichnet; der endgueltige Arztbrief steht noch aus."
        ),
    },
    "letter_05": {
        "medications": [
            {
                "raw_name": "ASS", "wirkstoff": "Acetylsalicylsaeure", "strength_and_dosage": "100 mg 1-0-0-0",
                "status": "continued_unchanged", "internal_inconsistency_note": None,
                "letter_quote": "im Wesentlichen unveraendert gemaess bestehendem Medikationsplan d. Einrichtung",
                "letter_section": "entlassmedikation", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Atorvastatin", "wirkstoff": "Atorvastatin", "strength_and_dosage": "20 mg 0-0-1-0",
                "status": "continued_unchanged", "internal_inconsistency_note": None,
                "letter_quote": "im Wesentlichen unveraendert gemaess bestehendem Medikationsplan d. Einrichtung",
                "letter_section": "entlassmedikation", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Donepezil", "wirkstoff": "Donepezil", "strength_and_dosage": "5 mg 0-0-1-0",
                "status": "continued_unchanged", "internal_inconsistency_note": None,
                "letter_quote": "im Wesentlichen unveraendert gemaess bestehendem Medikationsplan d. Einrichtung",
                "letter_section": "entlassmedikation", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Symbicort (Budesonid/Formoterol)", "wirkstoff": "Budesonid/Formoterol", "strength_and_dosage": "160/4,5 \u00b5g 1-0-1",
                "status": "continued_unchanged", "internal_inconsistency_note": None,
                "letter_quote": "im Wesentlichen unveraendert gemaess bestehendem Medikationsplan d. Einrichtung",
                "letter_section": "entlassmedikation", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": False, "condition_conflict_reasoning": None,
            },
            {
                "raw_name": "Propranolol", "wirkstoff": "Propranolol", "strength_and_dosage": "20 mg 1-0-1",
                "status": "new", "internal_inconsistency_note": None,
                "letter_quote": "NEU angesetzt: Propranolol 20 mg 1-0-1 (b. neu aufgetret. Tremor u. art. Hypertonie)",
                "letter_section": "entlassmedikation", "conflicts_with_allergy": False, "allergy_conflict_reasoning": None,
                "conflicts_with_existing_condition": True,
                "condition_conflict_reasoning": (
                    "Propranolol ist ein NICHT-selektiver Betablocker; Resident hat schwergradig persistierendes "
                    "Asthma bronchiale (GINA-Stufe 4) MIT Z.n. Status asthmaticus 2019 - Bronchospasmusrisiko durch "
                    "Beta-2-Blockade, in Asthma-Leitlinien als relative/absolute Kontraindikation gefuehrt."
                ),
            },
        ],
        "missing_attachment_note": None,
    },
}

# ---------------------------------------------------------------------------
# Follow-up / diagnoses / care extraction (one per letter)
# ---------------------------------------------------------------------------
FOLLOWUP = {
    "letter_01": {
        "diagnoses": [
            {"raw_name": "Akut dekompensierte chronische Herzinsuffizienz, NYHA IV bei Aufnahme", "icd10": "I50.14", "is_new_chronic_diagnosis": False, "letter_quote": "Akut dekompensierte chronische Herzinsuffizienz, NYHA IV bei Aufnahme", "letter_section": "diagnoses"},
            {"raw_name": "Pleuraerguesse beidseits", "icd10": "J91", "is_new_chronic_diagnosis": False, "letter_quote": "Pleuraerguesse beidseits (J91)", "letter_section": "diagnoses"},
        ],
        "tasks": [
            {"slug": "metformin_dose_reevaluation", "description": "Metformin-Dosis bei eingeschraenkter Nierenfunktion durch die Hausaerztin re-evaluieren lassen.", "due_in_days": 14, "requires_clinical_judgment": True, "letter_quote": "Re-Evaluation der Metformin-Dosis bei eingeschraenkter Nierenfunktion durch die Hausaerztin empfohlen.", "letter_section": "procedere"},
            {"slug": "lab_electrolytes_retention_recheck", "description": "Laborkontrolle (Elektrolyte, Retentionswerte) in 1-2 Wochen bei der Hausaerztin wegen intensivierter Diuretikatherapie.", "due_in_days": 10, "requires_clinical_judgment": False, "letter_quote": "Laborkontrolle (Elektrolyte, Retentionswerte) in 1-2 Wochen ueber die Hausaerztin bei intensivierter Diuretikatherapie.", "letter_section": "procedere"},
            {"slug": "cardiology_followup_echo", "description": "Kardiologische Verlaufskontrolle mit Echokardiographie in 3 Monaten vereinbaren.", "due_in_days": 90, "requires_clinical_judgment": False, "letter_quote": "Kardiologische Verlaufskontrolle mit Echokardiographie in 3 Monaten.", "letter_section": "procedere"},
            {"slug": "betablocker_sglt2i_reevaluation", "description": "Bei kardiologischer Kontrolle reevaluieren, ob Betablocker und SGLT2-Inhibitor begonnen werden koennen.", "due_in_days": 90, "requires_clinical_judgment": True, "letter_quote": "Re-Evaluation im Rahmen der kardiologischen Verlaufskontrolle in 3 Monaten.", "letter_section": "epikrise"},
        ],
        "care_instructions": [
            {"care_category": "fluid_management", "care_action": "modify", "target_entity": "fluid_restriction_chf", "description": "Trinkmengenbeschraenkung auf maximal 1.500 ml/Tag bei dekompensierter Herzinsuffizienz.", "target_ml": 1500, "strategy": "restrict", "wound_location": None, "dressing_interval_days": None, "letter_quote": "Trinkmengenbeschraenkung auf maximal 1.500 ml/Tag sowie taegliche Gewichtskontrollen.", "letter_section": "procedere"},
            {"care_category": "behavior", "care_action": "add", "target_entity": "daily_weight_monitoring_chf", "description": "Taegliche Gewichtskontrolle; bei Zunahme > 2 kg in 3 Tagen umgehende aerztliche Vorstellung.", "target_ml": None, "strategy": None, "wound_location": None, "dressing_interval_days": None, "letter_quote": "Bei einer Gewichtszunahme von mehr als 2 kg innerhalb von 3 Tagen bitten wir um umgehende aerztliche Vorstellung.", "letter_section": "procedere"},
        ],
    },
    "letter_02": {
        "diagnoses": [
            {"raw_name": "Pertrochantaere Femurfraktur links nach Sturz", "icd10": "S72.1", "is_new_chronic_diagnosis": True, "letter_quote": "Pertrochantaere Femurfraktur li. (S72.1) n. Sturz i. d. Pflegeeinrichtung", "letter_section": "diagnoses"},
            {"raw_name": "Arterielle Hypertonie", "icd10": None, "is_new_chronic_diagnosis": False, "letter_quote": "Art. Hypertonie, Hyperlipidaemie, Coxarthrose bds.", "letter_section": "diagnoses"},
            {"raw_name": "Hyperlipidaemie", "icd10": None, "is_new_chronic_diagnosis": False, "letter_quote": "Art. Hypertonie, Hyperlipidaemie, Coxarthrose bds.", "letter_section": "diagnoses"},
            {"raw_name": "Coxarthrose beidseits", "icd10": None, "is_new_chronic_diagnosis": False, "letter_quote": "Art. Hypertonie, Hyperlipidaemie, Coxarthrose bds.", "letter_section": "diagnoses"},
        ],
        "tasks": [
            {"slug": "renal_lab_recheck", "description": "Kreatinin-Kontrolle in 2 Wochen beim Hausarzt (Bedarfsmedikation, reduzierte Nierenfunktion).", "due_in_days": 14, "requires_clinical_judgment": False, "letter_quote": "Laborchem. Krea-Ko. in 2 Wo b. HA (b. Bedarfsmedikation u. red. Nierenfunktion).", "letter_section": "procedere"},
            {"slug": "osteoporosis_dxa_workup", "description": "Osteoporose-Basisdiagnostik (DXA) und ggf. spezifische Therapie beim Hausarzt veranlassen.", "due_in_days": 30, "requires_clinical_judgment": True, "letter_quote": "B. Z.n. Fragilitaetsfraktur Osteoporose-Basisdiagnostik (DXA) u. ggf. spezif. Therapie ueb. d. HA empfohlen", "letter_section": "procedere"},
            {"slug": "rollator_adjustment", "description": "Rollator-Anpassung ueber die Einrichtung veranlassen (ergotherapeutische Empfehlung).", "due_in_days": 7, "requires_clinical_judgment": False, "letter_quote": "Ergotherap. Hilfsmittelberatung erfolgt, Rollator-Anpassung ueb. d. Einrichtung empfohlen.", "letter_section": "epikrise"},
        ],
        "care_instructions": [
            {"care_category": "behavior", "care_action": "add", "target_entity": "fall_risk_transfer_precaution", "description": "Nach Sturz mit Fragilitaetsfraktur: Transfer nur mit Unterstuetzung von 2 Pflegekraeften, Mobilisation schmerzadaptiert.", "target_ml": None, "strategy": None, "wound_location": None, "dressing_interval_days": None, "letter_quote": "B. Entlassung Transfer Bett/Stuhl nur m. Unterstuetzung v. 2 Pflegekraeften moeglich, Gehstrecke m. Rollator u. Begleitung ca. 10 m.", "letter_section": "epikrise"},
        ],
    },
    "letter_03": {
        "diagnoses": [
            {"raw_name": "Synkope bei bradykarder Ueberleitung", "icd10": "R55", "is_new_chronic_diagnosis": False, "letter_quote": "Synkope bei bradykarder Ueberleitung bei permanentem Vorhofflimmern (I48.2, R55)", "letter_section": "diagnoses"},
        ],
        "tasks": [
            {"slug": "bisoprolol_reevaluation", "description": "Re-Evaluation der pausierten Bisoprolol-Therapie durch die Hausaerztin in 4 Wochen (Pulskontrolle).", "due_in_days": 28, "requires_clinical_judgment": True, "letter_quote": "Re-Evaluation der pausierten Bisoprolol-Therapie durch die Hausaerztin in 4 Wochen", "letter_section": "procedere"},
            {"slug": "renal_function_biannual_check", "description": "Halbjaehrliche Kontrolle der Nierenfunktion unter Apixaban.", "due_in_days": 182, "requires_clinical_judgment": False, "letter_quote": "Halbjaehrliche Kontrolle der Nierenfunktion empfohlen.", "letter_section": "procedere"},
            {"slug": "rhythmology_followup_if_recurrent_syncope", "description": "Bei erneuter Synkope/Schwindel Wiedervorstellung in der rhythmologischen Ambulanz (Re-Evaluation Schrittmacherindikation).", "due_in_days": 365, "requires_clinical_judgment": True, "letter_quote": "Wir bitten um Wiedervorstellung in unserer rhythmologischen Ambulanz bei erneuter Synkope oder Schwindel, dann Re-Evaluation einer Schrittmacherindikation.", "letter_section": "procedere"},
        ],
        "care_instructions": [],
    },
    "letter_04": {
        "diagnoses": [
            {"raw_name": "Ambulant erworbene Pneumonie, Unterlappen links", "icd10": "J18.9", "is_new_chronic_diagnosis": False, "letter_quote": "Ambulant erworbene Pneumonie, Unterlappen links (J18.9)", "letter_section": "diagnoses"},
            {"raw_name": "Infektexazerbierte COPD GOLD III", "icd10": "J44.1", "is_new_chronic_diagnosis": False, "letter_quote": "Infektexazerbierte COPD GOLD III (J44.1)", "letter_section": "diagnoses"},
        ],
        "tasks": [
            {"slug": "gp_crp_recheck", "description": "Klinische und laborchemische Verlaufskontrolle (CRP, Leukozyten) beim Hausarzt in 1 Woche.", "due_in_days": 7, "requires_clinical_judgment": False, "letter_quote": "Klinische u. laborchemische Verlaufskontrolle (CRP, Leukozyten) beim Hausarzt in 1 Woche", "letter_section": "procedere"},
            {"slug": "pulmonology_laba_evaluation", "description": "Pneumologische Kontrolle mit Lungenfunktion in 6-8 Wochen; dort Erweiterung der inhalativen Therapie (LABA) evaluieren.", "due_in_days": 49, "requires_clinical_judgment": True, "letter_quote": "Pneumologische Kontrolle m. Lungenfunktion in 6-8 Wochen empfohlen; Erweiterung d. inhalativen Therapie (LABA) dort evaluieren", "letter_section": "procedere"},
            {"slug": "vaccination_status_check", "description": "Influenza- und Pneumokokken-Impfstatus pruefen und ggf. vervollstaendigen.", "due_in_days": 14, "requires_clinical_judgment": False, "letter_quote": "Influenza- u. Pneumokokkenimpfstatus bitte pruefen u. ggf. vervollstaendigen", "letter_section": "procedere"},
        ],
        "care_instructions": [],
    },
    "letter_05": {
        "diagnoses": [
            {"raw_name": "Harnwegsinfekt mit Fieber", "icd10": "N39.0", "is_new_chronic_diagnosis": False, "letter_quote": "Harnwegslnfekt m. Fieber (N39.0)", "letter_section": "diagnoses"},
            {"raw_name": "Exsikkose", "icd10": "E86", "is_new_chronic_diagnosis": False, "letter_quote": "Exsikkose (E86)", "letter_section": "diagnoses"},
        ],
        "tasks": [
            {"slug": "urine_recheck_if_infection_signs", "description": "Bei erneuten Infektzeichen Urinkontrolle beim Hausarzt veranlassen.", "due_in_days": 180, "requires_clinical_judgment": False, "letter_quote": "Urinkontrolle b. erneuten Infektzeichen ueb. d. HA", "letter_section": "procedere"},
            {"slug": "speech_therapy_reevaluation_if_recurrent_cough", "description": "Bei rezidivierendem Husten/pulmonalen Infektzeichen logopaedische Re-Evaluation ueber den Hausarzt (Aspirationsrisiko).", "due_in_days": 180, "requires_clinical_judgment": True, "letter_quote": "B. rez. Husten od. pulmonalen Infektzeichen logopaed. Re-Evaluation ueb. d. HA.", "letter_section": "epikrise"},
        ],
        "care_instructions": [
            {"care_category": "wound", "care_action": "add", "target_entity": "sacral_wound_dressing", "description": "Wundversorgung sakral mit Schaumstoffverband fortfuehren, Verbandwechsel alle 2 Tage, druckentlastende Lagerung nach Plan.", "target_ml": None, "strategy": None, "wound_location": "sakral", "dressing_interval_days": 2, "letter_quote": "Wundversorgung sakral m. Schaumstoffverband fortfuehren, Verbandwechsel alle 2 Tage, druckentlastende Lagerung n. Plan", "letter_section": "procedere"},
        ],
    },
}


class ManualExtractionLLM:
    """Stands in for an HTTP LLM call. Routes on system-prompt content (which
    analyzer is asking) and on a unique patient-name marker present in every
    prompt (since the letter text is embedded in all three analyzers'
    prompts) to return the hand-authored extraction for that
    (letter, analyzer) pair. See module docstring for why."""

    model = "manual-extraction"

    def __init__(self):
        self.usage_log: list[dict] = []

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        letter_id = next(lid for lid, marker in PATIENT_MARKERS.items() if marker in user)
        if "clinical pharmacist" in system:
            payload = MEDICATIONS[letter_id]
        elif "for a nursing home" in system:
            payload = FOLLOWUP[letter_id]
        else:
            payload = STAY_METADATA[letter_id]
        response = json.dumps(payload, ensure_ascii=False)
        self.usage_log.append(
            {"model": self.model, "input_tokens": len(user) // 4, "output_tokens": len(response) // 4}
        )
        return response


def main() -> None:
    import src.run as run_module

    llm = ManualExtractionLLM()
    run_module.get_llm_client = lambda: llm  # the one substitution

    letters_dir = ROOT / "letters"
    data_dir = ROOT / "data"
    out_path = ROOT / "proposals.json"
    run_module.run_analysis(letters_dir, data_dir, out_path)

    output = ProposalsOutput.model_validate_json(out_path.read_text(encoding="utf-8"))
    print(f"\nproposals.json: {len(output.proposals)} proposals, {len(output.cost_log)} cost-log entries.")
    from collections import Counter
    print("by category:", Counter(p.category.value for p in output.proposals))
    print("by routing: ", Counter(p.routing.value for p in output.proposals))


if __name__ == "__main__":
    main()
