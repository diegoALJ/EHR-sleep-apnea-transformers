# EHR Sleep Apnea Prediction with Transformers

Research reproduction-oriented reconstruction of the master's thesis **"Pronóstico de la apnea del sueño mediante el uso de técnicas de procesamiento de lenguaje natural aplicadas a registros médicos electrónicos"**, developed with clinical electronic health records from **Hospital Universitario San Ignacio (Bogotá, Colombia)**.

This repository explores fine-tuning pretrained Transformers for binary obstructive sleep apnea (OSA) classification from longitudinal clinical records written in Spanish. It preserves the study's central approach while refactoring partial research notebooks into a configurable Python workflow.

> **Scientific provenance:** Performance values below are taken from the published study, **not** from executions of this reconstructed repository. The original clinical dataset is unavailable here, so the paper's results cannot be independently reproduced from this code alone.

## Project overview

The study compared three pretrained language models:

- **BETO** — Spanish BERT (`dccuchile/bert-base-spanish-wwm-uncased`)
- **XLM-RoBERTa** — multilingual RoBERTa (`xlm-roberta-base`)
- **Biomedical RoBERTa (Spanish)** — `PlanTL-GOB-ES/roberta-base-biomedical-es`

Each model was evaluated using two input representations: **full EHR text** and **clinical entities** (diagnoses, medications, procedures and laboratory tests extracted using MedSpaCy).

### Longitudinal record construction

The main methodological challenge was that one patient could have multiple visits and more text than a Transformer could process at once. Visits were aggregated chronologically per patient. For patients with recorded OSA, the final visit was the **last positive visit**; for patients without recorded OSA, the final visit was the **most recent available visit**. The resulting sequence prioritized recent records within a **512-token context window**.

The model uses a pretrained Transformer encoder followed by dropout and a binary classification head. Probabilities are obtained from the positive-class softmax score.

## Published results

The following results were reported in the paper:

| Pretrained model | EHR input | F1-score | ROC AUC | PR AUC |
|---|---|---:|---:|---:|
| BETO | Clinical entities | 0.70 | 0.79 | 0.58 |
| XLM-RoBERTa | Clinical entities | 0.55 | 0.69 | 0.48 |
| Biomedical RoBERTa | Clinical entities | 0.70 | 0.80 | 0.56 |
| **BETO** | **Full EHR** | **0.88** | **0.93** | **0.80** |
| XLM-RoBERTa | Full EHR | 0.70 | 0.78 | 0.59 |
| Biomedical RoBERTa | Full EHR | 0.86 | 0.90 | 0.79 |

The original study found that **complete EHR notes performed better than MedSpaCy-extracted clinical entities**, potentially because extraction missed clinically important information in Spanish records.

The published study selected dropout **0.3**, batch size **8**, and learning rate **1e-6** for clinical entities and **2e-6** for full EHR. The Grid Search was conducted with BETO before using the selected configurations with the other backbones.

## Dataset and data availability

**Important data availability note:** The source dataset is private clinical information from Hospital Universitario San Ignacio and is **not provided** in this repository. No patient records, anonymized individual clinical notes, model predictions tied to patients, or original dataset links are included. Access to the original data is governed by institutional approvals and confidentiality requirements.

Expected **visit-level** CSV schema for permitted data:

| Column | Meaning | Example (artificial) |
|---|---|---|
| `patient_id` | Non-identifying patient key | `SYN001` |
| `visit_date` | Clinical encounter date | `2021-04-01` |
| `note_text` | Visit-level clinical note | `Consulta de seguimiento sin hallazgos nuevos.` |
| `apnea_label` | Recorded apnea diagnosis at this visit (0/1) | `0` |
| `entities_text` | Validated extracted entities for this visit | `seguimiento` |

`entities_text` is required for the clinical-entities configuration. Its extraction must be validated independently for Spanish clinical notes; default MedSpaCy settings do not automatically reproduce the original study's ontology and rules.

## Project structure

```text
configs/                   Experiment parameters and model IDs
src/                       Temporal preprocessing, classifiers, training, metrics
scripts/run_experiments.py Automated grid search and six-model evaluation
notebooks/                 Lightweight, import-based research walkthroughs
tests/                     Synthetic-data unit tests
docs/methodology_notes.md  Original-method / reconstruction difference log
outputs/                   Private local artifacts (gitignored)
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m scripts.run_experiments --config configs/experiments.yaml
```

Configure `input_csv` to an **authorized local dataset** before running. `scripts/run_experiments.py` evaluates **three backbones × two representations**, with BETO used for development-set hyperparameter tuning. Training requires substantial GPU memory and time; the study used Google Colab Pro (NVIDIA A100, 32 GB RAM, 8 vCPUs).

**Privacy:** `outputs/` includes private test predictions; never publish those files. The repository excludes `data/` and all generated CSV files by default.

## Reproducibility and limitations

This is a **methodology-aligned reconstruction**, not a bitwise or performance-verified reproduction. The historical notebooks were incomplete, and some details of training / data construction were not recoverable. In particular:

- Patient-disjoint splitting occurs **before** train-only downsampling, whereas the article described downsampling before splitting. This is an intentional evaluation correction and may change measured scores.
- RoBERTa uses first-token hidden-state pooling rather than BETO's pooler, to work across different backbone families.
- Some training defaults (including epochs and model-selection rules) are documented reconstruction choices, not recovered paper settings.
- The original Spanish MedSpaCy extraction rules and full private data transformations are not included.
- This code reports **average precision**, which may differ numerically from the paper's PR AUC convention.
- Since the label-defining positive encounter may be included in the model input, these experiments should not be represented as verified pre-diagnosis prospective risk prediction.
- The original dataset included female patients only, limiting generalization to men and other populations.

See [Methodology notes](docs/methodology_notes.md) for the detailed difference log.


## Research context

The study used PyTorch, Hugging Face Transformers, MedSpaCy, and BertViz for exploratory attention visualization. Attention visualization is exploratory and **does not establish causal clinical importance**. Predictions from this repository are for research purposes only, **not clinical diagnosis**.

## Authors and Publication

| Organización   | Nombre del Miembro | Correo electronico | 
|----------|-------------|-------------|
| PUJ-Bogota |  Juan Pablo Pajaro Hernandez: Tutor Maestria |juanpajaro@javeriana.edu.co |
| PUJ-Bogota |  Jehico Bladimir Bolivar : Team member  | bolivarb-jehico@javeriana.edu.co |
| PUJ-Bogota |  Diego Montoya Tovar : Team member  | da.montoya@javeriana.edu.co |
| PUJ-Bogota |  Yohann Montaño Reyes : Team member  | ymontano@javeriana.edu.co |

This repository is based on the research conducted as part of a Master's thesis in Artificial Intelligence at Pontificia Universidad Javeriana, Bogotá, Colombia.

Title: Pronóstico de la apnea del sueño mediante el uso de técnicas de procesamiento de lenguaje natural aplicadas a registros médicos electrónicos.

The original publication describes the methodology, experimental setup, and results obtained using pretrained Transformer models for sleep apnea prediction from electronic health records.

Original publication: [Pontificia Universidad Javeriana Institutional Repository](https://repository.javeriana.edu.co/items/3bf9a713-ab12-4cc5-ad18-e2a62f60eb27)

The research was conducted using anonymized clinical data provided by Hospital Universitario San Ignacio, Bogotá, Colombia. The original dataset is not publicly available due to patient confidentiality and data privacy restrictions.
