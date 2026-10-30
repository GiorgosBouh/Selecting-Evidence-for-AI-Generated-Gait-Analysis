# Selecting Evidence for AI-Generated Gait Analysis Explanations

Reproducibility materials for the manuscript submitted to the *Journal of Healthcare Informatics Research*.

This repository contains runnable FR/TGR experiment code, fixed evidence resources, saved FR/TGR run records, the manuscript supplementary file, and scripts to regenerate the FR/TGR summary tables. The original gait datasets are not included. Download them from their original repositories and follow `data/README.md` for the expected local file paths.

## What the study compares

Each method selects short source passages for the same gait case and language model:

- **Flat Retrieval (FR):** selects passages using exact feature tags.
- **Taxonomy-Guided Retrieval (TGR):** selects passages using a gait category tree, then checks that all case features are covered.
- **Ontology-Guided Retrieval (OGR):** selects passages through concepts linked in the Gait Analysis Domain Ontology (GADO).

The saved primary FR/TGR runs are in `results/full_runs/`. The record keys `embedding_rag` and `vectorless_tree_rag` correspond to FR and TGR, respectively. Supplementary prompt templates, support-checking rules, and example audits are in `docs/reproducibility/`.

## Reproducibility status

The supplementary source and PDF report the OGR summary metrics and the supplementary checks (`supplementary/ESM_1.tex` and `.pdf`). The workspace does **not** contain the OGR execution script, the GADO OWL file, the feature-to-GADO mapping, or the full OGR run-level JSONL records. It also lacks the execution scripts and run-level records for several supplementary checks. Therefore, readers can inspect the reported tables, but cannot independently rerun OGR or regenerate all supplementary tables from raw run records using this release alone. The included FR/TGR full-run records and their analysis scripts are available for re-analysis.

The raw dataset archives are intentionally excluded. They must be obtained from the sources listed below. The Health&Gait data are large; the paper's experiment uses only three gait-parameter/participant CSV files, not the video and image archive.

## Repository contents

- `scripts/`: FR/TGR full-run, case-building, summary, and export scripts.
- `gait_rag/`: shared case, evidence, retrieval, explanation, and metric code.
- `resources/`: the literature corpus, feature-support tags, taxonomy, and pathology scope rules used by the available FR/TGR code.
- `results/full_runs/`: saved FR/TGR JSONL records and the summary/export files used to report them. These are derived model outputs, not raw dataset archives.
- `supplementary/`: the supplementary file source and PDF. It contains the reported OGR metrics and supplementary result tables, but not the underlying OGR or sensitivity-run records/scripts.
- `docs/reproducibility/`: exact prompt/schema, support-checking notes, OGR method notes, and representative audit examples. The OGR summary tables are in `supplementary/`; its executable code and full run records are not included.
- `data/README.md`: dataset descriptions, source links, and required local file layout.

## Software requirements

- Python 3.10 or newer
- `numpy` and `scipy` for the full-run loader and MATLAB workspaces
- Ollama for new model generations, with the three model tags below available locally

Install the Python dependencies:

```bash
python -m pip install numpy scipy
```

For a new generation run, install Ollama using its official instructions, start its local service, and pull the models:

```bash
ollama pull mistral-small:24b
ollama pull qwen3:14b
ollama pull llama3.1:8b
```

The runner calls `http://127.0.0.1:11434/api/generate` with JSON output, `temperature=0`, `stream=false`, and `think=false`. Local hardware must have enough memory for the selected model. The saved JSONL files can be summarized without Ollama.

## Reproduce the saved FR/TGR summary tables

From the repository root, regenerate the per-dataset summary JSON and Markdown files from the saved run records:

```bash
python scripts/summarize_full_dataset_batch.py \
  --input results/full_runs/data2_biomechanical_interpretation_full_2026-06-15.jsonl \
  --output-json /tmp/data2_summary.json \
  --output-md /tmp/data2_summary.md \
  --bootstrap-reps 2000
```

Repeat with the GAITEX, Health&Gait, and amputee JSONL files in `results/full_runs/`. To regenerate the long claim table and model-by-method table, use:

```bash
python scripts/export_batch_claim_tables.py \
  --input results/full_runs/data2_biomechanical_interpretation_full_2026-06-15.jsonl \
  --model-method-output /tmp/data2_model_method.csv \
  --pathology-model-method-output /tmp/data2_pathology_model_method.csv \
  --claims-output /tmp/data2_claims.csv
```

The saved summary and export files are included alongside each JSONL run so they can be compared with regenerated files.

## Run FR/TGR again from downloaded data

First download and place the source files as described in `data/README.md`. Then run from this repository root with `PYTHONPATH=scripts:.` so the main runner can import its helper module.

Health&Gait:

```bash
PYTHONPATH=scripts:. python scripts/run_full_dataset_batch.py \
  --datasets health \
  --models mistral-small:24b qwen3:14b llama3.1:8b \
  --output results/new/health_gait.jsonl
```

Clinical Gait Signals: build the participant-level feature packets first, then run the models:

```bash
PYTHONPATH=scripts:. python scripts/build_data2_biomechanical_cases.py \
  --zip Data_2_Clinical_Gait_Signals_with_Wearable_Sensors_from_Healthy_Neurological/dataset.zip \
  --output-jsonl outputs/data2_biomechanical_cases.jsonl

PYTHONPATH=scripts:. python scripts/run_full_dataset_batch.py \
  --datasets wearable --wearable-source biomechanical_cases \
  --biomechanical-cases outputs/data2_biomechanical_cases.jsonl \
  --models mistral-small:24b qwen3:14b llama3.1:8b \
  --output results/new/clinical_gait_signals.jsonl
```

GAITEX and the above-knee amputee dataset use the same runner, with `--datasets gaitex` or `--datasets amputee`. See `data/README.md` for the exact archive and directory names expected by the loaders. Full generation runs make many local model calls and may take substantial time.

## Data and citation

The raw datasets remain with their original custodians. This repository does not mirror them. Follow the original dataset terms and cite both the dataset paper and the dataset record as appropriate. Source links and the exact files used by each loader are listed in `data/README.md`.

The repository has no license file yet. A license must be selected by the authors before public release; dataset terms do not automatically license this code or the curated evidence resources.

## Before public release

1. Add and test the missing OGR runner, GADO mapping/ontology input, and primary-prompt OGR run records.
2. Add the scripts and run-level inputs needed to regenerate the supplementary ablations and retriever checks, or revise the manuscript and Data Availability Statement to match what is released.
3. Confirm that derived case-level output records may be redistributed under the source datasets' terms.
4. Choose a code/resource license, add the final repository URL, archive a version (for example on Zenodo), and add its DOI to the manuscript.
