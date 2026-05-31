# Post-Disaster Building Damage Assessment

This project combines satellite-image building damage classification, topology-based validation, and social-media disaster understanding into a fused situation-report pipeline.

For the full methodology, experiments, results, and week-by-week narrative, see [PROJECT_REPORT.md](PROJECT_REPORT.md).

## What It Does

- Classifies object-level building damage from pre-disaster and post-disaster image crops.
- Validates damage predictions with Week 13 topology features when structural change is visible.
- Classifies included crisis tweets for emotion and disaster type with zero-shot DeBERTa.
- Fuses satellite, topology, and social-media signals into a Week 15 event JSON.
- Builds retrieval-ready event documents and deterministic situation reports.

## Quick Start

Install dependencies:

```powershell
pip install -r requirements.txt
pip install -r docs/requirements.txt
```

Run the Streamlit app:

```powershell
streamlit run streamlit_app.py
```

## Main App Workflow

1. Choose a satellite input mode:
   - manual damage counts
   - uploaded building crop pairs
   - uploaded satellite JSON
2. For uploaded crop pairs, optionally enable Week 13 topology validation.
3. Paste or upload tweets and classify them with zero-shot DeBERTa.
4. Generate the fused event JSON, retrieval document, and situation report.

Outputs from the app are saved under:

```text
results/streamlit_app/
```

## Key Commands

Fit all-class Week 13 topology prototypes:

```powershell
python src\week13\week13_fit_topology_threshold.py --dataset-root data\week11_buildings_week8_extra --split train --topology-csv results\week13_topology\topology_features_train.csv --output-dir results\week13_topology\threshold --rebuild-topology-csv
```

Run zero-shot DeBERTa social classification:

```powershell
python src\week14\week14_zero_shot_text_classifier.py --crisismmd-root data\CrisisMMD_v2.0 --split val --event mexico_earthquake --model-name MoritzLaurer/deberta-v3-base-zeroshot-v2.0 --write-social-json --social-json results\week15_inputs\social_zero_shot_val.json
```

Fuse satellite and social JSON inputs:

```powershell
python src\week15\week15_fuse_event.py --event example_event --satellite-json results\week15_inputs\satellite.json --social-json results\week15_inputs\social_val.json --output results\week15_fusion\example_event.json
```

Generate a situation report:

```powershell
python src\week17\week17_generate_situation_report.py --input-json results\week15_fusion\example_event.json --output-md results\week17_reports\example_event.md
```

## Documentation

Read the Docs builds from `docs/` using `.readthedocs.yaml`.

The documentation includes:

- this short README
- the full [PROJECT_REPORT.md](PROJECT_REPORT.md)

## Repository Notes

- Raw datasets under `data/` are ignored.
- Result artifacts under `results/` are intended to be committed.
- Large model files may require Git LFS if they exceed GitHub's normal file-size limit.
