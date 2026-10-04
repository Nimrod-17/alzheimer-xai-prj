# alzheimer-xai-prj
Explainable AI for early Alzheimer diagnosis from 3D T1-weighted MRI (OASIS-3).

The model estimates whether a subject **has Alzheimer-type impairment or will develop it within 5 years**,
and explains each prediction with voxel- and region-level attribution maps.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Labels | `build_oasis3_metadata.py` | `data/processed/oasis3_ad_risk.csv` |
| 2. Preprocessing | `preprocess_oasis3.py` | `data/preprocessed/` (MNI space) |
| 3. Quality control | `quality_control.py` | `data/processed/qc_report.csv` + snapshots |
| 4. Training | `main.py` | `models/oasis3_resnet3d.pth` + `.json` |
| 5. Evaluation | `evaluate.py` | test metrics with 95% bootstrap CIs |
| 6. Explanations | `explain.py` | `results/explanations/<scan_id>/` |

**Labels** (`src/oasis3_parser.py`): each T1w session is matched to the closest clinical visit (≤ 180 days).
Label 1 = Alzheimer-type impairment at the scan or first AD diagnosis within 5 years;
label 0 = cognitively normal with ≥ 3 years of stable follow-up; ambiguous cases are excluded.
Scans without orientation in the NIfTI header (qform/sform = 0) are rejected.

**Preprocessing** (`src/mri_pipeline.py`): HD-BET skull stripping → N4 bias correction →
rigid + affine registration to the ICBM 2009 MNI152 1 mm template → brain z-score. Resumable.

**Data** (`src/data_module.py`): train/val/test split by subject (70/15/15, persisted in
`data/processed/subject_splits.csv`), one random scan per subject per training epoch,
one fixed scan per subject for validation and test. Network input: 128 × 160 × 128 at 1.25 mm.

**Model** (`src/model.py`): lightweight 3D ResNet with GroupNorm and a GAP + linear head.
Model selection on validation AUC; decision threshold chosen on validation (Youden).

**XAI** (`src/explainability.py`): Grad-CAM 3D and Integrated Gradients, mapped back to MNI space
and summarized over Harvard-Oxford atlas regions.

## Setup

Python 3.12. Install PyTorch for your GPU first (see `requirements.txt`), then:

```bash
pip install -r requirements.txt
```

OASIS-3 data is not included: it requires an approved Data Use Agreement (https://sites.wustl.edu/oasisbrains/).

## Tests

```bash
python -m unittest discover tests
```

The OASIS-1 prototype is preserved at the git tag `oasis1-baseline`.
