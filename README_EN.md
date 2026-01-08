# STHA Sports Analytics (Reproducibility Package)

This repository is a reproducibility-oriented reference implementation for:
- Rotation angle estimation via geometric torso-normal projection
- Gymnastics execution scoring via DTW-aligned skeleton distance mapped to a Code-of-Points-like penalty

It contains:
- Core algorithms (src/stha/)
- Preprocessing scripts (scripts/)
- Scoring/rotation evaluation scripts (scripts/)
- Anonymized sample skeleton sequences (data/sample/)
- A simple split file format (splits/)

Note: data/sample/ is synthetic/anonymized demo data to validate the pipeline end-to-end.
Replace it with your own extracted skeletons for real experiments.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data format
CSV columns: frame,joint,x,y,z (one row per joint per frame)

## Quick start
Rotation demo:
```bash
python scripts/evaluate_rotation.py --cfg configs/default.yaml --input data/sample/rotation_demo.csv
```

Gymnastics scoring demo:
```bash
python scripts/score_gymnastics.py --cfg configs/default.yaml \
  --test data/sample/gym_test.csv --template data/sample/gym_template.csv
```

## Reproducibility checklist
- Fixed seed (configs/default.yaml)
- Fixed resampling target fps
- Fixed normalization (root translation + shoulder-width scaling)
- Fixed DTW band (Sakoe–Chiba)
- Optional deterministic smoothing (Savitzky–Golay if SciPy is available)

## Structure
stha-sports-analytics/
  src/stha/        core library
  scripts/         runnable scripts
  configs/         yaml configs
  data/sample/     anonymized demo skeletons
  splits/          example split lists
  docs/            extra documentation
