# WavestarNet

Reproducibility materials for the WavestarNet plant leaf disease classification study.

## Datasets

- [AgriVision4](https://data.mendeley.com/datasets/8t6k37ztxc/1)
- [PlantDoc](https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset)
- [PlantWild v2](https://tqwei05.github.io/PlantWild/)

The complete package is in [`reproducibility/`](reproducibility/README.md). It contains the executed notebooks, source code, fixed dataset split records, result tables, integrity checks, and environment requirements used for the manuscript.

The three model-comparison checkpoint bundles are provided through the repository release rather than Git history. Their filenames and SHA-256 checksums are recorded in [`reproducibility/external_artifacts.csv`](reproducibility/external_artifacts.csv).

To validate the tracked package after cloning:

```bash
python reproducibility/verify_package.py
```
