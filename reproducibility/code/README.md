# Canonical source code

This directory is the single importable source tree for the reproducibility package. Add `reproducibility/code` to `PYTHONPATH`, or run the included evaluation and analysis notebooks from within `reproducibility/`; their setup cells add this directory automatically.

The code is organized by its original Python import paths:

```text
code/
├── agrivision4/
│   ├── models/
│   ├── scratch_comparison/
│   ├── evaluation/
│   ├── ablation_study/
│   ├── analysis/
│   └── interpretability/
├── plantwild/
│   ├── scratch_comparison/
│   ├── efficiency/
│   └── wavedown_ablation/
└── research_evaluation/
```

Dataset preparation scripts remain under each dataset's `data_protocol/code/` directory because they generate that dataset's audit and fixed split files.

The notebooks require the corresponding raw images and frozen checkpoints. These are deliberately not stored in Git. Download them through the dataset references and GitHub Release records in `../external_artifacts.csv`, then place them at the paths specified by the notebook configuration before executing a new evaluation.
