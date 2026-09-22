# WavestarNet Reproducibility Package

This package contains the code, fixed splits, executed notebooks, and compact results supporting the paper. All models were trained without pretrained weights on AgriVision4, PlantDoc, and PlantWild.

```text
code/          Canonical implementation
agrivision4/   Comparison, test evaluation, ablations, and Grad-CAM
plantdoc/      Comparison and test evaluation
plantwild/     Comparison, test evaluation, efficiency, and WaveDown study
```

Use each dataset folder's `README.md` for data access and experiment details. Run the notebooks with the fixed splits in `data_protocol/`; evaluation and analysis notebooks automatically load the implementation from `code/`.

Install the recorded GPU runtime with `pip install -r requirements.txt`.

Results, predictions, statistical outputs, and selected figures are included. Raw images, checkpoints, original output archives, and full Grad-CAM visualizations are external artifacts listed in `external_artifacts.csv`.

`paper_result_mapping.csv` links paper analyses to their source files. Verify file integrity with:

```bash
sha256sum --check checksums.sha256
```
