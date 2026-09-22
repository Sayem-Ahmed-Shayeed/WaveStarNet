# External artifacts

Large artifacts are stored outside the Git tree. They should be attached to the repository's GitHub Release without changing their filenames.

`external_artifacts.csv` records the local source path, size, SHA-256 checksum, dataset, experiment, and intended role of each external file. After the release is created, add its tag and download URL to that table.

### Checkpoint downloads

The three model-comparison checkpoint ZIP archives are split into four GitHub Release assets because of upload limits in the publication environment. Download every `*.part-01` through `*.part-04` asset for one dataset, then reconstruct and verify it:

```bash
cat agrivision4_model_comparison_checkpoints.zip.part-* > agrivision4_model_comparison_checkpoints.zip
sha256sum -c ARCHIVE_SHA256SUMS.txt
```

Use the equivalent filename for PlantDoc or PlantWild. The expected SHA-256 values are also recorded in `external_artifacts.csv`.

External files include:

- training checkpoints (`best_checkpoint.pt`);
- original Kaggle result ZIP files;
- full Grad-CAM heatmap and overlay collections;
- large experiment bundles used to preserve the original Kaggle outputs.

Raw dataset images are not release artifacts. They remain available through their dataset providers or Kaggle dataset pages. The fixed split files and audit records required to identify the evaluated samples are included in this directory.
