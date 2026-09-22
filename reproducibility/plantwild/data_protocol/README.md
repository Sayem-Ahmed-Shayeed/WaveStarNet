# PlantWild V2 Audit V1

This directory is a non-destructive audit of `/run/media/sha-weed/adc6c87e-0116-4a25-a6e9-8555193c1e51/Research/Agrivision4/plantwild/dataset/plantwild_v2`. It does not copy,
rename, recompress, or delete source images.

## Result

- Source images/classes: 11488 / 115
- Unreadable images: 0
- Exact/confirmed-near edges: 404 / 616
- Conflicting components/images excluded: 207 / 463
- Support-filtered classes/images: 10 / 125
- Benchmark images/classes: 10272 / 105
- Split counts: {'test': 1523, 'train': 7188, 'validation': 1561}
- Manual semantic/quality review queue: 2320 images
- Cross-label duplicate edges/review sheets: 280 / 35

## Interpretation

Exact and conservatively confirmed near-copy components cannot cross splits.
Every multi-label component is excluded. Quality and semantic heuristics are
review flags only; they do not alter the benchmark. The test manifest is sealed
by policy until all model-development decisions are frozen.

The local release has no fixed split, original URLs, or embedded license file.
See `audit/metadata_recovery.json` and `audit/validation.json`. Human or domain-
expert review of `review/manual_review_queue.csv` remains required before any
semantic relabeling or image exclusion.
