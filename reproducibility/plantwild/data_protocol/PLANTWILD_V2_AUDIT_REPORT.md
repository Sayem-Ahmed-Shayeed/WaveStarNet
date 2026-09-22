# PlantWild V2 Dataset Audit Report

## Material Passport

- **Artifact ID:** `plantwild-v2-audit-v2`
- **Artifact type:** Dataset integrity and reproducibility audit
- **Verification status:** `VERIFIED`
- **Source:** `plantwild/dataset/plantwild_v2`
- **Source archive SHA-256:** `5c3fd0c6ecf9a346c9d2dd03ff47769c71e581eb1170a68477d718ab43cfee5b`
- **Audit date:** 2026-07-18
- **Audit seed:** 42
- **Final benchmark policy:** Duplicate-aware 70/15/15 split; minimum 20 independent components per class

## Executive Decision

The raw PlantWild v2 tree is **not ready as a controlled 115-class benchmark**.
It contains extensive exact/near duplicates, cross-label conflicts, low-support
classes, and heterogeneous web imagery such as collages, screenshots, text,
watermarks, fruit images, and field scenes.

The audited output is ready for the planned seed-42 screening experiment as a
fixed **105-class, 10,272-image benchmark**. Use only the generated manifests.
Do not train by enumerating the raw class directories. The test manifest remains
sealed until the model and training recipe are frozen.

## Eight-Step Completion Matrix

| Step | Audit operation | Status | Verified result | Primary evidence |
|---|---|---|---|---|
| 1 | File inventory and full decode | Complete | 11,488 images in 115 folders; 0 unreadable | `audit/source_inventory.csv`, `audit/unreadable_images.csv` |
| 2 | Exact duplicate detection | Complete | 404 union edges; 402 file-identical and 404 decoded-pixel-identical edges | `audit/exact_duplicate_pairs.csv` |
| 3 | Conservative perceptual near-duplicate detection | Complete | 616 confirmed edges; 48 additional probable pairs held for review | `audit/confirmed_near_duplicate_pairs.csv`, `audit/probable_near_duplicate_review.csv` |
| 4 | Cross-class conflict resolution | Complete | 207 conflicting components containing 463 images excluded; 280 cross-label evidence edges | `audit/excluded_conflicting_components.csv`, `review/cross_label_duplicate_pairs.csv` |
| 5 | Quality and semantic-content screening | Complete automated screening; expert adjudication pending | 2,228 flagged images; 2,320-image review queue; 12 quality sheets and 35 duplicate sheets | `audit/quality_flags.csv`, `review/manual_review_queue.csv`, `review/*.jpg` |
| 6 | Metadata, source, and original split recovery | Complete | No embedded metadata, URLs, license file, or fixed split; official loader and training code inspected at pinned commit | `audit/metadata_recovery.json` |
| 7 | Duplicate-safe fixed manifests | Complete | Train 7,188; validation 1,561; sealed test 1,523; no component crosses splits | `manifests/*.csv`, `audit/validation.json` |
| 8 | Low-support class policy | Complete | Minimum 20 independent components; 10 classes and 125 canonical images excluded | `audit/support_policy.json`, `audit/excluded_under_supported_classes.csv` |

## Dataset Accounting

| Quantity | Count |
|---|---:|
| Source images | 11,488 |
| Source classes | 115 |
| Duplicate components | 764 |
| All connected components | 10,604 |
| Cross-label conflicting components | 207 |
| Images in conflicting components | 463 |
| Canonical images before support filtering | 10,397 |
| Classes removed by support policy | 10 |
| Canonical images removed by support policy | 125 |
| Final benchmark images | 10,272 |
| Final benchmark classes | 105 |

The accounting identity is:

`10,397 canonical non-conflicting images - 125 under-supported images = 10,272 benchmark images`.

## Duplicate and Label-Conflict Findings

Exact duplicates were detected with both compressed-file SHA-256 and decoded-
pixel SHA-256. Near copies were accepted only when perceptual hash, normalized
RGB difference, grayscale correlation, and aspect-ratio criteria jointly met
the conservative thresholds in `audit/audit_summary.json`.

The 280 cross-label evidence edges comprise 132 exact and 148 confirmed-near
edges. Visual inspection of sheets 1, 18, and 35 confirmed identical or nearly
identical scenes assigned to different labels, including examples across crop
and disease names. Excluding the entire connected component avoids choosing an
unsupported label and prevents leakage. Relabeling requires plant-pathology
expert review and is not performed by this audit.

## Quality and Semantic Screening

Quality flags overlap; one image may have several flags.

| Review signal | Images |
|---|---:|
| Possible collage seam | 900 |
| Low resolution | 788 |
| Low green fraction | 436 |
| Extreme aspect ratio | 377 |
| Very low resolution | 104 |
| Low entropy | 45 |
| Low contrast | 42 |
| Very bright | 14 |
| Very dark | 13 |
| Possible border/overlay | 6 |

The collage-seam heuristic is intentionally sensitive and produces false
positives on strong leaf boundaries, text insets, and black bars. Therefore,
none of these quality or semantic signals automatically removes an image. The
contact sheets confirm substantial domain heterogeneity: single leaves, fruit,
whole plants, field scenes, multi-panel figures, diagrams, screenshots,
watermarks, and non-natural backgrounds all occur.

## Excluded Low-Support Classes

| Class | Independent components |
|---|---:|
| bell pepper frogeye leaf spot | 18 |
| broccoli ring spot | 5 |
| carrot cercospora leaf blight | 15 |
| coffee black rot | 6 |
| coffee brown eye spot | 11 |
| peach anthracnose | 13 |
| peach rust | 8 |
| plum bacterial spot | 15 |
| raspberry leaf spot | 18 |
| tobacco frogeye leaf spot | 16 |

This is a pre-model support rule, not a result-driven class selection. The raw
taxonomy and all excluded files remain represented in the source inventory.

## Split and Access Policy

| Split | Images | Permitted use |
|---|---:|---|
| Train | 7,188 | Model fitting |
| Validation | 1,561 | Checkpoint selection and development comparison |
| Test | 1,523 | Sealed until architecture, recipe, and hyperparameters are frozen |

Splitting is class-stratified and deterministic at the connected-component
level. Every retained class occurs in all three splits. Exact and confirmed
near-copy components cannot cross split boundaries.

## Provenance and Original Protocol Audit

The local archive contains only class folders and images. It provides no fixed
split, original URLs, metadata table, or embedded license file. Filename tokens
suggest acquisition engines, not stable source identities: Google 4,638, Bing
3,286, Baidu 700, and unspecified 2,864.

The official MVPDR repository was inspected at commit
`fe7082d486d65ca1eac8b1a5eae1c9af87b3280b`. Its loader declares an approximate
70/10/20 split but samples indices over unsorted `os.listdir` results, so a seed
alone does not make the split portable. Its training procedure also evaluates
test features during epoch selection. Those procedures are not adopted here.
This audit uses a deterministic duplicate-aware 70/15/15 development protocol
with a sealed test set.

## Reproducibility Verification

Two independent full executions produced byte-identical SHA-256 hashes for all
14 core artifacts: the four manifests, class mapping, source inventory, all
three duplicate-pair tables, component membership, conflict exclusions,
support exclusions, class distribution, and split distribution.

`audit/validation.json` reports:

- `ok: true`
- no validation errors
- source tree unchanged
- 10,272 manifest paths and hashes verified
- one label and one split per retained component
- all retained classes represented in train, validation, and test

Every file in this audit package is covered by `artifact_checksums.sha256`; a
fresh `sha256sum -c artifact_checksums.sha256` verification passed.

## Reproduction Command

Run from the repository root:

```bash
PYTHONPATH="$PWD" /home/sha-weed/.local/share/mamba/envs/.venv_ensemble/bin/python \
  -m plantwild.code.audit_plantwild \
  --source-root plantwild/dataset/plantwild_v2 \
  --source-archive plantwild/dataset/plantwild_v2.zip \
  --output-root plantwild/audit/plantwild-v2-audit-v2-reproduction \
  --minimum-components-per-class 20
```

The implementation tests are in `tests/test_plantwild_audit.py`.

## Remaining Limitation

The computational audit is complete, but it is not a substitute for expert
plant-pathology review. Before a final paper claim about label correctness,
domain experts should adjudicate the 2,320-image review queue, prioritizing the
280 cross-label duplicate edges and probable near-duplicate table. Until then,
PlantWild should be described as an audited web-sourced benchmark with retained
semantic heterogeneity, not as a uniformly high-quality laboratory dataset.
