# AgriVision4 Grad-CAM and Error Analysis

## Scope

This report analyzes all 915 images in the locked AgriVision4 validation split for the seed-42 scratch checkpoints of WavestarNet V2 and MobileViT-S. Predictions exactly reproduce the archived validation outputs. Grad-CAM was generated for both the predicted and ground-truth classes.

The analysis supports statements about prediction behavior and spatial concentration of Grad-CAM maps. It does **not** establish lesion localization accuracy because AgriVision4 has no lesion masks or bounding-box annotations.

## Main Findings

| Finding | WavestarNet V2 | MobileViT-S | Interpretation |
|---|---:|---:|---|
| Accuracy | 0.6787 | 0.6667 | WavestarNet: +1.20 percentage points |
| Macro F1 | 0.6327 | 0.6246 | WavestarNet: +0.80 percentage points |
| Weighted F1 | 0.6828 | 0.6664 | WavestarNet: +1.64 percentage points |
| Correct predictions | 621 | 610 | WavestarNet makes 11 more correct predictions |
| Model-only correct cases | 91 | 80 | The models are complementary, but WavestarNet wins more disagreements |
| Incorrect predictions with confidence >= 0.90 | 10/294 (3.40%) | 41/305 (13.44%) | MobileViT-S is substantially more often confidently wrong |
| ECE, 10 equal-width bins | 0.0340 | 0.0777 | WavestarNet is better calibrated on this split |

Both models are correct on 530 images (57.92%) and wrong on 214 images (23.39%). They output the same class for 686 images (74.97%). Therefore, most remaining errors are shared rather than architecture-specific.

## Class-Level Behavior

WavestarNet has higher class F1 on 12 of 21 classes, ties on one, and is lower on eight.

Largest WavestarNet F1 gains over MobileViT-S:

| Class | Support | WavestarNet F1 | MobileViT-S F1 | Delta |
|---|---:|---:|---:|---:|
| Papaya Bacterial Blight | 31 | 0.4667 | 0.3571 | +0.1095 |
| Tomato Mosaic | 45 | 0.5882 | 0.4941 | +0.0941 |
| Papaya Carica Insect Hole | 72 | 0.5882 | 0.5000 | +0.0882 |
| Papaya Yellow Necrotic Spots/Holes | 12 | 0.5714 | 0.4878 | +0.0836 |
| Papaya Mosaic Virus | 21 | 0.4348 | 0.3556 | +0.0792 |

Largest WavestarNet F1 deficits:

| Class | Support | WavestarNet F1 | MobileViT-S F1 | Delta |
|---|---:|---:|---:|---:|
| Zucchini Downy Mildew | 7 | 0.7368 | 0.9231 | -0.1862 |
| Papaya Pathogen Symptoms | 66 | 0.6094 | 0.6753 | -0.0660 |
| Zucchini Yellow Mosaic Virus | 32 | 0.8438 | 0.9091 | -0.0653 |
| Bottle Gourd Fungal Damage Leaf | 9 | 0.2857 | 0.3333 | -0.0476 |
| Tomato Downy Mildew | 13 | 0.5385 | 0.5833 | -0.0449 |

WavestarNet's six lowest-F1 classes are Bottle Gourd Fungal Damage Leaf (0.2857), Papaya Mosaic Virus (0.4348), Tomato White Spot (0.4444), Bottle Gourd Early Alternaria Leaf Blight (0.4557), Papaya Bacterial Blight (0.4667), and Tomato Downy Mildew (0.5385).

Class support is positively associated with F1. The Spearman correlation is 0.554 for WavestarNet (p=0.009) and 0.408 for MobileViT-S (p=0.067). This is evidence that low-support classes remain a material weakness, especially for WavestarNet. The very large Zucchini Downy Mildew difference must be interpreted cautiously because the class contains only seven validation images.

## Dominant Error Modes

The largest WavestarNet confusions are:

| Ground truth -> prediction | Count |
|---|---:|
| Papaya Pathogen Symptoms -> Papaya Healthy Leaf | 16 |
| Papaya Carica Insect Hole -> Papaya Healthy Leaf | 13 |
| Papaya Carica Insect Hole -> Papaya Pathogen Symptoms | 11 |
| Bottle Gourd Anthracnose -> Bottle Gourd Alternaria Leaf Blight | 10 |
| Bottle Gourd Early Alternaria Leaf Blight -> Bottle Gourd Healthy | 8 |
| Papaya Curled Yellow Spot -> Papaya Pathogen Symptoms | 8 |
| Tomato Spot -> Tomato Mosaic | 8 |

These errors cluster around visually overlapping symptom families: mild disease versus healthy tissue, Papaya hole/spot/pathogen categories, and Bottle Gourd Alternaria/Anthracnose/Downy Mildew. Only 25 of WavestarNet's 294 errors (8.50%) cross crop species; for MobileViT-S, 29 of 305 errors (9.51%) cross species. Most errors therefore occur within the correct crop family and concern disease discrimination rather than crop recognition.

Representative shared failures:

| Sample | Ground truth | Both models predict | WavestarNet confidence | MobileViT-S confidence |
|---|---|---|---:|---:|
| `clean_002818` | Papaya Pathogen Symptoms | Papaya Healthy Leaf | 0.9734 | 0.9672 |
| `clean_000539` | Bottle Gourd Downy Mildew | Papaya Yellow Necrotic Spots/Holes | 0.9278 | 0.9957 |
| `clean_000746` | Bottle Gourd Early Alternaria Leaf Blight | Bottle Gourd Healthy | 0.9719 | 0.9072 |
| `clean_001771` | Papaya Carica Insect Hole | Papaya Healthy Leaf | 0.8928 | 0.9819 |

These cases should be used in the manuscript's failure panel because they expose both subtle symptom confusion and occasional crop-level shortcut behavior.

## Grad-CAM Spatial Behavior

For correctly predicted classes, WavestarNet maps have a greater top-10%-pixel mass fraction (0.2981 versus 0.1778) and lower normalized spatial entropy (0.9655 versus 0.9883). The same general pattern holds for incorrect predictions. This indicates that WavestarNet's Grad-CAM evidence is more spatially concentrated, whereas MobileViT-S produces broader, smoother activation.

Visual inspection of the case grids is consistent with those descriptors:

- WavestarNet frequently produces multiple localized responses around spots, damaged tissue, veins, or leaf boundaries.
- MobileViT-S commonly highlights a large contiguous region spanning much of the leaf and sometimes adjacent background.
- In WavestarNet-only-correct cases, localized responses often coincide with visible discoloration or structural damage.
- In MobileViT-S-only-correct cases, broad contextual evidence can still be effective, especially when symptoms cover a large fraction of the leaf.

This is a **descriptive architectural difference**, not proof that WavestarNet localizes lesions better. Absolute Grad-CAM values are not directly comparable across architectures, and the target layers have different spatial semantics.

## Background-Bias Assessment

Aggregate border-excess ratios do not show strong global border preference: WavestarNet is 0.976 for correct predictions and 0.938 for errors; MobileViT-S is 0.743 and 0.750, respectively. Values below one mean that the border receives less activation mass than expected from its area. Consequently, the data do not support a claim of pervasive border/background bias.

However, individual maps show attention on black backgrounds, stems, leaf silhouettes, and frame edges. These are valid qualitative shortcut-learning candidates. Without foreground or lesion masks, they cannot be labeled definitively as background failures. A publishable localization claim would require manual lesion/leaf masks for a stratified subset and a metric such as pointing-game accuracy, energy-inside-mask, or deletion/insertion AUC.

## Reliability Notes

- All 915 validation images were processed for both models.
- Prediction identities and probabilities match archived scratch results within numerical tolerance.
- WavestarNet produced no constant fused maps.
- MobileViT-S produced three constant ground-truth-class maps: `clean_000928`, `clean_001174`, and `clean_003810`. Their predicted-class maps were valid. These records remain in the prediction analysis but must not be interpreted as meaningful ground-truth localization maps.
- The displayed case grids are selected by predefined prediction categories and confidence criteria, not by manually choosing attractive heatmaps.

## Defensible Paper Conclusions

1. WavestarNet V2 provides a small but consistent aggregate advantage over MobileViT-S on this locked validation split.
2. WavestarNet wins more model-disagreement cases and is substantially less often confidently wrong.
3. WavestarNet Grad-CAM maps are more spatially concentrated; MobileViT-S maps are broader and smoother.
4. Both models remain weak on rare classes and on fine-grained, visually overlapping disease categories.
5. The current evidence supports an interpretability and error-analysis section, but not a claim of quantitatively superior lesion localization.

## Artifacts

- `predictions.csv`: complete predictions, confidence, entropy, map paths, and diagnostics.
- `per_class_model_comparison.csv`: aligned class-wise precision, recall, F1, support, and model deltas.
- `model_agreement_summary.csv`: both-correct, both-wrong, and model-only-correct counts.
- `confidence_error_summary.csv`: confidence and entropy conditioned on correctness.
- `gradcam_spatial_metrics.csv`: per-image heatmap distribution descriptors.
- `gradcam_spatial_summary.csv`: model-level spatial summaries.
- `top_error_pairs.csv`: confusion-pair counts.
- `figures/`: confusion matrices and predefined case grids.

