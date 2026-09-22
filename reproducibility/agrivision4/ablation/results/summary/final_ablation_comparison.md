# Final AgriVision4 Ablation Comparison

All values are validation results from three random-initialization runs using seeds 42, 2024, and 3407. The locked development split contains 1,810 training and 915 validation images. No test images were accessed. Macro-F1 deltas are percentage-point differences from the full WavestarNet reference.

| Component tested | Variant | Parameters | Macro-F1 (%) | Delta Macro-F1 (pp) | Accuracy (%) | Weighted F1 (%) |
|---|---|---:|---:|---:|---:|---:|
| Reference | **WavestarNet (full)** | 2,956,577 | **63.33 +/- 0.34** | 0.00 | **67.87 +/- 0.44** | **68.13 +/- 0.36** |
| WaveDown formulation | Fixed Haar WaveDown | 2,952,257 | **63.35 +/- 1.04** | +0.03 | 67.76 +/- 1.18 | 68.02 +/- 1.14 |
| WaveDown formulation | Capacity-matched conventional downsampling | 2,955,497 | 59.38 +/- 1.19 | -3.94 | 64.48 +/- 0.66 | 64.56 +/- 0.82 |
| Wavelet gating | No learned wavelet gates | 2,708,397 | 62.98 +/- 1.83 | -0.34 | 67.50 +/- 1.70 | 67.74 +/- 1.70 |
| Wavelet gating | All blocks wavelet-gated | 3,101,833 | 63.18 +/- 0.33 | -0.15 | 67.69 +/- 0.99 | 67.85 +/- 1.13 |
| Attention operator | Replace wavelet gates with SE | 2,767,793 | 62.82 +/- 0.71 | -0.51 | 67.69 +/- 0.46 | 67.99 +/- 0.60 |
| Fusion operator | Element-wise mean fusion | 2,956,577 | 63.19 +/- 1.31 | -0.13 | 66.67 +/- 0.85 | 67.07 +/- 0.87 |
| Classification head | Stage-4-only head | 2,949,953 | 61.01 +/- 1.65 | -2.31 | 65.50 +/- 1.43 | 65.68 +/- 1.51 |
| Supplemental downsampling control | Lower-capacity conventional downsampling | 2,661,737 | 59.26 +/- 1.54 | -4.07 | 64.55 +/- 0.79 | 64.49 +/- 0.81 |

## Interpretation

- Replacing WaveDown with a parameter-matched conventional operator decreases Macro-F1 by 3.94 points. This isolates a substantial benefit from wavelet-based downsampling rather than model capacity.
- Fixed Haar and learned lifting are effectively tied. The experiment supports wavelet decomposition, but it does not demonstrate that learned lifting is superior to fixed Haar.
- The multistage classification head contributes 2.31 Macro-F1 points relative to using Stage 4 alone.
- Selective wavelet gating is slightly better than gating every block, removing learned gates, or replacing them with SE, but these differences require paired statistical analysis before being described as definitive.
- Multiplicative Star fusion primarily improves accuracy and weighted F1 over element-wise mean fusion; its Macro-F1 difference is small.

The original lower-capacity conventional-downsampling result is supplemental because it removes approximately 9.97% of the reference parameters. The capacity-matched conventional result is the primary downsampling control.
