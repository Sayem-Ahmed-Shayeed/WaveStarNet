# AgriVision4 Ablation Statistical Analysis

Paired comparisons use the same 915-image locked validation set and matching seeds (42, 2024, 3407). Confidence intervals are based on 10,000 class-stratified paired bootstrap resamples. P-values use 10,000 paired randomization resamples and are Holm-corrected within each metric family.

| Variant | Params | Macro-F1 (%) | Delta (pp) | Bootstrap 95% CI (pp) | Holm p | Accuracy (%) | Delta (pp) | Bootstrap 95% CI (pp) | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WavestarNet (fixed Haar WaveDown) | 2,952,257 | 63.35 | +0.03 | [-2.04, +2.10] | 1.0000 | 67.76 | -0.11 | [-1.97, +1.82] | 1.0000 |
| WavestarNet (capacity-matched conventional downsampling) | 2,955,497 | 59.38 | -3.94 | [-6.29, -1.68] | 0.0105 | 64.48 | -3.39 | [-5.43, -1.31] | 0.0120 |
| WavestarNet (no learned wavelet gates) | 2,708,397 | 62.98 | -0.34 | [-2.05, +1.49] | 1.0000 | 67.50 | -0.36 | [-1.64, +0.91] | 1.0000 |
| WavestarNet (all blocks wavelet-gated) | 3,101,833 | 63.18 | -0.15 | [-1.62, +1.36] | 1.0000 | 67.69 | -0.18 | [-1.49, +1.09] | 1.0000 |
| WavestarNet (SE attention replacement) | 2,767,793 | 62.82 | -0.50 | [-2.11, +1.09] | 1.0000 | 67.69 | -0.18 | [-1.49, +1.13] | 1.0000 |
| WavestarNet (element-wise mean fusion) | 2,956,577 | 63.19 | -0.13 | [-2.08, +1.70] | 1.0000 | 66.67 | -1.20 | [-2.73, +0.33] | 0.7509 |
| WavestarNet (stage-4-only head) | 2,949,953 | 61.01 | -2.31 | [-4.15, -0.51] | 0.0996 | 65.50 | -2.37 | [-3.93, -0.73] | 0.0258 |
| WavestarNet (lower-capacity conventional downsampling) | 2,661,737 | 59.26 | -4.07 | [-6.40, -1.82] | 0.0088 | 64.55 | -3.32 | [-5.36, -1.31] | 0.0161 |

## Interpretation Rules

- Positive deltas favor the ablation; negative deltas favor full WavestarNet.
- Bootstrap intervals quantify validation-sample uncertainty for the observed three-seed mean.
- Seed-level t intervals are retained in the CSV and quantify training-run uncertainty, but are imprecise with only three seeds.
- Statistical significance does not replace practical effect-size interpretation.
