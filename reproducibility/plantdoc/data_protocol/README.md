# PlantDoc Clean Grouped V1

This private derivative was built from the supplied 2,922-image PlantDoc repack. It is not claimed to be the original 2,598-image/27-class paper release.

## Cleaning

- Source images inventoried: 2922
- Canonical images retained: 2898
- Conflicting-label images excluded: 16
- Under-supported canonical images excluded: 2
- Retained benchmark classes: 27
- Minimum independent components per retained class: 20
- Exact duplicate edges: 6
- Confirmed near-copy edges: 8
- Near-copy thresholds: perceptual Hamming <= 4, RGB MAD <= 0.001
- Split seed/fractions: 42 / 70-15-15

Every multi-label duplicate component is excluded. Every retained single-label component contributes one unchanged canonical source file. The test split must not be evaluated with legacy checkpoints because they may have trained on reassigned images.

## Provenance and license warning

The supplied archive contains no release identifier, license, or redistribution terms. This Kaggle dataset must remain private until those terms are verified. Kaggle license metadata is `other`; it is not a new license grant.

Source archive SHA-256: `2c5c079d13f09c263b0e530b87a85dc15d23c5a00a7d9c9a8b9e74c12c48dc15`

See `audit/` for exclusions, components, thresholds, distributions, and hashes. See `manifests/` for the fixed class mapping and splits.
