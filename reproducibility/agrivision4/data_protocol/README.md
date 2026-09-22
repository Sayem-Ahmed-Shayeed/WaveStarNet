# AgriVision 4 Clean Grouped V1

This private derivative was built from the supplied AgriVision 4 original-image archive.

## Cleaning

- Source images inventoried: 5246
- Canonical images retained: 3891
- Conflicting-label images excluded: 799
- Under-supported canonical images excluded: 45
- Retained benchmark classes: 21
- Minimum independent components per retained class: 20
- Exact duplicate edges: 942
- Confirmed near-copy edges: 164
- Near-copy thresholds: perceptual Hamming <= 4, RGB MAD <= 0.001
- Split seed/fractions: 42 / 70-15-15

Every multi-label duplicate component is excluded. Every retained single-label component contributes one unchanged canonical source file. The test split must not be evaluated with legacy checkpoints because they may have trained on reassigned images.

## Provenance and license warning

The source package identifies Mendeley dataset `8t6k37ztxc`, but the exact upstream version, license, and redistribution terms were not supplied. This Kaggle dataset must remain private until those terms are verified. Kaggle license metadata is `other`; it is not a new license grant.

Source archive SHA-256: `70853ddd987e0bf5d8553a5e767476ad6cb3ae41d1d0bbc37faa67d5b05b3a05`

See `audit/` for exclusions, components, thresholds, distributions, and hashes. See `manifests/` for the fixed class mapping and splits.
