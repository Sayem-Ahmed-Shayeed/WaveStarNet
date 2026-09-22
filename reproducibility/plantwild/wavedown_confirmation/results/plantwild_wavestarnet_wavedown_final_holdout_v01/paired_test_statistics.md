# PlantWild WaveDown Post-Selection Confirmation

This report-only analysis uses the saved three-seed test predictions.
The full and parameter-matched conventional variants were specified before this PlantWild test-set confirmation.
No PlantWild test-set result was used to select the WavestarNet architecture or training protocol.
Positive effects favor full WavestarNet. Confidence intervals use 10,000 class-stratified paired bootstrap resamples.
Paired label-swap randomization p-values are Holm-adjusted across Accuracy and Macro-F1 only.

| metric   |   full_wavestarnet_mean |   parameter_matched_conventional_mean |   full_minus_conventional |   full_minus_conventional_pp |    seed_sd |   bootstrap95_low |   bootstrap95_high |   bootstrap95_low_pp |   bootstrap95_high_pp |   paired_randomization_p_raw |   bootstrap_resamples |   randomization_resamples |   paired_randomization_p_holm | holm_reject_0_05   |
|:---------|------------------------:|--------------------------------------:|--------------------------:|-----------------------------:|-----------:|------------------:|-------------------:|---------------------:|----------------------:|-----------------------------:|----------------------:|--------------------------:|------------------------------:|:-------------------|
| accuracy |                0.518713 |                              0.510177 |                0.00853578 |                     0.853578 | 0.00626355 |       -0.00415846 |          0.02123   |            -0.415846 |               2.123   |                    0.19868   |                 10000 |                     10000 |                     0.19868   | False              |
| macro_f1 |                0.451628 |                              0.433726 |                0.0179022  |                     1.79022  | 0.00450402 |        0.00247034 |          0.0325219 |             0.247034 |               3.25219 |                    0.0280972 |                 10000 |                     10000 |                     0.0561944 | False              |
