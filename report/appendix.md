# Appendix: secondary results and investigations

Supporting material for `report.pdf`. Every number is copied from a generated file, named in each section.
All results in sections A–D are on **synthetic data**.

## A. Ablations (source: `results.md`, section A)

Direct model, temporal split, ETA MAE in hours.

| Variant | Booked | Departed | At port | Discharged | Customs cleared | Overall |
|---|--:|--:|--:|--:|--:|--:|
| All features | 22.7 | 22.8 | 18.8 | 13.9 | 7.0 | 17.0 |
| No interaction / calendar features | 25.8 | 25.7 | 22.6 | 21.1 | 15.3 | 22.1 |
| No weather | 22.7 | 22.8 | 18.8 | 13.9 | 7.2 | 17.1 |
| No missingness information | 22.6 | 22.7 | 18.9 | 13.8 | 7.1 | 17.0 |

Interaction and calendar features carry almost all the value (+5.1 h when removed). Weather and missingness features change the
error by at most 0.1 h. Ablations were run on the direct model only, not the chain.

## B. Robustness runs (source: `results.md`, sections B–D)

| Experiment | Blend MAE | Best baseline | 80% interval coverage | Note |
|---|--:|--:|---|---|
| 300-shipment subset (180 / 45 / 75) | 16.7 h (CI 13.5–21.1) | 24.2 h | 0.69–0.92 | About 22 shipments per calibration half; the calibrated classifier is unreliable (ECE 0.159) |
| Route hold-out, two unseen routes (1,289 / 322 / 389) | 22.3 h (CI 20.4–24.4) | 33.0 h | 0.62–0.78 | Early top-1 attribution (0.52, 0.54) no better than always blaming the port (0.52) |
| Three independent simulated worlds | 16.9 ± 1.1 h | 25.8 ± 2.0 h | 0.79 ± 0.02 | In one world the blend is no better than the direct model (18.06 vs 17.94 h) |

## C. Delay probability and intervals (source: `results.md`, section A)

| Estimator of P(delay > 48 h) | AUC | Brier | ECE |
|---|--:|--:|--:|
| Final (average of the two below) | 0.891 | 0.136 | 0.035 |
| Conformal, from the ETA distribution | 0.878 | 0.142 | 0.067 |
| Isotonic-calibrated classifier | 0.878 | 0.145 | 0.056 |
| Raw GBM classifier | 0.881 | 0.141 | 0.028 |

The raw classifier is the best calibrated; the calibration layer is not shown to help. Un-calibrated quantile-GBM intervals cover
0.75–0.79 against 0.77–0.87 for the conformal intervals, so conformalisation helps little on the temporal split.

## D. Chain-model investigations

### D1. Checkpoint-aware training (source: `chain_fix_effect.md`)

The chain was originally trained with every input, as if at the last checkpoint, and queried earlier with inputs missing.
LightGBM maps a missing value to 0 when training contained none, so the model saw "zero congestion" and predicted a mean
port-stage increment of −0.9 h where the true mean is 13.0 h. The fix trains one model per information regime.

| Chain-only MAE (h), test | Booked | Departed | At port | Discharged | Customs cleared | Overall |
|---|--:|--:|--:|--:|--:|--:|
| Old (trained as if at checkpoint 4) | 22.99 | 26.26 | 17.22 | 12.09 | 4.27 | 16.57 |
| New (per-regime models) | 27.06 | 26.72 | 17.22 | 12.09 | 4.27 | 17.47 |

The corrected chain is *worse* at booking: the old model's bias happened to help on right-skewed delays. The corrected version was kept.

### D2. Exposure bias (source: `chain_exposure_bias.md`; validation split only, test never read)

Stage models are trained with the true previous-stage delay but receive a rolled-forward prediction at inference.
The input noise is large:

| Stage (carry-in) | RMSE of predicted carry-in | SD of the true delay |
|---|--:|--:|
| 1 (departure delay) | 6.0 h | 5.9 h |
| 2 (port-arrival delay) | 11.4 h | 12.6 h |
| 3 (discharge delay) | 29.9 h | 37.2 h |
| 4 (customs delay) | 37.3 h | 43.8 h |

Three corrections were implemented and compared on validation data.

| Chain variant (chain-only MAE, h) | Booked | Departed | At port | Discharged | Customs cleared | Overall |
|---|--:|--:|--:|--:|--:|--:|
| A. Teacher forcing (kept) | 29.81 | 28.76 | 19.42 | 13.67 | 4.86 | 19.30 |
| B. Out-of-fold carry-in, target = true stage increment | 29.94 | 29.16 | 19.67 | 15.94 | 5.09 | 19.96 |
| C. Sequential residual chain: out-of-fold carry-in, target = true delay minus predicted carry-in, one model per stage and checkpoint | 28.37 | 30.62 | 21.86 | 16.24 | 5.79 | 20.57 |

| Combined with the direct model (MAE, h, validation half B) | Overall |
|---|--:|
| Direct model alone | 19.22 |
| Weight blend: direct + A (the shipped system) | 17.48 |
| Weight blend: direct + C | 18.63 |
| Stacked model: direct features + C's prediction as an input | 18.82 |

None of the corrections beats teacher forcing, alone or combined. A likely reason, not verified: propagation in this simulator is
close to linear in the carried delay, so plugging in the predicted delay is nearly unbiased, and the noise-aware variants mainly
lose data efficiency. The exposure bias is therefore measured and documented, but not removed.

## E. Do the simulator's mechanisms appear in real data? (source: `../realworld/simulator_check.md`)

Real: 1,639 port stays by 717 cargo and tanker vessels at 26 ports, from the AIS data. This checks direction and rough size of
effects only. Levels are not comparable: AIS measures how long a *vessel* stays, the simulator models when *cargo* is available,
and the regions differ. No simulator parameter was changed from these numbers.

| Mechanism assumed in the simulator | Real AIS | Simulator | Verdict |
|---|---|---|---|
| Congestion lengthens the port stage | Spearman −0.03 (CI −0.08 to +0.03); 0.92× when the port is busier than usual (CI 0.84–1.02) | Spearman +0.55; 1.43× | **Not detected** |
| Port-stage duration is right-skewed | mean/median 1.71 | mean/median 1.22 | Supported; real tail is heavier |
| Stays starting Fri–Sun are longer | 1.07× (CI 0.96–1.18) | modelled at customs only | Not detected |
| Arrival lateness has a late tail | 11% more than 6 h late vs 3% more than 6 h early | 62% more than 6 h late (vs a fixed schedule) | Supported |

The port-occupancy proxy is crude and these ports are rarely full, so this does not show congestion never matters. It does show
that the simulator's strongest hand-coded mechanism is **not externally validated**, and the ablation in section A — where
interaction features carry most of the value — partly reflects the model recovering rules that were written into the simulator.

## F. Audit findings and fixes

| Finding | Status |
|---|---|
| Chain trained with inputs not available at earlier checkpoints | Fixed: one model per information regime (D1) |
| "Booking" congestion was the departure-day value | Fixed: available from the departed checkpoint only |
| Blend weights and conformal residuals fitted on the same data | Fixed: separate halves of the calibration split |
| Actual timeline never validated; CLI accepted incomplete records | Fixed |
| Explanation driver used the wrong previous-stage delay and could name a negative contributor | Fixed |
| Unclosed file handles, unpinned dependencies, results silently overwritten | Fixed; runs are archived with metadata |
| Feature leakage, split isolation, bootstrap pairing | Checked and found sound (adversarial tests) |
| Attribution accounting chosen after seeing test fidelity | Disclosed; mitigated by replication on two fresh simulated worlds |
| Chain exposure bias | Measured; three corrections tested, none better (D2) |
