
## A. N=2000, temporal split (train past, test future)

shipments: train 1200, calibration 300, test 500 (2500 checkpoint rows)

### ETA error, MAE in hours (lower is better)

|                                           |   booked |   departed |   at_port |   discharged |   customs_cleared |   overall |
|:------------------------------------------|---------:|-----------:|----------:|-------------:|------------------:|----------:|
| schedule (delay=0)                        |     52   |       52   |      52   |         52   |              52   |      52   |
| route-median delay                        |     26.4 |       26.4 |      26.4 |         26.4 |              26.4 |      26.4 |
| carry-forward last delay                  |     52   |       47.4 |      41.7 |         36.6 |              18.9 |      39.3 |
| carry-forward + median remaining          |     29   |       29.2 |      25.8 |         21.9 |              15.1 |      24.2 |
| direct GBM                                |     22.7 |       22.8 |      18.8 |         13.9 |               7   |      17   |
| chain (propagation) model                 |     27.1 |       26.7 |      17.2 |         12.1 |               4.3 |      17.5 |
| blend (ours)                              |     23.1 |       23.8 |      16.9 |         11.8 |               4.2 |      16   |
| direct GBM, no interaction/calendar feats |     25.8 |       25.7 |      22.6 |         21.1 |              15.3 |      22.1 |
| direct GBM, no weather                    |     22.7 |       22.8 |      18.8 |         13.9 |               7.2 |      17.1 |
| direct GBM, no missingness info           |     22.6 |       22.7 |      18.9 |         13.8 |               7.1 |      17   |

Overall MAE 95% paired-bootstrap CI (500 resamples of test shipments): ours [14.7 17.3]; best_baseline [23.  25.5]; diff (baseline - ours) [7.1 9.5]; diff (direct - ours) [0.7 1.5]; diff (chain - ours) [1.  1.9]

### 80% interval coverage / mean width (h) / delay AUC by checkpoint

|                 |   conformal coverage |   width_h |   raw quantile-GBM coverage |   AUC(delay>48h) |
|:----------------|---------------------:|----------:|----------------------------:|-----------------:|
| booked          |                 0.77 |     86.31 |                        0.78 |             0.82 |
| departed        |                 0.8  |     82.48 |                        0.79 |             0.81 |
| at_port         |                 0.82 |     66.79 |                        0.78 |             0.86 |
| discharged      |                 0.79 |     34.35 |                        0.79 |             0.91 |
| customs_cleared |                 0.87 |      9.96 |                        0.75 |             0.99 |

### Delay probability (all checkpoints pooled)

|                              |   AUC |   Brier |   ECE |
|:-----------------------------|------:|--------:|------:|
| final (avg)                  | 0.891 |   0.136 | 0.035 |
| conformal-from-ETA           | 0.878 |   0.142 | 0.067 |
| calibrated direct classifier | 0.878 |   0.145 | 0.056 |
| raw GBM classifier           | 0.881 |   0.141 | 0.028 |

### Attribution fidelity vs simulator ground truth (stage level)

|                 |   spearman |   spearman_whatif |   top1_agree |   top1_if_always_port |   top1_if_always_customs |   mean_abs_err_h |
|:----------------|-----------:|------------------:|-------------:|----------------------:|-------------------------:|-----------------:|
| booked          |       0.29 |              0.19 |         0.48 |                  0.21 |                     0.32 |            12.42 |
| departed        |       0.31 |              0.18 |         0.5  |                  0.21 |                     0.32 |            12.44 |
| at_port         |       0.5  |              0.3  |         0.58 |                  0.21 |                     0.32 |             8.72 |
| discharged      |       0.64 |              0.36 |         0.71 |                  0.21 |                     0.32 |             6.21 |
| customs_cleared |       0.83 |              0.45 |         0.84 |                  0.21 |                     0.32 |             3.51 |

## B. N=300 subset (assignment size), temporal split

shipments: train 180, calibration 45, test 75 (375 checkpoint rows)

### ETA error, MAE in hours (lower is better)

|                                  |   booked |   departed |   at_port |   discharged |   customs_cleared |   overall |
|:---------------------------------|---------:|-----------:|----------:|-------------:|------------------:|----------:|
| schedule (delay=0)               |     50.2 |       50.2 |      50.2 |         50.2 |              50.2 |      50.2 |
| route-median delay               |     27.2 |       27.2 |      27.2 |         27.2 |              27.2 |      27.2 |
| carry-forward last delay         |     50.2 |       46.2 |      40.9 |         37.6 |              19.8 |      38.9 |
| carry-forward + median remaining |     27.7 |       28.2 |      26.4 |         22.6 |              15.8 |      24.2 |
| direct GBM                       |     23.3 |       23.2 |      21.7 |         16.1 |              12.2 |      19.3 |
| chain (propagation) model        |     26.4 |       29.7 |      23.4 |         11.1 |               4.9 |      19.1 |
| blend (ours)                     |     23.3 |       23.2 |      21.6 |         10.6 |               4.9 |      16.7 |

Overall MAE 95% paired-bootstrap CI (500 resamples of test shipments): ours [13.5 21.1]; best_baseline [20.8 28.5]; diff (baseline - ours) [ 5.1 10. ]; diff (direct - ours) [1.7 3.5]; diff (chain - ours) [0.6 4.1]

### 80% interval coverage / mean width (h) / delay AUC by checkpoint

|                 |   conformal coverage |   width_h |   raw quantile-GBM coverage |   AUC(delay>48h) |
|:----------------|---------------------:|----------:|----------------------------:|-----------------:|
| booked          |                 0.69 |     79.91 |                        0.73 |             0.8  |
| departed        |                 0.81 |     85.94 |                        0.65 |             0.83 |
| at_port         |                 0.72 |     64.1  |                        0.64 |             0.85 |
| discharged      |                 0.87 |     36.07 |                        0.6  |             0.86 |
| customs_cleared |                 0.92 |     25.92 |                        0.6  |             0.98 |

### Delay probability (all checkpoints pooled)

|                              |   AUC |   Brier |   ECE |
|:-----------------------------|------:|--------:|------:|
| final (avg)                  | 0.874 |   0.149 | 0.069 |
| conformal-from-ETA           | 0.878 |   0.142 | 0.084 |
| calibrated direct classifier | 0.825 |   0.194 | 0.159 |
| raw GBM classifier           | 0.842 |   0.166 | 0.06  |

### Attribution fidelity vs simulator ground truth (stage level)

|                 |   spearman |   spearman_whatif |   top1_agree |   top1_if_always_port |   top1_if_always_customs |   mean_abs_err_h |
|:----------------|-----------:|------------------:|-------------:|----------------------:|-------------------------:|-----------------:|
| booked          |       0.29 |              0.14 |         0.51 |                   0.2 |                     0.39 |            12.22 |
| departed        |       0.32 |              0.13 |         0.47 |                   0.2 |                     0.39 |            12.68 |
| at_port         |       0.47 |              0.32 |         0.59 |                   0.2 |                     0.39 |            10.04 |
| discharged      |       0.63 |              0.43 |         0.64 |                   0.2 |                     0.39 |             6.35 |
| customs_cleared |       0.83 |              0.43 |         0.8  |                   0.2 |                     0.39 |             3.53 |

## C. N=2000, route hold-out (Nhava Sheva>Hamburg and Chennai>Los Angeles unseen)

shipments: train 1289, calibration 322, test 389 (1945 checkpoint rows)

### ETA error, MAE in hours (lower is better)

|                                  |   booked |   departed |   at_port |   discharged |   customs_cleared |   overall |
|:---------------------------------|---------:|-----------:|----------:|-------------:|------------------:|----------:|
| schedule (delay=0)               |     79.9 |       79.9 |      79.9 |         79.9 |              79.9 |      79.9 |
| route-median delay               |     42.5 |       42.5 |      42.5 |         42.5 |              42.5 |      42.5 |
| carry-forward last delay         |     79.9 |       74.6 |      63.3 |         41   |              20.4 |      55.8 |
| carry-forward + median remaining |     42.5 |       42.6 |      35.9 |         26.3 |              17.8 |      33   |
| direct GBM                       |     34.2 |       33.1 |      25.5 |         18.8 |              11.1 |      24.5 |
| chain (propagation) model        |     35.6 |       34.9 |      22.3 |         18.5 |               7   |      23.7 |
| blend (ours)                     |     33.1 |       32.4 |      21.9 |         17.3 |               6.9 |      22.3 |

Overall MAE 95% paired-bootstrap CI (500 resamples of test shipments): ours [20.4 24.4]; best_baseline [30.5 35.5]; diff (baseline - ours) [ 9.1 12.2]; diff (direct - ours) [1.2 3.2]; diff (chain - ours) [0.8 2. ]

### 80% interval coverage / mean width (h) / delay AUC by checkpoint

|                 |   conformal coverage |   width_h |   raw quantile-GBM coverage |   AUC(delay>48h) |
|:----------------|---------------------:|----------:|----------------------------:|-----------------:|
| booked          |                 0.62 |     77.16 |                        0.78 |             0.77 |
| departed        |                 0.63 |     72.07 |                        0.76 |             0.76 |
| at_port         |                 0.72 |     62.49 |                        0.71 |             0.9  |
| discharged      |                 0.78 |     60.71 |                        0.7  |             0.95 |
| customs_cleared |                 0.74 |      9.51 |                        0.71 |             0.99 |

### Delay probability (all checkpoints pooled)

|                              |   AUC |   Brier |   ECE |
|:-----------------------------|------:|--------:|------:|
| final (avg)                  | 0.903 |   0.116 | 0.058 |
| conformal-from-ETA           | 0.869 |   0.125 | 0.065 |
| calibrated direct classifier | 0.894 |   0.124 | 0.069 |
| raw GBM classifier           | 0.897 |   0.111 | 0.021 |

### Attribution fidelity vs simulator ground truth (stage level)

|                 |   spearman |   spearman_whatif |   top1_agree |   top1_if_always_port |   top1_if_always_customs |   mean_abs_err_h |
|:----------------|-----------:|------------------:|-------------:|----------------------:|-------------------------:|-----------------:|
| booked          |       0.33 |              0.29 |         0.52 |                  0.52 |                      0.2 |            15.75 |
| departed        |       0.36 |              0.26 |         0.54 |                  0.52 |                      0.2 |            14.95 |
| at_port         |       0.52 |              0.4  |         0.61 |                  0.52 |                      0.2 |            10.41 |
| discharged      |       0.61 |              0.42 |         0.69 |                  0.52 |                      0.2 |             8.09 |
| customs_cleared |       0.82 |              0.55 |         0.83 |                  0.52 |                      0.2 |             5.17 |

## D. Replication over 3 independent simulated worlds (N=2000 each, temporal split)

|   sim_seed |   blend_MAE |   best_baseline_MAE |   direct_MAE |   chain_MAE |   mean_coverage |   AUC |   ECE |   attr_top1_after_customs |
|-----------:|------------:|--------------------:|-------------:|------------:|----------------:|------:|------:|--------------------------:|
|          0 |      15.974 |              24.198 |       17.037 |      17.47  |           0.811 | 0.891 | 0.035 |                     0.838 |
|          1 |      18.061 |              25.025 |       17.939 |      18.631 |           0.78  | 0.9   | 0.038 |                     0.804 |
|          2 |      16.736 |              28.034 |       18.688 |      17.265 |           0.782 | 0.918 | 0.03  |                     0.764 |

mean +- sd: blend_MAE 16.92+-1.06, best_baseline_MAE 25.75+-2.02, direct_MAE 17.89+-0.83, chain_MAE 17.79+-0.74, mean_coverage 0.79+-0.02, AUC 0.90+-0.01, ECE 0.03+-0.00, attr_top1_after_customs 0.80+-0.04