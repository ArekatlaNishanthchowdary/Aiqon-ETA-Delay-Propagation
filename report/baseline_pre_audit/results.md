
## N=2000, temporal split (train past, test future)  (test shipments: 500)

### ETA error, MAE in hours (lower is better)

|                                           |   booked |   departed |   at_port |   discharged |   customs_cleared |   overall |
|:------------------------------------------|---------:|-----------:|----------:|-------------:|------------------:|----------:|
| schedule (delay=0)                        |     52   |       52   |      52   |         52   |              52   |      52   |
| route-median delay                        |     26.4 |       26.4 |      26.4 |         26.4 |              26.4 |      26.4 |
| carry-forward last delay                  |     52   |       47.4 |      41.7 |         36.6 |              18.9 |      39.3 |
| carry-forward + median remaining          |     29   |       29.2 |      25.8 |         21.9 |              15.1 |      24.2 |
| direct GBM                                |     22.6 |       22.5 |      18.7 |         13.7 |               7   |      16.9 |
| chain (propagation) model                 |     25.1 |       26.3 |      17.2 |         12.1 |               4.3 |      17   |
| blend (ours)                              |     22.6 |       22.5 |      16.9 |         11.7 |               4.3 |      15.6 |
| direct GBM, no interaction/calendar feats |     25.9 |       25.7 |      22.7 |         21.2 |              15.3 |      22.1 |
| direct GBM, no weather                    |     22.7 |       22.6 |      18.8 |         13.9 |               7   |      17   |
| direct GBM, no missingness info           |     22.5 |       22.5 |      18.8 |         14   |               7.2 |      17   |

Overall MAE 95% bootstrap CI: ours [14.5 16.8], best baseline [23.1 25.5]

### 80% interval coverage / mean width (h) / delay AUC by checkpoint

|                 |   coverage |   width_h |   AUC(delay>48h) |
|:----------------|-----------:|----------:|-----------------:|
| booked          |       0.82 |     95.38 |             0.82 |
| departed        |       0.82 |     95.89 |             0.82 |
| at_port         |       0.83 |     70.82 |             0.86 |
| discharged      |       0.82 |     36.15 |             0.91 |
| customs_cleared |       0.84 |      8.77 |             0.99 |

### Delay probability (all checkpoints pooled)

|                              |   AUC |   Brier |   ECE |
|:-----------------------------|------:|--------:|------:|
| final (avg)                  | 0.894 |   0.132 | 0.023 |
| conformal-from-ETA           | 0.882 |   0.138 | 0.048 |
| calibrated direct classifier | 0.882 |   0.141 | 0.033 |
| raw GBM classifier           | 0.884 |   0.139 | 0.029 |

### Attribution fidelity vs simulator ground truth (stage level)

|                 |   spearman |   spearman_whatif |   top1_agree |   top1_if_always_port |   top1_if_always_customs |   mean_abs_err_h |
|:----------------|-----------:|------------------:|-------------:|----------------------:|-------------------------:|-----------------:|
| booked          |       0.31 |              0.22 |         0.51 |                  0.21 |                     0.32 |            11.79 |
| departed        |       0.33 |              0.22 |         0.51 |                  0.21 |                     0.32 |            12.03 |
| at_port         |       0.5  |              0.3  |         0.58 |                  0.21 |                     0.32 |             8.72 |
| discharged      |       0.64 |              0.36 |         0.71 |                  0.21 |                     0.32 |             6.21 |
| customs_cleared |       0.83 |              0.45 |         0.84 |                  0.21 |                     0.32 |             3.51 |

## N=300 subset (assignment size), temporal split  (test shipments: 75)

### ETA error, MAE in hours (lower is better)

|                                  |   booked |   departed |   at_port |   discharged |   customs_cleared |   overall |
|:---------------------------------|---------:|-----------:|----------:|-------------:|------------------:|----------:|
| schedule (delay=0)               |     50.2 |       50.2 |      50.2 |         50.2 |              50.2 |      50.2 |
| route-median delay               |     27.2 |       27.2 |      27.2 |         27.2 |              27.2 |      27.2 |
| carry-forward last delay         |     50.2 |       46.2 |      40.9 |         37.6 |              19.8 |      38.9 |
| carry-forward + median remaining |     27.7 |       28.2 |      26.4 |         22.6 |              15.8 |      24.2 |
| direct GBM                       |     23.6 |       23.4 |      21.5 |         16.9 |              12.5 |      19.6 |
| chain (propagation) model        |     25.7 |       25.3 |      23.4 |         11.1 |               4.9 |      18.1 |
| blend (ours)                     |     23.9 |       23.7 |      21.5 |         10.7 |               4.9 |      16.9 |

Overall MAE 95% bootstrap CI: ours [13.3 21.5], best baseline [20.6 28.5]

### 80% interval coverage / mean width (h) / delay AUC by checkpoint

|                 |   coverage |   width_h |   AUC(delay>48h) |
|:----------------|-----------:|----------:|-----------------:|
| booked          |       0.79 |     87.31 |             0.83 |
| departed        |       0.79 |     78.94 |             0.79 |
| at_port         |       0.75 |     75.75 |             0.85 |
| discharged      |       0.85 |     38.05 |             0.9  |
| customs_cleared |       0.84 |      9.22 |             0.99 |

### Delay probability (all checkpoints pooled)

|                              |   AUC |   Brier |   ECE |
|:-----------------------------|------:|--------:|------:|
| final (avg)                  | 0.882 |   0.14  | 0.042 |
| conformal-from-ETA           | 0.876 |   0.14  | 0.069 |
| calibrated direct classifier | 0.841 |   0.169 | 0.064 |
| raw GBM classifier           | 0.844 |   0.167 | 0.075 |

### Attribution fidelity vs simulator ground truth (stage level)

|                 |   spearman |   spearman_whatif |   top1_agree |   top1_if_always_port |   top1_if_always_customs |   mean_abs_err_h |
|:----------------|-----------:|------------------:|-------------:|----------------------:|-------------------------:|-----------------:|
| booked          |       0.31 |              0.2  |         0.47 |                   0.2 |                     0.39 |            11.41 |
| departed        |       0.29 |              0.15 |         0.44 |                   0.2 |                     0.39 |            12.06 |
| at_port         |       0.47 |              0.32 |         0.59 |                   0.2 |                     0.39 |            10.04 |
| discharged      |       0.63 |              0.43 |         0.64 |                   0.2 |                     0.39 |             6.35 |
| customs_cleared |       0.83 |              0.43 |         0.8  |                   0.2 |                     0.39 |             3.53 |

## N=2000, route hold-out (2 routes unseen in training)  (test shipments: 389)

### ETA error, MAE in hours (lower is better)

|                                  |   booked |   departed |   at_port |   discharged |   customs_cleared |   overall |
|:---------------------------------|---------:|-----------:|----------:|-------------:|------------------:|----------:|
| schedule (delay=0)               |     79.9 |       79.9 |      79.9 |         79.9 |              79.9 |      79.9 |
| route-median delay               |     42.5 |       42.5 |      42.5 |         42.5 |              42.5 |      42.5 |
| carry-forward last delay         |     79.9 |       74.6 |      63.3 |         41   |              20.4 |      55.8 |
| carry-forward + median remaining |     42.5 |       42.6 |      35.9 |         26.3 |              17.8 |      33   |
| direct GBM                       |     33.6 |       33.3 |      24.9 |         18.5 |              11.1 |      24.3 |
| chain (propagation) model        |     35.2 |       39.5 |      22.3 |         18.5 |               7   |      24.5 |
| blend (ours)                     |     32.9 |       33.4 |      21.9 |         17.3 |               6.9 |      22.5 |

Overall MAE 95% bootstrap CI: ours [20.5 24.3], best baseline [30.7 35.1]

### 80% interval coverage / mean width (h) / delay AUC by checkpoint

|                 |   coverage |   width_h |   AUC(delay>48h) |
|:----------------|-----------:|----------:|-----------------:|
| booked          |       0.67 |     73.66 |             0.75 |
| departed        |       0.61 |     70.92 |             0.75 |
| at_port         |       0.75 |     60.36 |             0.9  |
| discharged      |       0.72 |     36.07 |             0.95 |
| customs_cleared |       0.73 |      9.24 |             0.99 |

### Delay probability (all checkpoints pooled)

|                              |   AUC |   Brier |   ECE |
|:-----------------------------|------:|--------:|------:|
| final (avg)                  | 0.898 |   0.111 | 0.035 |
| conformal-from-ETA           | 0.874 |   0.121 | 0.051 |
| calibrated direct classifier | 0.893 |   0.117 | 0.049 |
| raw GBM classifier           | 0.895 |   0.113 | 0.033 |

### Attribution fidelity vs simulator ground truth (stage level)

|                 |   spearman |   spearman_whatif |   top1_agree |   top1_if_always_port |   top1_if_always_customs |   mean_abs_err_h |
|:----------------|-----------:|------------------:|-------------:|----------------------:|-------------------------:|-----------------:|
| booked          |       0.33 |              0.23 |         0.51 |                  0.52 |                      0.2 |            15.54 |
| departed        |       0.34 |              0.28 |         0.51 |                  0.52 |                      0.2 |            17.29 |
| at_port         |       0.52 |              0.4  |         0.61 |                  0.52 |                      0.2 |            10.41 |
| discharged      |       0.61 |              0.42 |         0.69 |                  0.52 |                      0.2 |             8.09 |
| customs_cleared |       0.82 |              0.55 |         0.83 |                  0.52 |                      0.2 |             5.17 |