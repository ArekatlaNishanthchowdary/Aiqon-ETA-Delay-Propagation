# Chain: previous-stage-delay train/inference mismatch (validation split only)

## Noise of the previous-stage delay the models receive when it is not yet observed (hours)

|                    |    n |   rmse_h |   sd_true_h |
|:-------------------|-----:|---------:|------------:|
| stage 1 (input d0) |  300 |      6   |         5.9 |
| stage 2 (input d1) |  580 |     11.4 |        12.6 |
| stage 3 (input d2) |  735 |     29.9 |        37.2 |
| stage 4 (input d3) | 1076 |     37.3 |        43.8 |

## Chain-only MAE on validation (hours)

|                                            |   booked |   departed |   at_port |   discharged |   customs_cleared |   overall |
|:-------------------------------------------|---------:|-----------:|----------:|-------------:|------------------:|----------:|
| teacher-forced (default)                   |    29.81 |      28.76 |     19.42 |        13.67 |              4.86 |     19.3  |
| out-of-fold rolled-forward + observed flag |    29.94 |      29.16 |     19.67 |        15.94 |              5.09 |     19.96 |
