# Chain exposure bias: experiments on the validation split only

## Noise of the carry-in delay at inference (training saw exact values)

|                       |   rows |   rmse_of_carry_in_h |   sd_of_true_delay_h |
|:----------------------|-------:|---------------------:|---------------------:|
| stage 1 (carry-in d0) |    300 |                  6   |                  5.9 |
| stage 2 (carry-in d1) |    580 |                 11.4 |                 12.6 |
| stage 3 (carry-in d2) |    735 |                 29.9 |                 37.2 |
| stage 4 (carry-in d3) |   1076 |                 37.3 |                 43.8 |

## Chain-only MAE (hours), all validation rows

|                                          |   booked |   departed |   at_port |   discharged |   customs_cleared |   overall |
|:-----------------------------------------|---------:|-----------:|----------:|-------------:|------------------:|----------:|
| A teacher forcing                        |    29.81 |      28.76 |     19.42 |        13.67 |              4.86 |     19.3  |
| B out-of-fold carry-in, increment target |    29.94 |      29.16 |     19.67 |        15.94 |              5.09 |     19.96 |
| C sequential residual chain              |    28.37 |      30.62 |     21.86 |        16.24 |              5.79 |     20.57 |

## Combined with the direct model: MAE (hours) on validation half B (combiner fitted on half A)

|                                                    |   booked |   departed |   at_port |   discharged |   customs_cleared |   overall |
|:---------------------------------------------------|---------:|-----------:|----------:|-------------:|------------------:|----------:|
| direct model alone                                 |    27.12 |      27.3  |     19.97 |        14.05 |              7.65 |     19.22 |
| A teacher forcing alone                            |    28.05 |      27.49 |     18.57 |        12.56 |              4.04 |     18.14 |
| weight blend: direct + A teacher forcing           |    26.61 |      26.84 |     17.87 |        11.98 |              4.08 |     17.48 |
| C sequential residual chain alone                  |    26.7  |      28.97 |     21.56 |        15.7  |              5.25 |     19.64 |
| weight blend: direct + C sequential residual chain |    26.45 |      27.3  |     20.03 |        14.02 |              5.37 |     18.63 |
| stacked: direct features + C as input              |    26.72 |      28    |     19.36 |        14.49 |              5.53 |     18.82 |
