# Chain model: effect of the checkpoint-aware training fix (chain-only MAE, hours)

|                                |   booked |   departed |   at_port |   discharged |   customs_cleared |   overall |
|:-------------------------------|---------:|-----------:|----------:|-------------:|------------------:|----------:|
| old (train as if checkpoint 4) |    22.99 |      26.26 |     17.22 |        12.09 |              4.27 |     16.57 |
| new (per-regime models)        |    27.06 |      26.72 |     17.22 |        12.09 |              4.27 |     17.47 |
