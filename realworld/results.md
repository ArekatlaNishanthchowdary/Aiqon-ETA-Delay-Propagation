# Real-data validation: AIS port-arrival ETA (Danish Maritime Authority open AIS)

## Descriptive statistics (R = 5 km, label cap 48 h)

rows 32042, vessel-port clusters 1094, ports 27, row times 2026-09-13 -> 2026-10-05; share of rows whose vessel arrives within the window: 0.82

Declared-ETA label (arrival - ETA capped at the cap; h) by claimed bucket, all rows:

| bucket      |   rows |   arrived |   mean |   median |
|:------------|-------:|----------:|-------:|---------:|
| late (<=0h) |   4260 |      0.55 |  29.68 |    33.17 |
| 0-3h        |   5166 |      0.88 |   7.56 |     0.08 |
| 3-12h       |  12626 |      0.88 |   7.57 |    -0.08 |
| 12-24h      |   7288 |      0.84 |   9.74 |     0.33 |
| 24-48h      |   2702 |      0.76 |  12.7  |     0.68 |

## Test performance (R = 5 km): clusters train/val/test = [670, 171, 337]; rows 15689/2678/6409; test row times from 2026-09-30 10:02

|                                             |   MAE |   median AE |   bias |   MAE late (<=0h) |   MAE 0-3h |   MAE 3-12h |   MAE 12-24h |   MAE 24-48h |
|:--------------------------------------------|------:|------------:|-------:|------------------:|-----------:|------------:|-------------:|-------------:|
| declared ETA as broadcast                   | 11.46 |        1.58 |  10.29 |             26.83 |       7.2  |        7.5  |        10.63 |        19.23 |
| median error by port x horizon (train only) |  7.97 |        1.42 |   1.27 |             12.36 |       5.55 |        6.47 |         8.81 |        11.51 |
| kinematic ETA (distance / speed)            | 14.03 |        3.4  |   8.43 |             19.03 |       6.16 |        9.98 |        18.19 |        31.31 |
| LightGBM                                    |  7.41 |        2.03 |   3.7  |             11.05 |       4.81 |        5.8  |         8.44 |        12.38 |

Paired bootstrap over test clusters, MAE(declared) - MAE(LightGBM): 95% CI [2.75, 5.41] h; MAE(median baseline) - MAE(LightGBM): 95% CI [-0.96, 2.14] h (positive = LightGBM better)

Conformal 80% interval half-width 17.98 h (validation); empirical test coverage 0.85

MAE split by outcome: rows whose vessel arrives within the window: declared 4.61 h vs LightGBM 4.04 h (n=5397); rows that never arrive (label = cap): declared 48.00 h vs LightGBM 25.36 h (n=1012)

LightGBM MAE on vessels seen in training: 4.77 h (rows 3558); unseen vessels: 10.70 h (rows 2851)

By port (test):

|       |   test clusters |   declared MAE |   LightGBM MAE |
|:------|----------------:|---------------:|---------------:|
| DEKEL |              18 |           4.29 |           3.14 |
| DELBC |               2 |          45.87 |          40.61 |
| DEPUT |               2 |           0.97 |           0.32 |
| DERSK |              33 |           4.18 |           3.18 |
| DETRV |              16 |           9.34 |           6.73 |
| DKAAL |               8 |           1.32 |           1.09 |
| DKAAR |              21 |           1.81 |           2.21 |
| DKCPH |              26 |           5.29 |           3.77 |
| DKEBJ |              27 |           2.47 |           2    |
| DKFDH |              10 |           8.7  |           7.63 |
| DKFRC |              10 |          16.38 |          12.8  |
| DKGED |               1 |           3.33 |           1.08 |
| DKGRE |               4 |           2.98 |           2.38 |
| DKHIR |               6 |           6.08 |           2.13 |
| DKKAL |               6 |           0.87 |           9.2  |
| DKKOG |               8 |           2.61 |           2.46 |
| DKNYB |               3 |           1.49 |           0.81 |
| DKODE |               4 |          16.49 |          12.77 |
| DKRDB |               1 |           0.84 |           1.61 |
| DKRNN |               7 |           1.71 |           1.87 |
| DKSKA |              38 |          34.64 |          18.64 |
| DKSVE |               1 |           0.33 |           5.15 |
| DKTHY |               4 |           0.85 |           5.02 |
| SEGOT |              53 |          19.35 |          11.76 |
| SEHEL |               6 |           2.1  |           3.51 |
| SEMMA |              12 |           8.55 |           5.11 |
| SETRG |              10 |           2.53 |           1.81 |
## Descriptive statistics (R = 10 km, label cap 48 h)

rows 30927, vessel-port clusters 1132, ports 27, row times 2026-09-13 -> 2026-10-05; share of rows whose vessel arrives within the window: 0.90

Declared-ETA label (arrival - ETA capped at the cap; h) by claimed bucket, all rows:

| bucket      |   rows |   arrived |   mean |   median |
|:------------|-------:|----------:|-------:|---------:|
| late (<=0h) |   2816 |      0.66 |  25.78 |    23.08 |
| 0-3h        |   4805 |      0.93 |   4.9  |    -0.25 |
| 3-12h       |  12920 |      0.94 |   4.02 |    -0.59 |
| 12-24h      |   7535 |      0.91 |   5.46 |    -0.25 |
| 24-48h      |   2851 |      0.88 |   5.42 |    -0.42 |

## Test performance (R = 10 km): clusters train/val/test = [692, 159, 341]; rows 15154/2452/6186; test row times from 2026-09-30 12:02

|                                             |   MAE |   median AE |   bias |   MAE late (<=0h) |   MAE 0-3h |   MAE 3-12h |   MAE 12-24h |   MAE 24-48h |
|:--------------------------------------------|------:|------------:|-------:|------------------:|-----------:|------------:|-------------:|-------------:|
| declared ETA as broadcast                   |  7.27 |        1.5  |   5.35 |             23.82 |       5.51 |        5    |         6.57 |         8.34 |
| median error by port x horizon (train only) |  6.5  |        1.58 |   3.34 |             14.95 |       5.39 |        4.83 |         6.34 |         9.14 |
| kinematic ETA (distance / speed)            | 10.2  |        2.35 |   4.34 |             16.56 |       4.38 |        7.14 |        13.74 |        19.28 |
| LightGBM                                    |  5.59 |        1.12 |   3.03 |             12.07 |       4.03 |        4.01 |         5.96 |         8.89 |

Paired bootstrap over test clusters, MAE(declared) - MAE(LightGBM): 95% CI [1.11, 2.25] h; MAE(median baseline) - MAE(LightGBM): 95% CI [0.45, 1.37] h (positive = LightGBM better)

Conformal 80% interval half-width 5.07 h (validation); empirical test coverage 0.78

MAE split by outcome: rows whose vessel arrives within the window: declared 3.93 h vs LightGBM 2.87 h (n=5716); rows that never arrive (label = cap): declared 48.00 h vs LightGBM 38.58 h (n=470)

LightGBM MAE on vessels seen in training: 3.08 h (rows 3411); unseen vessels: 8.67 h (rows 2775)

By port (test):

|       |   test clusters |   declared MAE |   LightGBM MAE |
|:------|----------------:|---------------:|---------------:|
| DEKEL |              18 |           3.53 |           2.06 |
| DELBC |               3 |          29.2  |          26.83 |
| DEPUT |               2 |           1.37 |           1.23 |
| DERSK |              32 |           3.24 |           2.6  |
| DETRV |              16 |           6.8  |           3.76 |
| DKAAL |               8 |           1.46 |           1.05 |
| DKAAR |              21 |           1.57 |           1.24 |
| DKCPH |              26 |           1.86 |           1.21 |
| DKEBJ |              27 |           2.6  |           1.64 |
| DKFDH |               9 |           9.68 |           8.17 |
| DKFRC |               9 |           3.03 |           2.21 |
| DKGED |               1 |           3.58 |           0.53 |
| DKGRE |               4 |           3.16 |           2.6  |
| DKHIR |               6 |           5.45 |           1.72 |
| DKKAL |               6 |           0.57 |           2.78 |
| DKKOG |               8 |           2.71 |           2.16 |
| DKNYB |               3 |           1.95 |           0.63 |
| DKODE |               4 |          16.35 |          12.3  |
| DKRDB |               1 |           1.63 |           1.1  |
| DKRNN |               7 |           2.04 |           1.42 |
| DKSKA |              46 |          11.09 |           9.99 |
| DKSVE |               1 |           0.17 |           1.3  |
| DKTHY |               3 |           0.78 |           0.34 |
| SEGOT |              52 |          18.57 |          13.88 |
| SEHEL |               6 |           0.86 |           0.97 |
| SEMMA |              12 |           6.44 |           4.25 |
| SETRG |              10 |           2.02 |           1.15 |
## Descriptive statistics (R = 15 km, label cap 48 h)

rows 30080, vessel-port clusters 1151, ports 27, row times 2026-09-13 -> 2026-10-05; share of rows whose vessel arrives within the window: 0.94

Declared-ETA label (arrival - ETA capped at the cap; h) by claimed bucket, all rows:

| bucket      |   rows |   arrived |   mean |   median |
|:------------|-------:|----------:|-------:|---------:|
| late (<=0h) |   2058 |      0.69 |  24.07 |    19.83 |
| 0-3h        |   4393 |      0.96 |   3.3  |    -0.42 |
| 3-12h       |  12973 |      0.96 |   2.13 |    -0.84 |
| 12-24h      |   7694 |      0.95 |   2.84 |    -0.83 |
| 24-48h      |   2962 |      0.93 |   1.98 |    -1.15 |

## Test performance (R = 15 km): clusters train/val/test = [705, 157, 346]; rows 14723/2337/6016; test row times from 2026-09-30 12:04

|                                             |   MAE |   median AE |   bias |   MAE late (<=0h) |   MAE 0-3h |   MAE 3-12h |   MAE 12-24h |   MAE 24-48h |
|:--------------------------------------------|------:|------------:|-------:|------------------:|-----------:|------------:|-------------:|-------------:|
| declared ETA as broadcast                   |  5.97 |        1.58 |   3.48 |             21.46 |       4.13 |        4.11 |         5.56 |         7.74 |
| median error by port x horizon (train only) |  5.32 |        1.33 |   2.55 |             14.34 |       3.98 |        3.82 |         5.36 |         7.9  |
| kinematic ETA (distance / speed)            |  8.81 |        1.92 |   2.91 |             14.9  |       2.88 |        5.86 |        12.33 |        17.96 |
| LightGBM                                    |  4.26 |        0.91 |   2.38 |             10.01 |       2.57 |        2.95 |         4.74 |         7.61 |

Paired bootstrap over test clusters, MAE(declared) - MAE(LightGBM): 95% CI [1.28, 2.17] h; MAE(median baseline) - MAE(LightGBM): 95% CI [0.76, 1.41] h (positive = LightGBM better)

Conformal 80% interval half-width 3.12 h (validation); empirical test coverage 0.76

MAE split by outcome: rows whose vessel arrives within the window: declared 3.87 h vs LightGBM 2.58 h (n=5729); rows that never arrive (label = cap): declared 48.00 h vs LightGBM 37.76 h (n=287)

LightGBM MAE on vessels seen in training: 2.41 h (rows 3296); unseen vessels: 6.51 h (rows 2720)

By port (test):

|       |   test clusters |   declared MAE |   LightGBM MAE |
|:------|----------------:|---------------:|---------------:|
| DEKEL |              17 |           3.48 |           1.83 |
| DELBC |               3 |           1.4  |           2.05 |
| DEPUT |               2 |           1.72 |           0.35 |
| DERSK |              31 |           2.34 |           1.45 |
| DETRV |              18 |           5.51 |           2.82 |
| DKAAL |               8 |           1.67 |           1.02 |
| DKAAR |              21 |           1.89 |           1.28 |
| DKCPH |              26 |           1.97 |           1.13 |
| DKEBJ |              27 |           2.56 |           1.48 |
| DKFDH |               9 |          10.06 |           8.34 |
| DKFRC |               9 |           3.18 |           2.24 |
| DKGED |               1 |           3.75 |           0.45 |
| DKGRE |               4 |           3.4  |           2.55 |
| DKHIR |               6 |           5.57 |           1.66 |
| DKKAL |               6 |           0.62 |           0.72 |
| DKKOG |               8 |           2.67 |           2.24 |
| DKNYB |               3 |           2.35 |           0.6  |
| DKODE |               4 |          13.53 |          12.49 |
| DKRDB |               1 |           1.97 |           0.85 |
| DKRNN |               7 |           2.31 |           1.23 |
| DKSKA |              50 |           4.66 |           3.58 |
| DKSVE |               1 |           0.59 |           0.93 |
| DKTHY |               3 |           1.08 |           0.3  |
| SEGOT |              52 |          18.63 |          14.02 |
| SEHEL |               7 |           0.86 |           1.09 |
| SEMMA |              12 |           5.33 |           4.03 |
| SETRG |              10 |           2.19 |           0.9  |
