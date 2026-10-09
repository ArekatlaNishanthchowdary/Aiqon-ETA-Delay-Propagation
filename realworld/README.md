# Real-data validation track: AIS port-arrival ETA (separate from the synthetic pipeline)

**Question:** can a model correct the ETA that vessels *declare*, on real arrivals, outside my own simulator?
**Scope:** port-arrival ETA only. It does **not** validate customs, final delivery, the five-milestone propagation model or
end-to-end shipment ETA; no public source links those milestones.

## Result (24 days, 2026-09-13 to 2026-10-06, Danish waters; test = last 20% of row times)

| arrival radius | declared ETA MAE | median baseline | kinematic ETA | LightGBM | LightGBM vs declared (95% CI) | LightGBM vs median baseline (95% CI) |
|---|---|---|---|---|---|---|
| 5 km  | 11.46 h | 7.97 h | 14.03 h | 7.41 h | +2.75 to +5.41 h | **-0.96 to +2.14 h (not significant)** |
| 10 km | 7.27 h | 6.50 h | 10.20 h | 5.59 h | +1.11 to +2.25 h | +0.45 to +1.37 h |
| 15 km | 5.97 h | 5.32 h | 8.81 h | 4.26 h | +1.28 to +2.17 h | +0.76 to +1.41 h |

About 30,000 vessel-hour rows, ~1,100 vessel-port pairs, 27 ports, ~340 test pairs. Full tables: `results.md`.

**How to read it honestly**
* The gain over the declared ETA is real but modest (about 1.5-2 h at 10-15 km) and depends on how "arrival" is defined.
  At 5 km the model is **not** distinguishable from a trivial per-port median correction, and its median error is worse than the
  declared ETA's (2.03 h vs 1.58 h).
* The typical declared ETA is already good: median absolute error ~1.5 h. MAE is driven by a tail of stale ETAs and no-shows.
  On vessels that do arrive the gain is ~1 h (3.93 -> 2.87 h at 10 km); most of the rest comes from recognising stale ETAs.
* A physics-only ETA (distance / speed) is **worse** than the declared ETA at every radius.
* The model is much weaker on vessels never seen in training (8.67 h vs 3.08 h at 10 km): part of the gain is vessel-specific habit.
* 80% conformal intervals cover 0.76-0.85 on test, i.e. only roughly calibrated.

## A first design was wrong, and why it matters
The first version kept only calls that *did* arrive within a fixed look-back window. It showed LightGBM at 12.75 h vs 24.63 h for
the declared ETA. A sensitivity check exposed it as an artifact: the declared error was 8 / 25 / 53 h for 48 / 96 / 168 h windows,
and the "gain" grew with the window, because conditioning on a future arrival tells the model that stale ETAs always resolve inside
the window. That result is retracted. In the current design rows are chosen using only information known at the prediction time,
vessels that never arrive are kept, and the same check is flat (declared 5.5 / 7.3 / 7.0 h for label caps of 24 / 48 / 72 h).
Two further bugs were found and fixed by checking against the data: the Fredericia/Frederikshavn port codes were swapped, and
vessels that sailed out of receiver coverage were being labelled as no-shows.

## Data
Danish Maritime Authority open AIS, daily files `aisdk-YYYY-MM-DD.zip` (~600 MB zipped, ~3 GB CSV, ~17M messages/day). Each position
message carries the vessel's *currently declared* `Destination` and `ETA` (with year), MMSI, ship type, speed. Download the files
yourself and keep them **outside the repo**; they and all derived data are git-ignored.

## Pipeline
```bash
python -I realworld/ais_extract.py C:/path/to/ais_days   # zips -> realworld/data/days/*.csv.gz (idempotent, ~70 s per day)
python realworld/validate.py 5 10 15                     # writes realworld/results.md (three radii = sensitivity to "arrival")
python -m pytest realworld -q                            # 10 tests on synthetic AIS tracks, no data needed
```

## Design
* **Row** = (vessel, hour t): at t the vessel declares a destination that maps to one of 27 ports, has an ETA and a speed, is
  outside the port's R-km circle, the ETA is between 12 h in the past and 48 h ahead, and ETA + 48 h is before the end of the
  data (no right-censoring). Only information known at t, or the declared ETA, decides whether a row exists.
* **Label** = min(arrival - declared ETA, 48 h), arrival = first observed entry into the circle after t (5-minute resolution).
  A vessel that never enters by ETA + 48 h gets 48 h, but only if it is still tracked in the last 12 h of that window;
  otherwise the outcome is unknown and the row is dropped, as are rows whose zone crossing was not observed (gap > 30 min).
* **Features (all as of t):** claimed horizon, remaining distance, speed, kinematic ETA and its gap to the claim, age and
  revisions of the declaration over the last 48 h, hours since the vessel was last inside the zone, navigational status,
  ETA hour/weekday, port, ship type, length, draught. The vessel identifier is not a feature.
* **Split:** by row time 60/20/20; a train/validation row is purged if its own label window (ETA + 48 h) reaches into the next
  period. Fixed hyper-parameters; validation only calibrates the conformal interval; test evaluated once.
* **Uncertainty:** paired bootstrap over (vessel, port) pairs, since a vessel's hourly rows are strongly dependent.
* **Ports:** approximate hand-entered centres, each verified against where vessels declaring that code actually moor (`ports.py`).

## Limitations
* One region, 24 days, one season; mixed ship types (ferries repeat the same call many times).
* "Arrival" is a zone entry, not berthing; crews may mean either. Results move with the radius, as shown above.
* Dropping unknown outcomes removes vessels that leave coverage; if those differ systematically, results are biased.
* Rows for the same vessel in train and test are not label-leaked (purge), but persistent habits make seen vessels easier.
* Port list, radius, the 48 h cap and the 12 h tracking rule are my choices; the cap and radius were varied, the others were not.
