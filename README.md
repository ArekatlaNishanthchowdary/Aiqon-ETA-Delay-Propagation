# Shipment ETA & Delay Propagation

Predicts a shipment's **final delivery ETA**, the **probability it is badly delayed** (> 48 h), and **which stage caused the delay**,
re-predicting at each point of the journey. Built for the Aiqon AI internship take-home.

> **Read this first:** all quantitative results come from a **synthetic** shipment simulator I wrote (only the port wind data is real).
> They demonstrate the method and let me *measure* explanation quality; they are **not** evidence of real-world ETA accuracy.

## What it does

```
booked -> departed -> at port -> discharged -> customs cleared -> delivered
   |          |          |            |              |
   +----------+----------+------------+--------------+--> prediction at every checkpoint, using only what is known then
```

For a shipment at any checkpoint it returns the predicted ETA with an 80% range, P(delay > 48 h), and a plain-English explanation:

```
Predicted ETA 2024-04-29 10:31 (+158h vs schedule, 80% range +130h to +197h); P(delay > 48h) = 100%.
Biggest contributors: port berth & terminal dwell +79h vs typical (expected; main driver: congestion x low terminal efficiency);
inland delivery +48h vs typical (expected; main driver: missed truck dispatch cut-off); ocean leg +11h vs typical (already incurred) ...
```

## How it works

| Layer | What | Why |
|---|---|---|
| Simulator (`simulate.py`) | Causal delay model `d_k = m_k·d_{k-1} + own_k`: slack absorbs part of upstream delay; congestion × terminal inefficiency; missed berth windows; customs office hours; dispatch cut-offs; inspections; real ERA5 wind | Known ground truth for every delay hour, so explanations can be scored |
| Data contract (`features.py`) | Schema + monotone scheduled/actual timelines; per-checkpoint "as-of" features; unobserved values are NaN, never future values | Prevents leakage; tested adversarially |
| Chain model (`models.py`) | One GBM per stage predicts that stage's increment from the previous stage's delay + its drivers, rolled forward to the final delay (one model per information regime) | Explicit delay propagation |
| Direct model | One GBM over all checkpoints | Strong early in the journey |
| Predictor | Per-checkpoint blend; split-conformal 80% intervals; isotonic + conformal delay probability; separate calibration halves | Honest uncertainty |
| Explanations (`explain.py`) | "Where hours were lost" per stage, indicative what-if (counterfactual re-roll), TreeSHAP top driver, text template (no LLM) | Actionable causes |
| Evaluation (`evaluate.py`) | Baselines, temporal split, 300-shipment subset, route hold-out, ablations, calibration, paired bootstrap, attribution fidelity, 3 simulated worlds | See results |

## Headline results (synthetic data, temporal split, 500 test shipments)

| | MAE (h) |
|---|---|
| Schedule as ETA | 52.0 |
| Best baseline (last delay + median remaining) | 24.2 |
| **This system** | **16.0** (95% CI 14.7-17.3) |

Error falls from 23.1 h at booking to 4.2 h after customs. Delay probability AUC 0.89; 80% intervals cover 0.77-0.87 by checkpoint.
Explanation top-1 stage agreement with ground truth: 0.48 at booking, 0.84 after customs (chance 0.20).

**Weak spots:** route hold-out interval coverage drops to 0.62-0.78; the chain model has a known previous-stage-delay train/inference
mismatch (a fix was tested and was not better); the blend's edge over the direct model alone is small (~1 h) and not uniform across
simulated worlds; the what-if view is not validated against true counterfactuals. All tables: `report/results.md`; write-up: `report/report.pdf`.

## Quickstart

```bash
pip install -r requirements.txt        # pinned versions, Python 3.13
python scripts/run_all.py              # simulate -> train -> evaluate (~1 min)
python -m pytest tests -q              # 13 tests
PYTHONPATH=src python -m aiqon.predict examples/at_port_extreme.json    # explained prediction
```

PowerShell: `$env:PYTHONPATH="src"; python -m aiqon.predict examples/at_port_extreme.json`.
The first run downloads real wind data from the free Open-Meteo archive API and caches it in `data/wind_daily.csv`.

`run_all.py` writes `report/results.md`, `report/fig_main.png`, `report/run_metadata.json` (seeds, parameters, package versions) and
`models/predictor.pkl`; earlier outputs are archived in `report/archive/`, never overwritten. Two consecutive runs give identical results.
Optional investigations: `python scripts/chain_fix_effect.py` and `python scripts/chain_dprev_mismatch.py` (chain-model studies, write to `report/`).

**Input format** (`examples/*.json`): scheduled times plus the actual timestamps known so far (`null` for unknown), static fields
(`origin`, `dest`, `dest_efficiency`, `customs_risk`, `docs_flag`, `transshipment`) and `cong_at_departure`; wind and arrival congestion
are used only once they would exist. Incomplete records or non-monotone timelines are rejected with an explicit error.

## Repository layout

```
src/aiqon/    simulate.py  weather.py  features.py  models.py  explain.py  evaluate.py  predict.py
scripts/      run_all.py (full pipeline)   chain_fix_effect.py   chain_dprev_mismatch.py
tests/        test_harness.py  (availability, adversarial leakage, feature consistency, decomposition, missing data,
              invalid timelines, split isolation, calibration, seeds, shapes)
examples/     four sample shipments (booked, at port, missing stage, clean)
report/       report.pdf/.html  results.md  chain_*.md  fig_main.png  run_metadata.json  baseline_pre_audit/ (results before the audit)
data/         shipments.csv (generated)  wind_daily.csv (cached real wind)
```

## Data, assumptions and limits

No public dataset has scheduled-vs-actual times for every stage of a shipment, so the data are simulated. Real inputs: port pairs, liner-like
transit times, ERA5 daily wind at each port. **Assumed** (not measured): terminal-efficiency tiers, congestion dynamics, all effect sizes,
the 48 h threshold, informative missingness. The models partly learn my own simulator rules. Validating on real records (for example
port-call ETA vs actual arrival feeds) would be the next step; it can only validate the port-arrival stage, not customs or delivery.
