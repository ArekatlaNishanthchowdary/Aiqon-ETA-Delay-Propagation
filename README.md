# Shipment ETA & Delay Propagation (Aiqon take-home)

Predicts a shipment's final delivery ETA, the probability it is badly delayed (> 48 h), and **explains which stage/driver
causes the delay**, re-predicting at each journey checkpoint (booked → departed → at port → discharged → customs cleared).

## Run
```bash
pip install -r requirements.txt            # pinned versions (Python 3.13)
python scripts/run_all.py                  # simulate -> train -> evaluate (~1 min). Writes report/results.md, fig_main.png, run_metadata.json,
                                           #   models/predictor.pkl. Previous outputs are archived in report/archive/, never overwritten.
python scripts/chain_fix_effect.py         # controlled old-vs-new chain training comparison -> report/chain_fix_effect.md
python scripts/chain_dprev_mismatch.py     # investigation of the chain's teacher-forcing mismatch (validation only) -> report/chain_dprev_mismatch.md
python -m pytest tests -q                  # 13 tests: availability, leakage (adversarial), feature consistency, decomposition,
                                           #   missing data, invalid timelines, split isolation, calibration, seeds, shapes
PYTHONPATH=src python -m aiqon.predict examples/at_port_extreme.json   # explained prediction for one shipment
```
(Windows PowerShell: `$env:PYTHONPATH="src"; python -m aiqon.predict examples/at_port_extreme.json`.)
First run downloads real wind data from the free Open-Meteo archive API and caches it in `data/wind_daily.csv`.
Input JSON = scheduled times + actual timestamps known so far (`null` for unknown); see `examples/`. Records missing required
fields or with non-monotone timelines are rejected with an explicit error. Seeds, model parameters and package versions of each
run are saved in `report/run_metadata.json`; `report/baseline_pre_audit/` holds the pre-audit results for comparison.

## Layout
| file | role |
|---|---|
| `src/aiqon/simulate.py` | causal simulator: delay propagation `d_k = m_k·d_{k-1} + own_k`, with named drivers + exact ground truth |
| `src/aiqon/weather.py` | real ERA5 daily max wind per port (Open-Meteo), cached |
| `src/aiqon/features.py` | data contract (`validate`), as-of checkpoint features (no leakage), stage features |
| `src/aiqon/models.py` | `Chain` (stage-to-stage propagation model), `Direct` (GBM), `Predictor` (blend + conformal + calibration) |
| `src/aiqon/explain.py` | stage attribution, what-if attribution, TreeSHAP top driver, plain-English text |
| `src/aiqon/evaluate.py` | baselines, temporal & route-holdout splits, bootstrap CI, ablations, calibration, attribution fidelity |
| `src/aiqon/predict.py` | CLI |
| `report/` | `report.pdf`, `results.md` (all tables), `fig_main.png`, `chain_fix_effect.md`, `run_metadata.json` |

## Data
No public dataset has full stage-level journeys (scheduled vs actual per stage, port processing, customs), so we use a
**synthetic simulator, anchored to real data where it exists**: real port pairs and realistic liner transit times, **real
historical wind at each port (ERA5)**, and assumed terminal-efficiency tiers/congestion. The simulator knows the true
cause of every delay hour, which lets us *measure* explanation quality. All simulator parameters other than wind are
assumptions, not measurements; see the report for limits.
