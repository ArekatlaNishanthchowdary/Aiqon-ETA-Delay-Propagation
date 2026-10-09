"""Simulate -> train -> evaluate -> figures + results.md + run_metadata.json.  Usage: python scripts/run_all.py
Previous results.md / figure are archived (not overwritten) under report/archive/."""
import sys, pickle, pathlib, json, time, platform, shutil
from importlib.metadata import version
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aiqon.simulate import simulate, THRESH as SIM_THRESH
from aiqon.evaluate import load, run, HOLDOUT_ROUTES
from aiqon.features import CKPT_NAMES
from aiqon.explain import stage_attribution
from aiqon.models import SEED, P

N, SEEDS, SUBSET = 2000, [0, 1, 2], 300
for d in ("data", "report", "models", "report/archive"):
    (root / d).mkdir(exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
for f in ("results.md", "fig_main.png", "run_metadata.json"):                 # archive instead of silently overwriting
    if (root / "report" / f).exists():
        shutil.copy(root / "report" / f, root / "report" / "archive" / f"{stamp}_{f}")

simulate(N, SEEDS[0]).to_csv(root / "data" / "shipments.csv", index=False)
df = load(root / "data" / "shipments.csv")
small = df.sample(SUBSET, random_state=1).sort_values("sched_departure").reset_index(drop=True)
md, main, t0 = [], None, time.time()
for name, d, mode in [("A. N=2000, temporal split (train past, test future)", df, "temporal"),
                      (f"B. N={SUBSET} subset (assignment size), temporal split", small, "temporal"),
                      (f"C. N=2000, route hold-out ({' and '.join(HOLDOUT_ROUTES)} unseen)", df, "route")]:
    r = run(d, mode, with_ablation=name.startswith("A."))
    sz = r["sizes"]
    md += [f"\n## {name}\n", f"shipments: train {sz['train']}, calibration {sz['calibration']}, test {sz['test']} "
           f"({sz['test_rows']} checkpoint rows)\n", "### ETA error, MAE in hours (lower is better)\n", r["MAE_table"].round(1).to_markdown(),
           "\nOverall MAE 95% paired-bootstrap CI (500 resamples of test shipments): " + "; ".join(f"{k} {np.round(v, 1)}" for k, v in r["MAE_ci"].items()) + "\n",
           "### 80% interval coverage / mean width (h) / delay AUC by checkpoint\n",
           pd.DataFrame({"conformal coverage": r["coverage_by_ckpt"], "width_h": r["width_by_ckpt"],
                         "raw quantile-GBM coverage": r["raw_quantile_coverage_by_ckpt"], "AUC(delay>48h)": r["prob_by_ckpt_AUC"]},
                        index=CKPT_NAMES).round(2).to_markdown(),
           "\n### Delay probability (all checkpoints pooled)\n", pd.DataFrame(r["prob"]).T.round(3).to_markdown(),
           "\n### Attribution fidelity vs simulator ground truth (stage level)\n", r["attribution_fidelity"].round(2).to_markdown()]
    print(f"done: {name}  ({time.time() - t0:.0f}s)", flush=True)
    if name.startswith("A."):
        main = r
        with open(root / "models" / "predictor.pkl", "wb") as fh:
            pickle.dump(r["model"], fh)

# ---- D. replication over independent simulated worlds (new seed => new congestion paths, delays, routes' timing)
rows = []
for sd in SEEDS:
    d = df if sd == SEEDS[0] else simulate(N, sd).pipe(lambda x: x.sort_values("sched_departure").reset_index(drop=True))
    r = main if sd == SEEDS[0] else run(d, "temporal", with_ablation=False)
    t = r["MAE_table"]
    rows.append(dict(sim_seed=sd, blend_MAE=t.loc["blend (ours)", "overall"], best_baseline_MAE=t.loc["carry-forward + median remaining", "overall"],
                     direct_MAE=t.loc["direct GBM", "overall"], chain_MAE=t.loc["chain (propagation) model", "overall"],
                     mean_coverage=float(np.mean(r["coverage_by_ckpt"])), AUC=r["prob"]["final (avg)"]["AUC"],
                     ECE=r["prob"]["final (avg)"]["ECE"], attr_top1_after_customs=r["attribution_fidelity"].loc["customs_cleared", "top1_agree"]))
rep = pd.DataFrame(rows)
md += ["\n## D. Replication over 3 independent simulated worlds (N=2000 each, temporal split)\n", rep.round(3).to_markdown(index=False),
       "\nmean +- sd: " + ", ".join(f"{c} {rep[c].mean():.2f}+-{rep[c].std():.2f}" for c in rep.columns[1:])]
(root / "report" / "results.md").write_text("\n".join(md), encoding="utf-8")
meta = dict(timestamp=stamp, python=platform.python_version(), seeds=dict(simulator=SEEDS, model=SEED, subset_sample=1, bootstrap=0),
            data=dict(n_shipments=N, delay_threshold_h=SIM_THRESH, split="temporal 60/15/25 by scheduled departure; route hold-out = " + ", ".join(HOLDOUT_ROUTES)),
            model=dict(lightgbm_params=P, calibration="half A: blend weights; half B: conformal + isotonic", interval="split-conformal 80%"),
            packages={p: version(p) for p in ("numpy", "pandas", "scikit-learn", "lightgbm", "scipy", "matplotlib")},
            runtime_s=round(time.time() - t0))
(root / "report" / "run_metadata.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
with open(root / "models" / "metadata.json", "w") as fh:
    json.dump(meta["packages"], fh)

# ---- figure: error curve + example stage attribution vs ground truth
tab = main["MAE_table"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for n in ["schedule (delay=0)", "carry-forward last delay", "carry-forward + median remaining", "direct GBM",
          "chain (propagation) model", "blend (ours)"]:
    ax[0].plot(CKPT_NAMES, tab.loc[n, CKPT_NAMES], marker="o", lw=3 if "ours" in n else 1.3, label=n)
ax[0].set_ylabel("ETA MAE (h)"); ax[0].set_title("ETA error shrinks as the journey progresses (synthetic data)")
ax[0].legend(fontsize=7); ax[0].tick_params(axis="x", rotation=20)
m, Lte, out = main["model"], main["Lte"], main["out"]
w = df.set_index("shipment_id"); k = 2
sub = Lte[Lte.ckpt == k]; sid = sub.shipment_id.iloc[int(np.argmax(out.loc[sub.index, "delay_h"].values))]
attr, _ = stage_attribution(m.chain, [(w.loc[sid], k)])
G = [f"gt_stage{j}" for j in range(5)]
gt = w.loc[sid, G].values.astype(float) - main["splits"][0][G].median().values   # excess over a typical shipment
x = np.arange(5); ax[1].bar(x - .2, attr[0], .4, label="model attribution"); ax[1].bar(x + .2, gt, .4, label="simulator ground truth (excess over typical)")
ax[1].set_xticks(x); ax[1].set_xticklabels(["departure", "ocean", "port", "customs", "inland"]); ax[1].set_ylabel("hours of final delay")
ax[1].set_title(f"Stage attribution, {sid} (at port arrival)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(root / "report" / "fig_main.png", dpi=150)
print("\n".join(md))
