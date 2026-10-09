"""Chronological validation of ETA correction on real AIS data.  Usage: python realworld/validate.py [R_km ...]

Rows are (vessel, hour) states chosen without looking at the outcome (see build_dataset.py); the label is
y = min(arrival - declared ETA, CAP_H) with y = CAP_H for vessels that never arrive within the window. A method predicts y;
corrected ETA = declared ETA + prediction (clipped to +-CAP_H).
Split: by row time t into 60% / 20% / 20%. A row's label looks ahead to its declared ETA + CAP_H, so train/validation rows
whose window crosses into the next period are purged; no label is determined by events inside a later split.
Fixed hyper-parameters; validation only calibrates the conformal interval; test is evaluated once. Bootstrap clusters are
(vessel, port) pairs, since rows of one vessel are strongly dependent.
Methods (information available at t only): declared ETA; median error by port x horizon bucket (train only); kinematic ETA
(remaining distance / speed); LightGBM (L1). Metrics are bucketed by the CLAIMED horizon. No performance table is produced with
fewer than MIN_TEST_CLUSTERS test clusters."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import build, load_days, CAP_H

OUT = Path(__file__).resolve().parent / "results.md"
MIN_TEST_CLUSTERS = 100
PURGE = pd.Timedelta(hours=CAP_H)
EDGES, LABELS = [-12, 0, 3, 12, 24, 48.001], ["late (<=0h)", "0-3h", "3-12h", "12-24h", "24-48h"]
NUM = ["claimed_h", "rem_km", "sog", "kin_h", "kin_gap_h", "eta_age_h", "n_prev", "rev_total_h", "since_inside_h", "eta_hour", "eta_wd",
       "length", "draught"]
CATS = ["port", "ship_type", "navstat"]
P = dict(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=30, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, verbose=-1, random_state=0, deterministic=True, force_row_wise=True, n_jobs=1)


def split_by_time(d):
    """Train | validation | test by row time (60/20/20). A row is purged from train/validation if its own label window
    (declared ETA + cap) reaches into the next period, so no label is determined by events inside a later split."""
    q1, q2 = d.t.quantile(0.6), d.t.quantile(0.8)
    end = d.eta + PURGE
    return d[end <= q1], d[(d.t > q1) & (end <= q2)], d[d.t > q2]


def design(tr, *others):
    maps = {c: {v: i for i, v in enumerate(sorted(tr[c].astype(str).unique()))} for c in CATS}   # fitted on train only
    def f(x):
        X = x[NUM].astype(float).copy()
        for c in CATS:
            X[c] = x[c].astype(str).map(maps[c]).fillna(-1).astype(int)
        return X
    return [f(tr)] + [f(o) for o in others]


def mu(a):
    return f"{a.mean():.2f}" if len(a) else "n/a"


def bucket(x):
    return pd.cut(x.claimed_h, EDGES, labels=LABELS, right=True)


def table(te, preds):
    b, rows = bucket(te), {}
    for name, p in preds.items():
        e = te.err_h.values - p
        r = {"MAE": np.abs(e).mean(), "median AE": np.median(np.abs(e)), "bias": e.mean()}
        for lab in LABELS:
            m = (b == lab).values
            r[f"MAE {lab}"] = np.abs(e[m]).mean() if m.any() else np.nan
        rows[name] = r
    return pd.DataFrame(rows).T.round(2)


def describe(d, R):
    return [f"## Descriptive statistics (R = {R:g} km, label cap {CAP_H:g} h)\n",
            f"rows {len(d)}, vessel-port clusters {d.cluster.nunique()}, ports {d.port.nunique()}, row times {d.t.min():%Y-%m-%d} -> {d.t.max():%Y-%m-%d}; "
            f"share of rows whose vessel arrives within the window: {d.arrived.mean():.2f}",
            "\nDeclared-ETA label (arrival - ETA capped at the cap; h) by claimed bucket, all rows:\n",
            d.assign(bucket=bucket(d)).groupby("bucket", observed=True).agg(rows=("err_h", "size"), arrived=("arrived", "mean"),
                                                                           mean=("err_h", "mean"), median=("err_h", "median")).round(2).to_markdown()]


def evaluate(d, R, seed=0):
    out = describe(d, R)
    tr, va, te = split_by_time(d)
    nc = [x.cluster.nunique() for x in (tr, va, te)]
    if nc[2] < MIN_TEST_CLUSTERS or min(len(tr), len(va)) < 200:
        return out + [f"\n**No performance table: test clusters {nc[2]} (< {MIN_TEST_CLUSTERS}) or too few train/validation rows "
                      f"(rows {len(tr)}/{len(va)}/{len(te)}).** Add more AIS days."]
    Xtr, Xva, Xte = design(tr, va, te)
    m = lgb.LGBMRegressor(objective="l1", **P).fit(Xtr, tr.err_h)
    med = tr.assign(b=bucket(tr)).groupby(["port", "b"], observed=True).err_h.median(); gm = tr.err_h.median()
    pm = np.array([med.get((p, b), gm) for p, b in zip(te.port, bucket(te))])
    kin = np.clip(np.nan_to_num(te.kin_gap_h.values), -CAP_H, CAP_H)
    pl = np.clip(m.predict(Xte), -CAP_H, CAP_H)
    preds = {"declared ETA as broadcast": np.zeros(len(te)), "median error by port x horizon (train only)": pm,
             "kinematic ETA (distance / speed)": kin, "LightGBM": pl}
    q = np.quantile(np.abs(va.err_h.values - np.clip(m.predict(Xva), -CAP_H, CAP_H)), 0.8)
    e_m, e_0 = np.abs(te.err_h.values - pl), np.abs(te.err_h.values)
    cl = te.cluster.values; u = np.unique(cl); pos = {c: np.where(cl == c)[0] for c in u}; r = np.random.default_rng(seed)
    e_b = np.abs(te.err_h.values - pm)
    diffs, diffs_b = [], []
    for _ in range(500):                                               # paired bootstrap over (vessel, port) clusters
        i = np.concatenate([pos[c] for c in r.choice(u, len(u))])
        diffs.append(e_0[i].mean() - e_m[i].mean()); diffs_b.append(e_b[i].mean() - e_m[i].mean())
    ci, ci_b = np.percentile(diffs, [2.5, 97.5]), np.percentile(diffs_b, [2.5, 97.5])
    seen = te.MMSI.isin(set(tr.MMSI)).values
    bp = pd.DataFrame({p: {"test clusters": te[te.port == p].cluster.nunique(), "declared MAE": e_0[(te.port == p).values].mean(),
                           "LightGBM MAE": e_m[(te.port == p).values].mean()} for p in sorted(te.port.unique())}).T.round(2)
    arr = te.arrived.values == 1
    return out + [f"\n## Test performance (R = {R:g} km): clusters train/val/test = {nc}; rows {len(tr)}/{len(va)}/{len(te)}; "
                  f"test row times from {te.t.min():%Y-%m-%d %H:%M}\n", table(te, preds).to_markdown(),
                  f"\nPaired bootstrap over test clusters, MAE(declared) - MAE(LightGBM): 95% CI [{ci[0]:.2f}, {ci[1]:.2f}] h; "
                  f"MAE(median baseline) - MAE(LightGBM): 95% CI [{ci_b[0]:.2f}, {ci_b[1]:.2f}] h (positive = LightGBM better)",
                  f"\nConformal 80% interval half-width {q:.2f} h (validation); empirical test coverage {(e_m <= q).mean():.2f}",
                  f"\nMAE split by outcome: rows whose vessel arrives within the window: declared {mu(e_0[arr])} h vs LightGBM {mu(e_m[arr])} h "
                  f"(n={int(arr.sum())}); rows that never arrive (label = cap): declared {mu(e_0[~arr])} h vs LightGBM {mu(e_m[~arr])} h (n={int((~arr).sum())})",
                  f"\nLightGBM MAE on vessels seen in training: {(f'{e_m[seen].mean():.2f}' if seen.any() else 'n/a')} h (rows {int(seen.sum())}); unseen vessels: "
                  f"{(f'{e_m[~seen].mean():.2f}' if (~seen).any() else 'n/a')} h (rows {int((~seen).sum())})",
                  "\nBy port (test):\n", bp.to_markdown()]


if __name__ == "__main__":
    radii = [float(x) for x in sys.argv[1:]] or [10.0]
    ais, res = load_days(), ["# Real-data validation: AIS port-arrival ETA (Danish Maritime Authority open AIS)\n"]
    for R in radii:
        d = build(ais, R)
        res += evaluate(d, R) if len(d) else [f"## R = {R:g} km\nno rows found"]
    text = "\n".join(res) + "\n"
    OUT.write_text(text, encoding="utf-8"); print(text)
