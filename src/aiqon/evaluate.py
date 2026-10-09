"""Evaluation harness: baselines, temporal + route-holdout splits, bootstrap CIs, ablations, calibration, attribution fidelity."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, brier_score_loss
from .features import FEATURES, CKPT_NAMES, build_long, validate
from .models import Predictor, Direct, THRESH
from .explain import stage_attribution, whatif_attribution

HOLDOUT_ROUTES = ["Nhava Sheva>Hamburg", "Chennai>Los Angeles"]
ABLATIONS = {"no interaction/calendar feats": ["cong_x_ineff", "d1_x_cong", "over_slack", "exp_open_wait", "exp_cut_wait",
                                               "dis_wd", "dis_hr", "cus_wd", "cus_hr"],
             "no weather": ["wind_dep", "wind_arr"], "no missingness info": ["n_missing", "last_known"]}


def load(path="data/shipments.csv"):
    cols = pd.read_csv(path, nrows=1).columns
    df = pd.read_csv(path, parse_dates=[c for c in cols if c.startswith(("sched_", "act_"))])
    return validate(df.sort_values("sched_departure").reset_index(drop=True))


def split(df, mode="temporal", seed=0):
    n = len(df)
    if mode == "temporal":                       # train on the past, test on the future
        a, b = int(.6 * n), int(.75 * n)
        return df.iloc[:a], df.iloc[a:b], df.iloc[b:]
    te = df[df.route.isin(HOLDOUT_ROUTES)]       # route hold-out: whole routes unseen in training
    rest = df[~df.route.isin(HOLDOUT_ROUTES)].sample(frac=1, random_state=seed)
    c = int(.2 * len(rest))
    return rest.iloc[c:], rest.iloc[:c], te


def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any())


def baselines(Ltr, Lte):
    rem = (Ltr.y - Ltr.last_delay).groupby(Ltr.ckpt).median()
    return {"schedule (delay=0)": np.zeros(len(Lte)),
            "route-median delay": Lte.route.map(Ltr.groupby("route").y.median()).fillna(Ltr.y.median()).values,
            "carry-forward last delay": Lte.last_delay.values,
            "carry-forward + median remaining": Lte.last_delay.values + Lte.ckpt.map(rem).values}


def run(df, mode="temporal", seed=0, with_ablation=True):
    tr, ca, te = split(df, mode, seed)
    L = build_long(df); sp = lambda d: L[L.shipment_id.isin(d.shipment_id)]
    Ltr, Lca, Lte = sp(tr), sp(ca), sp(te)
    m = Predictor().fit(tr, Ltr, ca, Lca)
    out = m.predict(Lte, df)
    y, ck = Lte.y.values, Lte.ckpt.values
    res = {"n_test_shipments": len(te)}
    preds = baselines(Ltr, Lte) | {"direct GBM": m.direct.predict(Lte)["q50"],
                                    "chain (propagation) model": m._chain(Lte, df), "blend (ours)": out.delay_h.values}
    if with_ablation:
        for name, drop in ABLATIONS.items():
            d = Direct([f for f in FEATURES if f not in drop]).fit(Ltr)
            preds[f"direct GBM, {name}"] = d.predict(Lte)["q50"]
    res["MAE_table"] = pd.DataFrame(
        {n: [np.abs(y - p)[ck == k].mean() for k in range(5)] + [np.abs(y - p).mean()] for n, p in preds.items()},
        index=CKPT_NAMES + ["overall"]).T
    # bootstrap CI of overall MAE, resampling shipments
    ids = Lte.shipment_id.values; u = np.unique(ids); rng = np.random.default_rng(0)
    idx = {s: np.where(ids == s)[0] for s in u}
    ae = {"ours": np.abs(y - preds["blend (ours)"]), "best_baseline": np.abs(y - preds["carry-forward + median remaining"])}
    ci = {k: [] for k in ae} | {"diff (baseline - ours)": []}
    for _ in range(500):                                     # paired bootstrap over shipments (all checkpoints of a shipment move together)
        pick = np.concatenate([idx[s] for s in rng.choice(u, len(u))])
        for k in ae:
            ci[k].append(ae[k][pick].mean())
        ci["diff (baseline - ours)"].append(ae["best_baseline"][pick].mean() - ae["ours"][pick].mean())
    res["MAE_ci"] = {k: np.percentile(v, [2.5, 97.5]) for k, v in ci.items()}
    # intervals + delay probability
    cov = (y >= out.lo.values) & (y <= out.hi.values)
    res["coverage_by_ckpt"] = [cov[ck == k].mean() for k in range(5)]
    dq = m.direct.predict(Lte)
    rawcov = (y >= dq["q10"]) & (y <= dq["q90"])             # un-calibrated quantile-GBM interval, for comparison
    res["raw_quantile_coverage_by_ckpt"] = [rawcov[ck == k].mean() for k in range(5)]
    res["sizes"] = dict(train=len(tr), calibration=len(ca), test=len(te), test_rows=len(Lte))
    res["width_by_ckpt"] = [(out.hi - out.lo).values[ck == k].mean() for k in range(5)]
    yb = (y > THRESH).astype(int)
    praw = m.direct.predict(Lte)["p_raw"]
    probs = {"final (avg)": out.p_delay.values, "conformal-from-ETA": out.p_conf.values,
             "calibrated direct classifier": out.p_direct.values, "raw GBM classifier": praw}
    res["prob"] = {n: dict(AUC=roc_auc_score(yb, p), Brier=brier_score_loss(yb, p), ECE=ece(yb, p)) for n, p in probs.items()}
    res["prob_by_ckpt_AUC"] = [roc_auc_score(yb[ck == k], out.p_delay.values[ck == k]) for k in range(5)]
    # attribution fidelity vs simulator ground truth (stage level); chance top-1 = 1/5
    fid, w = {}, df.set_index("shipment_id")
    G = [f"gt_stage{j}" for j in range(5)]
    typical = tr[G].median().values          # ground truth is "excess over a typical shipment" (matches the model's reference)
    for k in (0, 1, 2, 3, 4):
        sub = Lte[Lte.ckpt == k]
        pairs = [(w.loc[s], k) for s in sub.shipment_id]
        attr, _ = stage_attribution(m.chain, pairs)
        gt = w.loc[sub.shipment_id, G].values - typical
        wi = whatif_attribution(m.chain, pairs)
        fid[CKPT_NAMES[k]] = dict(spearman=np.nanmean([spearmanr(a, g)[0] for a, g in zip(attr, gt)]),
                                  spearman_whatif=np.nanmean([spearmanr(a, g)[0] for a, g in zip(wi, gt)]),
                                  top1_agree=np.mean(attr.argmax(1) == gt.argmax(1)),
                                  top1_if_always_port=np.mean(gt.argmax(1) == 2), top1_if_always_customs=np.mean(gt.argmax(1) == 3), mean_abs_err_h=np.abs(attr - gt).mean())
    res["attribution_fidelity"] = pd.DataFrame(fid).T
    res.update(model=m, splits=(tr, ca, te), out=out, Lte=Lte)
    return res
