"""Explanations: stage-level counterfactual attribution + feature-level drivers + plain-English report."""
import numpy as np, pandas as pd
from .features import stage_row

STAGES = ["origin departure", "ocean leg", "port berth & terminal dwell", "customs clearance", "inland delivery"]
LABEL = {
    "dprev": "delay carried over from the previous stage", "cong": "port congestion",
    "cong_x_ineff": "congestion x low terminal efficiency", "cong_sq_ineff": "berth queue (congestion x inefficiency)",
    "over_slack": "arrival outside the berth window", "dprev_x_cong": "late arrival into a congested port",
    "wind_dep": "weather at origin", "wind_arr": "weather at destination", "transit_h": "long route / transit time",
    "transship": "transshipment hop", "exp_open_wait": "customs closed (weekend / after hours)",
    "dis_wd": "day of week at discharge", "dis_hr": "time of day at discharge", "customs_risk": "high-risk cargo (inspection odds)",
    "docs_flag": "documentation issue", "exp_cut_wait": "missed truck dispatch cut-off", "cus_wd": "weekday of clearance",
    "cus_hr": "time of day at clearance", "eff": "terminal efficiency", "dest": "destination port", "origin": "origin port",
}


def stage_attribution(chain, pairs):
    """Where the hours were lost: per-stage increment (realised if the stage is done, expected otherwise) minus the typical
    increment. Sums to the shipment's delay beyond a typical journey. Returns (attr[n,5], predicted final delay)."""
    d, incs = chain.predict(pairs, return_stage=True)
    ref = np.array([chain.ref[j] for j in range(5)])
    return incs - ref, d


def whatif_attribution(chain, pairs):
    """Root-cause view: prediction minus prediction with stage j set to its typical increment and every downstream stage
    re-rolled, so delay that a stage *caused* downstream (e.g. late arrival -> missed berth window) is credited to it."""
    base = chain.predict(pairs)
    return np.column_stack([base - chain.predict(pairs, cf={j: chain.ref[j]}) for j in range(5)])


def top_driver(chain, pair, j, incs=None):
    """Largest positive TreeSHAP feature (LightGBM pred_contrib) of stage model g_j, evaluated at the delay the shipment
    is predicted to carry into stage j (realised if already observed, rolled forward otherwise)."""
    r, k = pair
    if incs is None:
        incs = chain.predict([pair], return_stage=True)[1][0]
    dprev = float(incs[:j].sum())
    key = chain.key(j, k)
    x = pd.DataFrame([stage_row(j, r, dprev, k)])[chain.cols[key]]
    c = chain.g[key].predict(x, pred_contrib=True)[0][:-1]
    i = int(np.argmax(c))
    return chain.cols[key][i], float(c[i])


def explain(chain, pair, point, lo, hi, p_delay, eta_sched):
    """Human-readable report for one shipment at one checkpoint."""
    attr, _ = stage_attribution(chain, [pair])
    incs = chain.predict([pair], return_stage=True)[1][0]
    wi = whatif_attribution(chain, [pair])[0]
    a = attr[0]; k = pair[1]
    order = np.argsort(-np.abs(a))
    parts = []
    for j in order[:3]:
        if abs(a[j]) < 2:
            continue
        when = "already incurred" if j <= k - 1 else "expected"
        if a[j] < 0:
            parts.append(f"{STAGES[j]} {a[j]:+.0f}h vs typical ({when}; running faster than usual)")
            continue
        f = top_driver(chain, pair, j, incs)[0] if j >= 1 else None
        drv = LABEL.get(f, f) if f else "gate-in / loading delay"
        parts.append(f"{STAGES[j]} {a[j]:+.0f}h vs typical ({when}; main driver: {drv})")
    jj = int(np.argmax(wi))
    gain = min(wi[jj], max(point, 0.0))                # a fix cannot remove more delay than there is
    if gain >= 5:
        parts.append(f"what-if (indicative): if {STAGES[jj]} had run normally the final delay would be ~{gain:.0f}h smaller (incl. knock-on effects)")
    eta = eta_sched + pd.Timedelta(hours=float(point))
    txt = (f"Predicted ETA {eta:%Y-%m-%d %H:%M} ({point:+.0f}h vs schedule, 80% range {lo:+.0f}h to {hi:+.0f}h); "
           f"P(delay > 48h) = {p_delay:.0%}. Biggest contributors: " + ("; ".join(parts) if parts else "none significant") + ".")
    return dict(eta=str(eta), delay_h=float(point), lo_h=float(lo), hi_h=float(hi), p_delay=float(p_delay),
                stage_attribution_h={STAGES[j]: float(a[j]) for j in range(5)},
                whatif_h={STAGES[j]: float(wi[j]) for j in range(5)}, text=txt)
