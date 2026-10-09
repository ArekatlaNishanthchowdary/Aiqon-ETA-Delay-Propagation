"""Data contract + as-of checkpoint features.

Checkpoints (what is known when the prediction is made):
  0 booked | 1 departed | 2 arrived at port | 3 discharged | 4 customs cleared
A row at checkpoint k only uses timestamps/measurements available by then (no leakage).
Target: final_delay_h = actual_delivery - scheduled_delivery (hours). ETA = sched_delivery + predicted delay.
"""
import numpy as np, pandas as pd
from .simulate import next_open, H, BERTH_SLACK

MILESTONES = ["departure", "port_arrival", "discharge", "customs", "delivery"]
CKPT_NAMES = ["booked", "departed", "at_port", "discharged", "customs_cleared"]
ORIGINS = ["Nhava Sheva", "Mundra", "Chennai"]
DESTS = ["Jebel Ali", "Singapore", "Rotterdam", "Hamburg", "Los Angeles"]
REQUIRED = [f"{p}_{m}" for p in ("sched", "act") for m in MILESTONES]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Data contract: required columns/types and monotone scheduled AND actual timelines (known timestamps only,
    missing ones skipped). Raises ValueError listing every problem."""
    miss = [c for c in REQUIRED + ["origin", "dest", "dest_efficiency", "wind_origin_dep", "wind_dest_arr",
                                   "cong_at_departure", "cong_at_arrival", "customs_risk", "docs_flag"] if c not in df]
    if miss:
        raise ValueError(f"missing columns: {miss}")
    problems = []
    for p in ("sched", "act"):
        ts = df[[f"{p}_{m}" for m in MILESTONES]].apply(pd.to_datetime)
        if p == "sched" and ts.isna().any().any():
            problems.append("scheduled timestamps must all be present")
        step = ts.ffill(axis=1).diff(axis=1).iloc[:, 1:]       # each known timestamp vs the previous known one
        bad = int((step < pd.Timedelta(0)).any(axis=1).sum())
        if bad:
            problems.append(f"{bad} rows with non-monotone {p} timeline")
    if problems:
        raise ValueError("; ".join(problems))
    return df


def hrs(a, b):
    return (a - b) / H if pd.notna(a) and pd.notna(b) else np.nan


def delays(r) -> list:
    """Cumulative delay (h) at each milestone vs schedule; NaN if the timestamp is missing."""
    return [hrs(r[f"act_{m}"], r[f"sched_{m}"]) for m in MILESTONES]


def cal_feats(r, d_prev_dis, d_prev_cus):
    """Calendar features from EXPECTED timestamps (actual if known else schedule + best-known delay)."""
    t_dis = r["sched_discharge"] + d_prev_dis * H
    t_cus = r["sched_customs"] + d_prev_cus * H
    return dict(exp_open_wait=hrs(next_open(t_dis), t_dis), dis_wd=t_dis.weekday(), dis_hr=t_dis.hour,
                exp_cut_wait=max(hrs(next_open(t_cus + 2 * H), t_cus) - 2, 0.0) if (t_cus.hour >= 15 or t_cus.weekday() >= 5) else 0.0,
                cus_wd=t_cus.weekday(), cus_hr=t_cus.hour)


def regime(k):
    """Information regime of a checkpoint for exogenous inputs: 0 = booked (nothing observed), 1 = departed
    (origin wind + destination congestion on the departure date), 2 = at port or later (arrival congestion/wind)."""
    return min(k, 2)


def stage_row(j, r, dprev, k, obs=None):
    """Features for stage-model g_j (predicts the increment of milestone j given delay at j-1).
    Only exogenous values observable at checkpoint k are used; unobserved ones are NaN (never a future value)."""
    eff = r["dest_efficiency"]
    ci = r["cong_at_arrival"] if k >= 2 else (r["cong_at_departure"] if k >= 1 else np.nan)
    wa = r["wind_dest_arr"] if k >= 2 else np.nan
    wd = r["wind_origin_dep"] if k >= 1 else np.nan
    # dprev_obs: is the previous milestone's delay observed (exact) at checkpoint k, or a rolled-forward prediction?
    base = dict(dprev=dprev, dprev_obs=float(k >= j) if obs is None else obs, dest=DESTS.index(r["dest"]), origin=ORIGINS.index(r["origin"]), eff=eff)
    if j == 1:
        return {**base, "wind_dep": wd, "wind_arr": wa, "transit_h": hrs(r["sched_port_arrival"], r["sched_departure"]),
                "transship": r["transshipment"]}
    if j == 2:
        return {**base, "cong": ci, "cong_x_ineff": ci * (1 - eff), "cong_sq_ineff": ci ** 2 * (1.3 - eff),
                "wind_arr": wa, "over_slack": float(dprev > BERTH_SLACK), "dprev_x_cong": dprev * ci}
    c = cal_feats(r, dprev, dprev)
    if j == 3:
        return {**base, "exp_open_wait": c["exp_open_wait"], "dis_wd": c["dis_wd"], "dis_hr": c["dis_hr"],
                "customs_risk": r["customs_risk"], "docs_flag": r["docs_flag"], "cong": ci}
    return {**base, "exp_cut_wait": c["exp_cut_wait"], "cus_wd": c["cus_wd"], "cus_hr": c["cus_hr"]}


def build_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (shipment, checkpoint) with only information known at that checkpoint."""
    rows = []
    for _, r in df.iterrows():
        d = delays(r)
        for k in range(5):
            known = [j for j in range(4) if k >= j + 1 and pd.notna(d[j])]
            ls = max(known) if known else -1               # last milestone with a known timestamp
            last = d[ls] if ls >= 0 else 0.0
            sched_last = r[f"sched_{MILESTONES[ls]}"] if ls >= 0 else r["sched_departure"]
            f = dict(shipment_id=r["shipment_id"], route=r["route"], dest=r["dest"], ckpt=k,
                     sched_departure=r["sched_departure"], sched_delivery=r["sched_delivery"],
                     y=r["final_delay_h"], delayed=r["delayed"],
                     last_delay=last, last_known=ls, n_missing=sum(pd.isna(d[j]) for j in range(4) if k >= j + 1),
                     remaining_sched_h=hrs(r["sched_delivery"], sched_last),
                     transit_h=hrs(r["sched_port_arrival"], r["sched_departure"]),
                     dest_eff=r["dest_efficiency"], transship=r["transshipment"], customs_risk=r["customs_risk"],
                     docs_flag=r["docs_flag"],
                     cong_dep=r["cong_at_departure"] if k >= 1 else np.nan,   # departure-day index: not known at booking
                     dest_code=DESTS.index(r["dest"]), origin_code=ORIGINS.index(r["origin"]),
                     dep_month=r["sched_departure"].month, dep_wd=r["sched_departure"].weekday())
            for j in range(4):                              # delay at milestone j is known iff checkpoint k >= j+1
                f[f"d{j}"] = d[j] if k >= j + 1 else np.nan
            for j in (1, 2, 3):                             # stage increments
                a, b = f[f"d{j}"], f[f"d{j-1}"]
                f[f"inc{j}"] = a - b if pd.notna(a) and pd.notna(b) else np.nan
            f["wind_dep"] = r["wind_origin_dep"] if k >= 1 else np.nan
            if k >= 2:
                f.update(cong_arr=r["cong_at_arrival"], wind_arr=r["wind_dest_arr"],
                         cong_x_ineff=r["cong_at_arrival"] * (1 - r["dest_efficiency"]),
                         over_slack=float(f["d1"] > BERTH_SLACK) if pd.notna(f["d1"]) else np.nan,
                         d1_x_cong=f["d1"] * r["cong_at_arrival"] if pd.notna(f["d1"]) else np.nan)
            else:
                f.update(cong_arr=np.nan, wind_arr=np.nan, over_slack=np.nan, d1_x_cong=np.nan,
                         cong_x_ineff=r["cong_at_departure"] * (1 - r["dest_efficiency"]) if k >= 1 else np.nan)
            f.update(cal_feats(r, last, last))
            rows.append(f)
    return pd.DataFrame(rows)


FEATURES = ["ckpt", "last_delay", "last_known", "n_missing", "remaining_sched_h", "transit_h", "dest_eff",
            "transship", "customs_risk", "docs_flag", "cong_dep", "dest_code", "origin_code",
            "d0", "d1", "d2", "d3", "inc1", "inc2", "inc3", "wind_dep", "cong_arr", "wind_arr",
            "cong_x_ineff", "over_slack", "d1_x_cong", "exp_open_wait", "dis_wd", "dis_hr",
            "exp_cut_wait", "cus_wd", "cus_hr"]
