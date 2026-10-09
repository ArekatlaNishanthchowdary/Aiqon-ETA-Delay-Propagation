"""Causal shipment simulator.

Cumulative delay at milestone k (hours vs schedule):  d_k = m_k * d_{k-1} + own_k
  m_k in [0,1]  : share of upstream delay that survives (schedule slack / speed-up recovery)
  own_k         : new delay created at stage k (sum of named drivers)
So the final delay decomposes EXACTLY:  d_final = sum_k own_k * prod_{j>k} m_j   (ground truth for explanations).

Stages: S0 departure | S1 ocean leg | S2 port (berth wait + terminal dwell) | S3 customs | S4 inland delivery
All parameters below are ASSUMED (plausible, loosely anchored to public figures), except wind = real ERA5 data.
"""
from pathlib import Path
import numpy as np, pandas as pd
from .weather import load_wind

# (origin, dest): scheduled ocean transit hours (typical liner schedules incl. 2024 Red Sea diversion), transshipment
ROUTES = {
    ("Nhava Sheva", "Jebel Ali"): (120, 0), ("Mundra", "Jebel Ali"): (96, 0),
    ("Chennai", "Singapore"): (216, 0), ("Nhava Sheva", "Singapore"): (312, 0),
    ("Nhava Sheva", "Rotterdam"): (600, 0), ("Mundra", "Rotterdam"): (576, 0),
    ("Nhava Sheva", "Hamburg"): (624, 1), ("Chennai", "Hamburg"): (672, 1),
    ("Nhava Sheva", "Los Angeles"): (840, 1), ("Chennai", "Los Angeles"): (792, 1),
}
# destination terminal efficiency (0..1, assumed tiers inspired by World Bank CPPI) and base congestion level
DEST = {"Jebel Ali": (0.80, 0.30), "Singapore": (0.90, 0.25), "Rotterdam": (0.75, 0.35),
        "Hamburg": (0.55, 0.45), "Los Angeles": (0.45, 0.55)}
SCHED_DWELL, SCHED_CUSTOMS, SCHED_INLAND, BERTH_SLACK, THRESH = 48.0, 36.0, 24.0, 12.0, 48.0
H = pd.Timedelta(hours=1)
DRIVERS = ["departure_gate_delay", "sea_weather", "sea_noise", "berth_congestion_wait", "terminal_dwell_excess",
           "missed_berth_window", "customs_calendar_wait", "customs_inspection", "customs_docs_error",
           "customs_noise", "inland_cutoff_wait", "inland_noise"]


def _congestion(rng, dates):
    """Shared daily congestion index per destination: AR(1) + seasonal peak (Q4 peak season)."""
    out = {}
    for p, (_, base) in DEST.items():
        x, z = np.zeros(len(dates)), 0.0
        for i in range(len(dates)):
            z = 0.93 * z + rng.normal(0, 0.09)
            x[i] = z
        seas = 0.12 * np.sin(2 * np.pi * (dates.dayofyear - 240) / 365)
        out[p] = pd.Series(np.clip(base + seas + x, 0.02, 1.0), index=dates)
    return pd.DataFrame(out)


def next_open(t):
    """Customs/terminal office hours: Mon-Fri 09:00-17:00. Returns next time work can start."""
    if t.weekday() >= 5:
        t = (t + pd.Timedelta(days=7 - t.weekday())).normalize() + 9 * H
    elif t.hour >= 17:
        t = (t + pd.Timedelta(days=1)).normalize() + 9 * H
        if t.weekday() >= 5:
            t = (t + pd.Timedelta(days=7 - t.weekday())).normalize() + 9 * H
    elif t.hour < 9:
        t = t.normalize() + 9 * H
    return t


def simulate(n=2000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wind = load_wind()
    cong = _congestion(rng, wind.index)
    keys = list(ROUTES)
    rows = []
    for i in range(n):
        o, d = keys[rng.integers(len(keys))]
        transit, trans = ROUTES[(o, d)]
        eff, _ = DEST[d]
        std = pd.Timestamp("2023-01-01") + pd.Timedelta(hours=float(rng.integers(0, 24 * 540)))
        transit *= rng.uniform(0.97, 1.05)
        sta = std + transit * H
        s_dis = sta + SCHED_DWELL * H
        s_cus = s_dis + SCHED_CUSTOMS * H
        s_del = s_cus + SCHED_INLAND * H
        cong_dep, cong_arr = cong.at[std.normalize(), d], cong.at[sta.normalize(), d]
        w_dep, w_arr = wind.at[std.normalize(), o], wind.at[sta.normalize(), d]
        risk = rng.beta(2, 8)                       # importer/commodity risk score (observed at booking)
        docs_issue = rng.random() < 0.10            # documentation problem flagged at booking

        # ---- S0 departure: gate-in / loading delays, worse in bad weather
        c = {"departure_gate_delay": rng.gamma(1.2, 4.0) * (1 + 0.5 * (rng.random() < 0.15)) + 0.4 * max(0, w_dep - 35)}
        d0 = sum(c.values()); comps = [c]
        # ---- S1 ocean leg: weather at both ends (wind > 30 km/h) + noise; vessel recovers part of upstream delay (<=24h)
        c = {"sea_weather": 0.9 * max(0, w_dep - 30) + 1.2 * max(0, w_arr - 30),
             "sea_noise": transit / 100 * rng.exponential(1.0) * (1 + trans) + rng.normal(0, 3)}
        m1 = 1.0 if d0 <= 0 else 1 - min(rng.uniform(0.2, 0.6), 24 / d0)
        d1 = m1 * d0 + sum(c.values()); comps.append(c)
        # ---- S2 port: berth queue + terminal dwell (congestion x inefficiency interaction) + missed berth window
        berth = 72 * cong_arr ** 2 * (1.3 - eff) * rng.lognormal(0, 0.3)
        dwell_x = 36.0 * (1 + 1.5 * cong_arr * (1 - eff)) * rng.lognormal(0, 0.25) - SCHED_DWELL
        m2 = rng.uniform(0.8, 1.0)
        pre = m2 * d1
        missed = rng.uniform(24, 48) if (pre > BERTH_SLACK and rng.random() < 0.5 + 0.4 * cong_arr) else 0.0
        c = {"berth_congestion_wait": berth, "terminal_dwell_excess": dwell_x, "missed_berth_window": missed}
        d2 = pre + sum(c.values()); comps.append(c)
        # ---- S3 customs: office hours (weekend/evening), random inspection, doc errors
        t_dis = s_dis + d2 * H
        wk = (next_open(t_dis) - t_dis) / H
        insp_p = 0.04 + 0.35 * risk
        c = {"customs_calendar_wait": wk,
             "customs_inspection": rng.lognormal(3.6, 0.5) if rng.random() < insp_p else 0.0,
             "customs_docs_error": rng.uniform(12, 36) if (docs_issue and rng.random() < 0.7) else 0.0,
             "customs_noise": rng.normal(0, 3)}
        m3 = rng.uniform(0.85, 1.0)
        d3 = m3 * d2 + sum(c.values()); comps.append(c)
        # ---- S4 inland: truck dispatch cut-off (cleared after 15:00 / weekend -> next working morning), noise
        t_cus = s_cus + d3 * H
        cutoff = (next_open(t_cus + 2 * H) - t_cus) / H - 2 if (t_cus.hour >= 15 or t_cus.weekday() >= 5) else 0.0
        c = {"inland_cutoff_wait": max(cutoff, 0.0), "inland_noise": rng.gamma(1.5, 2.0) - 3}
        m4 = rng.uniform(0.9, 1.0)
        d4 = m4 * d3 + sum(c.values()); comps.append(c)

        ms = [None, m1, m2, m3, m4]
        row = dict(shipment_id=f"SH{i:05d}", origin=o, dest=d, route=f"{o}>{d}", transshipment=trans,
                   sched_departure=std, sched_port_arrival=sta, sched_discharge=s_dis,
                   sched_customs=s_cus, sched_delivery=s_del,
                   act_departure=std + d0 * H, act_port_arrival=sta + d1 * H, act_discharge=s_dis + d2 * H,
                   act_customs=s_cus + d3 * H, act_delivery=s_del + d4 * H,
                   dest_efficiency=eff, cong_at_departure=cong_dep + rng.normal(0, 0.03),
                   cong_at_arrival=cong_arr + rng.normal(0, 0.02), wind_origin_dep=w_dep, wind_dest_arr=w_arr,
                   customs_risk=risk, docs_flag=int(docs_issue), final_delay_h=d4, delayed=int(d4 > THRESH))
        # ground-truth attribution: driver contribution to FINAL delay
        for k, ck in enumerate(comps):
            surv = float(np.prod(ms[k + 1:])) if k < 4 else 1.0
            for name, v in ck.items():
                row[f"gt_{name}"] = v * surv
            row[f"gt_stage{k}"] = sum(ck.values()) * surv
        rows.append(row)
    df = pd.DataFrame(rows)
    # missing intermediate timestamps (worse at low-efficiency ports -> informative missingness)
    for col, base in [("act_port_arrival", 0.05), ("act_discharge", 0.10), ("act_customs", 0.10)]:
        p = base + (0.25 * (1 - df.dest_efficiency) if col == "act_discharge" else 0)
        df.loc[rng.random(len(df)) < p, col] = pd.NaT
    return df


if __name__ == "__main__":
    df = simulate()
    df.to_csv(Path(__file__).resolve().parents[2] / "data" / "shipments.csv", index=False)
    print(df.final_delay_h.describe(), "\ndelayed rate:", df.delayed.mean())
    gt = [c for c in df if c.startswith("gt_stage")]
    print(df[gt].mean().round(1)); print("decomp err", (df[gt].sum(axis=1) - df.final_delay_h).abs().max())
    print(df.isna().mean()[lambda s: s > 0])
