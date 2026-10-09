"""Do the mechanisms hand-coded in the simulator show up in real AIS data?   Usage: python realworld/simulator_check.py

This is a check of MECHANISMS (direction and rough size), not a parameter fit: AIS measures how long a VESSEL stays in a port,
while the simulator's port stage models when CARGO becomes available, and the regions differ (Danish short-sea vs simulated
India->Europe/US deep-sea), so levels are not comparable and nothing is re-fitted from these numbers.
Real quantities (cargo and tanker vessels, 5 km port zones, entry and exit both observed):
  stay_h      time between entering and leaving a port zone
  occupancy   number of cargo/tanker vessels inside that zone in the hour of entry (a congestion proxy)
Simulator counterparts come from data/shipments.csv. Writes realworld/simulator_check.md."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import load_days, hav_km, build
from ports import PORTS

ROOT = Path(__file__).resolve().parents[1]
R, GAP = 5.0, pd.Timedelta(minutes=30)
rng = np.random.default_rng(0)


def visits(ais):
    a = ais[ais["Ship type"].isin(["Cargo", "Tanker"])].sort_values(["MMSI", "ts"]).reset_index(drop=True)
    lat, lon = a.Latitude.values, a.Longitude.values
    best, bd = np.full(len(a), -1), np.full(len(a), 1e9)
    for i, (plat, plon) in enumerate(PORTS.values()):
        d = hav_km(lat, lon, plat, plon); m = d < bd
        bd[m], best[m] = d[m], i
    a["zone"] = np.where(bd <= R, best, -1)
    same = a.MMSI.eq(a.MMSI.shift()) & (a.ts - a.ts.shift() <= GAP)
    a["run"] = (~(same & a.zone.eq(a.zone.shift()))).cumsum()
    prev_out = same & a.zone.shift().eq(-1)                                   # previous message: same vessel, outside, recent
    nxt_same = a.MMSI.eq(a.MMSI.shift(-1)) & (a.ts.shift(-1) - a.ts <= GAP)
    next_out = nxt_same & a.zone.shift(-1).eq(-1)
    g = a[a.zone >= 0].groupby("run")
    v = g.agg(MMSI=("MMSI", "first"), zone=("zone", "first"), start=("ts", "first"), end=("ts", "last"), ship=("Ship type", "first"))
    v["entry_seen"], v["exit_seen"] = prev_out[g.head(1).index].values, next_out[g.tail(1).index].values
    v = v[v.entry_seen & v.exit_seen].copy()
    v["stay_h"] = (v.end - v.start) / pd.Timedelta(hours=1)
    v = v[(v.stay_h >= 1) & (v.stay_h <= 240)]
    occ = a[a.zone >= 0].assign(hour=lambda x: x.ts.dt.floor("h")).groupby(["zone", "hour"]).MMSI.nunique().rename("occupancy")
    v = v.join(occ, on=["zone", v.start.dt.floor("h").rename("hour")])
    v["port"] = v.zone.map(dict(enumerate(PORTS)))
    return v.dropna(subset=["occupancy"])


def within(df, col, by):
    return df[col] - df.groupby(by)[col].transform("mean")


def boot_ci(df, fn, cluster, n=500):
    u = df[cluster].unique(); pos = {c: np.where(df[cluster].values == c)[0] for c in u}
    return np.percentile([fn(df.iloc[np.concatenate([pos[c] for c in rng.choice(u, len(u))])]) for _ in range(n)], [2.5, 97.5])


def q(s):
    return f"median {s.median():.1f}, IQR {s.quantile(.25):.1f}-{s.quantile(.75):.1f}, p90 {s.quantile(.9):.1f}, mean/median {s.mean() / s.median():.2f}"


if __name__ == "__main__":
    ais = load_days()
    v = visits(ais)
    v["ls"], v["wk"] = np.log(v.stay_h), v.start.dt.weekday.isin([4, 5, 6]).astype(float)
    v["ls_w"], v["occ_w"], v["wk_w"] = within(v, "ls", "port"), within(v, "occupancy", "port"), within(v, "wk", "port")
    rho = lambda d: spearmanr(d.occ_w, d.ls_w)[0]
    wk = lambda d: np.exp(d.ls_w[d.wk == 1].mean() - d.ls_w[d.wk == 0].mean())
    hi = lambda d: np.exp(d.ls_w[d.occ_w > 0].mean() - d.ls_w[d.occ_w <= 0].mean())
    # simulator counterparts
    s = pd.read_csv(ROOT / "data" / "shipments.csv", parse_dates=["act_port_arrival", "act_discharge", "sched_port_arrival"])
    s = s.dropna(subset=["act_port_arrival", "act_discharge"]).copy()
    s["stay_h"] = (s.act_discharge - s.act_port_arrival) / pd.Timedelta(hours=1)
    s["ls"] = np.log(s.stay_h.clip(lower=1)); s["ls_w"], s["occ_w"] = within(s, "ls", "dest"), within(s, "cong_at_arrival", "dest")
    s_rho = spearmanr(s.occ_w, s.ls_w)[0]
    s_hi = np.exp(s.ls_w[s.occ_w > 0].mean() - s.ls_w[s.occ_w <= 0].mean())
    s_late = (s.act_port_arrival - s.sched_port_arrival) / pd.Timedelta(hours=1)
    # real arrival lateness: declared ETA 12-24 h ahead, vessels that arrive (10 km zone)
    d = build(ais, 10.0); r_late = d[(d.arrived == 1) & (d.claimed_h > 12) & (d.claimed_h <= 24)].err_h
    tail = lambda x: f"median {x.median():+.1f} h, p90 {x.quantile(.9):+.1f} h, share later than +6 h {(x > 6).mean():.2f}, share earlier than -6 h {(x < -6).mean():.2f}"
    ci_r, ci_h, ci_w = boot_ci(v, rho, "MMSI"), boot_ci(v, hi, "MMSI"), boot_ci(v, wk, "MMSI")
    # verdicts by rule: an effect counts as supported only if its 95% CI excludes "no effect"
    v_cong = "supported" if ci_r[0] > 0 else ("contradicted" if ci_r[1] < 0 else "NOT detected in real data")
    rs, ss = v.stay_h.mean() / v.stay_h.median(), s.stay_h.mean() / s.stay_h.median()
    v_skew = f"supported; real tail is {'heavier' if rs > ss else 'lighter'} than simulated" if rs > 1.1 else "not supported"
    v_cal = "supported" if ci_w[0] > 1 else ("contradicted" if ci_w[1] < 1 else "not detected (CI includes no effect)")
    v_late = "supported (late tail)" if (r_late > 6).mean() > 2 * (r_late < -6).mean() else "not supported"
    out = "\n".join([
        "# Do the simulator's hand-coded mechanisms appear in real AIS data?\n",
        f"Real: {len(v)} port stays by {v.MMSI.nunique()} cargo/tanker vessels at {v.port.nunique()} ports, {v.start.min():%Y-%m-%d} to {v.start.max():%Y-%m-%d} "
        f"(5 km zones, entry and exit observed, 1-240 h). Simulator: {len(s)} shipments. Levels are NOT comparable (vessel stay vs cargo availability; "
        "short-sea vs deep-sea); only direction and rough size of effects are.\n",
        "| Mechanism assumed in the simulator | Real AIS | Simulator | Verdict |", "|---|---|---|---|",
        f"| Congestion lengthens the port stage (within port) | Spearman {rho(v):+.2f} (95% CI {ci_r[0]:+.2f} to {ci_r[1]:+.2f}); stays {hi(v):.2f}x as long when occupancy is above the port's average (CI {ci_h[0]:.2f}-{ci_h[1]:.2f}) | Spearman {s_rho:+.2f}; {s_hi:.2f}x | {v_cong} |",
        f"| Port-stage duration is right-skewed (heavy tail) | {q(v.stay_h)} | {q(s.stay_h)} | {v_skew} |",
        f"| Calendar effect: stays starting Fri-Sun are longer | {wk(v):.2f}x vs Mon-Thu (CI {ci_w[0]:.2f}-{ci_w[1]:.2f}) | modelled at customs (office hours), not at the port stage | {v_cal} |",
        f"| Arrival lateness is asymmetric with a late tail | vs ETA declared 12-24 h ahead: {tail(r_late)} | vs schedule: {tail(s_late)} | {v_late}; simulated lateness is much larger but uses a fixed schedule, not an updated ETA |",
        "",
        "Reading: occupancy is a crude congestion proxy and these ports are small and rarely full, so the absence of a congestion effect",
        "here does not show that congestion never matters; it shows the simulator's congestion mechanism is NOT externally validated.",
        "No simulator parameter was changed on the basis of these numbers.", ""])
    (Path(__file__).resolve().parent / "simulator_check.md").write_text(out, encoding="utf-8"); print(out)
