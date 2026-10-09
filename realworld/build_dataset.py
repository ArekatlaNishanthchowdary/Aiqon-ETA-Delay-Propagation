"""Turn extracted AIS days into a prediction table whose rows are chosen WITHOUT looking at the outcome.

A row = (vessel, hour t). It exists whenever, at hour t, the vessel's message in the :00-:04 five-minute bin shows
  * a declared Destination that maps to a port, an ETA, and a speed;  * the vessel is outside that port's R-km zone;
  * the declared ETA is between 12 h in the past and 48 h ahead (claimed_h in (-12, 48]);
  * eta + CAP_H <= end of data (so the outcome window is fully observed: no right-censoring).
All of these are known at t or depend only on the declared ETA and the data end. Vessels that never arrive are KEPT.

Label: y = min(arrival - declared ETA, CAP_H) in hours, where arrival = first observed zone entry after t. If the vessel does
not enter the zone by eta + CAP_H (or changes plans), y = CAP_H and arrived = 0, PROVIDED the vessel is still tracked in the
last 12 h of that window; if it left receiver coverage the outcome is unknown and the row is dropped. Rows whose window contains
an entry that went unobserved (message gap > 30 min while crossing) are also dropped. Hovering re-entries within 2 h are ignored.
Earlier versions that kept only calls that did arrive within a fixed window were rejected: that conditions on the future and the
error metric then depends on the window length (see README).
Features use only data up to t: ETA/position/speed at t, declaration history over the previous 48 h, time since the vessel
last was inside the zone, navigational status, static vessel fields."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ports import PORTS, locode

DAYS = Path(__file__).resolve().parent / "data" / "days"
H = np.timedelta64(1, "h")
CAP_H, CLAIM_MIN, CLAIM_MAX = 48.0, -12.0, 48.0
MAX_GAP = np.timedelta64(30, "m")
TRACK_TAIL = np.timedelta64(12, "h")
FMT = "%d/%m/%Y %H:%M:%S"
OUT_COLS = ["MMSI", "port", "cluster", "t", "eta", "claimed_h", "err_h", "arrived", "rem_km", "sog", "kin_h", "kin_gap_h", "eta_age_h",
            "n_prev", "rev_total_h", "since_inside_h", "navstat", "eta_hour", "eta_wd", "ship_type", "length", "draught"]


def hav_km(lat, lon, plat, plon):
    p = np.pi / 180
    a = np.sin((lat - plat) * p / 2) ** 2 + np.cos(lat * p) * np.cos(plat * p) * np.sin((lon - plon) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def load_days() -> pd.DataFrame:
    files = sorted(DAYS.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no extracted days in {DAYS}; run ais_extract.py first")
    return pd.concat((pd.read_csv(f, parse_dates=["ts"], low_memory=False) for f in files), ignore_index=True)


def build(ais: pd.DataFrame, R: float = 10.0, cap_h: float = CAP_H, end=None) -> pd.DataFrame:
    ais = ais.copy()
    ais["port"] = ais.Destination.map(locode)
    ais["eta"] = pd.to_datetime(ais.ETA, format=FMT, errors="coerce")
    ais = ais.sort_values(["MMSI", "ts"])
    end = np.datetime64(pd.Timestamp(end) if end is not None else ais.ts.max())
    cap = np.timedelta64(int(cap_h * 60), "m")
    rows = []
    for mmsi, g in ais.groupby("MMSI", sort=False):
        g = g.reset_index(drop=True)
        ts, port_v, eta_v = g.ts.values, g.port.values, g.eta.values
        top = np.where(pd.DatetimeIndex(ts).minute < 5)[0]          # the :00-:04 five-minute bin = one state per hour
        ports = g.port.dropna().unique()
        if len(top) == 0 or len(ports) == 0:
            continue
        key = g.eta.astype("int64").astype(str) + "|" + g.Destination.astype(str)
        run = (key != key.shift()).cumsum().values
        run_start = pd.Series(ts).groupby(run).transform("min").values
        for p in ports:
            d = hav_km(g.Latitude.values, g.Longitude.values, *PORTS[p])
            inside = d <= R
            last_in = pd.Series(np.where(inside, ts, np.datetime64("NaT"))).ffill().values
            ent = np.where(inside[1:] & ~inside[:-1])[0] + 1
            ent = [i for i in ent if not (pd.notna(last_in[i - 1]) and ts[i] - last_in[i - 1] < np.timedelta64(2, "h"))]   # hover re-entry
            ent_t = ts[ent]
            observed = np.array([ts[i] - ts[i - 1] <= MAX_GAP for i in ent], dtype=bool)
            idx = top[(port_v[top] == p) & ~inside[top] & ~pd.isna(eta_v[top]) & g.SOG.notna().values[top]]
            if len(idx) == 0:
                continue
            t, eta = ts[idx], eta_v[idx]
            claimed = (eta - t) / H
            ok = (claimed > CLAIM_MIN) & (claimed <= CLAIM_MAX) & (eta + cap <= end)
            if not ok.any():
                continue
            idx, t, eta, claimed = idx[ok], t[ok], eta[ok], claimed[ok]
            n = np.searchsorted(ent_t, t, side="left")
            has = n < len(ent_t)
            if len(ent_t):
                nn = np.where(has, n, 0)
                t_next = np.where(has, ent_t[nn], np.datetime64("NaT"))
                was_observed = np.where(has, observed[nn], False)
            else:
                t_next, was_observed = np.full(len(t), np.datetime64("NaT")), np.zeros(len(t), dtype=bool)
            within = has & (t_next <= eta + cap)
            ambiguous = within & ~was_observed                   # entry happened but its time is unknown
            err = np.where(within & was_observed, (t_next - eta) / H, cap_h)
            err = np.minimum(err, cap_h)
            # "never arrived" is only asserted if the vessel is still being tracked near the end of its window; a vessel
            # that sailed out of receiver coverage has an UNKNOWN outcome and is dropped rather than labelled as a no-show
            j_last = np.searchsorted(ts, eta + cap, side="right") - 1
            tracked = ts[j_last] >= eta + cap - TRACK_TAIL
            keep = ~ambiguous & ((within & was_observed) | tracked)
            if not keep.any():
                continue
            idx, t, eta, claimed, err = idx[keep], t[keep], eta[keep], claimed[keep], err[keep]
            arrived = ((within & was_observed)[keep]).astype(int)
            win0 = np.searchsorted(ts, t - 48 * H, side="left")
            win0 = np.minimum(win0, idx)
            rem = d[idx] - R
            sog = g.SOG.values[idx]
            kin = np.maximum(rem, 0) / (np.maximum(sog, 1.0) * 1.852)
            same = (port_v[win0] == p) & ~pd.isna(eta_v[win0])
            rows.append(pd.DataFrame(dict(
                MMSI=mmsi, port=p, cluster=f"{mmsi}|{p}", t=t, eta=eta, claimed_h=claimed, err_h=err, arrived=arrived, rem_km=rem, sog=sog,
                kin_h=kin, kin_gap_h=kin - claimed, eta_age_h=(t - run_start[idx]) / H, n_prev=run[idx] - run[win0],
                rev_total_h=np.where(same, (eta - eta_v[win0]) / H, 0.0),
                since_inside_h=np.minimum(np.nan_to_num((t - last_in[idx]) / H, nan=168.0), 168.0),
                navstat=g["Navigational status"].values[idx], eta_hour=pd.DatetimeIndex(eta).hour, eta_wd=pd.DatetimeIndex(eta).weekday,
                ship_type=g["Ship type"].values[idx], length=g.Length.values[idx], draught=g.Draught.values[idx])))
    return pd.concat(rows, ignore_index=True)[OUT_COLS] if rows else pd.DataFrame(columns=OUT_COLS)


if __name__ == "__main__":
    R = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    d = build(load_days(), R)
    print(f"R={R} km: {len(d)} rows, {d.cluster.nunique()} vessel-port clusters, arrived share {d.arrived.mean():.2f}")
    if len(d):
        d.to_csv(DAYS.parent / f"dataset_R{int(R)}.csv.gz", index=False, compression="gzip")
