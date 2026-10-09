"""python -m pytest realworld -q   (no data or network needed: synthetic AIS tracks)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from ports import PORTS, locode
from build_dataset import build, FMT, hav_km, CAP_H
from validate import evaluate, split_by_time, PURGE

T0 = pd.Timestamp("2026-09-01 00:00:00")
KNOT = 1.852
HOUR = pd.Timedelta(hours=1)


def _frame(ts, mmsi, lat, lon, sog, dest, etas):
    return pd.DataFrame({"ts": ts, "MMSI": mmsi, "Latitude": lat, "Longitude": lon, "SOG": sog, "IMO": "1", "Length": 100.0, "Draught": 6.0,
                         "Destination": dest, "ETA": [e.strftime(FMT) for e in etas], "Ship type": "Cargo",
                         "Navigational status": "Under way using engine"})


def track(mmsi, t_in, bias_h=0.0, port="DKSKA", dest="DKSKA", speed_kn=12.0, start_h=30, eta_fn=None, R=10.0):
    """Vessel steaming straight at `port` at constant speed; it enters the R-km zone at t_in. 5-minute messages."""
    plat, plon = PORTS[port]
    ts = pd.date_range(t_in - start_h * HOUR, t_in + HOUR, freq="5min")
    h = (t_in - ts) / HOUR
    dist = np.maximum(R + speed_kn * KNOT * h, 0.5)
    eta = t_in - bias_h * HOUR                                   # declared ETA is bias_h BEFORE the real arrival -> error +bias_h
    return _frame(ts, mmsi, plat + dist / 111.19, plon, speed_kn, dest, [(eta_fn(t) if eta_fn else eta) for t in ts])


def loiter(mmsi, t0, t1, eta, dest="DKSKA", km_away=300.0):
    """Vessel that declares `dest` but stays far away for the whole period (never arrives)."""
    plat, plon = PORTS[dest]
    ts = pd.date_range(t0, t1, freq="5min")
    return _frame(ts, mmsi, plat + km_away / 111.19, plon, 0.2, dest, [eta] * len(ts))


def test_locode_including_the_pair_that_was_once_swapped():
    assert locode("DK SKA") == "DKSKA" and locode("Skagen") == "DKSKA" and locode("SEGOT>DKSKA") == "DKSKA"
    assert locode("Fredericia") == "DKFRC" and locode("Frederikshavn") == "DKFDH"
    assert PORTS["DKFDH"][0] > 57 and PORTS["DKFRC"][0] < 56          # Frederikshavn is the northern one
    assert locode("DEHAM") is None and locode("Unknown") is None and locode(np.nan) is None


def test_geometry_helper():
    assert abs(hav_km(55.0, 10.0, 56.0, 10.0) - 111.19) < 0.5


def test_labels_for_an_arriving_vessel():
    t_in = T0 + 40 * HOUR
    d = build(track(1, t_in, bias_h=2.0), end=t_in + 100 * HOUR)
    assert len(d) > 10 and (d.arrived == 1).all()
    assert (d.t < t_in + pd.Timedelta(minutes=6)).all()
    assert ((d.err_h >= 2.0 - 1e-9) & (d.err_h <= 2.0 + 5 / 60 + 1e-9)).all()      # arrival = first message inside: 5-min resolution
    assert ((d.claimed_h > -12) & (d.claimed_h <= 48)).all()
    assert np.allclose(d.kin_h, d.rem_km / (12 * KNOT), atol=1e-3)                   # physics ETA matches the constructed track


def test_non_arriving_vessel_is_kept_with_the_cap_label_when_still_tracked():
    eta = T0 + 30 * HOUR
    d = build(loiter(2, T0, eta + 60 * HOUR, eta), end=eta + 60 * HOUR)
    assert len(d) > 10 and (d.arrived == 0).all() and np.allclose(d.err_h, CAP_H)   # rows are NOT conditioned on arriving


def test_unknown_outcome_is_dropped_not_labelled_as_no_show():
    eta = T0 + 30 * HOUR
    lost = loiter(3, T0, eta + 5 * HOUR, eta)                                       # vessel leaves coverage long before eta + cap
    assert build(lost, end=eta + 60 * HOUR).empty


def test_right_censored_rows_are_excluded():
    t_in = T0 + 40 * HOUR
    assert build(track(4, t_in, bias_h=2.0), end=t_in + 10 * HOUR).empty            # eta + cap lies beyond the data end


def test_unmapped_destination_and_unobserved_entry():
    t_in = T0 + 40 * HOUR
    assert build(track(5, t_in, dest="DEHAM"), end=t_in + 100 * HOUR).empty
    tr = track(6, t_in, bias_h=1.0)
    gap = tr[(tr.ts < t_in - HOUR) | (tr.ts > t_in + pd.Timedelta(minutes=30))]     # the crossing itself is not observed
    assert build(gap, end=t_in + 100 * HOUR).empty                                   # arrival time unknown -> ambiguous -> dropped


def test_future_revisions_do_not_leak_into_earlier_rows():
    t_in = T0 + 40 * HOUR
    cut = t_in - 10 * HOUR
    base = track(7, t_in, bias_h=1.0)
    revised = track(7, t_in, bias_h=1.0, eta_fn=lambda t: (t_in - HOUR) if t <= cut else (t_in + 7 * HOUR))
    revised["SOG"] = np.where(revised.ts > cut, 3.0, revised.SOG)                    # future speed changes too
    a, b = build(base, end=t_in + 100 * HOUR), build(revised, end=t_in + 100 * HOUR)
    pd.testing.assert_frame_equal(a[a.t <= cut].reset_index(drop=True), b[b.t <= cut].reset_index(drop=True))
    assert (b[b.t > cut].claimed_h > 5).all()                                        # the revision is visible once broadcast


def _world(n, bias=2.0, seed=0):
    rng = np.random.default_rng(seed)
    ports = ["DKSKA", "DKEBJ", "DKCPH", "DERSK"]
    return pd.concat([track(1000 + i, T0 + (40 + 1.2 * i) * HOUR, bias_h=bias + rng.normal(0, 1.0), port=ports[i % 4], dest=ports[i % 4],
                            speed_kn=float(rng.uniform(8, 16)), start_h=20) for i in range(n)], ignore_index=True)


def test_time_split_is_ordered_and_purged():
    d = build(_world(600))                                                           # ~30 days: long enough to survive the purge
    tr, va, te = split_by_time(d)
    q1, q2 = d.t.quantile(0.6), d.t.quantile(0.8)
    assert len(tr) and len(va) and len(te)
    assert (tr.eta + PURGE <= q1).all() and (va.t > q1).all() and (va.eta + PURGE <= q2).all() and (te.t > q2).all()
    assert tr.t.max() < va.t.min() and va.t.max() < te.t.min()
    assert not (set(tr.cluster) & set(te.cluster))                                   # one voyage never straddles train and test here


def test_refuses_small_samples_and_learns_an_injected_bias():
    assert "No performance table" in "\n".join(evaluate(build(_world(80)), 10.0))
    big = "\n".join(evaluate(build(_world(700)), 10.0))
    assert "Test performance" in big and "empirical test coverage" in big
    row = lambda name: float(next(l for l in big.splitlines() if l.startswith(f"| {name}")).split("|")[2])
    assert row("LightGBM") < row("declared ETA as broadcast")                        # systematic lateness is learnt on later, held-out rows
