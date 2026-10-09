"""CLI: python -m aiqon.predict examples/at_port.json   -> JSON with ETA, 80% range, P(delay>48h), stage attribution, text."""
import sys, json, pickle, pathlib
import numpy as np, pandas as pd
from .features import build_long, validate, MILESTONES
from .explain import explain

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_model(fh):
    """Unpickle the trained Predictor, warning if installed package versions differ from those it was trained with."""
    import warnings
    from importlib.metadata import version
    model = pickle.load(fh)
    try:
        with open(ROOT / "models" / "metadata.json") as mf:
            trained = json.load(mf)
        diff = {p: (v, version(p)) for p, v in trained.items() if version(p) != v}
        if diff:
            warnings.warn(f"package versions differ from training (trained, installed): {diff}; re-run scripts/run_all.py")
    except FileNotFoundError:
        pass
    return model


def predict_one(model, rec: dict) -> dict:
    """rec: one shipment (scheduled times + actuals known so far, null for unknown) -> explained prediction."""
    need = ["origin", "dest", "sched_departure", "sched_port_arrival", "sched_discharge", "sched_customs", "sched_delivery",
            "transshipment", "dest_efficiency", "cong_at_departure", "customs_risk", "docs_flag"]
    absent = [c for c in need if rec.get(c) is None]
    if absent:
        raise ValueError(f"shipment record is missing required fields: {absent}")
    r = dict(rec); r.setdefault("route", f"{r['origin']}>{r['dest']}")
    for c in ("wind_origin_dep", "wind_dest_arr", "cong_at_arrival"):   # observations that may not exist yet
        r.setdefault(c, np.nan)
    for m in MILESTONES:                                   # timestamps -> datetime (null -> NaT)
        for p in ("sched", "act"):
            r[f"{p}_{m}"] = pd.Timestamp(r.get(f"{p}_{m}")) if r.get(f"{p}_{m}") else pd.NaT
    known = [j for j, m in enumerate(MILESTONES[:4]) if pd.notna(r[f"act_{m}"])]
    k = max(known) + 1 if known else 0                     # checkpoint = latest milestone reached
    r.update(act_delivery=r["sched_delivery"], final_delay_h=0.0, delayed=0, shipment_id="query")
    wide = pd.DataFrame([r]); validate(wide)
    L = build_long(wide); L = L[L.ckpt == k]
    out = model.predict(L, wide).iloc[0]
    pair = (wide.set_index("shipment_id").loc["query"], k)
    res = explain(model.chain, pair, out.delay_h, out.lo, out.hi, out.p_delay, r["sched_delivery"])
    res["checkpoint"] = ["booked", "departed", "at_port", "discharged", "customs_cleared"][k]
    return res


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m aiqon.predict <shipment.json>")
    with open(ROOT / "models" / "predictor.pkl", "rb") as fh:
        model = load_model(fh)
    with open(sys.argv[1]) as fh:
        res = predict_one(model, json.load(fh))
    print(json.dumps(res, indent=2)); print("\n" + res["text"])
