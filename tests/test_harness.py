"""Tests: python -m pytest tests -q    (self-contained: simulates a small dataset and trains a small model once; ~1 min).
Needs the cached wind file data/wind_daily.csv (created on first `python scripts/run_all.py`, or fetched from Open-Meteo)."""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
import numpy as np, pandas as pd, pytest
from aiqon.simulate import simulate
from aiqon.features import build_long, validate, FEATURES, stage_row, regime, MILESTONES
from aiqon.models import Predictor, Chain
from aiqon.evaluate import split, HOLDOUT_ROUTES, load
from aiqon.explain import stage_attribution
from aiqon.predict import predict_one

G = [f"gt_stage{j}" for j in range(5)]


def _sorted(df):
    return df.sort_values("sched_departure").reset_index(drop=True)


@pytest.fixture(scope="module")
def world():
    df = _sorted(simulate(400, seed=7))
    tr, ca, te = split(df, "temporal")
    L = build_long(df); sp = lambda d: L[L.shipment_id.isin(d.shipment_id)]
    m = Predictor().fit(tr, sp(tr), ca, sp(ca))
    return dict(df=df, tr=tr, ca=ca, te=te, L=L, sp=sp, m=m)


def _future_altered(df, k):
    """Copy of df where everything that happens AFTER checkpoint k is changed drastically."""
    a = df.copy()
    for j, name in enumerate(MILESTONES):               # milestone j is reached at checkpoint j+1
        if j + 1 > k:
            a[f"act_{name}"] = a[f"act_{name}"] + pd.Timedelta(hours=400)
    if k < 2:                                            # arrival-time observations do not exist yet
        a["cong_at_arrival"] = 0.99; a["wind_dest_arr"] = 99.0
    if k < 1:
        a["wind_origin_dep"] = 99.0; a["cong_at_departure"] = 0.99
    a["final_delay_h"] += 400; a["delayed"] = 1
    return a


# 1. checkpoint feature availability
def test_checkpoint_feature_availability(world):
    L = world["L"]
    for k in range(5):
        r = L[L.ckpt == k]
        for j in range(4):                               # delay at milestone j only known once checkpoint >= j+1
            if k < j + 1:
                assert r[f"d{j}"].isna().all(), (k, j)
        if k < 2:
            assert r[["cong_arr", "wind_arr", "d1_x_cong", "over_slack"]].isna().all().all()
        if k < 1:
            assert r[["wind_dep", "cong_dep", "d0"]].isna().all().all()
    assert (L[L.ckpt == 4].d3.notna().mean() > 0.7)


# 2. future-timestamp leakage in the feature table
def test_no_feature_leakage(world):
    df, base = world["df"].head(80), build_long(world["df"].head(80))
    for k in range(4):
        new = build_long(_future_altered(df, k))
        feats = [c for c in base.columns if c not in ("y", "delayed")]
        pd.testing.assert_frame_equal(base[base.ckpt == k][feats].reset_index(drop=True),
                                      new[new.ckpt == k][feats].reset_index(drop=True))


# 2b. adversarial end-to-end: predictions at checkpoint k are unchanged when the future is altered
def test_predictions_invariant_to_future(world):
    m, df = world["m"], world["te"].head(40)
    L0 = build_long(df)
    for k in range(4):
        a = _future_altered(df, k)
        La = build_long(a)
        p0 = m.predict(L0[L0.ckpt == k], df); p1 = m.predict(La[La.ckpt == k], a)
        np.testing.assert_allclose(p0.values, p1.values, rtol=0, atol=1e-9, err_msg=f"checkpoint {k}")
        attr0, _ = stage_attribution(m.chain, [(df.set_index("shipment_id").loc[s], k) for s in L0[L0.ckpt == k].shipment_id])
        attr1, _ = stage_attribution(m.chain, [(a.set_index("shipment_id").loc[s], k) for s in La[La.ckpt == k].shipment_id])
        np.testing.assert_allclose(attr0, attr1, atol=1e-9)


# 3. training / inference feature consistency
def test_train_inference_feature_consistency(world):
    m, df = world["m"], world["df"]
    r = df.iloc[0]
    for j in range(1, 5):
        for k in range(5):
            assert list(stage_row(j, r, 5.0, k)) == m.chain.cols[(j, regime(k))], (j, k)   # same columns, same order
    L = world["L"]
    assert set(FEATURES) <= set(L.columns)
    assert all(f in m.direct.feats for f in FEATURES)
    assert np.isnan(stage_row(2, r, 5.0, 0)["cong"]) and np.isnan(stage_row(2, r, 5.0, 1)["wind_arr"])
    assert stage_row(2, r, 5.0, 1)["cong"] == r["cong_at_departure"] and stage_row(2, r, 5.0, 2)["cong"] == r["cong_at_arrival"]


# 4. exact delay decomposition
def test_exact_decomposition():
    df = simulate(150, seed=3)
    assert (df[G].sum(axis=1) - df.final_delay_h).abs().max() < 1e-6
    assert ((df.act_delivery - df.sched_delivery) / pd.Timedelta(hours=1) - df.final_delay_h).abs().max() < 1e-6


# 5. missing intermediate timestamps
def test_missing_timestamps(world):
    m, df = world["m"], world["te"].head(30).copy()
    rng = np.random.default_rng(0)
    for c in ("act_port_arrival", "act_discharge", "act_customs"):
        df.loc[rng.random(len(df)) < 0.6, c] = pd.NaT
    L = build_long(validate(df))
    out = m.predict(L, df)
    assert np.isfinite(out.values).all()
    assert (L.n_missing >= 0).all() and L.n_missing.max() > 0


# 6. invalid / non-monotonic timelines and schema
def test_validate_rejects_bad_input(world):
    df = world["df"].head(5).copy()
    bad = df.copy(); bad.loc[0, "sched_delivery"] = bad.loc[0, "sched_departure"] - pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="sched"):
        validate(bad)
    bad = df.copy(); bad.loc[1, "act_customs"] = bad.loc[1, "act_departure"] - pd.Timedelta(hours=5)
    with pytest.raises(ValueError, match="act"):
        validate(bad)
    with pytest.raises(ValueError, match="missing columns"):
        validate(df.drop(columns=["dest_efficiency"]))
    validate(df)                                          # valid data passes


def test_predict_one_validates_and_handles_gaps(world):
    m, r = world["m"], world["te"].iloc[3]
    rec = {k: (str(v) if isinstance(v, pd.Timestamp) else (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)))
           for k, v in r.items() if not k.startswith("gt_")}
    for k in ("act_discharge", "act_customs", "act_delivery"):
        rec[k] = None
    out = predict_one(m, rec)
    assert np.isfinite(out["delay_h"]) and out["lo_h"] <= out["hi_h"] and 0 <= out["p_delay"] <= 1
    with pytest.raises(ValueError, match="missing required"):
        predict_one(m, {**rec, "dest_efficiency": None})


# 7. split isolation
def test_split_isolation():
    df = _sorted(simulate(300, seed=2))
    tr, ca, te = split(df, "temporal")
    assert tr.sched_departure.max() <= ca.sched_departure.min() <= ca.sched_departure.max() <= te.sched_departure.min()
    assert not (set(tr.shipment_id) & set(ca.shipment_id)) and not (set(ca.shipment_id) & set(te.shipment_id))
    tr, ca, te = split(df, "route")
    assert set(te.route) <= set(HOLDOUT_ROUTES)
    assert not (set(tr.route) | set(ca.route)) & set(HOLDOUT_ROUTES)
    assert not set(tr.shipment_id) & (set(ca.shipment_id) | set(te.shipment_id)) and not set(ca.shipment_id) & set(te.shipment_id)


def test_calibration_data_not_used_for_fitting(world):
    m = world["m"]
    assert m.config["n_cal_weights"] + m.config["n_cal_conformal"] == len(world["ca"])


# 8. calibration / interval behaviour
def test_intervals_and_probabilities(world):
    m, te, sp = world["m"], world["te"], world["sp"]
    Lt = sp(te); out = m.predict(Lt, world["df"])
    assert (out.lo <= out.hi).all() and ((out.p_delay >= 0) & (out.p_delay <= 1)).all()
    assert ((out.lo <= out.delay_h) | (out.delay_h <= out.hi)).all()
    cov = ((Lt.y.values >= out.lo.values) & (Lt.y.values <= out.hi.values)).mean()
    assert 0.65 < cov < 0.95, cov                          # nominal 0.80, small test set
    w = (out.hi - out.lo).groupby(Lt.ckpt.values).mean()
    assert w.iloc[-1] < w.iloc[0]                          # uncertainty shrinks as the journey progresses
    for k in range(5):                                     # P(delay) rises with predicted delay at a fixed checkpoint
        s = out[Lt.ckpt.values == k].sort_values("delay_h")
        assert (np.diff(s.p_conf.values) >= -1e-12).all()      # exactly non-decreasing by construction


# 9. reproducibility under fixed seeds
def test_reproducibility():
    a, b = simulate(120, seed=11), simulate(120, seed=11)
    pd.testing.assert_frame_equal(a, b)
    df = _sorted(a); tr, ca, te = split(df, "temporal")
    L = build_long(df); sp = lambda d: L[L.shipment_id.isin(d.shipment_id)]
    p = [Predictor().fit(tr, sp(tr), ca, sp(ca)).predict(sp(te), df).values for _ in range(2)]
    np.testing.assert_array_equal(p[0], p[1])


# 10. finite, correctly shaped predictions
def test_prediction_shapes(world):
    m, te, sp = world["m"], world["te"], world["sp"]
    Lt = sp(te); out = m.predict(Lt, world["df"])
    assert out.shape == (len(Lt), 6) and np.isfinite(out.values).all()
    w = world["df"].set_index("shipment_id")
    attr, d = stage_attribution(m.chain, [(w.loc[s], k) for s, k in zip(Lt.shipment_id.head(25), Lt.ckpt.head(25))])
    assert attr.shape == (25, 5) and np.isfinite(attr).all() and d.shape == (25,)
