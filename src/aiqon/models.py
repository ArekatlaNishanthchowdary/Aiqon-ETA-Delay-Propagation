"""Models: structured chain (propagation) model, direct quantile GBM, and the conformal/blending wrapper."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from .features import FEATURES, stage_row, delays, regime

THRESH = 48.0
SEED = 0
P = dict(n_estimators=250, learning_rate=0.04, num_leaves=8, min_child_samples=15, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=SEED,
         deterministic=True, force_row_wise=True, n_jobs=1)   # bit-reproducible training


class Chain:
    """d_j = d_{j-1} + g_j(d_{j-1}, stage exogenous features). Rolling the stage models forward gives the final delay;
    swapping one stage's increment for a reference value and re-rolling gives a counterfactual (explain.py).

    Information regimes: exogenous inputs (origin wind, congestion, arrival wind) become observable at different
    checkpoints, so each stage has one model per regime (0 booked / 1 departed / 2 at port or later), trained on
    exactly the features that exist in that regime. This removes the train/inference mismatch of a single model
    trained with all (eventually-known) inputs but queried before they are known."""

    def __init__(self, oof_folds=0):
        self.oof_folds = oof_folds

    def fit(self, wide: pd.DataFrame):
        """Stage 1: teacher-forced fit (dprev = true delay). If oof_folds > 0, stage 2 refits every stage model on the delay
        the chain would actually carry into that stage at inference (out-of-fold rolled-forward prediction), so training
        inputs match the inference-time input distribution (otherwise sensitivity to dprev is overestimated)."""
        self.g, self.cols, self.ref = {}, {}, {}
        D = np.array([delays(r) for _, r in wide.iterrows()], dtype=float)
        self.ref[0] = float(np.nanmedian(D[:, 0]))
        self.d0_mean = float(np.nanmean(D[:, 0]))
        ok = {j: [i for i in range(len(wide)) if np.isfinite(D[i, j]) and np.isfinite(D[i, j - 1])] for j in range(1, 5)}
        for j in range(1, 5):
            self.ref[j] = float(np.median([D[i, j] - D[i, j - 1] for i in ok[j]]))
        ks = {0: [0], 1: [1], 2: [2, 3, 4]}                  # checkpoints served by each information regime
        self._fit_stage_models(wide, D, ok, lambda i, j, k: D[i, j - 1], {0: [0], 1: [1], 2: [2]}, obs=1.0)
        if self.oof_folds:
            dhat = self._oof_rollouts(wide)
            self._fit_stage_models(wide, D, ok, lambda i, j, k: dhat[(i, k)][j - 1], ks)
        return self

    def _fit_stage_models(self, wide, D, ok, dprev_fn, ks, obs=None):
        for j in range(1, 5):
            for reg, kk in ks.items():
                rows, y = [], []
                for k in kk:
                    for i in ok[j]:
                        rows.append(stage_row(j, wide.iloc[i], dprev_fn(i, j, k), k, obs)); y.append(D[i, j] - D[i, j - 1])
                X = pd.DataFrame(rows)
                self.cols[(j, reg)] = list(X.columns)
                self.g[(j, reg)] = lgb.LGBMRegressor(**P).fit(X, y)

    def _oof_rollouts(self, wide):
        """dhat[(i, k)][j] = delay at milestone j predicted for training shipment i at checkpoint k by a chain that never saw i."""
        n, dhat = len(wide), {}
        fold = np.arange(n) % self.oof_folds
        for f in range(self.oof_folds):
            c = Chain(0).fit(wide.iloc[np.where(fold != f)[0]])
            held = np.where(fold == f)[0]
            for k in range(5):
                _, incs = c.predict([(wide.iloc[i], k) for i in held], return_stage=True)
                for a, i in enumerate(held):
                    dhat[(i, k)] = np.cumsum(incs[a])
        return dhat

    def _states(self, pairs):
        """Imputed factual delays at observed milestones (carry-forward / interpolate for missing timestamps).
        Only milestones already reached at checkpoint k are read."""
        out = []
        for r, k in pairs:
            d = delays(r); obs = [j for j in range(4) if k >= j + 1]
            s = {}
            for j in obs:
                if np.isfinite(d[j]):
                    s[j] = d[j]
                else:
                    nxt = next((d[i] for i in obs if i > j and np.isfinite(d[i])), None)
                    prv = s.get(j - 1, 0.0)
                    s[j] = prv if nxt is None else (prv + nxt) / 2
            out.append(s)
        return out

    def stage_pred(self, j, pairs, dvec):
        """g_j evaluated for each (row, checkpoint) with the model of that checkpoint's information regime."""
        out = np.zeros(len(pairs))
        regs = np.array([regime(k) for _, k in pairs])
        for reg in range(3):
            idx = np.where(regs == reg)[0]
            if len(idx):
                X = pd.DataFrame([stage_row(j, pairs[i][0], dvec[i], pairs[i][1]) for i in idx])[self.cols[(j, reg)]]
                out[idx] = self.g[(j, reg)].predict(X)
        return out

    def predict(self, pairs, cf: dict | None = None, return_stage=False):
        """pairs: list of (wide_row, checkpoint). cf: {stage: reference increment} counterfactual override.
        Returns final-delay predictions (and per-stage factual increments if return_stage)."""
        n = len(pairs); S = self._states(pairs)
        d_f = np.array([s.get(0, self.d0_mean) for s in S]); d_c = d_f.copy()
        if cf and 0 in cf:
            d_c = np.full(n, cf[0])
        incs = np.zeros((n, 5)); incs[:, 0] = d_f
        for j in range(1, 5):
            pf = self.stage_pred(j, pairs, d_f)
            base = np.array([s[j] - s[j - 1] if j in s else np.nan for s in S])
            base = np.where(np.isnan(base), pf, base)       # observed increment, else model expectation
            if cf and j in cf:
                inc_cf = np.full(n, cf[j])
            elif cf:                                        # downstream response to the changed upstream delay
                inc_cf = base + self.stage_pred(j, pairs, d_c) - pf
            else:
                inc_cf = base
            d_f, d_c, incs[:, j] = d_f + base, d_c + inc_cf, base
        return (d_c, incs) if return_stage else d_c


class Direct:
    """Single GBM over all checkpoints (checkpoint is a feature; NaN = unobserved/missing, handled natively)."""

    def __init__(self, feats=None):
        self.feats = feats or FEATURES

    def fit(self, L: pd.DataFrame):
        X, y = L[self.feats], L["y"]
        self.q = {a: lgb.LGBMRegressor(objective="quantile", alpha=a, **P).fit(X, y) for a in (0.1, 0.5, 0.9)}
        self.clf = lgb.LGBMClassifier(**P).fit(X, (y > THRESH).astype(int))
        return self

    def predict(self, L):
        X = L[self.feats]
        return {f"q{int(a*100)}": m.predict(X) for a, m in self.q.items()} | {"p_raw": self.clf.predict_proba(X)[:, 1]}


class Predictor:
    """Harness model: chain + direct, blended; per-checkpoint conformal intervals; calibrated delay probability.
    Data roles (no overlap): train -> fit chain/direct; calibration half A -> per-checkpoint blend weights;
    calibration half B -> conformal residuals + isotonic calibration. Test data is never touched in fit()."""

    def __init__(self):
        self.w = {k: 0.5 for k in range(5)}   # chain weight per checkpoint

    def fit(self, wide_tr, L_tr, wide_cal, L_cal):
        self.chain = Chain().fit(wide_tr)
        self.direct = Direct().fit(L_tr)
        ids = np.array(sorted(L_cal["shipment_id"].unique()))
        half = set(ids[::2])                                  # deterministic alternating split by shipment
        A = L_cal[L_cal.shipment_id.isin(half)]; B = L_cal[~L_cal.shipment_id.isin(half)]
        c, d = self._chain(A, wide_cal), self.direct.predict(A)["q50"]
        for k in range(5):
            m = (A["ckpt"] == k).values
            self.w[k] = min(np.linspace(0, 1, 11), key=lambda w: np.abs(A["y"].values[m] - (w * c[m] + (1 - w) * d[m])).mean())
        res = B["y"].values - self.point(B, wide_cal)
        self.res = {k: res[(B["ckpt"] == k).values] for k in range(5)}
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(self.direct.predict(B)["p_raw"], (B["y"] > THRESH).astype(int))
        self.config = dict(seed=SEED, lgbm=P, blend_weights=self.w, n_cal_weights=len(half), n_cal_conformal=len(ids) - len(half))
        return self

    def _chain(self, L, wide):
        w = wide.set_index("shipment_id")
        return self.chain.predict([(w.loc[s], k) for s, k in zip(L["shipment_id"], L["ckpt"])])

    def point(self, L, wide):
        c = self._chain(L, wide)
        d = self.direct.predict(L)["q50"]
        w = L["ckpt"].map(self.w).values
        return w * c + (1 - w) * d

    def predict(self, L, wide):
        pt = self.point(L, wide)
        ck = L["ckpt"].values
        lo = pt + np.array([np.quantile(self.res[k], 0.1) for k in ck])
        hi = pt + np.array([np.quantile(self.res[k], 0.9) for k in ck])
        prob = np.array([(p + self.res[k] > THRESH).mean() for p, k in zip(pt, ck)])
        pdir = self.iso.predict(self.direct.predict(L)["p_raw"])
        return pd.DataFrame(dict(delay_h=pt, lo=lo, hi=hi, p_delay=0.5 * prob + 0.5 * pdir, p_conf=prob, p_direct=pdir),
                            index=L.index)
