"""Exposure bias in the chain model: experiments on the VALIDATION (calibration) split only; the test split is never read.

The chain's stage models are trained with the TRUE previous-stage delay (teacher forcing) but receive a rolled-forward PREDICTION
beyond the last observed milestone. Variants compared (chain-only MAE, hours):
  A  teacher forcing (baseline)
  B  out-of-fold carry-in, target = true stage increment                  (earlier attempt; cannot correct upstream error)
  C  sequential residual chain: out-of-fold carry-in, target = true delay at the stage MINUS the predicted carry-in, fitted
     stage by stage so each stage sees the carry-in the upstream stages actually produce and can correct their error
And for the combination with the direct model (half A of validation fits the combiner, half B scores it):
  per-checkpoint weight blend   vs   stacked model (direct features + chain prediction as an input, out-of-fold)
Writes report/chain_exposure_bias.md."""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
import numpy as np, pandas as pd, lightgbm as lgb
from aiqon.evaluate import load, split
from aiqon.features import build_long, delays, stage_row, FEATURES, CKPT_NAMES
from aiqon.models import Chain, Direct, P

class SeqChain(Chain):
    """Sequential residual chain (exposure-bias correction). Stage j is trained on the carry-in delay the upstream stages actually
    produce (out-of-fold, so a row's carry-in never comes from a model that saw it) and predicts the TRUE delay at milestone j
    minus that carry-in. Each stage can therefore correct accumulated upstream error instead of assuming an exact input, and
    training inputs match what predict() feeds at inference. Stages are fitted in order so stage j sees stage j-1's real output.
    One model per (stage, checkpoint): the carry-in is exact at some checkpoints and a noisy forecast at others, and a shared
    model cannot serve both. Where stage j is already observed the model is only used by the what-if view and is teacher-forced."""

    def __init__(self, folds=5):
        self.folds = folds

    def key(self, j, k):
        return (j, k)

    def fit(self, wide: pd.DataFrame):
        n = len(wide)
        rows = [wide.iloc[i] for i in range(n)]
        D = np.array([delays(r) for r in rows], dtype=float)
        self.g, self.cols, self.ref = {}, {}, {0: float(np.nanmedian(D[:, 0]))}
        self.d0_mean = float(np.nanmean(D[:, 0]))
        S = {k: self._states([(r, k) for r in rows]) for k in range(5)}
        carry = {k: np.array([s.get(0, self.d0_mean) for s in S[k]]) for k in range(5)}       # delay carried INTO stage 1
        fold = np.arange(n) % self.folds
        for j in range(1, 5):
            okj = np.where(np.isfinite(D[:, j]))[0]
            both = [i for i in okj if np.isfinite(D[i, j - 1])]
            self.ref[j] = float(np.median([D[i, j] - D[i, j - 1] for i in both]))
            new = {}
            for k in range(5):
                if k > j:                                          # stage j observed at checkpoint k: what-if only, teacher-forced
                    X = pd.DataFrame([stage_row(j, rows[i], D[i, j - 1], k, 1.0) for i in both])
                    self.cols[(j, k)] = list(X.columns)
                    self.g[(j, k)] = lgb.LGBMRegressor(**P).fit(X, [D[i, j] - D[i, j - 1] for i in both])
                    new[k] = np.array([s[j] for s in S[k]])        # carry the (imputed) observation forward
                    continue
                X = pd.DataFrame([stage_row(j, rows[i], carry[k][i], k) for i in okj])
                yy = D[okj, j] - carry[k][okj]
                self.cols[(j, k)] = list(X.columns)
                self.g[(j, k)] = lgb.LGBMRegressor(**P).fit(X, yy)
                new[k] = carry[k].copy()
                Xall = pd.DataFrame([stage_row(j, rows[i], carry[k][i], k) for i in range(n)])[self.cols[(j, k)]]
                for f in range(self.folds):                        # out-of-fold delay AFTER stage j, fed to stage j+1
                    tr_m = fold[okj] != f
                    m = lgb.LGBMRegressor(**P).fit(X[tr_m], yy[tr_m])
                    held = np.where(fold == f)[0]
                    new[k][held] = carry[k][held] + m.predict(Xall.iloc[held])
            carry = new
        self.oof_final = carry                                     # out-of-fold final-delay prediction per checkpoint (for stacking)
        return self


df = load(root / "data" / "shipments.csv")
tr, ca, te = split(df, "temporal")                       # te is not used anywhere below
L = build_long(df)
Ltr, Lca = L[L.shipment_id.isin(tr.shipment_id)], L[L.shipment_id.isin(ca.shipment_id)]
w = df.set_index("shipment_id")
pairs = lambda LL: [(w.loc[s], k) for s, k in zip(LL.shipment_id, LL.ckpt)]
ck, y = Lca.ckpt.values, Lca.y.values


def by_ckpt(pred, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    e = np.abs(y - pred)
    return [e[m & (ck == k)].mean() for k in range(5)] + [e[m].mean()]


# how noisy is the carry-in the stage models actually receive when the previous milestone is not yet observed?
_, incs = Chain(0).fit(tr).predict(pairs(Lca), return_stage=True)
dhat, Dv = np.cumsum(incs, axis=1), np.array([delays(r) for r, _ in pairs(Lca)], dtype=float)
noise = {}
for j in range(1, 5):
    m = np.isfinite(Dv[:, j - 1]) & (ck < j)
    noise[f"stage {j} (carry-in d{j-1})"] = dict(rows=int(m.sum()), rmse_of_carry_in_h=float(np.sqrt(np.mean((dhat[m, j - 1] - Dv[m, j - 1]) ** 2))),
                                                 sd_of_true_delay_h=float(Dv[m, j - 1].std()))
t0 = pd.DataFrame(noise).T.round(1)

chains = {"A teacher forcing": Chain(0).fit(tr), "B out-of-fold carry-in, increment target": Chain(5).fit(tr),
          "C sequential residual chain": SeqChain(5).fit(tr)}
cp = {n: c.predict(pairs(Lca)) for n, c in chains.items()}
t1 = pd.DataFrame({n: by_ckpt(p) for n, p in cp.items()}, index=CKPT_NAMES + ["overall"]).T.round(2)

# combination with the direct model: fit combiner on half A, score on half B (both halves are validation data)
ids = np.array(sorted(Lca.shipment_id.unique())); A = Lca.shipment_id.isin(set(ids[::2])).values; B = ~A
direct = Direct().fit(Ltr); dp = direct.predict(Lca)["q50"]
rows = {"direct model alone": by_ckpt(dp, B)}
for n in ("A teacher forcing", "C sequential residual chain"):
    c = cp[n]; rows[f"{n} alone"] = by_ckpt(c, B)
    wts = {k: min(np.linspace(0, 1, 11), key=lambda q: np.abs(y[A & (ck == k)] - (q * c[A & (ck == k)] + (1 - q) * dp[A & (ck == k)])).mean()) for k in range(5)}
    rows[f"weight blend: direct + {n}"] = by_ckpt(np.array([wts[k] for k in ck]) * c + (1 - np.array([wts[k] for k in ck])) * dp, B)
# stacked: direct features + out-of-fold chain prediction on the training rows
seq = chains["C sequential residual chain"]
oof = np.array([seq.oof_final[k][i] for i, k in zip(Ltr.shipment_id.map({s: i for i, s in enumerate(tr.shipment_id)}).values, Ltr.ckpt.values)])
st = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **P).fit(Ltr[FEATURES].assign(chain=oof), Ltr.y)
rows["stacked: direct features + C as input"] = by_ckpt(st.predict(Lca[FEATURES].assign(chain=cp["C sequential residual chain"])), B)
t2 = pd.DataFrame(rows, index=CKPT_NAMES + ["overall"]).T.round(2)
out = "\n\n".join(["# Chain exposure bias: experiments on the validation split only",
                    "## Noise of the carry-in delay at inference (training saw exact values)", t0.to_markdown(),
                    "## Chain-only MAE (hours), all validation rows", t1.to_markdown(),
                    "## Combined with the direct model: MAE (hours) on validation half B (combiner fitted on half A)", t2.to_markdown()]) + "\n"
(root / "report" / "chain_exposure_bias.md").write_text(out, encoding="utf-8")
print(out)
