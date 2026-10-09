"""Controlled comparison: chain stage models trained the OLD way (all exogenous inputs, as if checkpoint 4, one model reused at every
checkpoint) vs the NEW way (one model per information regime). Same data, same split, same corrected availability rules at inference.
Writes report/chain_fix_effect.md. Chain-only MAE (hours) on the temporal test split."""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
import numpy as np, pandas as pd, lightgbm as lgb
from aiqon.evaluate import load, split
from aiqon.features import build_long, stage_row, delays, CKPT_NAMES
from aiqon.models import Chain, P


class OldChain(Chain):
    def fit(self, wide):
        self.g, self.cols, self.ref = {}, {}, {}
        D = np.array([delays(r) for _, r in wide.iterrows()], dtype=float)
        self.ref[0] = float(np.nanmedian(D[:, 0])); self.d0_mean = float(np.nanmean(D[:, 0]))
        for j in range(1, 5):
            ok = [i for i in range(len(wide)) if np.isfinite(D[i, j]) and np.isfinite(D[i, j - 1])]
            y = [D[i, j] - D[i, j - 1] for i in ok]
            self.ref[j] = float(np.median(y))
            X = pd.DataFrame([stage_row(j, wide.iloc[i], D[i, j - 1], 4) for i in ok])   # old: always full information
            mdl = lgb.LGBMRegressor(**P).fit(X, y)
            for reg in range(3):
                self.g[(j, reg)], self.cols[(j, reg)] = mdl, list(X.columns)
        return self


df = load(root / "data" / "shipments.csv")
tr, ca, te = split(df, "temporal")
L = build_long(df); Lte = L[L.shipment_id.isin(te.shipment_id)]
w = df.set_index("shipment_id"); pairs = [(w.loc[s], k) for s, k in zip(Lte.shipment_id, Lte.ckpt)]
res = {}
for name, cls in [("old (train as if checkpoint 4)", OldChain), ("new (per-regime models)", Chain)]:
    pred = cls().fit(tr).predict(pairs)
    res[name] = [np.abs(Lte.y.values - pred)[Lte.ckpt.values == k].mean() for k in range(5)]
    res[name].append(np.abs(Lte.y.values - pred).mean())
t = pd.DataFrame(res, index=CKPT_NAMES + ["overall"]).T.round(2)
out = "# Chain model: effect of the checkpoint-aware training fix (chain-only MAE, hours)\n\n" + t.to_markdown() + "\n"
(root / "report" / "chain_fix_effect.md").write_text(out, encoding="utf-8")
print(out)
