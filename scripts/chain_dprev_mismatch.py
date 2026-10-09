"""Investigation: the chain's stage models are trained with the TRUE previous-stage delay (teacher forcing) but, past the last
observed milestone, receive a ROLLED-FORWARD PREDICTED delay at inference. (1) quantify that input noise; (2) test a fix (train on
out-of-fold rolled-forward delays + an 'is previous delay observed' flag). Everything is measured on the VALIDATION (calibration)
split only; the test split is not touched. Writes report/chain_dprev_mismatch.md."""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
import numpy as np, pandas as pd
from aiqon.evaluate import load, split
from aiqon.features import build_long, delays, CKPT_NAMES
from aiqon.models import Chain

df = load(root / "data" / "shipments.csv")
tr, ca, te = split(df, "temporal")
L = build_long(df); Lc = L[L.shipment_id.isin(ca.shipment_id)]
w = df.set_index("shipment_id")
pairs = [(w.loc[s], k) for s, k in zip(Lc.shipment_id, Lc.ckpt)]
ck, y = Lc.ckpt.values, Lc.y.values

# (1) error of the delay that stage models actually receive at inference vs the true value they saw in training (0)
base = Chain(0).fit(tr)
_, incs = base.predict(pairs, return_stage=True)
dhat = np.cumsum(incs, axis=1)
D = np.array([delays(r) for r, _ in pairs], dtype=float)
rows = {}
for j in range(1, 5):                                   # stage j receives d_{j-1}; observed iff checkpoint k >= j
    m = np.isfinite(D[:, j - 1]) & (ck < j)
    rows[f"stage {j} (input d{j-1})"] = dict(n=int(m.sum()), rmse_h=float(np.sqrt(np.mean((dhat[m, j - 1] - D[m, j - 1]) ** 2))),
                                           sd_true_h=float(D[m, j - 1].std()))
t1 = pd.DataFrame(rows).T.round(1)

# (2) chain-only MAE on validation: teacher-forced vs out-of-fold-trained
res = {}
for name, c in [("teacher-forced (default)", base), ("out-of-fold rolled-forward + observed flag", Chain(5).fit(tr))]:
    e = np.abs(y - c.predict(pairs))
    res[name] = [e[ck == k].mean() for k in range(5)] + [e.mean()]
t2 = pd.DataFrame(res, index=CKPT_NAMES + ["overall"]).T.round(2)
out = ("# Chain: previous-stage-delay train/inference mismatch (validation split only)\n\n"
       "## Noise of the previous-stage delay the models receive when it is not yet observed (hours)\n\n" + t1.to_markdown() +
       "\n\n## Chain-only MAE on validation (hours)\n\n" + t2.to_markdown() + "\n")
(root / "report" / "chain_dprev_mismatch.md").write_text(out, encoding="utf-8")
print(out)
