"""由最精确三分位估计 τ² = (R_low − 1)·mean v_low（截 0），推算仅有异质性时最不精确三分位的预期 R = 1 + τ²/mean v_high；
自助法 95% CI（5,000 次，seed 42，三分位成员固定）。另报告 SE 与研究级 n 的描述。输出 results/09_expected_r_high.json。
用法：python -I code/09_expected_r_high.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; BAD = {40596905, 41648663}; out = {}
def f(H):
    d = (H.auc_a - H.auc_ext).to_numpy(float); v = (((H.hi_a - H.lo_a) / 3.92) ** 2 + ((H.hi_ext - H.lo_ext) / 3.92) ** 2).to_numpy(float)
    t = pd.qcut(np.sqrt(v), 3, labels=False); Rl = np.var(d[t == 0], ddof=1) / v[t == 0].mean(); tau2 = max(0.0, (Rl - 1) * v[t == 0].mean())
    rng = np.random.default_rng(42); L, Hh = np.where(t == 0)[0], np.where(t == 2)[0]; bs = []
    for _ in range(5000):   # 自助法 CI（三分位成员固定，组内有放回抽研究）
        a = rng.choice(L, len(L)); b = rng.choice(Hh, len(Hh)); t2 = max(0.0, (np.var(d[a], ddof=1) / v[a].mean() - 1) * v[a].mean())
        bs.append(1 + t2 / v[b].mean())
    return {"k": len(d), "tau_from_low": round(float(np.sqrt(tau2)), 4), "expected_R_high": round(float(1 + tau2 / v[t == 2].mean()), 3),
            "expected_R_high_ci95": [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])],
            "median_auc_int": round(float(H.auc_a.median()), 3), "median_auc_ext": round(float(H.auc_ext.median()), 3), "share_ext_ge_int": round(float((d <= 0).mean()), 3)}
P = []
for tag, d_ in (("radiomics", "data/radiomics_FT"), ("DL", "data/dl_FT")):
    p = pd.read_csv(ROOT / d_ / "pairs.csv"); H = p[p.headline & p.comp.eq("int_ext")]; P.append(H)
    out[tag] = {"abstract_only": f(H[H.source.eq("abstract")]), "combined": f(H), "combined_minus_2_errors": f(H[~H.pmid.isin(BAD)])}
    if tag == "radiomics": out[tag]["fulltext_added"] = f(H[H.source.eq("fulltext")])
Q = pd.concat(P); out["pooled_minus_errors"] = f(Q[~Q.pmid.isin(BAD)].drop_duplicates("pmid"))
(ROOT / "results/09_expected_r_high.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
