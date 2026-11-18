"""放射组学全文合并样本（k = 181）未加权 R 的敏感性分析：(a) 去掉内部或外部 AUC ≥ 0.95；(b) logit 尺度（SE/(a(1 − a))）；
(c) 训练–外部对照。自助法 5,000 次，seed 42，三分位方法同 08。输出 results/10_sensitivity.json。
用法：python -I code/10_sensitivity.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
def stats(d, v):
    rng = np.random.default_rng(42); k = len(d); o = {"k": k, "R": round(float(np.var(d, ddof=1) / v.mean()), 3)}
    o["ci95"] = [round(float(x), 3) for x in np.percentile([np.var(d[i], ddof=1) / v[i].mean() for i in (rng.integers(0, k, k) for _ in range(5000))], [2.5, 97.5])]
    t = pd.qcut(np.sqrt(v), 3, labels=False)
    for g, nm in enumerate(("low_se", "mid_se", "high_se")):
        dg, vg = d[t == g], v[t == g]; kg = len(dg)
        o[nm] = {"k": kg, "R": round(float(np.var(dg, ddof=1) / vg.mean()), 3), "ci95": [round(float(x), 3) for x in np.percentile([np.var(dg[i], ddof=1) / vg[i].mean() for i in (rng.integers(0, kg, kg) for _ in range(5000))], [2.5, 97.5])]}
    return o
p = pd.read_csv(ROOT / "data/radiomics_FT/pairs.csv"); out = {}
for nm, H in (("no_ceiling", None), ("logit", None), ("train_ext_control", None)):
    if nm == "train_ext_control": H = p[p.headline & p.comp.eq("train_ext")]
    else: H = p[p.headline & p.comp.eq("int_ext")]
    if nm == "no_ceiling": H = H[(H.auc_a < 0.95) & (H.auc_ext < 0.95)]
    a, e = H.auc_a.to_numpy(float), H.auc_ext.to_numpy(float); sa, se = ((H.hi_a - H.lo_a) / 3.92).to_numpy(float), ((H.hi_ext - H.lo_ext) / 3.92).to_numpy(float)
    if nm == "logit":
        ok = (a < 1) & (e < 1); a, e, sa, se = a[ok], e[ok], sa[ok], se[ok]
        d = np.log(a / (1 - a)) - np.log(e / (1 - e)); v = (sa / (a * (1 - a))) ** 2 + (se / (e * (1 - e))) ** 2
    else: d = a - e; v = sa ** 2 + se ** 2
    out[nm] = stats(d, v)
(ROOT / "results/10_sensitivity.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
for k, x in out.items(): print(k, x["k"], x["R"], x["ci95"], "| tertiles", [(x[t]["R"], x[t]["ci95"]) for t in ("low_se", "mid_se", "high_se")])
