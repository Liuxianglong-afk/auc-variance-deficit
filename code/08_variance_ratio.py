"""四个集合（放射组学 / 深度学习 × 仅摘要 / 摘要 + 全文）的未加权方差比 R = Var(d)/mean(v)（自助法 95% CI，5,000 次，seed 42），
以及按 √v 三分位的 R（三分位在原样本内固定，组内重抽研究）。敏感性：剔除人工核对中确认抽取错误的 2 对。输出 results/08_variance_ratio.json。
用法：python -I code/08_variance_ratio.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
BAD = {40596905, 41648663}
def stats(H, seed=42):
    rng = np.random.default_rng(seed)
    d = (H.auc_a - H.auc_ext).to_numpy(float); v = (((H.hi_a - H.lo_a) / 3.92) ** 2 + ((H.hi_ext - H.lo_ext) / 3.92) ** 2).to_numpy(float); k = len(d)
    R = float(np.var(d, ddof=1) / v.mean()); bs = [np.var(d[i], ddof=1) / v[i].mean() for i in (rng.integers(0, k, k) for _ in range(5000))]
    o = {"k": k, "mean_drop": round(float(d.mean()), 4), "R": round(R, 3), "ci95": [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])], "share_lt1": round(float(np.mean(np.array(bs) < 1)), 3)}
    if k >= 15:
        t = pd.qcut(np.sqrt(v), 3, labels=False)
        for g, nm in enumerate(("low_se", "mid_se", "high_se")):
            dg, vg = d[t == g], v[t == g]; kg = len(dg)
            b = [np.var(dg[i], ddof=1) / vg[i].mean() for i in (rng.integers(0, kg, kg) for _ in range(5000))]
            o[nm] = {"k": kg, "se_median": round(float(np.median(np.sqrt(vg))), 4), "R": round(float(np.var(dg, ddof=1) / vg.mean()), 3), "ci95": [round(float(x), 3) for x in np.percentile(b, [2.5, 97.5])], "share_lt1": round(float(np.mean(np.array(b) < 1)), 3)}
    return o
out = {}
for tag, d_ in (("radiomics", "data/radiomics_FT"), ("DL", "data/dl_FT")):
    p = pd.read_csv(ROOT / d_ / "pairs.csv"); H = p[p.headline & p.comp.eq("int_ext")]
    out[tag] = {"abstract_only": stats(H[H.source.eq("abstract")]), "fulltext_added": stats(H[H.source.eq("fulltext")]),
                "combined": stats(H), "combined_minus_2_errors": stats(H[~H.pmid.isin(BAD)])}
P = pd.concat([pd.read_csv(ROOT / d_ / "pairs.csv") for d_ in ("data/radiomics_FT", "data/dl_FT")]); P = P[P.headline & P.comp.eq("int_ext") & ~P.pmid.isin(BAD)].drop_duplicates("pmid")
out["pooled_both_samples_minus_errors"] = stats(P)
(ROOT / "results/08_variance_ratio.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
for k1, x in out.items():
    for k2, y in (x.items() if "k" not in x else [("", x)]):
        print(k1, k2, y["k"], "R", y["R"], y["ci95"], "| high", y.get("high_se", {}).get("R"), y.get("high_se", {}).get("ci95"), "| low", y.get("low_se", {}).get("R"))
