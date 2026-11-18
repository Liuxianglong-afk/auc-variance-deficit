"""同一研究报告 ≥ 2 个模型配对时，比较选中模型（内部 AUC 最高者；另做外部 AUC 最高者版本）与其余对照模型的方差比 R
（研究层面自助法 5,000 次，seed 42），以及研究内选中模型 |z| 是否小于对照 |z| 中位数（符号检验）。输出 results/07_selected_vs_comparator.json。
用法：python -I code/07_selected_vs_comparator.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; rng = np.random.default_rng(42)
rows = []
for tag, d_ in (("radiomics", "data/radiomics"), ("DL", "data/dl")):
    v = pd.read_csv(ROOT / d_ / "auc_values.csv"); v = v[v.ci_ok]
    for (pm, si), g in v.groupby(["pmid", "sent"]):
        a, e = g[g.cohort.eq("int")], g[g.cohort.eq("ext")]
        if len(a) and len(e):
            x, y = a.iloc[0], e.iloc[0]
            rows.append({"sample": tag, "pmid": pm, "sent": si, "auc_int": x.auc, "auc_ext": y.auc,
                         "v": ((x.hi - x.lo) / 3.92) ** 2 + ((y.hi - y.lo) / 3.92) ** 2})
m = pd.DataFrame(rows).drop_duplicates(["sample", "pmid", "auc_int", "auc_ext"])
m = m[m.groupby(["sample", "pmid"]).pmid.transform("size") >= 2].copy()
m["d"] = m.auc_int - m.auc_ext; m["z"] = m.d / np.sqrt(m.v)
def R(s): return float(np.var(s.d, ddof=1) / s.v.mean())
def bootR(s):
    u = s.pmid.unique(); g = {k: s[s.pmid == k] for k in u}
    bs = [R(pd.concat([g[k] for k in rng.choice(u, len(u))])) for _ in range(5000)]
    return [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])]
out = {}
for crit in ("auc_int", "auc_ext"):
    m["selected"] = m.groupby(["sample", "pmid"])[crit].transform("max").eq(m[crit]) & ~m.duplicated(["sample", "pmid", crit])
    for tag in ("all", "radiomics", "DL"):
        s = m if tag == "all" else m[m["sample"].eq(tag)]
        if s.pmid.nunique() < 5: continue
        sel, com = s[s.selected], s[~s.selected]
        wins = [(abs(g[g.selected].z.iloc[0]) < np.median(abs(g[~g.selected].z))) for _, g in s.groupby("pmid")]
        bt = stats.binomtest(int(sum(wins)), len(wins), 0.5)
        out[f"{crit}_{tag}"] = {"k_studies": int(s.pmid.nunique()), "k_comparator_pairs": int(len(com)),
            "R_selected": round(R(sel), 3), "R_selected_ci95": bootR(sel), "R_comparators": round(R(com), 3), "R_comparators_ci95": bootR(com),
            "mean_drop_selected": round(float(sel.d.mean()), 4), "mean_drop_comparators": round(float(com.d.mean()), 4),
            "share_selected_absz_below_comparator_median": round(float(np.mean(wins)), 3), "sign_test_p": float(f"{bt.pvalue:.3g}")}
out["note"] = "model pairs = sentences with both int and ext AUC (+95% CI); selected = max within study by the named criterion"
(ROOT / "results/07_selected_vs_comparator.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
for k, x in out.items():
    if isinstance(x, dict): print(k, x)
