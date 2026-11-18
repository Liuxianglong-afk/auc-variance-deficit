"""由 auc_values.csv 组成研究级配对：同一句内（同一模型）取第一个内部值与第一个外部值；训练–外部同理。
主模型：combined 优先，其次 imaging；clinical 单列为对照。每篇每个模型类别只取第一个满足条件的句子，要求 CI 完整。
输出 <DATA_DIR>/pairs.csv。
用法：python -I code/03_build_pairs.py"""
# 环境变量：DATA_DIR 数据目录（默认 data/radiomics）、RESULT_TAG 结果文件后缀、POOL_FILE 检索池 json。深度学习样本：DATA_DIR=data/dl POOL_FILE=data/dl/pool_pmids.json RESULT_TAG=_DL
import os, sys
from pathlib import Path
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; W = ROOT / os.environ.get("DATA_DIR", "data/radiomics")
v = pd.read_csv(W / "auc_values.csv"); v = v[v.ci_ok]
rows = []
for (pm, si), g in v.groupby(["pmid", "sent"], sort=True):
    mc = g.model_cls.iloc[0]
    for a, b in (("int", "ext"), ("train", "ext")):
        ga, gb = g[g.cohort.eq(a)], g[g.cohort.eq(b)]
        if len(ga) and len(gb):
            x, y = ga.iloc[0], gb.iloc[0]
            rows.append({"pmid": pm, "sent": si, "model_cls": mc, "comp": f"{a}_ext", "auc_a": x.auc, "lo_a": x.lo, "hi_a": x.hi, "auc_ext": y.auc, "lo_ext": y.lo, "hi_ext": y.hi,
                         "n_ext_in_sent": len(gb), "model_txt": g.model_txt.iloc[0][-100:]})
p = pd.DataFrame(rows)
p = p.sort_values(["pmid", "comp", "model_cls", "sent"]).drop_duplicates(["pmid", "comp", "model_cls"])
p["headline"] = False
for (pm, comp), g in p[p.model_cls.ne("clinical")].groupby(["pmid", "comp"]):
    i = g[g.model_cls.eq("combined")].index if (g.model_cls == "combined").any() else g.index
    p.loc[i[0], "headline"] = True
p.to_csv(W / "pairs.csv", index=False, encoding="utf-8")
print("pairs", len(p), "| headline int_ext", int((p.headline & p.comp.eq("int_ext")).sum()), "| headline train_ext", int((p.headline & p.comp.eq("train_ext")).sum()),
      "| clinical int_ext", int((p.model_cls.eq("clinical") & p.comp.eq("int_ext")).sum()))
MANUAL = ROOT / "data/manual_coding.csv"
if MANUAL.exists():
    m = pd.read_csv(MANUAL); m = m[m.status.eq("coded")]
    h = p[p.headline & p.comp.eq("int_ext")].set_index("pmid")
    chk = []
    for _, r in m.iterrows():
        if pd.isna(r.auc_int) or pd.isna(r.auc_ext) or "mean of" in str(r.note): continue
        if r.pmid in h.index:
            e = h.loc[r.pmid]; ok = abs(e.auc_a - r.auc_int) <= .0015 and abs(e.auc_ext - r.auc_ext) <= .0015
            chk.append({"pmid": r.pmid, "manual_int": r.auc_int, "manual_ext": r.auc_ext, "auto_int": e.auc_a, "auto_ext": e.auc_ext, "match": ok})
        else: chk.append({"pmid": r.pmid, "manual_int": r.auc_int, "manual_ext": r.auc_ext, "auto_int": None, "auto_ext": None, "match": None})
    c = pd.DataFrame(chk); c.to_csv(W / "extraction_check_vs_manual.csv", index=False, encoding="utf-8")
    print("manual comparable", len(c), "| auto found", int(c.match.notna().sum()), "| match", int((c.match == True).sum()))
    print(c[c.match == False].to_string())
