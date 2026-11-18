"""检查报告的 CI 是否宽于样本量所能解释的宽度。从摘要（无则 PMC 全文）抽取内部与外部队列人数，
计算 Hanley–McNeil 类别均衡 SE（给定 AUC 与 n 时可能的最小 SE），报告 SE_CI / SE_HMmin 的分布，并用 HM 最小 SE 替代报告 CI 重算方差比。
输出 <DATA_DIR>/pairs_with_n.csv 与 results/06_ci_vs_n<RESULT_TAG>.json。需要 01 的 abstracts_all.json。
用法：python -I code/06_ci_vs_n.py"""
# 环境变量：DATA_DIR 数据目录（默认 data/radiomics）、RESULT_TAG 结果文件后缀、POOL_FILE 检索池 json。深度学习样本：DATA_DIR=data/dl POOL_FILE=data/dl/pool_pmids.json RESULT_TAG=_DL
import json, os, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; W = ROOT / os.environ.get("DATA_DIR", "data/radiomics")
A = json.load(open(W / "abstracts_all.json", encoding="utf-8")); p = pd.read_csv(W / "pairs.csv")
H = p[p.headline & p.comp.eq("int_ext")].copy()
INT = r"(?:internal(?:[- ]validation| test(?:ing)?)?|validation|test(?:ing)?)"; EXT = r"(?:external(?:[- ](?:validation|test(?:ing)?))?|independent(?: external)?(?: validation| test(?:ing)?)?)"
UNIT = r"(?:cohort|set|dataset|group|sample)s?"
def find_n(t, L):
    pats = [L + r"\s+" + UNIT + r"\s*(?:[\(,:]|\[(?=\s*[nN]\s*=))\s*(?:n\s*=\s*|N\s*=\s*)?(\d{2,5})\b(?!\.\d)(?!\s*(?:(?:patients|cases|lesions|participants|subjects)\s+)?(?:with|had|achieved|showed|developed|were|was|of whom|experienced|\]|%))",
            r"(?<![\[,]\s)\b(\d{2,5})\s+(?:patients|cases|participants|subjects|lesions|nodules|women|men|tumou?rs)\s+(?:were\s+)?(?:in|for|as|formed|constituted|comprised)\s+(?:the\s+|an?\s+)?" + L]
    for q in pats:
        m = re.search(q, t, re.I)
        if m:
            n = int(m.group(1))
            if 10 <= n <= 20000: return n
    return None
def hm(a, n):
    n1 = n2 = n / 2; q1 = a / (2 - a); q2 = 2 * a * a / (1 + a)
    return np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2))
import time, urllib.request, xml.etree.ElementTree as ET
(W / "fulltext").mkdir(exist_ok=True)
def body(pm):
    pmc = A[str(pm)]["pmcid"]
    if not pmc: return ""
    f = W / f"fulltext/{pmc}.xml"
    if not f.exists():
        try: f.write_bytes(urllib.request.urlopen(urllib.request.Request(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc[3:]}&retmode=xml", headers={"User-Agent": "Mozilla/5.0"}), timeout=90).read()); time.sleep(0.4)
        except Exception: return ""
    try: b = ET.fromstring(f.read_bytes()).find(".//body")
    except Exception: return ""
    return re.sub(r"\s+", " ", " ".join(b.itertext())) if b is not None else ""
src = []
for c_ in ("int", "ext"):
    L = r"(?<!external )" + INT if c_ == "int" else EXT
    vals, srcs = [], []
    for x in H.pmid:
        n = find_n(A[str(x)]["abstract"], L); s_ = "abstract"
        if n is None: n = find_n(body(x), L); s_ = "fulltext" if n else ""
        vals.append(n); srcs.append(s_)
    H["n_" + c_] = vals; H["n_" + c_ + "_src"] = srcs
H.to_csv(W / "pairs_with_n.csv", index=False, encoding="utf-8")
c = H.dropna(subset=["n_int", "n_ext"]).copy()
c["se_ci_int"], c["se_ci_ext"] = (c.hi_a - c.lo_a) / 3.92, (c.hi_ext - c.lo_ext) / 3.92
c["se_hm_int"], c["se_hm_ext"] = hm(c.auc_a, c.n_int), hm(c.auc_ext, c.n_ext)
ratio = pd.concat([c.se_ci_int / c.se_hm_int, c.se_ci_ext / c.se_hm_ext])
d = (c.auc_a - c.auc_ext).to_numpy(float)
v_ci = (c.se_ci_int ** 2 + c.se_ci_ext ** 2).to_numpy(float); v_hm = (c.se_hm_int ** 2 + c.se_hm_ext ** 2).to_numpy(float)
rng = np.random.default_rng(42); k = len(d)
bs = [np.var(d[i], ddof=1) / v_hm[i].mean() for i in (rng.integers(0, k, k) for _ in range(5000))]
out = {"k_with_both_n": int(k), "k_headline": int(len(H)),
       "se_ratio_ci_over_hmmin": {"median": round(float(ratio.median()), 2), "q25_q75": [round(float(x), 2) for x in ratio.quantile([.25, .75])], "share_lt_0.9": round(float((ratio < .9).mean()), 3)},
       "R_reported_ci": round(float(np.var(d, ddof=1) / v_ci.mean()), 3), "R_hm_min_upper_bound": round(float(np.var(d, ddof=1) / v_hm.mean()), 3),
       "R_hm_ci95": [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])], "share_boot_R_hm_lt1": round(float(np.mean(np.array(bs) < 1)), 3),
       "n_int_median": float(c.n_int.median()), "n_ext_median": float(c.n_ext.median())}
(ROOT / ("results/06_ci_vs_n" + os.environ.get("RESULT_TAG", "") + ".json")).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
print(c[["pmid", "auc_a", "n_int", "auc_ext", "n_ext"]].head(12).to_string())
