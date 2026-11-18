"""补充样本：摘要报告了主模型内部与外部 AUC 点值、但未给 95% CI 的研究，从 PMC 开放全文（正文与表格）中查找同一数值的 95% CI。
摘要点值抽取沿用 02 的句子切分、AUC 指标词与队列标签规则，只是数值不要求后接 CI；丢弃 < 0.5 的值和紧跟 P 值的值。
输出 <DATA_DIR>/pairs_ft.csv、pairs_ft_stat.json；与 pairs.csv 合并后写入 <DATA_DIR>_FT/pairs.csv 与 auc_values.csv。
用法：python -I code/04_fulltext_ci.py"""
import json, os, re, sys, time, shutil, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; DIRN = os.environ.get("DATA_DIR", "data/radiomics"); W = ROOT / DIRN
A = json.load(open(W / "abstracts_all.json", encoding="utf-8"))
NUM = r"(0\.\d{2,4}|1\.0{1,4}|1)"
CI = r"\s*[\(\[]\s*(?:95\s*%\s*(?:CI|confidence intervals?)\s*(?:\[CI\]|\(CI\))?\s*[:=,]?\s*)?" + NUM + r"\s*(?:-|–|—|−|~|to|,)\s*" + NUM + r"\s*[\)\];,]"
VCI = re.compile(NUM + CI, re.I)
VAL = re.compile(r"(?<![\d.])" + r"(0\.\d{2,4}|1\.0{1,4})(?![\d])", re.I)
LAB = re.compile(r"(training|train|derivation|development|discovery|primary|modell?ing|internal(?:[- ]validation| test(?:ing)?| validation)?|validation|test(?:ing)?|external(?:[- ](?:validation|test(?:ing)?))?|independent(?: external)?(?: validation| test(?:ing)?)?|temporal(?: validation)?|prospective(?: validation)?)", re.I)
METRIC = re.compile(r"(AUCs?|AUROCs?|area under|ROC)|(sensitivit|specificit|accurac|precision|F1|NRI|IDI|C-?index|concordance|hazard|(?-i:\bHR\b)|odds|(?-i:\bOR\b)|kappa|ICC|Brier|PPV|NPV|recall|correlation|slope|intercept)", re.I)
META = re.compile(r"pooled|meta-analys|systematic review|summary receiver", re.I)
def cls(lab):
    l = lab.lower()
    if re.search(r"external|independent|temporal|prospective", l): return "ext"
    if re.search(r"train|derivation|development|discovery|primary|model", l): return "train"
    if re.search(r"internal|validation|test", l): return "int"
    return "unk"
def mcls(t):
    t = t.lower()
    if re.search(r"combin|nomogram|integrat|fusion|hybrid|joint|merged|ensemble|stack|clinical[- ]radiomic|radiomic[s]?[- ]clinical|clinic(al)?-?radiolog", t) and re.search(r"radiom|deep|dl|imag|signature|rad-?score|habitat|model", t): return "combined"
    if re.search(r"clinic", t) and not re.search(r"radiom|deep|\bdl|rad-?score|signature|habitat|imaging", t): return "clinical"
    return "imaging"
def point_values(s):
    out = []
    for v in VAL.finditer(s):
        a = float(v.group(1))
        if a < 0.5: continue
        if re.search(r"\b[Pp]\s*[<=>≤≥]\s*$", s[max(0, v.start() - 6): v.start()]): continue
        if re.search(r"(?:-|–|—|−|~|to)\s*$", s[max(0, v.start() - 4): v.start()]): continue   # CI 上限
        if re.search(r"[\(\[]\s*(?:95\s*%[^\d]{0,30})?$", s[max(0, v.start() - 40): v.start()]) and re.match(r"\s*(?:-|–|—|−|~|to|,)\s*" + NUM, s[v.end():]): continue   # CI 下限
        ms = list(METRIC.finditer(s[:v.start()]))
        if ms and ms[-1].group(1): out.append(v)
    return out
def label_values(s, vals):   # 同 02_extract_auc.py 的标签规则
    labs = [None] * len(vals)
    m = re.search(r"respectively", s[vals[0].end():], re.I)
    nb = sum(1 for v in vals if v.end() <= vals[0].end() + m.start()) if m else 0
    if m and nb >= 2:
        seg = s[vals[nb - 1].end(): vals[0].end() + m.start()]
        items = [x for x in re.split(r",\s*(?:and\s+)?|\s+and\s+|/", seg) if LAB.search(x)]
        if len(items) == nb: labs[:nb] = [LAB.search(x).group(0) for x in items]
        else:
            pre = s[:vals[0].start()]
            k = max(pre.lower().rfind(" in the "), pre.lower().rfind(" in "), pre.lower().rfind(" for the "), pre.lower().rfind(" across "))
            if k >= 0:
                items = [x for x in re.split(r",\s*(?:and\s+)?|\s+and\s+|/", pre[k:]) if LAB.search(x)]
                if len(items) == nb: labs[:nb] = [LAB.search(x).group(0) for x in items]
    lead = re.match(r"^\W*(?:RESULTS?:\s*)?(?:In|For|On|Within)\s+(?:the\s+)?(?:\w+\s+){0,2}?" + LAB.pattern + r"[^,]{0,40},", s, re.I)
    for j, v in enumerate(vals):
        if labs[j] is None:
            seg0 = s[max(0, (vals[j - 1].end() if j else 0)): v.start()]
            mb = re.search(r"\b(?:in|for|on)\s+(?:the\s+)?" + LAB.pattern + r"\s+(?:cohort|set|dataset|group|sample)s?\s*,?\s*(?:with|had|showed|yielded|achieved|reached|demonstrated)?\s*(?:an?\s+|the\s+)?(?:AUC|AUROC|area under the (?:ROC )?curve)\s*(?:of|=|was|:|value of)?\s*$", seg0, re.I)
            if mb: labs[j] = mb.group(1)
        if labs[j] is None:
            mc = re.search(LAB.pattern + r"\s*[:=]\s*(?:AUC\s*[:=]?\s*)?$", s[max(0, v.start() - 14): v.start()], re.I)
            if mc: labs[j] = mc.group(1)
        if labs[j] is None:
            vend = v.end(); mci = re.match(CI, s[vend:], re.I)
            if mci: vend += mci.end()
            mm = re.match(r"[^.,]{0,25}?\b(?:in|for|on|of)\s+(?:the\s+)?(?:\w+\s+){0,2}?" + LAB.pattern, s[vend: vend + 70], re.I)
            if mm: labs[j] = LAB.search(mm.group(0)).group(0)
        if labs[j] is None and lead: labs[j] = LAB.search(lead.group(0)).group(0)
        if labs[j] is None:
            before = s[max(0, (vals[j - 1].end() if j else 0)): v.start()]
            hits = [h for h in LAB.finditer(before) if re.search(r"(cohort|set|dataset|group|sample|centre|center)", before[h.end(): h.end() + 25], re.I)]
            if hits: labs[j] = hits[-1].group(0)
    return labs
old = pd.read_csv(W / "pairs.csv"); have = set(old[old.headline & old.comp.eq("int_ext")].pmid.astype(str))
cand = []
for pm, r in A.items():
    if pm in have or META.search(r["abstract"]): continue
    sents = re.split(r"(?<=[.;])\s+(?=[A-Z(])", r["abstract"]); found = {}
    for si, s in enumerate(sents):
        if re.search(r"macro|micro|multi-?class|ranged? from|ranging from", s, re.I): continue
        vals = point_values(s)
        if not vals: continue
        labs = label_values(s, vals); mc = mcls(s[:vals[0].start()][-160:] + " " + s[:200])
        ints = [float(v.group(1)) for v, l in zip(vals, labs) if l and cls(l) == "int"]; exts = [float(v.group(1)) for v, l in zip(vals, labs) if l and cls(l) == "ext"]
        if ints and exts and mc not in found: found[mc] = (si, ints[0], exts[0], s)
    if not found: continue
    mc = "combined" if "combined" in found else ("imaging" if "imaging" in found else None)
    if mc: si, ai, ae, s = found[mc]; cand.append({"pmid": pm, "pmcid": r["pmcid"], "sent": si, "model_cls": mc, "auc_a": ai, "auc_ext": ae, "abs_sent": s})
print("abstract candidates (point int+ext, no CI pair)", len(cand))
(W / "fulltext").mkdir(exist_ok=True)
def fulltext(pmc):
    if not pmc: return ""
    f = W / f"fulltext/{pmc}.xml"
    if not f.exists():
        for _ in range(2):
            try: f.write_bytes(urllib.request.urlopen(urllib.request.Request(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc[3:]}&retmode=xml", headers={"User-Agent": "Mozilla/5.0"}), timeout=90).read()); time.sleep(0.4); break
            except Exception: time.sleep(3)
        else: return ""
    try: x = ET.fromstring(f.read_bytes())
    except Exception: return ""
    parts = [e for tag in ("body", "floats-group") for e in x.iter(tag)]
    return re.sub(r"\s+", " ", " ".join(" ".join(p.itertext()) for p in parts))
rows, stat = [], {"no_text": 0, "int_val_eq_ext": 0, "ci_missing_or_ambiguous": 0, "accepted": 0}
for c in cand:
    if c["auc_a"] == c["auc_ext"]: stat["int_val_eq_ext"] += 1; continue
    t = fulltext(c["pmcid"])
    if not t: stat["no_text"] += 1; continue
    cis = {"a": set(), "ext": set()}
    for m in VCI.finditer(t):
        a, lo, hi = (float(x) for x in m.groups())
        if not (lo <= a <= hi and hi - lo < 0.6): continue
        for k_, val in (("a", c["auc_a"]), ("ext", c["auc_ext"])):
            if a == val: cis[k_].add((lo, hi))
    if len(cis["a"]) == 1 and len(cis["ext"]) == 1:
        (la, ha), = cis["a"]; (le, he), = cis["ext"]; stat["accepted"] += 1
        rows.append({**{k: c[k] for k in ("pmid", "sent", "model_cls")}, "comp": "int_ext", "auc_a": c["auc_a"], "lo_a": la, "hi_a": ha, "auc_ext": c["auc_ext"], "lo_ext": le, "hi_ext": he,
                     "n_ext_in_sent": 1, "model_txt": c["abs_sent"][:100], "headline": True, "source": "fulltext"})
    else: stat["ci_missing_or_ambiguous"] += 1
ft = pd.DataFrame(rows); ft.to_csv(W / "pairs_ft.csv", index=False, encoding="utf-8")
print(stat)
W2 = ROOT / (DIRN + "_FT"); W2.mkdir(exist_ok=True)
old["source"] = "abstract"; pd.concat([old, ft], ignore_index=True).to_csv(W2 / "pairs.csv", index=False, encoding="utf-8")
shutil.copy(W / "auc_values.csv", W2 / "auc_values.csv")
json.dump(stat | {"candidates": len(cand), "k_abstract_headline": len(have)}, open(W / "pairs_ft_stat.json", "w", encoding="utf-8"), ensure_ascii=False)
print("merged headline int_ext:", len(have), "+", len(ft), "->", W2)
