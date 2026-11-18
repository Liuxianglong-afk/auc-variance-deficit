"""按固定规则从摘要中抽取带 95% CI 的 AUC 及其队列标签。
句式 A：'… AUCs of v1 (CI), v2 (CI) and v3 (CI) in the training, internal validation and external validation cohorts, respectively'，值个数等于队列标签个数时按顺序配对。
句式 B：'… v (CI) in the external validation cohort' / 'In the external validation cohort, … AUC of v (CI)'，取值后 60 字符内或同句值前的队列标签。
每个值记录：句号、模型标签文本（句首到第一个值）、队列类别（train / int / ext / unk）、模型类别（clinical / imaging / combined，按关键词判定）。
输出 <DATA_DIR>/auc_values.csv。
用法：python -I code/02_extract_auc.py"""
# 环境变量：DATA_DIR 数据目录（默认 data/radiomics）、RESULT_TAG 结果文件后缀、POOL_FILE 检索池 json。深度学习样本：DATA_DIR=data/dl POOL_FILE=data/dl/pool_pmids.json RESULT_TAG=_DL
import json, os, re, sys
from pathlib import Path
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
W = Path(__file__).resolve().parents[1] / os.environ.get("DATA_DIR", "data/radiomics")
R = json.load(open(W / "abstracts_all.json", encoding="utf-8"))
NUM = r"(0\.\d{2,4}|1\.0{1,4}|1)"
VCI = re.compile(NUM + r"\s*[\(\[]\s*(?:95\s*%\s*(?:CI|confidence intervals?)\s*(?:\[CI\]|\(CI\))?\s*[:=,]?\s*)?" + NUM + r"\s*(?:-|–|—|−|~|to|,)\s*" + NUM + r"\s*[\)\];,]", re.I)
LAB = re.compile(r"(training|train|derivation|development|discovery|primary|modell?ing|internal(?:[- ]validation| test(?:ing)?| validation)?|validation|test(?:ing)?|external(?:[- ](?:validation|test(?:ing)?))?|independent(?: external)?(?: validation| test(?:ing)?)?|temporal(?: validation)?|prospective(?: validation)?)", re.I)
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
METRIC = re.compile(r"(AUCs?|AUROCs?|area under|ROC)|(sensitivit|specificit|accurac|precision|F1|NRI|IDI|C-?index|concordance|hazard|(?-i:\bHR\b)|odds|(?-i:\bOR\b)|kappa|ICC|Brier|PPV|NPV|recall|correlation|slope|intercept)", re.I)
META = re.compile(r"pooled|meta-analys|systematic review|summary receiver", re.I)
rows = []
for pm, r in R.items():
    if META.search(r["abstract"]): continue
    sents = re.split(r"(?<=[.;])\s+(?=[A-Z(])", r["abstract"])
    for si, s in enumerate(sents):
        if re.search(r"macro|micro|multi-?class|ranged? from|ranging from", s, re.I): continue
        vals = []
        for v in VCI.finditer(s):
            ms = list(METRIC.finditer(s[:v.start()]))
            if ms and ms[-1].group(1): vals.append(v)
        if not vals: continue
        model_txt = s[:vals[0].start()][-160:]
        labs = [None] * len(vals)
        m = re.search(r"respectively", s[vals[0].end():], re.I)
        nb = sum(1 for v in vals if v.end() <= vals[0].end() + m.start()) if m else 0   # 'respectively' 只作用于其前面的值
        if m and nb >= 2:
            seg = s[vals[nb - 1].end(): vals[0].end() + m.start()]
            items = [x for x in re.split(r",\s*(?:and\s+)?|\s+and\s+|/", seg) if LAB.search(x)]
            if len(items) == nb: labs[:nb] = [LAB.search(x).group(0) for x in items]
            else:   # 标签列在数值之前
                pre = s[:vals[0].start()]
                k = max(pre.lower().rfind(" in the "), pre.lower().rfind(" in "), pre.lower().rfind(" for the "), pre.lower().rfind(" across "))
                if k >= 0:
                    items = [x for x in re.split(r",\s*(?:and\s+)?|\s+and\s+|/", pre[k:]) if LAB.search(x)]
                    if len(items) == nb: labs[:nb] = [LAB.search(x).group(0) for x in items]
        lead = re.match(r"^\W*(?:RESULTS?:\s*)?(?:In|For|On|Within)\s+(?:the\s+)?(?:\w+\s+){0,2}?" + LAB.pattern + r"[^,]{0,40},", s, re.I)
        for j, v in enumerate(vals):
            if labs[j] is None:   # 'in the X cohort with an AUC of v' 优先
                seg0 = s[max(0, (vals[j - 1].end() if j else 0)): v.start()]
                mb = re.search(r"\b(?:in|for|on)\s+(?:the\s+)?" + LAB.pattern + r"\s+(?:cohort|set|dataset|group|sample)s?\s*,?\s*(?:with|had|showed|yielded|achieved|reached|demonstrated)?\s*(?:an?\s+|the\s+)?(?:AUC|AUROC|area under the (?:ROC )?curve)\s*(?:of|=|was|:|value of)?\s*$", seg0, re.I)
                if mb: labs[j] = mb.group(1)
            if labs[j] is None:
                pre20 = s[max(0, v.start() - 14): v.start()]
                mc = re.search(LAB.pattern + r"\s*[:=]\s*(?:AUC\s*[:=]?\s*)?$", pre20, re.I)
                if mc: labs[j] = mc.group(1)
            if labs[j] is None:
                after = s[v.end(): v.end() + 70]
                mm = re.match(r"[^.,]{0,25}?\b(?:in|for|on|of)\s+(?:the\s+)?(?:\w+\s+){0,2}?" + LAB.pattern, after, re.I)
                if mm: labs[j] = LAB.search(mm.group(0)).group(0)
            if labs[j] is None and lead: labs[j] = LAB.search(lead.group(0)).group(0)
            if labs[j] is None:
                before = s[max(0, (vals[j - 1].end() if j else 0)): v.start()]
                hits = [h for h in LAB.finditer(before) if re.search(r"(cohort|set|dataset|group|sample|centre|center)", before[h.end(): h.end() + 25], re.I)]
                if hits: labs[j] = hits[-1].group(0)
            a, lo, hi = (float(x) for x in v.groups())
            ok = lo <= a <= hi and hi - lo < 0.6
            rows.append({"pmid": pm, "sent": si, "pos": j, "auc": a, "lo": lo, "hi": hi, "ci_ok": ok, "label": labs[j] or "", "cohort": cls(labs[j]) if labs[j] else "unk",
                         "model_txt": model_txt.strip(), "model_cls": mcls(model_txt + " " + s[:200])})
d = pd.DataFrame(rows); d.to_csv(W / "auc_values.csv", index=False, encoding="utf-8")
print("values", len(d), "studies", d.pmid.nunique(), "| cohort counts", d.cohort.value_counts().to_dict(), "| ci_ok", int(d.ci_ok.sum()))
