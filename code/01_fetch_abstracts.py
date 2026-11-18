"""取检索池（pool_pmids.json）全部 PubMed 记录的题名、期刊、年份与摘要（efetch）。
输出 <DATA_DIR>/abstracts_all.json。
用法：python -I code/01_fetch_abstracts.py"""
# 环境变量：DATA_DIR 数据目录（默认 data/radiomics）、RESULT_TAG 结果文件后缀、POOL_FILE 检索池 json。深度学习样本：DATA_DIR=data/dl POOL_FILE=data/dl/pool_pmids.json RESULT_TAG=_DL
import json, os, sys, time, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; W = ROOT / os.environ.get("DATA_DIR", "data/radiomics"); W.mkdir(parents=True, exist_ok=True)
pool = json.load(open(ROOT / os.environ.get("POOL_FILE", "data/radiomics/pool_pmids.json"), encoding="utf-8"))["pmids"]
recs = {}
for i in range(0, len(pool), 200):
    for _ in range(3):
        try:
            x = urllib.request.urlopen(urllib.request.Request("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id=" + ",".join(pool[i:i+200]), headers={"User-Agent": "Mozilla/5.0"}), timeout=180).read(); break
        except Exception: time.sleep(5)
    for a in ET.fromstring(x).iter("PubmedArticle"):
        pm = a.findtext(".//PMID")
        recs[pm] = {"pmid": pm, "pmcid": next((e.text for e in a.iter("ArticleId") if e.get("IdType") == "pmc"), None),
                    "year": a.findtext(".//PubDate/Year") or (a.findtext(".//PubDate/MedlineDate") or "")[:4],
                    "journal": a.findtext(".//Journal/ISOAbbreviation"), "title": "".join(a.find(".//ArticleTitle").itertext()),
                    "abstract": " ".join(((t.get("Label") + ": ") if t.get("Label") else "") + "".join(t.itertext()) for t in a.iter("AbstractText"))}
    time.sleep(0.5)
(W / "abstracts_all.json").write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
print("fetched", len(recs), "of", len(pool))
