"""离散度分析。d_i = AUC_int − AUC_ext；SE 由报告的 95% CI 反推 = (上限 − 下限)/3.92；v_i = SE_int² + SE_ext²。
Cochran's Q = Σ (d_i − d̄_w)²/v_i（w = 1/v_i），无选择时 E[Q] ≥ k − 1；报告 H² = Q/(k − 1)、χ²(k − 1) 下尾 P；
另报告以 0 为中心的 Q0、logit 尺度、研究层面自助法 H² 95% CI（5,000 次，seed 42）与 d 对 √v 的加权回归斜率。
对照：训练–外部、clinical 模型、全部模型。输出 results/05_dispersion<RESULT_TAG>.json（全文合并样本用 DATA_DIR=data/radiomics_FT RESULT_TAG=_FT，深度学习用 data/dl_FT 与 _DL_FT）。
用法：python -I code/05_dispersion.py"""
# 环境变量：DATA_DIR 数据目录（默认 data/radiomics）、RESULT_TAG 结果文件后缀、POOL_FILE 检索池 json。深度学习样本：DATA_DIR=data/dl POOL_FILE=data/dl/pool_pmids.json RESULT_TAG=_DL
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; W = ROOT / os.environ.get("DATA_DIR", "data/radiomics")
p = pd.read_csv(W / "pairs.csv"); v = pd.read_csv(W / "auc_values.csv")
lab = v[v.cohort.eq("int")].groupby(["pmid", "sent"]).label.first()
p["int_label"] = [lab.get((a, b), "") for a, b in zip(p.pmid, p.sent)]
rng = np.random.default_rng(42)
def Q(d, vv):
    w = 1 / vv; m = np.sum(w * d) / np.sum(w); return float(np.sum(w * (d - m) ** 2)), float(m)
def run(s, logit=False, boot=True):
    a, e = s.auc_a.to_numpy(float), s.auc_ext.to_numpy(float)
    sa, se = (s.hi_a - s.lo_a).to_numpy(float) / 3.92, (s.hi_ext - s.lo_ext).to_numpy(float) / 3.92
    if logit:
        a_, e_ = np.clip(a, .001, .999), np.clip(e, .001, .999)
        sa, se = sa / (a_ * (1 - a_)), se / (e_ * (1 - e_)); a, e = np.log(a_ / (1 - a_)), np.log(e_ / (1 - e_))
    d, vv = a - e, sa ** 2 + se ** 2; k = len(d)
    q, m = Q(d, vv); q0 = float(np.sum(d ** 2 / vv))
    eg = np.polyfit(np.sqrt(vv), d, 1, w=1 / np.sqrt(vv))[0]
    out = {"k_pairs": k, "k_studies": int(s.pmid.nunique()), "mean_drop": round(float(d.mean()), 4), "weighted_mean_drop": round(m, 4),
           "observed_sd_drop": round(float(d.std(ddof=1)), 4), "rms_sampling_sd": round(float(np.sqrt(vv.mean())), 4),
           "Q": round(q, 1), "df": k - 1, "H2": round(q / (k - 1), 3), "P_lower": float(f"{stats.chi2.cdf(q, k - 1):.3g}"),
           "Q0_centered_zero": round(q0, 1), "P0_lower": float(f"{stats.chi2.cdf(q0, k):.3g}"), "egger_slope_d_on_se": round(float(eg), 3),
           "share_ext_ge_a": round(float(np.mean(e >= a)), 3)}
    if boot:
        pm = s.pmid.to_numpy(); u = np.unique(pm); hs = []
        for _ in range(5000):
            pick = rng.choice(u, len(u)); idx = np.concatenate([np.where(pm == x)[0] for x in pick])
            hs.append(Q(d[idx], vv[idx])[0] / (len(idx) - 1))
        out["H2_boot_ci95"] = [round(float(x), 3) for x in np.percentile(hs, [2.5, 97.5])]
    return out
H = p[p.headline & p.comp.eq("int_ext")]
res = {"primary_headline_int_ext": run(H), "primary_logit": run(H, logit=True, boot=False),
       "control_train_ext_headline": run(p[p.headline & p.comp.eq("train_ext")]),
       "control_clinical_int_ext": run(p[p.model_cls.eq("clinical") & p.comp.eq("int_ext")], boot=False) if (p.model_cls.eq("clinical") & p.comp.eq("int_ext")).sum() >= 3 else "k < 3",
       "all_models_int_ext": run(p[p.comp.eq("int_ext")])}
hi = (H[["auc_a", "auc_ext"]].max(axis=1) >= .95) | (H.hi_a >= .999) | (H.hi_ext >= .999)
res["sens_no_ceiling"] = run(H[~hi], boot=False)
res["sens_int_label_internal_or_test"] = run(H[H.int_label.str.contains("internal|test", case=False)], boot=False)
both = p[p.comp.eq("int_ext") & p.headline].merge(p[p.comp.eq("train_ext") & p.headline][["pmid", "auc_a", "lo_a", "hi_a"]], on="pmid", suffixes=("", "_tr"))
res["same_studies_train_vs_int"] = {"k": int(len(both)),
    "int_ext": run(both, boot=False),
    "train_ext": run(both.assign(auc_a=both.auc_a_tr, lo_a=both.lo_a_tr, hi_a=both.hi_a_tr), boot=False)}
res["extraction_precision"] = {"tuning_set": "38/39 after tuning (verify40_gold.csv; 1 multiclass removed)", "independent_fresh": "28/30 before final fix (verify30_fresh_ids.csv)"}
res["note"] = f"data {W.relative_to(ROOT).as_posix()}; PubMed pool (2020-2026, PMC); rule-based abstract extraction requiring 95% CI; one headline model per study (combined > imaging), first external cohort"
(ROOT / ("results/05_dispersion" + os.environ.get("RESULT_TAG", "") + ".json")).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
for k, x in res.items():
    if isinstance(x, dict) and "H2" in x: print(k, {kk: x[kk] for kk in ("k_pairs", "mean_drop", "observed_sd_drop", "rms_sampling_sd", "H2", "P_lower", "egger_slope_d_on_se") if kk in x}, x.get("H2_boot_ci95", ""))
for kk in ("int_ext", "train_ext"): x = res["same_studies_train_vs_int"][kk]; print("same_studies", kk, x["k_pairs"], x["H2"], x["P_lower"])
