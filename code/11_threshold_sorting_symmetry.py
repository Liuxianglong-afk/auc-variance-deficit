"""(a) 外部 AUC 比内部高 ≥ 0.03 的研究数与比例。
(b) 排序效应模拟：三分位按估计的 SE 划分，SE 有误差时最不精确组偏向收进 SE 被高估的研究。以观测 v 为真值，d ~ N(均值, τ² + v)，
    SE_hat = SE·exp(σz − σ²/2)（内部、外部各自独立），按 SE_hat 分三分位后算无选择时的 R（2,000 次，seed 42）。
(c) 报告区间的对称性：|上半宽 − 下半宽| ≤ 0.01 视为对称。输出 results/11_threshold_sorting_symmetry.json。
用法：python -I code/11_threshold_sorting_symmetry.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
out = {}
def load(d):
    p = pd.read_csv(ROOT / d / "pairs.csv"); H = p[p.headline & p.comp.eq("int_ext")]
    se_i = ((H.hi_a - H.lo_a) / 3.92).to_numpy(float); se_e = ((H.hi_ext - H.lo_ext) / 3.92).to_numpy(float)
    return (H.auc_a - H.auc_ext).to_numpy(float), se_i, se_e
for tag, d_ in (("radiomics", "data/radiomics_FT"), ("DL", "data/dl_FT")):
    d, si, se = load(d_)
    out[f"a_{tag}"] = {"k": int(len(d)), "ext_ge_int": int((d <= 0).sum()), "ext_ge_int_plus_0.03": int((d <= -0.03).sum()),
                       "pct_ext_ge_int_plus_0.03": round(100 * float((d <= -0.03).mean()), 1)}
d, si, se = load("data/radiomics_FT")
rng = np.random.default_rng(42); k = len(d); mu = d.mean(); vt = si ** 2 + se ** 2
def R(dd, vv): return dd.var(ddof=1) / vv.mean()
sim = {}
for tau in (0.0, 0.029, 0.05):
    for sig in (0.0, 0.1, 0.2, 0.3, 0.5):
        acc = []
        for _ in range(2000):
            dd = mu + rng.normal(0, np.sqrt(tau ** 2 + vt))
            vh = (si * np.exp(sig * rng.normal(size=k) - sig ** 2 / 2)) ** 2 + (se * np.exp(sig * rng.normal(size=k) - sig ** 2 / 2)) ** 2
            o = np.argsort(np.sqrt(vh)); g = np.array_split(o, 3)
            acc.append([R(dd, vh)] + [R(dd[i], vh[i]) for i in g])
        a = np.array(acc); m = a.mean(0)
        sim[f"tau{tau}_sigma{sig}"] = {"R_all": round(m[0], 3), "R_low": round(m[1], 3), "R_mid": round(m[2], 3), "R_high": round(m[3], 3),
                                       "p_Rhigh_le_0.32": round(float((a[:, 3] <= 0.32).mean()), 4)}
out["b_sorting_artifact_radiomics"] = {"observed": {"R_all": 0.52, "R_high": 0.32}, "seed": 42, "reps": 2000, "sim": sim}
# (c) 报告区间的对称性：|上半宽 − 下半宽| ≤ 0.01 视为对称（AUC 尺度 Wald/DeLong）；下半宽 > 上半宽 + 0.01 为 logit/自助法式左偏
pp = pd.read_csv(ROOT / "data/radiomics_FT/pairs.csv"); H = pp[pp.headline & pp.comp.eq("int_ext")]
asym = np.concatenate([((H.hi_a - H.auc_a) - (H.auc_a - H.lo_a)).to_numpy(float), ((H.hi_ext - H.auc_ext) - (H.auc_ext - H.lo_ext)).to_numpy(float)])
out["c_ci_symmetry_radiomics"] = {"n_intervals": int(len(asym)), "pct_symmetric_le0.01": round(100 * float((np.abs(asym) <= 0.01).mean()), 1),
                                  "pct_lower_longer": round(100 * float((asym < -0.01).mean()), 1), "pct_upper_longer": round(100 * float((asym > 0.01).mean()), 1),
                                  "n_symmetric": int((np.abs(asym) <= 0.01).sum()), "n_lower_longer": int((asym < -0.01).sum()), "n_upper_longer": int((asym > 0.01).sum())}
p = ROOT / "results/11_threshold_sorting_symmetry.json"; p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
