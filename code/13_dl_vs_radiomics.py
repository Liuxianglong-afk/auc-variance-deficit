"""放射组学与深度学习两样本对比（深度学习样本整体 R 高于 1 的来源）。两样本用同一方法：
(1) 描述：k、差值均值与中位数、SE_d 中位数（IQR）、外部 ≥ 内部的比例；
(2) 异质性：最精确三分位估计 τ = sqrt((R_low − 1)·mean v_low)（截 0）；
(3) 极端值影响：去掉 |d| > 0.3 的研究后的整体 R；
(4) 最不精确三分位观测 R ÷ 仅有异质性时的预期 R（1 + τ²/mean v_high），自助法 95% CI（5,000 次，seed 42，三分位成员固定，组内有放回重抽，与 09 相同）；
敏感性：去掉 |d| > 0.3 的研究后重新分三分位并重算 (2)–(4)。
输出 results/13_dl_vs_radiomics.json。用法：python -I code/13_dl_vs_radiomics.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; out = {}
def load(d):
    p = pd.read_csv(ROOT / d / "pairs.csv"); H = p[p.headline & p.comp.eq("int_ext")]
    d_ = (H.auc_a - H.auc_ext).to_numpy(float); v = (((H.hi_a - H.lo_a) / 3.92) ** 2 + ((H.hi_ext - H.lo_ext) / 3.92) ** 2).to_numpy(float); return d_, v
def R(d, v): return d.var(ddof=1) / v.mean()
def parts(d, v, g):
    lo, hi = g[0], g[2]; Rl = R(d[lo], v[lo]); tau2 = max(Rl - 1, 0) * v[lo].mean(); exp_h = 1 + tau2 / v[hi].mean(); Rh = R(d[hi], v[hi])
    return Rl, tau2, exp_h, Rh
for tag, dir_ in (("radiomics", "data/radiomics_FT"), ("DL", "data/dl_FT")):
    d, v = load(dir_); se = np.sqrt(v); g = np.array_split(np.argsort(se, kind="stable"), 3)
    # 三分位划分与 08 一致：pd.qcut
    t = pd.qcut(se, 3, labels=False); g = [np.where(t == i)[0] for i in range(3)]
    Rl, tau2, exp_h, Rh = parts(d, v, g)
    rng = np.random.default_rng(42); bs = []
    for _ in range(5000):
        gb = [rng.choice(ix, len(ix), replace=True) for ix in g]
        a = parts(d, v, gb); bs.append(a[3] / a[2])
    bs = np.array(bs); keep = np.abs(d) <= 0.3
    q = lambda x: [round(float(y), 3) for y in np.percentile(x, [50, 25, 75])]
    out[tag] = {"k": int(len(d)), "mean_d": round(float(d.mean()), 4), "median_d_iqr": q(d), "se_d_median_iqr": q(se),
                "share_ext_ge_int": round(float((d <= 0).mean()), 3), "R_all": round(float(R(d, v)), 3),
                "n_abs_d_gt_0.3": int((~keep).sum()), "R_all_excl_abs_d_gt_0.3": round(float(R(d[keep], v[keep])), 3),
                "R_low": round(float(Rl), 3), "tau": round(float(np.sqrt(tau2)), 4), "expected_R_high": round(float(exp_h), 3), "R_high": round(float(Rh), 3),
                "obs_over_expected_R_high": round(float(Rh / exp_h), 3), "obs_over_expected_ci95": [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])],
                "share_boot_ratio_lt1": round(float((bs < 1).mean()), 4), "tertile_sizes": [int(len(x)) for x in g]}
out["note"] = "tertiles by pd.qcut of SE_d as in 08_variance_ratio.py; bootstrap 5,000, seed 42, tertile membership fixed"
(ROOT / "results/13_dl_vs_radiomics.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))

# 敏感性：去掉 |d| > 0.3 的研究后重新分三分位，重算 τ、预期值与观测/预期比（深度学习 1 项，差值 0.41，位于最精确三分位）
sens = {}
for tag, dir_ in (("radiomics", "data/radiomics_FT"), ("DL", "data/dl_FT")):
    d, v = load(dir_); keep = np.abs(d) <= 0.3; d, v = d[keep], v[keep]; se = np.sqrt(v)
    t = pd.qcut(se, 3, labels=False); g = [np.where(t == i)[0] for i in range(3)]
    Rl, tau2, exp_h, Rh = parts(d, v, g); rng = np.random.default_rng(42); bs = []
    for _ in range(5000):
        gb = [rng.choice(ix, len(ix), replace=True) for ix in g]; a = parts(d, v, gb); bs.append(a[3] / a[2])
    bs = np.array(bs)
    sens[tag] = {"k": int(len(d)), "R_all": round(float(R(d, v)), 3), "R_low": round(float(Rl), 3), "tau": round(float(np.sqrt(tau2)), 4),
                 "expected_R_high": round(float(exp_h), 3), "R_high": round(float(Rh), 3), "obs_over_expected_R_high": round(float(Rh / exp_h), 3),
                 "obs_over_expected_ci95": [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])], "share_boot_ratio_lt1": round(float((bs < 1).mean()), 4)}
out["sensitivity_excl_abs_d_gt_0.3"] = sens
(ROOT / "results/13_dl_vs_radiomics.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(sens, indent=1, ensure_ascii=False))
