"""选择模拟：δ ~ N(μ, τ²)；d = δ + ε，ε ~ N(0, v)，内部与外部 SE 从 181 项放射组学研究成对有放回抽取；
SE 以对数正态误差观测（σ = 0、0.11、0.2），三分位与 R 按观测值计算。报告规则：无选择；单侧（d ≤ c 才报告）；双侧（|d − μ| ≤ c）。
网格 μ 0–0.12（步长 0.01）、τ 0–0.12、c 0.01–0.20（步长 0.01）；距离 = sqrt(mean([Δlog R, Δlog R_low, Δlog R_mid, Δlog R_high, Δ均值/0.01]²))。
每格 20,000 项研究，seed 42。需要 08 的结果。输出 results/12_selection_simulation.json。
用法：python -I code/12_selection_simulation.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
p = pd.read_csv(ROOT / "data/radiomics_FT/pairs.csv"); H = p[p.headline & p.comp.eq("int_ext")]
si = ((H.hi_a - H.lo_a) / 3.92).to_numpy(float); se_ = ((H.hi_ext - H.lo_ext) / 3.92).to_numpy(float)
X = json.load(open(ROOT / "results/08_variance_ratio.json", encoding="utf-8"))["radiomics"]["combined"]
target = np.array([X["R"], X["low_se"]["R"], X["mid_se"]["R"], X["high_se"]["R"]]); mean_obs = float((H.auc_a - H.auc_ext).mean())
N = 20000
def summ(d, vh):
    t = pd.qcut(np.sqrt(vh), 3, labels=False)
    return np.array([np.var(d, ddof=1) / vh.mean()] + [np.var(d[t == g], ddof=1) / vh[t == g].mean() for g in range(3)])
res = {"k": int(len(H)), "observed": {**dict(zip(["R", "R_low", "R_mid", "R_high"], target.round(3).tolist())), "mean_drop": round(mean_obs, 4)},
       "grid": {"mu": "0-0.12 step 0.01", "tau": [0, .01, .02, .03, .04, .05, .06, .08, .10, .12], "c": "0.01-0.20 step 0.01"}, "n_per_cell": N, "seed": 42, "fits": {}}
for sig in (0.0, 0.11, 0.2):
    rng = np.random.default_rng(42); fit = {}
    for rule in ("none", "one_sided", "two_sided"):
        best = None
        for mu in np.round(np.arange(0, .121, .01), 2):
            for tau in (0, .01, .02, .03, .04, .05, .06, .08, .10, .12):
                for c in ([None] if rule == "none" else np.round(np.arange(.01, .201, .01), 2)):
                    idx = rng.integers(0, len(si), N); a, b = si[idx], se_[idx]
                    d = rng.normal(mu, tau, N) + rng.normal(0, np.sqrt(a ** 2 + b ** 2))
                    vh = (a * np.exp(sig * rng.normal(size=N) - sig ** 2 / 2)) ** 2 + (b * np.exp(sig * rng.normal(size=N) - sig ** 2 / 2)) ** 2
                    keep = np.ones(N, bool) if rule == "none" else (d <= c if rule == "one_sided" else np.abs(d - mu) <= c)
                    if keep.sum() < 300: continue
                    s = summ(d[keep], vh[keep]); dist = float(np.sqrt((np.sum((np.log(s) - np.log(target)) ** 2) + ((d[keep].mean() - mean_obs) / 0.01) ** 2) / 5))
                    if best is None or dist < best["dist"]:
                        best = {"mu": float(mu), "tau": tau, "c": None if c is None else float(c), "share_reported": round(float(keep.mean()), 3), "mean_drop": round(float(d[keep].mean()), 4),
                                "R": round(float(s[0]), 3), "R_low": round(float(s[1]), 3), "R_mid": round(float(s[2]), 3), "R_high": round(float(s[3]), 3), "dist": round(dist, 3)}
        best["at_grid_edge"] = bool(best["mu"] in (0.0, 0.12) or best["tau"] in (0, .12) or best["c"] in (0.01, 0.2))
        fit[rule] = best
    res["fits"][f"sigma{sig}"] = fit
(ROOT / "results/12_selection_simulation.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res["fits"], indent=1, ensure_ascii=False))
