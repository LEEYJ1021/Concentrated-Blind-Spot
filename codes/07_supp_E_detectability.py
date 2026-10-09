# -*- coding: utf-8 -*-
"""
SUPPLEMENT E (after D).  (E1) is the OCG size-screen failure noise?  (E2) analytic detection floor of top-K leakage
(E3) detectability map: power of the exactly calibrated OCG statistic vs N, out-of-scope share, planted fraction.
PRE-SPECIFIED: commit before running.  After E the analysis is FROZEN.   QUICK=1 = smoke test only.
"""
import os, json, math, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import comb
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("RAIL_ROOT", "/home/yjlee/Research/AI_Rail_OM")); R1 = ROOT / "Reanalysis_v1"
OUT = R1 / "supp_e"; TAB, FIG = OUT / "tables", OUT / "figures"
for d in (OUT, TAB, FIG): d.mkdir(parents=True, exist_ok=True)
QUICK = os.environ.get("QUICK", "0") == "1"; SEED = 20261013; rng = np.random.default_rng(SEED)
ALPHA, NS, FOUT, PF, R_VEC = 0.05, (53, 100, 200), (0.2, 0.35, 0.5, 0.64, 0.8), (0.05, 0.10), 3
N_NULL, R_SCOPE = (1000, 300) if QUICK else (5000, 1500)
KEY = {"seed": SEED, "quick_mode": QUICK, "config": {"alpha": ALPHA, "N": NS, "f_out": FOUT, "planted_fraction": PF, "n_vector_realisations": R_VEC}}
rd = lambda p: pd.read_csv(p, encoding="utf-8-sig")
def hdr(t): print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90, flush=True)
def savet(df, nm): df.to_csv(TAB / f"{nm}.csv", index=False, encoding="utf-8-sig")
def p_up(ns, g): return (1 + len(ns) - np.searchsorted(ns, g, side="left")) / (len(ns) + 1)

# ---------------------------------------------------------------- E1  size screen: noise or real miscalibration?
hdr("[E1] Size at m=0 for every statistic x indicator (Bonferroni-adjusted bound over all cells)")
D1 = rd(R1 / "supp_d" / "tables" / "D1_size_power_statistics.csv"); m0 = D1[D1.planted_m == 0].copy(); zb = norm.ppf(1 - ALPHA / len(m0))
m0["bound_2mcse"] = ALPHA + 2 * m0.mcse; m0["bound_bonf"] = ALPHA + zb * m0.mcse; m0["exceeds_2mcse"] = m0.reject > m0.bound_2mcse; m0["exceeds_bonf"] = m0.reject > m0.bound_bonf
print(m0[["indicator", "statistic", "reject", "mcse", "exceeds_2mcse", "exceeds_bonf"]].round(4).to_string(index=False))
print(f"\nn cells = {len(m0)}; exceeding 2*mcse: {int(m0.exceeds_2mcse.sum())} (chance expectation ~{0.025 * len(m0):.1f}); exceeding Bonferroni bound: {int(m0.exceeds_bonf.sum())}")
savet(m0, "E1_size_screen"); KEY["size_screen"] = {"n_cells": int(len(m0)), "n_exceed_2mcse": int(m0.exceeds_2mcse.sum()), "n_exceed_bonf": int(m0.exceeds_bonf.sum()),
                                                   "ocg_exceeds_bonf": bool(m0[m0.statistic == "pool_share (OCG)"].exceeds_bonf.any())}

# ---------------------------------------------------------------- E2  analytic detection floor of top-K leakage
hdr("[E2] Smallest attainable p-value of top-K leakage under a random scope: C(n_pool,K)/C(N,K)")
rows = []
for Np in NS:
    for f in FOUT:
        n_ = int(round(f * Np)); pm = [(K, comb(n_, K) / comb(Np, K)) for K in range(1, 21)]
        rows.append({"N": Np, "f_out": f, "n_pool": n_, "p_min_K5": comb(n_, 5) / comb(Np, 5), "K_min_for_p<=0.05": next((K for K, p in pm if p <= ALPHA), np.nan)})
E2 = pd.DataFrame(rows); savet(E2, "E2_leakage_floor"); print(E2.round(4).to_string(index=False)); KEY["leakage_floor"] = E2.round(5).to_dict("records")

# ---------------------------------------------------------------- E3  detectability map for the calibrated OCG statistic
hdr("[E3] Power of the OCG (pool share) test: bootstrap-resampled empirical indicator vectors")
V = rd(R1 / "tables" / "T02_station_indicators.csv").set_index("station_kor"); IND = [c for c in ("eff_loss_dist", "betw_dist") if c in V.columns]
def ocg_power(v):
    Np = len(v); tot = v.sum(); order = np.lexsort((rng.random(Np), -v)); out = {}
    for f in FOUT:
        n_ = int(round(f * Np)); ns = np.sort(v[np.argsort(rng.random((N_NULL, Np)), axis=1)[:, :n_]].sum(1) / tot)
        for pf in PF:
            m = max(1, int(round(pf * Np)))
            if m > n_: out[(f, pf)] = (m, np.nan); continue
            forced, rest = order[:m], order[m:]; picks = rest[np.argsort(rng.random((R_SCOPE, len(rest))), axis=1)[:, :n_ - m]]
            T = (v[forced].sum() + v[picks].sum(1)) / tot; out[(f, pf)] = (m, float((p_up(ns, T) <= ALPHA).mean()))
    return out
rows = []
for k in IND:
    base = V[k].to_numpy(float)
    for Np in NS:
        acc = {}
        for _ in range(R_VEC):
            v = rng.choice(base, Np, replace=True)
            if v.sum() <= 0: continue
            for key, (m, pw) in ocg_power(v).items(): acc.setdefault(key, {"m": m, "p": []})["p"].append(pw)
        for (f, pf), d in acc.items(): rows.append({"indicator": k, "N": Np, "f_out": f, "planted_fraction": pf, "planted_m": d["m"], "power": float(np.nanmean(d["p"]))})
E3 = pd.DataFrame(rows); savet(E3, "E3_detectability_map")
for k in IND:
    print(f"\n--- {k}: power of OCG test (rows: N, planted fraction; columns: out-of-scope share)")
    print(E3[E3.indicator == k].pivot_table(index=["N", "planted_fraction"], columns="f_out", values="power").round(3).to_string())
ok = E3[E3.power >= 0.8]; KEY["detectable_region"] = ok.groupby(["indicator", "N", "planted_fraction"]).f_out.max().reset_index().to_dict("records") if len(ok) else []
print("\nlargest out-of-scope share with >=80% power (per indicator, N, planted fraction):", KEY["detectable_region"] if KEY["detectable_region"] else "none")

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
for a, k in zip(ax[:2], IND):
    for Np in NS:
        g = E3[(E3.indicator == k) & (E3.N == Np) & (E3.planted_fraction == max(PF))].sort_values("f_out"); a.plot(g.f_out, g.power, "o-", label=f"N={Np}")
    a.axhline(.8, color="k", ls=":"); a.axvline(0.64, color="grey", ls="--", lw=.8); a.set_xlabel("Out-of-scope share"); a.set_ylabel(f"Power (planted {int(100 * max(PF))}% of stations)"); a.set_title(f"{k}"); a.legend(fontsize=7)
for Np in NS: g = E2[E2.N == Np]; ax[2].plot(g.f_out, g["K_min_for_p<=0.05"], "o-", label=f"N={Np}")
ax[2].set_xlabel("Out-of-scope share"); ax[2].set_ylabel("Top-K all out of scope needed for p<=0.05"); ax[2].set_title("Analytic floor of top-K leakage test"); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "FE1_detectability.png", dpi=300); fig.savefig(FIG / "FE1_detectability.pdf"); plt.close(fig)
(OUT / "e_numbers.json").write_text(json.dumps(KEY, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8"); print("\nDONE ->", OUT)
