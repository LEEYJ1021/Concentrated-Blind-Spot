# -*- coding: utf-8 -*-
"""
SUPPLEMENT D (run after C).  PRE-SPECIFIED (commit before running; do not edit afterwards):
 primary statistic = among statistics whose size at m=0 is <= ALPHA+2*mcse for ALL indicators, the one with the highest
 mean power at m=PLANT_REF (averaged over indicators).  Selection uses simulation only, never the Korean outcomes.  QUICK=1 = smoke test.
"""
import os, json, math, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("RAIL_ROOT", "/home/yjlee/Research/AI_Rail_OM"))
B, R1 = ROOT / "Data_bundle", ROOT / "Reanalysis_v1"; OUT = R1 / "supp_d"; TAB, FIG = OUT / "tables", OUT / "figures"
for d in (OUT, TAB, FIG): d.mkdir(parents=True, exist_ok=True)
QUICK = os.environ.get("QUICK", "0") == "1"; SEED = 20261012; rng = np.random.default_rng(SEED)
ALPHA, MS, PLANT_REF = 0.05, (0, 1, 2, 3, 5), 3
R_SCOPE, N_NULL = (300, 2000) if QUICK else (3000, 20000)
FAM = ["eff_loss_dist", "betw_dist", "service_km"]; ORIG_FOUR = ["경부선", "충북선", "영동선", "중앙선"]
KEY = {"seed": SEED, "quick_mode": QUICK, "config": {"alpha": ALPHA, "planted_m": MS, "plant_ref": PLANT_REF}}
rd = lambda p: pd.read_csv(p, encoding="utf-8-sig")
def hdr(t): print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90, flush=True)
def savet(df, nm): df.to_csv(TAB / f"{nm}.csv", index=False, encoding="utf-8-sig")
def gini_rows(X):
    X = np.sort(X, axis=1); n = X.shape[1]; i = np.arange(1, n + 1); s = X.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"): return 2 * (X * i).sum(1) / (n * s) - (n + 1) / n
def p_up(ns, g): return (1 + len(ns) - np.searchsorted(ns, g, side="left")) / (len(ns) + 1)
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj

master = rd(B / "01_station_master.csv"); nodes = master.station_kor.tolist(); N = len(nodes); eng = master.station_eng.fillna(master.station_kor).tolist()
V = rd(R1 / "tables" / "T02_station_indicators.csv").set_index("station_kor").reindex(nodes)
IND = [c for c in ("eff_loss_dist", "betw_dist", "betw_unw", "eig_unw", "service_km") if c in V.columns]
w0 = master.w_original.fillna(0).to_numpy(float); n_pool = int((w0 < 0.5).sum()); KEY["n_pool_manuscript"] = n_pool
K1, K2 = max(3, round(0.10 * N)), max(5, round(0.20 * N))

def prep(v):                                                              # per-indicator fixed helpers
    order = np.lexsort((rng.random(N), -v)); tm = {}
    for k in (K1, K2): m_ = np.zeros(N, bool); m_[order[:k]] = True; tm[k] = m_
    return order, tm, rankdata(v) / N
def stat_table(v, pools, tm, rk):
    R, n_ = pools.shape; M = np.zeros((R, N), bool); M[np.arange(R)[:, None], pools] = True
    comp = np.where(~M)[1].reshape(R, N - n_); gp = gini_rows(v[pools])
    out = {"gini_pool": gp, "gini_diff_out_minus_in": gp - gini_rows(v[comp]), "pool_share (OCG)": v[pools].sum(1) / v.sum(), "pool_mean_rank": rk[pools].mean(1)}
    for k, m_ in tm.items(): out[f"top{k}_leakage"] = m_[pools].sum(1).astype(float)
    return out

# ---------------------------------------------------------------- D1  size / power of candidate audit statistics (random scope vs planted blind spots)
hdr(f"[D1] Which statistic detects planted blind spots? (n_pool={n_pool}, N={N})")
perm = np.argsort(rng.random((N_NULL, N)), axis=1)[:, :n_pool]; rows = []
for k in IND:
    v = V[k].to_numpy(float); order, tm, rk = prep(v); nst = stat_table(v, perm, tm, rk); nulls = {s: np.sort(a[np.isfinite(a)]) for s, a in nst.items()}
    for m in MS:
        forced, rest = order[:m], order[m:]; picks = rest[np.argsort(rng.random((R_SCOPE, len(rest))), axis=1)[:, :n_pool - m]]
        pools = np.hstack([np.tile(forced, (R_SCOPE, 1)), picks]); st = stat_table(v, pools, tm, rk)
        for s, a in st.items():
            a = a[np.isfinite(a)]; r = float((p_up(nulls[s], a) <= ALPHA).mean())
            rows.append({"indicator": k, "planted_m": m, "statistic": s, "reject": r, "mcse": math.sqrt(max(r * (1 - r), 1e-12) / len(a)), "mean_stat_planted": float(a.mean()), "mean_stat_null": float(nulls[s].mean())})
D1 = pd.DataFrame(rows); savet(D1, "D1_size_power_statistics")
print(D1[D1.statistic == "gini_pool"].round(4).to_string(index=False))
piv = D1.pivot_table(index="statistic", columns="planted_m", values="reject", aggfunc="mean").round(3); print("\nmean rejection over indicators (m = planted blind spots):\n", piv.to_string())
elig = {}
for s, g in D1[D1.planted_m == 0].groupby("statistic"): elig[s] = bool(((g.reject - (ALPHA + 2 * g.mcse)) <= 0).all())
pw = D1[D1.planted_m == PLANT_REF].groupby("statistic").reject.mean(); cand = [s for s in pw.index if elig.get(s)]
CHOSEN = pw[cand].idxmax() if cand else None; KEY["size_eligible"] = elig; KEY["power_at_ref"] = pw.round(4).to_dict(); KEY["chosen_statistic"] = CHOSEN
print("\nsize-eligible:", elig, "\nchosen primary statistic (pre-specified rule):", CHOSEN)
mde = D1[D1.reject >= .8].groupby(["statistic", "indicator"]).planted_m.min().unstack(); print("\nminimum planted m with >=80% power:\n", mde.to_string()); KEY["mde_m"] = mde.to_dict() if len(mde) else {}

# ---------------------------------------------------------------- D2  Korean application with the calibrated statistic(s)
hdr("[D2] Korean scopes: observed audit statistics vs random-scope null (upper tail), Holm over the pre-specified family")
tt = rd(B / "03a_timetable_trains.csv"); tt["n_days"] = tt["n_days"].astype(float)
lg = pd.concat([tt[["origin", "main_line_kor", "n_days"]].rename(columns={"origin": "st"}), tt[["dest", "main_line_kor", "n_days"]].rename(columns={"dest": "st"})])
TD = lg.pivot_table(index="st", columns="main_line_kor", values="n_days", aggfunc="sum", fill_value=0).reindex(index=nodes).fillna(0)
sh = TD.div(TD.sum(1).replace(0, np.nan), axis=0).fillna(0); orig = [l for l in ORIG_FOUR if l in sh.columns]
scopes = {"manuscript line-tag (34)": w0}
if "w_branch_to_trunk" in master: scopes["branch->trunk tags"] = master.w_branch_to_trunk.fillna(0).to_numpy(float)
scopes["original four, service share"] = sh[orig].sum(1).to_numpy(); rows = []
for snm, w in scopes.items():
    pool = np.where(w < 0.5)[0]; n_p = len(pool)
    if n_p < 8 or n_p > N - 8: continue
    permS = np.argsort(rng.random((N_NULL, N)), axis=1)[:, :n_p]
    for k in IND:
        v = V[k].to_numpy(float); order, tm, rk = prep(v); nst = stat_table(v, permS, tm, rk); ob = stat_table(v, pool[None, :], tm, rk)
        for s in nst:
            ns = np.sort(nst[s][np.isfinite(nst[s])]); o = float(ob[s][0])
            rows.append({"scope": snm, "n_pool": n_p, "indicator": k, "statistic": s, "observed": o, "null_mean": float(ns.mean()), "effect": o - float(ns.mean()),
                         "p_upper": float(p_up(ns, o)) if np.isfinite(o) else np.nan, "is_primary": s == CHOSEN})
D2 = pd.DataFrame(rows); D2["p_holm_family"] = np.nan
for (snm, s), g in D2[D2.indicator.isin(FAM)].groupby(["scope", "statistic"]): D2.loc[g.index, "p_holm_family"] = holm(g.p_upper.to_numpy())
savet(D2, "D2_korea_audit_statistics")
show = D2[D2.is_primary] if CHOSEN else D2[D2.statistic == "pool_share (OCG)"]
print(show.drop(columns=["is_primary"]).round(4).to_string(index=False)); KEY["korea_primary"] = show.round(5).to_dict("records")

# ---------------------------------------------------------------- figure + summary
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for s in piv.index: ax[0].plot(piv.columns, piv.loc[s], "o-", label=s)
ax[0].axhline(.8, color="k", ls=":"); ax[0].axhline(ALPHA, color="grey", ls=":"); ax[0].set_xlabel("Planted blind spots m"); ax[0].set_ylabel("Mean rejection over indicators"); ax[0].set_title("A. Size (m=0) and power of audit statistics"); ax[0].legend(fontsize=6)
g_ = D1[D1.statistic == "gini_pool"]
for k, g in g_.groupby("indicator"): ax[1].plot(g.planted_m, g.mean_stat_planted - g.mean_stat_null, "o-", label=k)
ax[1].axhline(0, color="k", lw=.7); ax[1].set_xlabel("Planted blind spots m"); ax[1].set_ylabel("Mean pool Gini minus null mean"); ax[1].set_title("B. Does pool Gini move with planted blind spots?"); ax[1].legend(fontsize=6)
fig.tight_layout(); fig.savefig(FIG / "FD1_statistics.png", dpi=300); fig.savefig(FIG / "FD1_statistics.pdf"); plt.close(fig)
(OUT / "d_numbers.json").write_text(json.dumps(KEY, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8"); print("\nDONE ->", OUT)
