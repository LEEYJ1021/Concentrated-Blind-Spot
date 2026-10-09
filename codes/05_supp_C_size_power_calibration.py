# -*- coding: utf-8 -*-
"""
SUPPLEMENT C (run after S and P).  PRE-SPECIFIED: commit this file to git BEFORE running; do not edit thresholds afterwards.
alpha, planted counts MS, equivalence margin DELTA, core threshold CORE_FREQ are fixed below.  QUICK=1 = smoke test only.
"""
import os, json, math, warnings
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("RAIL_ROOT", "."))
B, R1 = ROOT / "Data_bundle", ROOT / "Reanalysis_v1"; OUT = R1 / "supp_c"; TAB, FIG = OUT / "tables", OUT / "figures"
for d in (OUT, TAB, FIG): d.mkdir(parents=True, exist_ok=True)
QUICK = os.environ.get("QUICK", "0") == "1"; SEED = 20261011; rng = np.random.default_rng(SEED)
ALPHA, DELTA, CORE_FREQ, MS = 0.05, 0.05, 0.8, (0, 1, 2, 3, 5)
R_SCOPE, N_NULL, N_NET, R_NETPOOL = (300, 2000, 6, 100) if QUICK else (3000, 20000, 30, 500)
KEY = {"seed": SEED, "quick_mode": QUICK, "config": {"alpha": ALPHA, "delta": DELTA, "core_freq": CORE_FREQ, "planted_m": MS}}
rd = lambda p: pd.read_csv(p, encoding="utf-8-sig")
def hdr(t): print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90, flush=True)
def savet(df, nm): df.to_csv(TAB / f"{nm}.csv", index=False, encoding="utf-8-sig")
def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x); s = x.sum()
    return np.nan if n == 0 or s <= 0 else 2 * (np.arange(1, n + 1) * x).sum() / (n * s) - (n + 1) / n
def gini_rows(X):
    X = np.sort(X, axis=1); n = X.shape[1]; i = np.arange(1, n + 1); s = X.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"): return 2 * (X * i).sum(1) / (n * s) - (n + 1) / n
def p_up(ns, g): return (1 + len(ns) - np.searchsorted(ns, g, side="left")) / (len(ns) + 1)      # ns sorted; upper tail
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj

master = rd(B / "01_station_master.csv"); nodes = master.station_kor.tolist(); N = len(nodes)
eng = master.station_eng.fillna(master.station_kor).tolist()
V = rd(R1 / "tables" / "T02_station_indicators.csv").set_index("station_kor").reindex(nodes)
w0 = master.w_original.fillna(0).to_numpy(float); pool0 = np.where(w0 < 0.5)[0]; n_pool = len(pool0)
IND = [c for c in ("eff_loss_dist", "betw_dist", "betw_unw", "eig_unw", "service_km") if c in V.columns]
KEY["n_stations"], KEY["n_pool_manuscript"], KEY["indicators"] = N, n_pool, IND

# ---------------------------------------------------------------- C1  size and power of the concentration test (HA, HB)
hdr("[C1] Size / power of concentration tests: random scopes (m=0) and m planted blind spots")
gd = gini_rows(rng.dirichlet(np.ones(n_pool), N_NULL)); gd = np.sort(gd[np.isfinite(gd)])
perm = np.argsort(rng.random((N_NULL, N)), axis=1)[:, :n_pool]; rows = []
for k in IND:
    v = V[k].to_numpy(float); gr = gini_rows(v[perm]); gr = np.sort(gr[np.isfinite(gr)])
    order = np.lexsort((rng.random(N), -v))                                   # top by value, random tie-break
    for m in MS:
        forced, rest = order[:m], order[m:]
        picks = rest[np.argsort(rng.random((R_SCOPE, len(rest))), axis=1)[:, :n_pool - m]]
        pools = np.hstack([np.tile(forced, (R_SCOPE, 1)), picks]); g = gini_rows(v[pools]); g = g[np.isfinite(g)]
        rd_, rr_ = float((p_up(gd, g) <= ALPHA).mean()), float((p_up(gr, g) <= ALPHA).mean())
        rows.append({"indicator": k, "planted_m": m, "n_pool": n_pool, "n_sim": len(g), "reject_dirichlet": rd_, "reject_randompool": rr_,
                     "mcse_dirichlet": math.sqrt(max(rd_ * (1 - rd_), 1e-12) / len(g)), "mcse_randompool": math.sqrt(max(rr_ * (1 - rr_), 1e-12) / len(g))})
C1 = pd.DataFrame(rows); savet(C1, "C1_size_power"); print(C1.round(4).to_string(index=False))
z = C1[C1.planted_m == 0]; KEY["size_at_m0"] = {"dirichlet_min": float(z.reject_dirichlet.min()), "dirichlet_max": float(z.reject_dirichlet.max()),
                                                   "randompool_min": float(z.reject_randompool.min()), "randompool_max": float(z.reject_randompool.max())}
KEY["min_detectable_m_80pct_power"] = {k: (int(g[g.reject_randompool >= .8].planted_m.min()) if (g.reject_randompool >= .8).any() else None) for k, g in C1.groupby("indicator")}
print("\nsize at m=0:", KEY["size_at_m0"], "\nminimum planted m with >=80% power (randompool test):", KEY["min_detectable_m_80pct_power"])

# ---------------------------------------------------------------- C2  typicality vs degree-preserving rewiring (HD)
hdr(f"[C2] Is the observed concentration typical of rewired networks? (95% prediction interval; margin delta={DELTA})")
rr = rd(R1 / "supp" / "tables" / "S4_rewiring_realizations.csv"); rs = rd(R1 / "supp" / "tables" / "S4_rewiring_summary.csv"); rows = []
for variant, g in rr.groupby("variant"):
    for st in ("gini_all53", "gini_pool", "top3_all53"):
        x = g[st].dropna().to_numpy(float); o = float(rs[(rs.variant == variant) & (rs.statistic == st)].observed.iloc[0]); lo, hi = np.percentile(x, [2.5, 97.5])
        rows.append({"variant": variant, "statistic": st, "observed": o, "ensemble_mean": x.mean(), "pi_lo": lo, "pi_hi": hi, "inside_95PI": bool(lo <= o <= hi),
                     "abs_diff": abs(o - x.mean()), "within_delta": bool(abs(o - x.mean()) <= DELTA), "n_valid": len(x)})
C2 = pd.DataFrame(rows); savet(C2, "C2_typicality"); print(C2.round(4).to_string(index=False)); KEY["typicality"] = C2.round(5).to_dict("records")

# ---------------------------------------------------------------- C3  invariant core and external consistency (HE)
hdr(f"[C3] Specification-invariant core (frequency >= {CORE_FREQ}) and exogenous consistency")
spc = rd(R1 / "tables" / "T25_specification_curve.csv"); scn = [s for s in spc.scope.unique() if "line-tag" in s][0]; sub = spc[spc.scope == scn]
fr = pd.Series("|".join(sub.top5).split("|")).value_counts() / len(sub); T = pd.DataFrame({"station": fr.index, "frequency": fr.values})
T["tier"] = np.where(T.frequency >= CORE_FREQ, "A (core)", np.where(T.frequency >= 0.3, "B", "C")); savet(T, "C3_candidate_tiers"); print(T.head(12).round(3).to_string(index=False))
ix_eng = {e: i for i, e in enumerate(eng)}; core = [ix_eng[s] for s in T[T.tier.str.startswith("A")].station if s in ix_eng]
exo = pd.DataFrame(index=range(N))
for c in ("dep_events", "arr_events", "loaded_wagons", "req_wagons_dep"):
    if c in master: exo[c] = master[c].fillna(0).to_numpy(float)
exo["wagon_flow"] = V.wagon_flow.fillna(0).to_numpy(float); rows = []
for c in exo.columns:
    msk = pool0[exo[c].to_numpy()[pool0] > 0]; ms_set = set(msk.tolist()); ci = [i for i in core if i in ms_set]; miss = [eng[i] for i in core if i not in ms_set]
    row = {"exogenous": c, "n_pool_with_data": len(msk), "core_with_data": len(ci), "core_missing": "|".join(miss)}
    if len(ci) >= 2 and len(msk) >= 8:
        pr = pd.Series(exo[c].to_numpy()[msk]).rank(pct=True).to_numpy(); pos = {s: j for j, s in enumerate(msk)}; obs = float(np.mean([pr[pos[i]] for i in ci]))
        nl = np.array([pr[rng.choice(len(msk), len(ci), replace=False)].mean() for _ in range(max(2000, N_NULL // 4))])
        row.update({"core_mean_percentile": obs, "null_mean": 0.5, "p_one_sided": (1 + (nl >= obs).sum()) / (len(nl) + 1), "core_percentiles": "|".join(f"{eng[i]}:{pr[pos[i]]:.2f}" for i in ci)})
    rows.append(row)
C3 = pd.DataFrame(rows); ok = C3.p_one_sided.notna(); C3["p_holm"] = np.nan
if ok.any(): C3.loc[ok, "p_holm"] = holm(C3.loc[ok, "p_one_sided"])
savet(C3, "C3_core_exogenous"); print(C3.round(4).to_string(index=False)); KEY["core"] = T[T.tier.str.startswith("A")].station.tolist(); KEY["core_exogenous"] = C3.round(5).to_dict("records")

# ---------------------------------------------------------------- C4  does the Dirichlet size problem generalise? (HA across topologies)
hdr("[C4] Size of the Dirichlet test on synthetic networks (random scope = 64% of stations)")
def make(kind, n, seed):
    for t in range(50):
        s = seed + 1000 * t
        G = nx.barabasi_albert_graph(n, 2, seed=s) if kind == "BA" else nx.connected_watts_strogatz_graph(n, 4, 0.1, seed=s) if kind == "WS" else nx.gnp_random_graph(n, 2 * math.log(n) / n, seed=s)
        if nx.is_connected(G): return G
    return None
rows = []
for n in (53, 100):
    np_ = int(round(0.64 * n)); gdn = np.sort(gini_rows(rng.dirichlet(np.ones(np_), max(2000, N_NULL // 4))))
    for kind in ("BA", "WS", "ER"):
        acc = {"degree": [], "betweenness": [], "closeness": []}
        for s in range(N_NET):
            G = make(kind, n, int(rng.integers(1, 10 ** 6)))
            if G is None: continue
            vals = {"degree": np.array([d for _, d in G.degree()], float), "betweenness": np.array(list(nx.betweenness_centrality(G).values())),
                    "closeness": np.array(list(nx.closeness_centrality(G).values()))}
            idx = np.argsort(rng.random((R_NETPOOL, n)), axis=1)[:, :np_]
            for ind, v in vals.items():
                g = gini_rows(v[idx]); g = g[np.isfinite(g)]; acc[ind].append(float((p_up(gdn, g) <= ALPHA).mean()))
        for ind, a in acc.items():
            if a: rows.append({"topology": kind, "n": n, "indicator": ind, "n_networks": len(a), "dirichlet_size_mean": np.mean(a), "dirichlet_size_min": np.min(a), "dirichlet_size_max": np.max(a)})
C4 = pd.DataFrame(rows); savet(C4, "C4_synthetic_size"); print(C4.round(3).to_string(index=False)); KEY["synthetic_size"] = C4.round(4).to_dict("records")

# ---------------------------------------------------------------- figure + summary
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
a = ax[0]; z = C1[C1.planted_m == 0]; x_ = np.arange(len(z)); a.bar(x_ - .2, z.reject_dirichlet, .4, label="Dirichlet(1) null"); a.bar(x_ + .2, z.reject_randompool, .4, label="random-pool null")
a.axhline(ALPHA, color="k", ls=":"); a.set_xticks(x_); a.set_xticklabels(z.indicator, rotation=45, ha="right", fontsize=7); a.set_ylabel("Rejection rate under irrelevant (random) scope"); a.set_title(f"A. Empirical size (n_pool={n_pool})"); a.legend(fontsize=7)
a = ax[1]
for k, g in C1.groupby("indicator"): a.plot(g.planted_m, g.reject_randompool, "o-", label=k)
a.axhline(.8, color="k", ls=":"); a.set_xlabel("Planted blind spots m (top-m stations forced out of scope)"); a.set_ylabel("Power (random-pool null)"); a.set_title("B. Power at n = %d" % n_pool); a.legend(fontsize=6)
a = ax[2]; q = C4[C4.indicator == "betweenness"]; lab = [f"{r.topology}-{r.n}" for r in q.itertuples()]
a.bar(lab, q.dirichlet_size_mean, yerr=[q.dirichlet_size_mean - q.dirichlet_size_min, q.dirichlet_size_max - q.dirichlet_size_mean], capsize=3, color="#7f7f7f")
a.axhline(ALPHA, color="k", ls=":"); a.set_ylabel("Dirichlet test size (betweenness)"); a.set_title("C. Synthetic topologies"); a.tick_params(axis="x", rotation=45, labelsize=7)
fig.tight_layout(); fig.savefig(FIG / "FC1_calibration.png", dpi=300); fig.savefig(FIG / "FC1_calibration.pdf"); plt.close(fig)
(OUT / "c_numbers.json").write_text(json.dumps(KEY, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
print("\nDONE ->", OUT)
