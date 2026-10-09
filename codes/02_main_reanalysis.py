# -*- coding: utf-8 -*-
"""
FULL RE-ANALYSIS PIPELINE  (Korea freight-rail service network: scope-design audit)
====================================================================================
Run :  python rail_reanalysis_full.py            (or paste into one Jupyter cell)
Quick smoke test (minutes -> seconds):  QUICK=1 python rail_reanalysis_full.py
Input :  <ROOT>/Data_bundle/  (v3 bundle: 01_station_master.csv, 03a_timetable_trains.csv,
         05e_reservations_slim.csv, 05l_special_terms_slim.csv, raw_github/cascade_deg.csv ...)
Output:  <ROOT>/Reanalysis_v1/{tables,figures}/  + results_summary.md + manuscript_numbers.json

DESIGN PRINCIPLES (each answers a concrete reviewer / audit finding)
 - The 53-node graph is a TRAIN-SERVICE CONNECTION network (timetable), not track topology.
 - No claim that four corridors are officially designated. A "scope" S is any subset of lines;
   the original four lines and a data-defined reference are just two points in the design space.
 - Station coverage w_i(S) = share of the station's weekly train-days running on lines in S.
 - Value indicators are separated into STRUCTURAL / SERVICE / DEMAND layers.
 - All tests report effect sizes + Monte-Carlo error; null models preserve what they should.
 - Stage C uses a facility-location objective (monotone submodular BY CONSTRUCTION; checked
   numerically) and is compared with the exact optimum by full enumeration.
 - "Hypotheses" are re-stated (H1'..H5') in the final summary with data-driven verdicts.
"""
import os, sys, json, math, itertools, warnings
from itertools import combinations, islice
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx
from scipy.stats import spearmanr, pearsonr, norm, hypergeom
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================================ 0. CONFIG
ROOT = Path(os.environ.get("RAIL_ROOT", "."))
B = ROOT / "Data_bundle"
OUT = ROOT / "Reanalysis_v1"; TAB = OUT / "tables"; FIG = OUT / "figures"
for d in (OUT, TAB, FIG): d.mkdir(parents=True, exist_ok=True)

QUICK = os.environ.get("QUICK", "0") == "1"
SEED = 20261009
N_NULL = 300 if QUICK else 5000          # Gini nulls
N_MORAN = 199 if QUICK else 9999         # Moran permutations
N_REWIRE = 40 if QUICK else 1000         # rewired networks per variant
N_WDRAW = 100 if QUICK else 1000         # Stage-C weight draws
N_POWER = 30 if QUICK else 300           # spatial power simulations per cell
N_H3 = 2000 if QUICK else 50000          # conditional H3 null
N_BOOT = 300 if QUICK else 2000
K_MAIN, TAU, M_REF, LAMBDA = 5, 0.5, 4, 0.30
ORIG_FOUR = ["경부선", "충북선", "영동선", "중앙선"]       # NOT a policy claim: one reference point
rng = np.random.default_rng(SEED)
KEY = {"seed": SEED, "quick_mode": QUICK}

LINE_EN = {"경부선": "Gyeongbu", "중앙선": "Jungang", "충북선": "Chungbuk", "전라선": "Jeolla", "장항선": "Janghang",
           "태백선": "Taebaek", "경전선": "Gyeongjeon", "영동선": "Yeongdong", "대구선": "Daegu", "호남선": "Honam",
           "동해선": "Donghae", "경북선": "Gyeongbuk", "경원선": "Gyeongwon", "경의선": "Gyeongui"}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})
COL = {"blue": "#1f77b4", "red": "#d62728", "grey": "#7f7f7f", "green": "#2ca02c", "orange": "#ff7f0e", "purple": "#9467bd"}

def log(*a): print(*a, flush=True)
def hdr(t): log("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)
def rd(p): return pd.read_csv(p, encoding="utf-8-sig")
def savet(df, name): df.to_csv(TAB / f"{name}.csv", index=False, encoding="utf-8-sig")
def savefig(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=300); fig.savefig(FIG / f"{name}.pdf"); plt.close(fig)
def mc_p(count, n): return (count + 1) / (n + 1)
def mc_se(p, n): return math.sqrt(max(p * (1 - p), 1e-12) / n)
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1, run)
    return adj
def bh(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps); adj = np.empty(m); prev = 1
    for r in range(m - 1, -1, -1):
        i = o[r]; prev = min(prev, ps[i] * m / (r + 1)); adj[i] = prev
    return adj

def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x); s = x.sum()
    if n == 0 or s <= 0: return np.nan
    i = np.arange(1, n + 1); return 2 * (i * x).sum() / (n * s) - (n + 1) / n
def gini_rows(X):
    X = np.sort(X, axis=1); n = X.shape[1]; i = np.arange(1, n + 1); s = X.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 2 * (X * i).sum(1) / (n * s) - (n + 1) / n
def top_share(x, k): x = np.sort(np.asarray(x, float))[::-1]; return x[:k].sum() / x.sum() if x.sum() > 0 else np.nan
def n_for(x, p=0.8):
    x = np.sort(np.asarray(x, float))[::-1]; c = np.cumsum(x) / x.sum(); return int(np.searchsorted(c, p) + 1)
assert abs(gini([1, 1, 1, 1])) < 1e-12 and abs(gini([0, 0, 0, 1]) - 0.75) < 1e-12   # unit test of Eq.(2) replacement
def haversine(lat, lon):
    la, lo = np.radians(lat), np.radians(lon); dla = la[:, None] - la[None, :]; dlo = lo[:, None] - lo[None, :]
    h = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))
def top_idx(v, k, pool=None):
    pool = np.arange(len(v)) if pool is None else np.asarray(pool)
    o = np.lexsort((pool, -np.asarray(v)[pool])); return pool[o[:k]]       # deterministic tie-break
def pct(s): return pd.Series(s).rank(pct=True).to_numpy()

# ============================================================================ 1. LOAD
hdr("[1] LOAD DATA")
master = rd(B / "01_station_master.csv"); tt = rd(B / "03a_timetable_trains.csv")
nodes = master.station_kor.tolist(); N = len(nodes); ix = {n: i for i, n in enumerate(nodes)}
eng = master.station_eng.fillna(master.station_kor).tolist()
lat, lon = master.latitude.to_numpy(float), master.longitude.to_numpy(float)
Dkm = haversine(lat, lon)
miss = (set(tt.origin) | set(tt.dest)) - set(nodes)
assert not miss, f"timetable stations missing from master: {miss}"
log(f"stations={N}, trains={len(tt)}, lines={tt.main_line_kor.nunique()}")
KEY.update(n_stations=N, n_trains=int(len(tt)), timetable_reference_year="UNKNOWN - author must state (R1-8)")

# ============================================================================ 2. SERVICE-CONNECTION NETWORK
hdr("[2] NETWORK (undirected service-connection graph; edge length = scheduled distance, min rule)")
e = tt.groupby(["origin", "dest"], as_index=False).distance_km.first()
e["a"] = np.where(e.origin < e.dest, e.origin, e.dest); e["b"] = np.where(e.origin < e.dest, e.dest, e.origin)
EDG = e.groupby(["a", "b"], as_index=False).distance_km.agg(["min", "max"]).reset_index()
KEY["n_edges_undirected"] = int(len(EDG)); KEY["edges_with_asymmetric_distance"] = int((EDG["min"] != EDG["max"]).sum())
A_dist = np.zeros((N, N))
for r in EDG.itertuples():
    A_dist[ix[r.a], ix[r.b]] = A_dist[ix[r.b], ix[r.a]] = r.min
A_hop = (A_dist > 0).astype(float)
G_unw = nx.from_numpy_array(A_hop); G_dist = nx.from_numpy_array(A_dist)
nx.relabel_nodes(G_unw, dict(enumerate(nodes)), copy=False); nx.relabel_nodes(G_dist, dict(enumerate(nodes)), copy=False)
for u, v, d in G_dist.edges(data=True): d["dist"] = d["weight"]
KEY["connected"] = bool(nx.is_connected(G_unw)); log(f"N={N}, E={len(EDG)}, connected={KEY['connected']}")

# ============================================================================ 3. NODE INDICATORS
hdr("[3] INDICATORS: structural / service / demand layers")
def removal_metrics(A, weighted):
    """For each node v: topo = (1 + nodes cut off from the largest component)/n ;
    eff_loss = relative loss of pairwise efficiency among the OTHER n-1 nodes (rerouting+disconnection)."""
    n = A.shape[0]
    D0 = shortest_path(csr_matrix(A), method="D", directed=False, unweighted=not weighted)
    inv0 = np.where(np.isfinite(D0) & (D0 > 0), 1.0 / np.where(D0 > 0, D0, 1), 0.0)
    topo, loss = np.zeros(n), np.zeros(n)
    for v in range(n):
        idx = np.delete(np.arange(n), v); sub = csr_matrix(A[np.ix_(idx, idx)])
        _, lab = connected_components(sub, directed=False)
        topo[v] = (1 + (n - 1) - np.bincount(lab).max()) / n
        D = shortest_path(sub, method="D", directed=False, unweighted=not weighted)
        inv = np.where(np.isfinite(D) & (D > 0), 1.0 / np.where(D > 0, D, 1), 0.0)
        base = inv0[np.ix_(idx, idx)].sum(); loss[v] = 1 - inv.sum() / base if base > 0 else 0.0
    return topo, loss

def cascade_overload(G, fail, load, alpha=0.5, floor=1e-4, maxit=40):
    """Legacy simplified Motter-Lai cascade (uniform tolerance; load NOT recomputed; deterministic order)."""
    L = {n: max(load[n], floor) for n in G}; cap = {n: (1 + alpha) * v for n, v in L.items()}
    cur = dict(L); failed = {fail}; H = G.copy(); H.remove_node(fail)
    nb = list(G.neighbors(fail))
    for x in nb: cur[x] += L[fail] / len(nb)
    for _ in range(maxit):
        new = {n for n in H if cur[n] > cap[n]}
        if H.number_of_nodes() > 1:
            for c in sorted(nx.connected_components(H), key=len, reverse=True)[1:]: new |= set(c)
        new -= failed
        if not new: break
        for n in sorted(new):
            failed.add(n); al = [x for x in H.neighbors(n) if x not in failed]
            for x in al: cur[x] += cur[n] / len(al)
            if n in H: H.remove_node(n)
    return len(failed) / G.number_of_nodes()

def eig_c(G):
    try: return nx.eigenvector_centrality_numpy(G)
    except Exception: return nx.eigenvector_centrality(G, max_iter=5000)

V = pd.DataFrame(index=nodes)
V["deg"] = pd.Series(nx.degree_centrality(G_unw))
V["betw_unw"] = pd.Series(nx.betweenness_centrality(G_unw)); V["betw_dist"] = pd.Series(nx.betweenness_centrality(G_dist, weight="dist"))
V["close_unw"] = pd.Series(nx.closeness_centrality(G_unw)); V["close_dist"] = pd.Series(nx.closeness_centrality(G_dist, distance="dist"))
V["eig_unw"] = pd.Series(eig_c(G_unw))                      # unweighted on purpose (legacy used distance as strength = inverted)
V["topo"], V["eff_loss_hop"] = removal_metrics(A_hop, False)
_, V["eff_loss_dist"] = removal_metrics(A_dist, True)
for a in (0.2, 0.5, 0.8):
    V[f"overload_a{int(a*10):02d}"] = [cascade_overload(G_unw, n, V.betw_unw.to_dict(), alpha=a) for n in nodes]
V["topo_alpha_inf"] = [cascade_overload(G_unw, n, V.betw_unw.to_dict(), alpha=1e6) for n in nodes]
V["overload_extra_a05"] = V.overload_a05 - V.topo_alpha_inf

# service layer
tt["n_days"] = tt["n_days"].astype(float); tt["tkm"] = tt.n_days * tt.distance_km
lines = sorted(tt.main_line_kor.unique()); L = len(lines); li = {l: i for i, l in enumerate(lines)}
len_en = [LINE_EN.get(l, f"Line{i}") for i, l in enumerate(lines)]
line_stats = tt.groupby("main_line_kor").agg(trains=("train_no", "nunique"), train_days=("n_days", "sum"), train_km_week=("tkm", "sum")).reindex(lines)
line_stats["share_km"] = line_stats.train_km_week / line_stats.train_km_week.sum(); line_stats["line_en"] = len_en
savet(line_stats.reset_index(), "T01_line_statistics")
long = pd.concat([tt[["origin", "main_line_kor", "n_days", "tkm"]].rename(columns={"origin": "st"}),
                  tt[["dest", "main_line_kor", "n_days", "tkm"]].rename(columns={"dest": "st"})])
TD = long.pivot_table(index="st", columns="main_line_kor", values="n_days", aggfunc="sum", fill_value=0).reindex(index=nodes, columns=lines).fillna(0)
TKM = long.pivot_table(index="st", columns="main_line_kor", values="tkm", aggfunc="sum", fill_value=0).reindex(index=nodes, columns=lines).fillna(0)
V["service_days"] = TD.sum(1); V["service_km"] = TKM.sum(1)
V["wagon_flow"] = master.get("wagon_flow", pd.Series(0, index=master.index)).fillna(0).to_numpy()
savet(V.assign(station_kor=nodes, station=eng).reset_index(drop=True), "T02_station_indicators")

# reproduction checks against legacy artefacts
rep = {}
for c_new, c_old in [("betw_dist", "betweenness"), ("close_dist", "closeness"), ("deg", "degree")]:
    if c_old in master: rep[f"spearman({c_new}, legacy {c_old})"] = float(spearmanr(V[c_new], master[c_old])[0])
gh = B / "raw_github" / "cascade_deg.csv"
if gh.exists():
    g = rd(gh).set_index("node")["cascade_impact"]; mine = pd.Series(V.overload_a05.to_numpy(), index=eng).reindex(g.index)
    rep["legacy_cascade_reproduction_max_abs_diff"] = float(np.abs(mine - g).max())
KEY["reproduction"] = rep; log(json.dumps(rep, indent=1))
KEY["overload_extra_nodes"] = int((V.overload_extra_a05 > 1e-12).sum())
KEY["alpha_mean_impact"] = {a: float(V[f"overload_a{a}"].mean()) for a in ("02", "05", "08")}
KEY["top_collapse_nodes"] = [eng[i] for i in top_idx(V.overload_a05.to_numpy(), 3)]

# ============================================================================ 4. DESIGN SPACE
hdr("[4] SCOPE DESIGN SPACE (all line subsets)")
TDn = (TD.div(TD.sum(1).replace(0, np.nan), axis=0)).fillna(0).to_numpy()          # (N, L) shares
masks, sizes = [], []
for r_ in range(1, L):
    for comb in combinations(range(L), r_):
        m = np.zeros(L, bool); m[list(comb)] = True; masks.append(m); sizes.append(r_)
masks = np.array(masks); sizes = np.array(sizes); nS = len(masks)
Wm = TDn @ masks.T.astype(float)                                                  # (N, nS) coverage w_i(S)
LA = ((TD.to_numpy() > 0).T.astype(int) @ (TD.to_numpy() > 0).astype(int)) > 0     # line adjacency (share a station)
def is_connected(m):
    idx = np.where(m)[0]; seen = {idx[0]}; st = [idx[0]]
    while st:
        u = st.pop()
        for w_ in idx:
            if w_ not in seen and LA[u, w_]: seen.add(w_); st.append(w_)
    return len(seen) == len(idx)
conn = np.array([is_connected(m) for m in masks])
def design_id(line_list):
    m = np.zeros(L, bool); m[[li[x] for x in line_list if x in li]] = True
    return int(np.where((masks == m).all(1))[0][0])
ref_lines = line_stats.sort_values("train_km_week", ascending=False).index[:M_REF].tolist()
d_ref = design_id(ref_lines); orig_present = [x for x in ORIG_FOUR if x in li]; d_orig = design_id(orig_present)
KEY.update(n_lines=L, n_designs=int(nS), reference_lines=[LINE_EN.get(x, x) for x in ref_lines],
           original_four_present=[LINE_EN.get(x, x) for x in orig_present],
           reference_equals_original=bool(set(ref_lines) == set(orig_present)))
log(f"lines={len_en}\ndesigns={nS}; reference(top-{M_REF} by train-km)={[LINE_EN.get(x, x) for x in ref_lines]}; original four={[LINE_EN.get(x, x) for x in orig_present]}")

# ============================================================================ 5. OCG OVER THE DESIGN SPACE (exact permutation moments)
hdr("[5] OCG over all designs: exact permutation mean / variance (reviewer R3-stat-3)")
PRIM = "eff_loss_dist"
def ocg_design_table(v):
    v = np.asarray(v, float); a = 1 - Wm; T = (a * v[:, None]).sum(0); ocg = T / v.sum(); base = a.mean(0)
    var = ((a - a.mean(0)) ** 2).sum(0) * ((v - v.mean()) ** 2).sum() / (N - 1)
    z = (T - base * v.sum()) / np.sqrt(np.maximum(var, 1e-18))
    return ocg, base, z
ocg, base_, zO = ocg_design_table(V[PRIM])
cov = 1 - ocg
DT = pd.DataFrame({"design": [" + ".join(len_en[j] for j in np.where(m)[0]) for m in masks], "size": sizes, "connected": conn,
                   "OCG": ocg, "baseline_1_minus_mean_w": base_, "effect": ocg - base_, "z_perm": zO, "p_upper_normal": 1 - norm.cdf(zO),
                   "coverage_value": cov, "km_share_covered": (masks.astype(float) @ line_stats.share_km.to_numpy())})
savet(DT, "T03_design_space_OCG")
def rank_in_size(col, d, m=M_REF, asc=False):
    sub = DT[DT["size"] == m][col]; return int((sub > DT.loc[d, col]).sum() + 1) if not asc else int((sub < DT.loc[d, col]).sum() + 1)
KEY["ocg"] = {"reference": float(ocg[d_ref]), "original_four": float(ocg[d_orig]),
              "reference_baseline": float(base_[d_ref]), "original_baseline": float(base_[d_orig]),
              "reference_effect": float(ocg[d_ref] - base_[d_ref]), "original_effect": float(ocg[d_orig] - base_[d_orig]),
              "reference_z": float(zO[d_ref]), "original_z": float(zO[d_orig]),
              "original_coverage_rank_among_size4": rank_in_size("coverage_value", d_orig),
              "reference_coverage_rank_among_size4": rank_in_size("coverage_value", d_ref),
              "n_size4_designs": int((DT["size"] == M_REF).sum())}
log(json.dumps(KEY["ocg"], indent=1))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for m_ in range(2, 7): ax[0].hist(DT[DT["size"] == m_].OCG, bins=25, alpha=.55, label=f"m={m_}")
ax[0].axvline(ocg[d_ref], color=COL["red"], lw=2, label="Reference (top-4 by train-km)"); ax[0].axvline(ocg[d_orig], color="k", ls="--", lw=2, label="Original four lines")
ax[0].set_xlabel("Optimization coverage gap (OCG)"); ax[0].set_ylabel("Number of designs"); ax[0].set_title("A. OCG across the full design space"); ax[0].legend(fontsize=6)
sub = DT[DT["size"] == M_REF]
ax[1].scatter(sub.coverage_value, sub.km_share_covered, s=14, c=np.where(sub.connected, COL["blue"], COL["grey"]), alpha=.7)
ax[1].scatter(DT.loc[d_ref, "coverage_value"], DT.loc[d_ref, "km_share_covered"], s=90, c=COL["red"], marker="*", label="Reference")
ax[1].scatter(DT.loc[d_orig, "coverage_value"], DT.loc[d_orig, "km_share_covered"], s=90, c="k", marker="D", label="Original four")
ax[1].set_xlabel(f"Covered structural value ({PRIM})"); ax[1].set_ylabel("Share of train-km on designated lines"); ax[1].set_title("B. Size-4 designs (grey = lines not mutually connected)"); ax[1].legend()
savefig(fig, "F02_design_space_OCG")

# ============================================================================ 6. CONCENTRATION + THREE NULLS + IN-SCOPE CONTRAST
hdr("[6] CONCENTRATION of out-of-scope value (reference scope) with three null models")
IND = ["eff_loss_dist", "eff_loss_hop", "topo", "betw_dist", "betw_unw", "close_dist", "eig_unw", "deg", "service_km", "overload_a05"]
def pool_of(w, tau=TAU): return np.where(np.asarray(w) < tau)[0]
w_ref = Wm[:, d_ref]; pool = pool_of(w_ref); inscope = np.setdiff1d(np.arange(N), pool); n = len(pool)
KEY["pool_size_reference"] = int(n); log(f"reference pool size = {n} (in-scope = {len(inscope)})")
rows = []; draws = {}
for k in IND:
    v = V[k].to_numpy(float); x = v[pool]; g = gini(x)
    ridx = np.argsort(rng.random((N_NULL, N)), axis=1)[:, :n]; g_rp = gini_rows(v[ridx])                 # random pool of same size
    g_dir = gini_rows(rng.dirichlet(np.ones(n), N_NULL))                                                 # legacy Dirichlet(1)
    g_bs = gini_rows(v[rng.integers(0, N, (N_NULL, n))])                                                 # empirical-distribution bootstrap
    draws[k] = (g_rp, g_dir, g_bs)
    p_rp, p_dir, p_bs = [mc_p(int(np.nansum(gg >= g)), N_NULL) for gg in (g_rp, g_dir, g_bs)]
    rows.append({"indicator": k, "n_pool": n, "n_zero": int((x == 0).sum()), "gini_pool": g, "gini_inscope": gini(v[inscope]) if len(inscope) > 1 else np.nan,
                 "gini_all53": gini(v), "top3_share": top_share(x, 3), "n_for_80pct": n_for(x) if x.sum() > 0 else np.nan,
                 "null_randompool_mean": np.nanmean(g_rp), "effect_vs_randompool": g - np.nanmean(g_rp), "p_randompool": p_rp, "p_randompool_mcse": mc_se(p_rp, N_NULL),
                 "null_dirichlet_mean": np.nanmean(g_dir), "p_dirichlet": p_dir, "null_bootstrap_mean": np.nanmean(g_bs), "p_bootstrap": p_bs})
CT = pd.DataFrame(rows)
FAM = ["eff_loss_dist", "betw_dist", "service_km"]                           # pre-specified confirmatory family
CT["p_randompool_holm"] = np.nan; sel = CT.indicator.isin(FAM); CT.loc[sel, "p_randompool_holm"] = holm(CT.loc[sel, "p_randompool"])
savet(CT, "T04_concentration_three_nulls"); log(CT.round(4).to_string(index=False))
KEY["concentration"] = CT.set_index("indicator")[["gini_pool", "gini_inscope", "top3_share", "effect_vs_randompool", "p_randompool", "p_randompool_holm", "p_dirichlet", "p_bootstrap"]].round(5).to_dict("index")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for k, c in zip(["eff_loss_dist", "betw_dist", "service_km", "overload_a05"], [COL["blue"], COL["red"], COL["green"], COL["purple"]]):
    x = np.sort(V[k].to_numpy()[pool]); cs = np.r_[0, np.cumsum(x) / x.sum()]; ax[0].plot(np.linspace(0, 1, n + 1), cs, color=c, label=f"{k} (G={gini(x):.2f})")
    xd = np.sort(V[k].to_numpy()[pool])[::-1]; ax[1].plot(np.arange(0, n + 1), np.r_[0, np.cumsum(xd) / xd.sum()], color=c, label=k)
ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[0].set_xlabel("Cumulative share of out-of-scope stations (ascending)"); ax[0].set_ylabel("Cumulative share of value")
ax[0].set_title("A. Lorenz curves (standard ascending order)"); ax[0].legend()
ax[1].axvline(3, color="k", ls=":"); ax[1].set_xlabel("Number of top stations (descending)"); ax[1].set_ylabel("Cumulative share of value")
ax[1].set_title("B. Top-share curves (top-3 marked) - same vectors as panel A"); ax[1].legend()
savefig(fig, "F03_lorenz_topshare")

fig, ax = plt.subplots(3, 3, figsize=(10, 7.5))
for r_, k in enumerate(FAM):
    for c_, (nm, gg) in enumerate(zip(["Random pool of equal size", "Dirichlet(1) (legacy)", "Empirical bootstrap"], draws[k])):
        a = ax[r_, c_]; a.hist(gg[np.isfinite(gg)], bins=40, color=COL["grey"], alpha=.7); a.axvline(CT.set_index("indicator").loc[k, "gini_pool"], color=COL["red"], lw=2)
        a.set_title(f"{k} | {nm}", fontsize=7); a.set_xlabel("Gini")
savefig(fig, "F04_gini_null_models")

# ============================================================================ 7. DESIGN-INVARIANT BLIND SPOTS (DIBI) + PARETO + CORRIDOR vs STATION
hdr("[7] DIBI, Pareto frontier, corridor-vs-station comparison")
def dibi_table(m, tau, vmetric, only_connected=False):
    sel_ = (sizes == m) & (conn if only_connected else True)
    q = (Wm[:, sel_] < tau).mean(1); v = pct(V[vmetric]); return q, v, q * v
q, vv, dibi = dibi_table(M_REF, TAU, PRIM)
DB = pd.DataFrame({"station_kor": nodes, "station": eng, "q_uncovered_share": q, "v_structural_pct": vv, "DIBI": dibi,
                   "naive_rank": pd.Series(-vv).rank(method="first").astype(int).to_numpy()}); DB["DIBI_rank"] = DB.DIBI.rank(ascending=False, method="first").astype(int)
DB["in_reference_pool"] = np.isin(np.arange(N), pool); DB["invariant_blind_spot"] = (DB.q_uncovered_share >= 0.8) & (DB.v_structural_pct >= 0.75)
DB = DB.sort_values("DIBI", ascending=False); savet(DB, "T05_DIBI")
naive = top_idx(vv, 10); dtop = top_idx(dibi, 10); refonly = top_idx(vv, 10, pool)
def jac(a, b): a, b = set(a), set(b); return len(a & b) / len(a | b)
KEY["dibi"] = {"spearman_DIBI_vs_naive": float(spearmanr(dibi, vv)[0]), "jaccard_top10_DIBI_vs_naive": jac(dtop, naive),
               "jaccard_top10_DIBI_vs_reference_only": jac(dtop, refonly), "n_invariant_blind_spots": int(DB.invariant_blind_spot.sum()),
               "invariant_blind_spots": DB[DB.invariant_blind_spot].station.tolist(), "top5_DIBI": DB.station.head(5).tolist()}
sens = []
for m_ in (3, 4, 5):
    for tau_ in (0.25, 0.5, 0.75):
        for vm in (PRIM, "betw_dist", "topo", "service_km"):
            for oc in (False, True):
                q_, v_, d_ = dibi_table(m_, tau_, vm, oc); sens.append({"m": m_, "tau": tau_, "value": vm, "connected_only": oc,
                    "spearman_vs_main": spearmanr(d_, dibi)[0], "top5_jaccard_vs_main": jac(top_idx(d_, 5), top_idx(dibi, 5)),
                    "top5": "|".join(eng[i] for i in top_idx(d_, 5))})
SD = pd.DataFrame(sens); savet(SD, "T06_DIBI_sensitivity"); KEY["dibi"]["sensitivity_median_top5_jaccard"] = float(SD.top5_jaccard_vs_main.median())
freq = pd.Series("|".join(SD.top5).split("|")).value_counts() / len(SD); KEY["dibi"]["top5_frequency_over_specs"] = freq.head(8).round(3).to_dict()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
sc = ax[0].scatter(DB.q_uncovered_share, DB.v_structural_pct, s=20 + 300 * DB.DIBI, c=DB.DIBI, cmap="viridis", alpha=.8, edgecolor="k", lw=.3)
for r in DB.head(10).itertuples(): ax[0].annotate(r.station, (r.q_uncovered_share, r.v_structural_pct), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax[0].axvline(.8, color=COL["grey"], ls=":"); ax[0].axhline(.75, color=COL["grey"], ls=":")
ax[0].set_xlabel(f"q: share of size-{M_REF} scope designs leaving the station uncovered (w<{TAU})"); ax[0].set_ylabel("v: structural value (percentile)"); ax[0].set_title("A. Design-invariant blind-spot index (DIBI)")
h = DB.head(15).iloc[::-1]; ax[1].barh(h.station, h.DIBI, color=COL["blue"])
for i, r in enumerate(h.itertuples()): ax[1].text(r.DIBI + .005, i, f"naive rank {r.naive_rank}", va="center", fontsize=6)
ax[1].set_xlabel("DIBI = q x v"); ax[1].set_title("B. Top-15 stations"); savefig(fig, "F05_DIBI")

# Pareto frontier
par = []
for m_ in range(1, L):
    s_ = np.where(sizes == m_)[0]; j = s_[np.argmax(cov[s_])]; par.append({"m": m_, "best_design": DT.design[j], "best_coverage": cov[j],
        "median_coverage": np.median(cov[s_]), "reference_coverage": cov[d_ref] if m_ == M_REF else np.nan, "original_coverage": cov[d_orig] if m_ == len(orig_present) else np.nan})
PF = pd.DataFrame(par); savet(PF, "T07_pareto_frontier")
KEY["pareto"] = {"best_at_m4": PF.loc[PF.m == M_REF, "best_design"].iloc[0], "best_cov_m4": float(PF.loc[PF.m == M_REF, "best_coverage"].iloc[0]),
                 "reference_gap_to_frontier": float(PF.loc[PF.m == M_REF, "best_coverage"].iloc[0] - cov[d_ref]),
                 "original_gap_to_frontier": float(PF.loc[PF.m == len(orig_present), "best_coverage"].iloc[0] - cov[d_orig])}
# marginal corridor (add/remove one line from the reference)
mr = []
for l_ in range(L):
    m2 = masks[d_ref].copy(); m2[l_] = ~m2[l_]; d2 = int(np.where((masks == m2).all(1))[0][0])
    mr.append({"line": len_en[l_], "action": "remove" if masks[d_ref][l_] else "add", "delta_coverage": cov[d2] - cov[d_ref], "delta_km_share": DT.km_share_covered[d2] - DT.km_share_covered[d_ref]})
MR = pd.DataFrame(mr).sort_values("delta_coverage", key=np.abs, ascending=False); savet(MR, "T08_marginal_corridor")
KEY["marginal_corridor"] = MR.head(3).round(4).to_dict("records")

# corridor expansion vs station-level addition (same number of newly covered stations)
vtot = V[PRIM].sum(); cs_rows = []
for l_ in range(L):
    if masks[d_ref][l_]: continue
    m2 = masks[d_ref].copy(); m2[l_] = True; d2 = int(np.where((masks == m2).all(1))[0][0]); w2 = Wm[:, d2]
    newly = np.where((w_ref < TAU) & (w2 >= TAU))[0]; gain = float((V[PRIM].to_numpy() * (w2 - w_ref)).sum() / vtot)
    k_ = len(newly); st_sel = top_idx(V[PRIM].to_numpy(), k_, pool) if k_ else []
    st_gain = float((V[PRIM].to_numpy()[st_sel] * (1 - w_ref[st_sel])).sum() / vtot) if k_ else 0.0
    cs_rows.append({"added_line": len_en[l_], "newly_covered_stations": k_, "corridor_gain": gain, "station_level_gain_same_count": st_gain,
                    "ratio_station_over_corridor": st_gain / gain if gain > 0 else np.nan})
CS = pd.DataFrame(cs_rows).sort_values("corridor_gain", ascending=False); savet(CS, "T09_corridor_vs_station")
ksw = [{"K": k_, "station_level_gain": float((V[PRIM].to_numpy()[top_idx(V[PRIM].to_numpy(), k_, pool)] * (1 - w_ref[top_idx(V[PRIM].to_numpy(), k_, pool)])).sum() / vtot)} for k_ in range(1, 16)]
KS = pd.DataFrame(ksw); savet(KS, "T10_station_addition_sweep")
KEY["corridor_vs_station"] = CS.head(3).round(4).to_dict("records")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
ax[0].plot(PF.m, PF.best_coverage, "o-", color=COL["blue"], label="Best design (exhaustive)"); ax[0].plot(PF.m, PF.median_coverage, "s--", color=COL["grey"], label="Median design")
ax[0].scatter([M_REF], [cov[d_ref]], c=COL["red"], marker="*", s=120, label="Reference"); ax[0].scatter([len(orig_present)], [cov[d_orig]], c="k", marker="D", label="Original four")
ax[0].set_xlabel("Number of designated lines m"); ax[0].set_ylabel("Covered structural value"); ax[0].set_title("A. Pareto frontier of scope design"); ax[0].legend()
mm = MR.iloc[::-1]; ax[1].barh([f"{a} {b}" for a, b in zip(mm.action, mm.line)], mm.delta_coverage, color=[COL["green"] if a == "add" else COL["red"] for a in mm.action])
ax[1].set_xlabel("Change in covered value vs reference"); ax[1].set_title("B. Marginal corridor")
ax[2].plot(KS.K, KS.station_level_gain, "o-", color=COL["blue"], label="Station-level additions (top-K by value)")
for r in CS.itertuples(): ax[2].scatter([max(r.newly_covered_stations, .5)], [r.corridor_gain], s=40, marker="s", label=f"+{r.added_line}")
ax[2].set_xlabel("Stations newly covered / added"); ax[2].set_ylabel("Gain in covered value"); ax[2].set_title("C. Corridor vs station-level expansion"); ax[2].legend(fontsize=6)
savefig(fig, "F06_pareto_marginal_station_vs_corridor")

# ============================================================================ 8. STAGE C (facility-location, exact optimum)
hdr("[8] STAGE C: monotone-submodular facility-location objective, exact enumeration, benchmarks")
FEAT_FULL = ["close_dist", "eig_unw", "service_km", "betw_dist"]; FEAT_DISTINCT = ["close_dist", "eig_unw"]; FEAT_SERVICE = ["close_dist", "eig_unw", "service_km"]
Fp = pd.DataFrame({k: pct(V[k]) for k in set(FEAT_FULL)}, index=nodes)
nbr = (A_hop + np.eye(N)) > 0
J = (nbr[:, None, :] & nbr[None, :, :]).sum(2) / np.maximum((nbr[:, None, :] | nbr[None, :, :]).sum(2), 1)       # closed-neighbourhood Jaccard
def smat(pool_idx, lam):
    S = lam * J[np.ix_(pool_idx, pool_idx)]; np.fill_diagonal(S, 1.0); return S
def fval(S, u, sel): return float((S[list(sel)].max(0) * u).sum()) if len(sel) else 0.0
def greedy(S, u, K):
    n_ = len(u); cur = np.zeros(n_); sel = []; gains = []
    for _ in range(min(K, n_)):
        g = (np.maximum(cur[None, :], S) * u).sum(1) - (cur * u).sum(); g[sel] = -np.inf; j = int(np.argmax(g)); sel.append(j); gains.append(float(g[j])); cur = np.maximum(cur, S[j])
    return sel, gains
def exact(S, u, K, cap=3_000_000):
    n_ = len(u)
    if math.comb(n_, K) > cap: return None, None
    best, bs = -1, None; it = combinations(range(n_), K)
    while True:
        ch = np.array(list(islice(it, 20000)))
        if len(ch) == 0: break
        vals = (S[ch].max(1) * u).sum(1); j = int(np.argmax(vals))
        if vals[j] > best: best, bs = float(vals[j]), tuple(ch[j])
    return bs, best
def utility(feats, w=None):
    w = np.ones(len(feats)) / len(feats) if w is None else w; return (Fp[feats].to_numpy() * w).sum(1)
S_c = smat(pool, LAMBDA); pe = [eng[i] for i in pool]
cfgs = {"Full (closeness, eigenvector, service, betweenness)": FEAT_FULL, "Service-augmented (closeness, eigenvector, service)": FEAT_SERVICE, "Distinct only (closeness, eigenvector)": FEAT_DISTINCT}
bench, stageC = [], {}
b_scores = V[PRIM].to_numpy()[pool]
for nm, feats in cfgs.items():
    u = utility(feats)[pool]; tot = u.sum(); sel_g, gains = greedy(S_c, u, K_MAIN); sel_e, f_e = exact(S_c, u, K_MAIN)
    stageC[nm] = {"u": u, "greedy": sel_g, "gains": gains, "exact": sel_e, "f_exact": f_e, "f_greedy": fval(S_c, u, sel_g), "total": tot}
    cand = {"Greedy (proposed)": sel_g, "Exact optimum": list(sel_e) if sel_e else None, "Stage-B top-K": list(top_idx(b_scores, K_MAIN)),
            "Composite top-K (no overlap discount)": list(top_idx(u, K_MAIN))}
    for k_ in feats: cand[f"Single feature: {k_}"] = list(top_idx(Fp[k_].to_numpy()[pool], K_MAIN))
    rv = [fval(S_c, u, rng.choice(len(u), K_MAIN, replace=False)) for _ in range(2000)]; cand["Random (mean of 2000)"] = None
    for m_, s_ in cand.items():
        f = np.mean(rv) if m_.startswith("Random") else (fval(S_c, u, s_) if s_ is not None else np.nan)
        bench.append({"feature_set": nm, "method": m_, "set": "|".join(pe[i] for i in s_) if s_ else "", "objective": f, "coverage_of_total": f / tot,
                      "ratio_to_exact": f / f_e if f_e else np.nan})
BM = pd.DataFrame(bench); savet(BM, "T11_stageC_benchmarks"); log(BM.round(4).to_string(index=False))
main_nm = list(cfgs)[0]; sc = stageC[main_nm]
KEY["stageC"] = {"pool_n": n, "K": K_MAIN, "lambda": LAMBDA, "greedy_set": [pe[i] for i in sc["greedy"]], "exact_set": [pe[i] for i in sc["exact"]] if sc["exact"] else None,
                 "greedy_coverage": sc["f_greedy"] / sc["total"], "exact_coverage": (sc["f_exact"] / sc["total"]) if sc["f_exact"] else None,
                 "greedy_over_exact": (sc["f_greedy"] / sc["f_exact"]) if sc["f_exact"] else None, "guarantee_(1-1/e)": 1 - 1 / math.e,
                 "marginal_gains_pct_of_total": [round(100 * g / sc["total"], 2) for g in sc["gains"]], "denominator": "sum of station utilities in the out-of-scope pool"}
# monotone/submodular numerical check
viol_sub = viol_mono = 0
for _ in range(3000):
    nsz = rng.integers(2, min(8, n)); Bset = list(rng.choice(n, nsz, replace=False)); Aset = Bset[:rng.integers(1, nsz)]; x = int(rng.integers(0, n))
    if x in Bset: continue
    u = sc["u"]
    if fval(S_c, u, Aset + [x]) - fval(S_c, u, Aset) < fval(S_c, u, Bset + [x]) - fval(S_c, u, Bset) - 1e-12: viol_sub += 1
    if fval(S_c, u, Bset + [x]) < fval(S_c, u, Bset) - 1e-12: viol_mono += 1
KEY["stageC"]["submodularity_violations_in_3000_random_checks"] = viol_sub; KEY["stageC"]["monotonicity_violations"] = viol_mono
# lambda sensitivity + K sweep
ls_rows = []
for lam in (0.0, 0.15, 0.30, 0.50):
    Sl = smat(pool, lam); u = sc["u"]; g_, _ = greedy(Sl, u, K_MAIN); e_, fe = exact(Sl, u, K_MAIN)
    ls_rows.append({"lambda": lam, "greedy_set": "|".join(pe[i] for i in g_), "exact_set": "|".join(pe[i] for i in e_) if e_ else "", "greedy_over_exact": fval(Sl, u, g_) / fe if fe else np.nan,
                    "jaccard_vs_lambda0.30": np.nan})
LS = pd.DataFrame(ls_rows); savet(LS, "T12_lambda_sensitivity")
ksw = []
for k_ in range(1, 16):
    g_, gg = greedy(S_c, sc["u"], k_); ev = None
    if k_ <= 5: _, ev = exact(S_c, sc["u"], k_)
    ksw.append({"K": k_, "greedy_coverage": fval(S_c, sc["u"], g_) / sc["total"], "exact_coverage": ev / sc["total"] if ev else np.nan, "marginal_gain": gg[-1] / sc["total"]})
KSW = pd.DataFrame(ksw); savet(KSW, "T13_K_sweep")
# stability under explicit weight distributions
stab = {}
for nm, feats in cfgs.items():
    for alpha_d in (0.5, 1.0, 5.0):
        cnt = np.zeros(n); setc = {}
        for _ in range(N_WDRAW):
            w = rng.dirichlet(np.ones(len(feats)) * alpha_d); u = utility(feats, w)[pool]; s_, _ = greedy(S_c, u, K_MAIN)
            for i in s_: cnt[i] += 1
            key_ = tuple(sorted(s_)); setc[key_] = setc.get(key_, 0) + 1
        ref_set = tuple(sorted(stageC[nm]["greedy"]))
        stab[(nm, alpha_d)] = {"freq": cnt / N_WDRAW, "exact_set_freq": setc.get(ref_set, 0) / N_WDRAW, "n_distinct_sets": len(setc)}
SB = pd.DataFrame([{"feature_set": k[0], "dirichlet_alpha": k[1], "station": pe[i], "selection_freq": v["freq"][i], "exact_set_freq": v["exact_set_freq"], "n_distinct_sets": v["n_distinct_sets"]}
                   for k, v in stab.items() for i in np.argsort(-v["freq"])[:10]]); savet(SB, "T14_stageC_stability")
KEY["stageC"]["stability_main_alpha1"] = {"exact_set_freq": stab[(main_nm, 1.0)]["exact_set_freq"], "n_draws": N_WDRAW,
        "top_station_freq": {pe[i]: round(float(stab[(main_nm, 1.0)]["freq"][i]), 3) for i in np.argsort(-stab[(main_nm, 1.0)]["freq"])[:7]}}
# cost-weighted (knapsack) version if yard data exist
if "n_tracks" in master and master.n_tracks.notna().sum() >= 0.8 * n:
    cost = master.n_tracks.fillna(master.n_tracks.median()).clip(lower=1).to_numpy()[pool]; Bud = K_MAIN * np.median(cost); u = sc["u"]
    sel, cur, spent = [], np.zeros(n), 0.0
    while True:
        g = (np.maximum(cur[None, :], S_c) * u).sum(1) - (cur * u).sum(); ratio = np.where((np.array([i in sel for i in range(n)])) | (spent + cost > Bud), -np.inf, g / cost)
        if not np.isfinite(ratio.max()): break
        j = int(np.argmax(ratio)); sel.append(j); spent += cost[j]; cur = np.maximum(cur, S_c[j])
    KEY["stageC"]["cost_weighted_set"] = [pe[i] for i in sel]; KEY["stageC"]["cost_proxy"] = "number of station tracks (n_tracks); budget = K x median cost"
else:
    KEY["stageC"]["cost_weighted_set"] = None; KEY["stageC"]["cost_proxy"] = "unavailable for >=80% of pool: uniform cost; K is a cardinality limit, NOT a monetary budget"

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
a = ax[0, 0]; gg = np.array(sc["gains"]) / sc["total"] * 100; a.bar([pe[i] for i in sc["greedy"]], gg, color=COL["blue"]); a.set_ylabel("Marginal gain (% of pool total)"); a.set_title(f"A. Greedy insertion order (K={K_MAIN})"); a.tick_params(axis="x", rotation=30)
a = ax[0, 1]; sub = BM[BM.feature_set == main_nm]; sub = sub.iloc[::-1]; a.barh(sub.method, sub.coverage_of_total * 100, color=COL["grey"]); a.set_xlabel("Objective (% of pool total)"); a.set_title("B. Benchmarks on identical objective / pool"); a.tick_params(axis="y", labelsize=6)
a = ax[1, 0]; a.plot(KSW.K, KSW.greedy_coverage * 100, "o-", color=COL["blue"], label="Greedy"); a.plot(KSW.K, KSW.exact_coverage * 100, "s", color=COL["red"], label="Exact (K<=5)")
a.set_xlabel("K"); a.set_ylabel("Objective (% of pool total)"); a.set_title("C. K sweep (diminishing returns)"); a.legend()
a = ax[1, 1]; fr = stab[(main_nm, 1.0)]["freq"]; o = np.argsort(-fr)[:12][::-1]; a.barh([pe[i] for i in o], fr[o], color=COL["green"]); a.set_xlabel(f"Selection frequency over {N_WDRAW} Dirichlet(1) weight draws")
a.set_title(f"D. Conditional sensitivity (exact-set recurrence = {stab[(main_nm, 1.0)]['exact_set_freq']:.2f})")
savefig(fig, "F07_stageC")

# ============================================================================ 9. H3' : convergence with shared information
hdr("[9] H3': Stage-B vs Stage-C agreement (exact hypergeometric + information-matched null) and dependence audit")
SBset = set(top_idx(b_scores, K_MAIN)); h3 = []
for nm, feats in cfgs.items():
    ov = len(SBset & set(stageC[nm]["greedy"])); p_ex = float(hypergeom.sf(ov - 1, n, K_MAIN, K_MAIN)) if ov > 0 else 1.0
    comp = utility(feats)[pool]; rho = float(spearmanr(b_scores, comp)[0]); zb = (pd.Series(b_scores).rank().to_numpy() - 1) / (n - 1)
    zb = norm.ppf(np.clip((zb * (n - 1) + 0.5) / n, 1e-6, 1 - 1e-6)); cnt = 0
    for _ in range(N_H3):
        zc = rho * zb + math.sqrt(max(1 - rho ** 2, 0)) * rng.standard_normal(n); cnt += len(SBset & set(top_idx(zc, K_MAIN))) >= ov
    p_c = mc_p(cnt, N_H3)
    h3.append({"feature_set": nm, "overlap": ov, "K": K_MAIN, "pool": n, "p_uniform_exact": p_ex, "spearman_B_vs_composite": rho, "p_information_matched": p_c,
               "p_information_matched_mcse": mc_se(p_c, N_H3), "expected_overlap_uniform": K_MAIN * K_MAIN / n})
H3 = pd.DataFrame(h3); savet(H3, "T15_H3_overlap"); log(H3.round(5).to_string(index=False)); KEY["h3"] = H3.round(5).to_dict("records")
dep_feats = ["close_dist", "eig_unw", "service_km", "betw_dist", "deg"]; dep = []
for tgt in (PRIM, "betw_dist", "overload_a05"):
    for f in dep_feats:
        for scope_nm, idx in (("pool", pool), ("all53", np.arange(N))):
            a_, b_ = V[f].to_numpy()[idx], V[tgt].to_numpy()[idx]
            if f == tgt or np.std(a_) == 0 or np.std(b_) == 0: continue
            dep.append({"target": tgt, "feature": f, "population": scope_nm, "n": len(idx), "spearman": spearmanr(a_, b_)[0], "pearson": pearsonr(a_, b_)[0], "shared_rank_variance": spearmanr(a_, b_)[0] ** 2})
DEP = pd.DataFrame(dep); savet(DEP, "T16_dependence_audit")
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
a = ax[0]; kk = np.arange(0, K_MAIN + 1); a.bar(kk, hypergeom.pmf(kk, n, K_MAIN, K_MAIN), color=COL["grey"], alpha=.8, label="Exact hypergeometric null")
for r in H3.itertuples(): a.axvline(r.overlap + 0.05 * (list(cfgs).index(r.feature_set) - 1), lw=2, label=f"{r.feature_set.split(' (')[0]}: {r.overlap}/{K_MAIN}")
a.set_xlabel("Overlap with Stage-B top-K"); a.set_ylabel("Probability"); a.set_title("A. Uniform null (exact)"); a.legend(fontsize=6)
a = ax[1]; a.bar(H3.feature_set.str.split(" \\(").str[0], H3.p_uniform_exact, width=.35, label="uniform exact", color=COL["grey"]); a.bar(np.arange(len(H3)) + .35, H3.p_information_matched, width=.35, label="information-matched", color=COL["red"])
a.set_yscale("log"); a.set_ylabel("p-value"); a.axhline(.05, color="k", ls=":"); a.set_title("B. Control for shared information"); a.legend(); a.tick_params(axis="x", rotation=15)
a = ax[2]; M_ = pd.DataFrame({k: V[k] for k in [PRIM, "betw_dist", "overload_a05", "close_dist", "eig_unw", "service_km", "deg"]}).iloc[pool].corr(method="spearman")
im = a.imshow(M_, vmin=-1, vmax=1, cmap="RdBu_r"); a.set_xticks(range(len(M_))); a.set_xticklabels(M_.columns, rotation=60, ha="right", fontsize=6); a.set_yticks(range(len(M_))); a.set_yticklabels(M_.index, fontsize=6)
for i in range(len(M_)):
    for j in range(len(M_)): a.text(j, i, f"{M_.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5)
a.set_title("C. Spearman matrix (out-of-scope pool)"); savefig(fig, "F08_H3_dependence")

# ============================================================================ 10. SPATIAL ANALYSIS
hdr("[10] SPATIAL: Moran's I (k=3..8 ALL reported), railway-neighbour weights, LISA, Gi*, top-K clustering, power")
vp = V[PRIM].to_numpy()[pool]; Dp = Dkm[np.ix_(pool, pool)]; EI = -1 / (n - 1)
def knn_W(D, k, rowstd=True):
    m = len(D); W = np.zeros((m, m)); DD = D + np.diag(np.full(m, np.inf))
    for i in range(m): W[i, np.argsort(DD[i], kind="stable")[:k]] = 1
    return W / W.sum(1, keepdims=True) if rowstd else W
def moran(z, W, P, rg):
    m = len(z); zc = z - z.mean(); S0 = W.sum(); den = (zc ** 2).sum()
    if S0 == 0 or den == 0: return None
    I = (m / S0) * (zc @ W @ zc) / den; perms = np.argsort(rg.random((P, m)), axis=1); Zp = zc[perms]; Ip = (m / S0) * ((Zp @ W) * Zp).sum(1) / den
    return {"I": I, "E_I": -1 / (m - 1), "perm_mean": Ip.mean(), "z": (I - Ip.mean()) / Ip.std(), "p_two": mc_p(int((np.abs(Ip - Ip.mean()) >= abs(I - Ip.mean())).sum()), P), "p_pos": mc_p(int((Ip >= I).sum()), P)}
sp_rows = []
for k in range(3, 9):
    r = moran(vp, knn_W(Dp, k), N_MORAN, rng); sp_rows.append({"weights": f"geographic kNN k={k}", "k": k, **r})
Wadj = A_hop[np.ix_(pool, pool)]; r = moran(vp, Wadj, N_MORAN, rng)
if r: sp_rows.append({"weights": "service adjacency (pool-pool edges)", "k": np.nan, **r})
Wsh = ((A_hop @ A_hop)[np.ix_(pool, pool)] > 0).astype(float); np.fill_diagonal(Wsh, 0); r = moran(vp, Wsh, N_MORAN, rng)
if r: sp_rows.append({"weights": "railway neighbour (share a service neighbour)", "k": np.nan, **r})
SP = pd.DataFrame(sp_rows); SP["p_two_bh"] = bh(SP.p_two); savet(SP, "T17_moran_all_weights"); log(SP.round(4).to_string(index=False))
KEY["moran"] = {"n": n, "expected_I": EI, "I_range_knn": [float(SP.I[SP.k.notna()].min()), float(SP.I[SP.k.notna()].max())],
                "p_two_range_knn": [float(SP.p_two[SP.k.notna()].min()), float(SP.p_two[SP.k.notna()].max())], "min_p_positive_any_weights_raw": float(SP.p_pos.min()),
                "k_values_reported": list(range(3, 9))}
# LISA and Gi*
W5 = knn_W(Dp, 5); zc = vp - vp.mean(); m2 = (zc ** 2).mean(); lisa = []
for i in range(n):
    oth = np.delete(np.arange(n), i); Wi = W5[i, oth]; li_ = zc[i] * (Wi @ zc[oth]) / m2
    perm = np.argsort(rng.random((N_MORAN // 10 + 99, n - 1)), axis=1); Ip = zc[i] * (zc[oth][perm] @ Wi) / m2
    lisa.append({"station": pe[i], "local_I": li_, "p_pos": mc_p(int((Ip >= li_).sum()), len(Ip)), "quadrant": ("HH" if zc[i] > 0 and (W5[i] @ zc) > 0 else "LL" if zc[i] < 0 and (W5[i] @ zc) < 0 else "HL/LH")})
LI = pd.DataFrame(lisa); LI["p_bh"] = bh(LI.p_pos)
Wg = knn_W(Dp, 5, rowstd=False) + np.eye(n); xbar, s_ = vp.mean(), vp.std(); sw = Wg.sum(1); sw2 = (Wg ** 2).sum(1)
LI["Gi_star_z"] = (Wg @ vp - xbar * sw) / (s_ * np.sqrt((n * sw2 - sw ** 2) / (n - 1))); savet(LI, "T18_local_clustering")
KEY["lisa"] = {"n_significant_raw": int((LI.p_pos < .05).sum()), "n_significant_BH": int((LI.p_bh < .05).sum()), "max_Gi_star_z": float(LI.Gi_star_z.max())}
# clustering of the top-5 stations
t5 = top_idx(vp, 5); obs_d = Dp[np.ix_(t5, t5)][np.triu_indices(5, 1)].mean()
rnd = np.array([Dp[np.ix_(c, c)][np.triu_indices(5, 1)].mean() for c in (rng.choice(n, 5, replace=False) for _ in range(N_NULL))])
KEY["top5_spatial"] = {"mean_pairwise_km": float(obs_d), "random_mean_km": float(rnd.mean()), "p_lower_tail": mc_p(int((rnd <= obs_d).sum()), N_NULL)}
# power under pre-specified clustering scenarios
power = []; sorted_v = np.sort(vp); Wgen = knn_W(Dp, 5)
for rho_ in (0.0, 0.3, 0.5, 0.7, 0.9):
    Linv = np.linalg.inv(np.eye(n) - rho_ * Wgen)
    for kt in (5, 8):
        Wt = knn_W(Dp, kt); hit = 0
        for _ in range(N_POWER):
            y = Linv @ rng.standard_normal(n); y = sorted_v[pd.Series(y).rank(method="first").astype(int).to_numpy() - 1]
            hit += moran(y, Wt, 499, rng)["p_pos"] <= 0.05
        power.append({"true_rho": rho_, "test_k": kt, "detection_rate": hit / N_POWER})
PW = pd.DataFrame(power); savet(PW, "T19_spatial_power"); KEY["spatial_power"] = PW.round(3).to_dict("records")

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
a = ax[0, 0]; s_ = SP.reset_index(drop=True); 
a.scatter(range(len(s_)), s_.I, c=COL["blue"], zorder=3, label="Observed I"); a.scatter(range(len(s_)), s_.perm_mean, marker="x", c=COL["grey"], label="Permutation mean")
a.scatter(range(len(s_)), s_.E_I, marker="_", s=200, c=COL["red"], label=f"E[I] = {EI:.3f}"); a.set_xticks(range(len(s_))); a.set_xticklabels([w.replace("geographic ", "").replace("railway neighbour (share a service neighbour)", "rail-nbr").replace("service adjacency (pool-pool edges)", "svc-adj") for w in s_.weights], rotation=45, ha="right", fontsize=6)
a.set_ylabel("Moran's I"); a.set_title("A. Moran's I vs null expectation (all k = 3-8 shown)"); a.legend()
a = ax[0, 1]; a.bar(range(len(s_)), -np.log10(s_.p_two), color=COL["grey"]); a.axhline(-np.log10(.05), color=COL["red"], ls=":"); a.set_ylabel("-log10 p (two-sided)"); a.set_title("B. No detected global autocorrelation?"); a.set_xticks(range(len(s_))); a.set_xticklabels(range(1, len(s_) + 1))
a = ax[1, 0]
for kt, c in ((5, COL["blue"]), (8, COL["orange"])): q_ = PW[PW.test_k == kt]; a.plot(q_.true_rho, q_.detection_rate, "o-", color=c, label=f"test with kNN k={kt}")
a.axhline(.8, color="k", ls=":"); a.set_xlabel("Injected spatial dependence rho (pre-specified scenarios)"); a.set_ylabel("Detection rate (alpha=.05, one-sided)"); a.set_title("C. Power of Moran's I at N = %d" % n); a.legend()
a = ax[1, 1]; sc_ = a.scatter(lon[pool], lat[pool], c=LI.Gi_star_z, cmap="coolwarm", s=40 + 400 * pct(vp), edgecolor="k", lw=.4)
for i in top_idx(vp, 6): a.annotate(eng[i], (lon[i], lat[i]), fontsize=6, xytext=(3, 3), textcoords="offset points")
plt.colorbar(sc_, ax=a, label="Getis-Ord Gi* z"); a.set_xlabel("Longitude"); a.set_ylabel("Latitude"); a.set_title("D. Local hot/cold spots (marker size = value)")
savefig(fig, "F09_spatial")

# ============================================================================ 11. H2' REWIRING ENSEMBLE
hdr("[11] H2': degree-preserving rewiring ensemble (non-spatial and distance-constrained), fixed station labels")
edges0 = [(ix[r.a], ix[r.b]) for r in EDG.itertuples()]; edges0 = [(min(a, b), max(a, b)) for a, b in edges0]
def bfs_conn(adj):
    seen = {0}; st = [0]
    while st:
        u = st.pop()
        for w_ in adj[u]:
            if w_ not in seen: seen.add(w_); st.append(w_)
    return len(seen) == len(adj)
def rewire(edges, rg, nswap, tol=None):
    E_ = list(edges); S_ = set(E_); adj = {i: set() for i in range(N)}
    for a, b in E_: adj[a].add(b); adj[b].add(a)
    done = tries = 0
    while done < nswap and tries < nswap * 60:
        tries += 1; i, j = rg.integers(len(E_), size=2)
        if i == j: continue
        (a, b), (c, d) = E_[i], E_[j]
        if rg.random() < .5: c, d = d, c
        if len({a, b, c, d}) < 4: continue
        n1, n2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if n1 in S_ or n2 in S_: continue
        if tol is not None:
            old = Dkm[a, b] + Dkm[c, d]; new = Dkm[a, d] + Dkm[c, b]
            if abs(new - old) > tol * old: continue
        old1, old2 = E_[i], E_[j]
        for x, y in (old1, old2): adj[x].discard(y); adj[y].discard(x)
        for x, y in (n1, n2): adj[x].add(y); adj[y].add(x)
        if not bfs_conn(adj):
            for x, y in (n1, n2): adj[x].discard(y); adj[y].discard(x)
            for x, y in (old1, old2): adj[x].add(y); adj[y].add(x)
            continue
        S_.discard(old1); S_.discard(old2); S_.add(n1); S_.add(n2); E_[i], E_[j] = n1, n2; done += 1
    return E_, done
obs_hop = V.eff_loss_hop.to_numpy(); obs_topo = V.topo.to_numpy(); obs_set = set(top_idx(obs_hop, 5, pool))
OBS = {"gini_eff": gini(obs_hop[pool]), "gini_topo": gini(obs_topo[pool]), "top3_eff": top_share(obs_hop[pool], 3)}
rw_all = []
for variant, tol in (("degree-preserving", None), ("degree-preserving + length-constrained (25%)", 0.25)):
    rows_ = []; sw_done = []
    for r_ in range(N_REWIRE):
        Er, dn = rewire(edges0, rng, 5 * len(edges0), tol); sw_done.append(dn); A = np.zeros((N, N))
        for a, b in Er: A[a, b] = A[b, a] = 1
        tp, ef = removal_metrics(A, False)
        rows_.append({"variant": variant, "gini_eff": gini(ef[pool]), "gini_topo": gini(tp[pool]), "top3_eff": top_share(ef[pool], 3), "overlap_top5": len(obs_set & set(top_idx(ef, 5, pool))),
                      "max_eff": ef.max(), "n_nonzero_pool": int((ef[pool] > 1e-12).sum())})
    R_ = pd.DataFrame(rows_); R_["swaps_done"] = np.mean(sw_done); rw_all.append(R_)
RW = pd.concat(rw_all); savet(RW, "T20_rewiring_realizations")
rs = []
for variant, g in RW.groupby("variant"):
    for stat in ("gini_eff", "gini_topo", "top3_eff"):
        ob = OBS[stat]; x = g[stat].to_numpy(); p1 = mc_p(int((x >= ob).sum()), len(x))
        rs.append({"variant": variant, "statistic": stat, "observed": ob, "ensemble_mean": x.mean(), "ensemble_sd": x.std(), "z": (ob - x.mean()) / x.std() if x.std() > 0 else np.nan,
                   "percentile_of_observed": float((x < ob).mean()), "p_upper": p1, "p_upper_mcse": mc_se(p1, len(x)), "mean_swaps": g.swaps_done.iloc[0], "n_rewired": len(x),
                   "mean_top5_overlap": g.overlap_top5.mean(), "expected_overlap_random": 25 / n})
RS = pd.DataFrame(rs); savet(RS, "T21_rewiring_summary"); log(RS.round(4).to_string(index=False)); KEY["rewiring"] = RS.round(4).to_dict("records")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
for a, stat, ttl in zip(ax, ("gini_eff", "top3_eff", "overlap_top5"), ("Gini of efficiency loss (out-of-scope pool)", "Top-3 share", "Top-5 identity overlap with observed")):
    for variant, c in zip(RW.variant.unique(), (COL["blue"], COL["orange"])):
        x = RW[RW.variant == variant][stat]; a.hist(x, bins=30 if stat != "overlap_top5" else np.arange(-.5, 6.5, 1), alpha=.55, color=c, label=variant if stat == "gini_eff" else None)
    if stat in OBS: a.axvline(OBS[stat], color=COL["red"], lw=2, label="Observed")
    a.set_xlabel(ttl); a.set_ylabel(f"Rewired networks (n={N_REWIRE} per variant)")
ax[0].legend(fontsize=6); savefig(fig, "F10_rewiring_H2")

# ============================================================================ 12. EXOGENOUS VALIDATION, LAYERS, TEMPORAL STABILITY
hdr("[12] Exogenous validation (data NOT used to build rankings), multi-layer robust candidates, temporal stability")
exo = pd.DataFrame(index=nodes)
spf = B / "05l_special_terms_slim.csv"
if spf.exists():
    sp = rd(spf); vc = pd.concat([sp[c] for c in sp.columns if "경유" in c]).dropna().value_counts(); exo["via_count_special_terms"] = pd.Series(vc).reindex(nodes).fillna(0)
for c in ("dep_events", "arr_events", "loaded_wagons", "req_wagons_dep"):
    if c in master: exo[c] = master[c].fillna(0).to_numpy()
if "loaded_wagons" in exo and "req_wagons_dep" in exo: exo["completion_ratio"] = np.where(exo.req_wagons_dep > 0, exo.loaded_wagons / exo.req_wagons_dep, np.nan)
exo["wagon_flow"] = V.wagon_flow
vrows = []
for s_name in ("betw_dist", "eff_loss_dist", "close_dist", "service_km"):
    for xo in exo.columns:
        msk = (exo[xo].fillna(0) > 0).to_numpy()
        if msk.sum() < 8: continue
        a_, b_ = V[s_name].to_numpy()[msk], exo[xo].to_numpy()[msk]
        if np.std(a_) == 0 or np.std(b_) == 0: continue
        r0 = spearmanr(a_, b_)[0]; bs = []
        for _ in range(N_BOOT):
            ii = rng.integers(0, msk.sum(), msk.sum()); bs.append(spearmanr(a_[ii], b_[ii])[0])
        vrows.append({"structural_or_service": s_name, "exogenous": xo, "n_stations_with_data": int(msk.sum()), "spearman": r0, "ci_lo": np.nanpercentile(bs, 2.5), "ci_hi": np.nanpercentile(bs, 97.5)})
VX = pd.DataFrame(vrows); savet(VX, "T22_exogenous_validation"); KEY["exogenous"] = VX.round(3).to_dict("records")[:12]
KEY["stations_without_reservation_data"] = [eng[i] for i in range(N) if V.wagon_flow.iloc[i] == 0]
# layers
LY = pd.DataFrame({"station": eng, "structural_rank": pd.Series(-V[PRIM]).rank(method="first").astype(int).to_numpy(), "service_rank": pd.Series(-V.service_km).rank(method="first").astype(int).to_numpy(),
                   "demand_rank": np.where(V.wagon_flow > 0, pd.Series(-V.wagon_flow).rank(method="first").to_numpy(), np.nan), "in_reference_pool": np.isin(np.arange(N), pool)})
TOPQ = 8; pr = LY[LY.in_reference_pool].copy()
for c in ("structural_rank", "service_rank", "demand_rank"):
    pr[c.replace("rank", "top")] = pr[c].rank(method="first") <= TOPQ
pr["layers_in_top"] = pr[["structural_top", "service_top", "demand_top"]].sum(1); pr = pr.sort_values("layers_in_top", ascending=False); savet(pr, "T23_multilayer_candidates")
KEY["robust_candidates_in_>=2_layers"] = pr[pr.layers_in_top >= 2].station.tolist()
# temporal stability of reservation flows
rsf = B / "05e_reservations_slim.csv"
if rsf.exists():
    rs_ = rd(rsf); rs_["q"] = pd.to_datetime(rs_.rsv_date).dt.to_period("Q").astype(str); tq = []
    qs = sorted(rs_.q.unique())
    cnt_q = {q_: pd.concat([rs_[rs_.q == q_].dep_station, rs_[rs_.q == q_].arr_station]).value_counts().reindex(nodes).fillna(0) for q_ in qs}
    for q1, q2 in zip(qs[:-1], qs[1:]):
        both = (cnt_q[q1] > 0) & (cnt_q[q2] > 0)
        if both.sum() >= 5: tq.append({"quarter_1": q1, "quarter_2": q2, "n_stations": int(both.sum()), "spearman": spearmanr(cnt_q[q1][both], cnt_q[q2][both])[0]})
    TQ = pd.DataFrame(tq); savet(TQ, "T24_temporal_stability"); KEY["temporal_stability"] = TQ.round(3).to_dict("records")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
a = ax[0]
if len(VX):
    v_ = VX.sort_values("spearman").reset_index(drop=True); a.errorbar(v_.spearman, range(len(v_)), xerr=[v_.spearman - v_.ci_lo, v_.ci_hi - v_.spearman], fmt="o", color=COL["blue"]); a.set_yticks(range(len(v_)))
    a.set_yticklabels([f"{s} vs {x} (n={n_})" for s, x, n_ in zip(v_.structural_or_service, v_.exogenous, v_.n_stations_with_data)], fontsize=6); a.axvline(0, color="k", lw=.7)
a.set_xlabel("Spearman rho (bootstrap 95% CI)"); a.set_title("A. Structural / service indicators vs exogenous operational data")
a = ax[1]; p2 = pr.head(15).iloc[::-1]; a.barh(p2.station, p2.layers_in_top, color=COL["green"]); a.set_xlabel(f"Number of layers with the station in top-{TOPQ} of the pool"); a.set_title("B. Multi-layer robust candidates")
savefig(fig, "F11_exogenous_layers")

# ============================================================================ 13. SPECIFICATION CURVE
hdr("[13] Specification curve: scope definition x threshold x indicator")
scopes = {"Reference top-4 (service share)": Wm[:, d_ref], "Original four (service share)": Wm[:, d_orig]}
if "w_original" in master: scopes["Original four (line-tag weights)"] = master.w_original.fillna(0).to_numpy()
if "w_branch_to_trunk" in master: scopes["Original four (branch->trunk tags)"] = master.w_branch_to_trunk.fillna(0).to_numpy()
spec = []; SPM = ["eff_loss_dist", "eff_loss_hop", "topo", "betw_dist", "betw_unw", "close_dist", "eig_unw", "service_km"]
for sc_nm, w in scopes.items():
    for tau_ in (0.25, 0.5, 0.75):
        pl = pool_of(w, tau_); nn = len(pl)
        if nn < 8 or nn > N - 3: continue
        for mt in SPM:
            v = V[mt].to_numpy(float); g = gini(v[pl]); ridx = np.argsort(rng.random((N_NULL // 2, N)), axis=1)[:, :nn]; gn = gini_rows(v[ridx])
            a_ = 1 - w; T_ = (a_ * v).sum(); spec.append({"scope": sc_nm, "tau": tau_, "indicator": mt, "pool_n": nn, "gini": g, "effect_vs_randompool": g - np.nanmean(gn),
                "p_randompool": mc_p(int(np.nansum(gn >= g)), len(gn)), "OCG": T_ / v.sum(), "OCG_effect": T_ / v.sum() - a_.mean(), "top5": "|".join(eng[i] for i in top_idx(v, 5, pl))})
SPC = pd.DataFrame(spec).sort_values("effect_vs_randompool").reset_index(drop=True); savet(SPC, "T25_specification_curve")
KEY["spec_curve"] = {"n_specs": int(len(SPC)), "share_p_below_05": float((SPC.p_randompool < .05).mean()), "share_effect_positive": float((SPC.effect_vs_randompool > 0).mean()),
                     "median_effect": float(SPC.effect_vs_randompool.median()), "top5_frequency": (pd.Series("|".join(SPC.top5).split("|")).value_counts() / len(SPC)).head(8).round(3).to_dict()}
fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 2.2]})
ax[0].scatter(SPC.index, SPC.effect_vs_randompool, c=np.where(SPC.p_randompool < .05, COL["red"], COL["grey"]), s=14); ax[0].axhline(0, color="k", lw=.7)
ax[0].set_ylabel("Gini(out-of-scope) - random-pool null mean"); ax[0].set_title("Specification curve (red: p < 0.05 vs random pool of equal size)")
rowsel = [("scope", s_) for s_ in scopes] + [("tau", t_) for t_ in (0.25, 0.5, 0.75)] + [("indicator", m_) for m_ in SPM]
for r_, (col_, val_) in enumerate(rowsel): hit = SPC.index[SPC[col_] == val_]; ax[1].scatter(hit, np.full(len(hit), r_), marker="|", s=40, c="k")
ax[1].set_yticks(range(len(rowsel))); ax[1].set_yticklabels([f"{c}: {v}" for c, v in rowsel], fontsize=6); ax[1].invert_yaxis(); ax[1].set_xlabel("Specifications (sorted by effect)")
savefig(fig, "F12_specification_curve")

# ============================================================================ 14. NETWORK MAP + CASCADE DECOMPOSITION FIGURES
hdr("[14] Map and cascade-decomposition figures")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
a = ax[0]
for r in EDG.itertuples(): a.plot([lon[ix[r.a]], lon[ix[r.b]]], [lat[ix[r.a]], lat[ix[r.b]]], color="#bbbbbb", lw=.6, zorder=1)
sc_ = a.scatter(lon, lat, c=w_ref, cmap="RdYlBu", vmin=0, vmax=1, s=30 + 500 * pct(V[PRIM]) ** 3, edgecolor="k", lw=.4, zorder=2)
for i in top_idx(V[PRIM].to_numpy(), 8, pool): a.annotate(eng[i], (lon[i], lat[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")
plt.colorbar(sc_, ax=a, shrink=.7, label="Coverage w_i under reference scope"); a.set_title("A. Service-connection network (edges are train services, not track)\nmarker size = structural value; labels = top-8 uncovered stations"); a.set_xlabel("Longitude"); a.set_ylabel("Latitude")
a = ax[1]; o = np.argsort(-V.overload_a05.to_numpy())[:15][::-1]
a.barh([eng[i] for i in o], V.topo_alpha_inf.to_numpy()[o], color=COL["blue"], label="Topological disconnection only (alpha = inf)")
a.barh([eng[i] for i in o], V.overload_extra_a05.to_numpy()[o], left=V.topo_alpha_inf.to_numpy()[o], color=COL["orange"], label="Additional overload cascade (alpha = 0.5)")
a.set_xlabel("Fraction of stations failed after single-node removal"); a.set_title("B. Decomposition of legacy 'cascade impact'"); a.legend(fontsize=7)
savefig(fig, "F01_map_and_cascade_decomposition")

# ============================================================================ 15. SUMMARY / RECALIBRATED HYPOTHESES
hdr("[15] RESULTS SUMMARY AND RECALIBRATED HYPOTHESES")
cs = CT.set_index("indicator")
h1_ok = bool((cs.loc[FAM, "p_randompool_holm"] < .05).any() and (cs.loc[PRIM, "gini_pool"] > cs.loc[PRIM, "gini_inscope"] if len(inscope) > 1 else True))
rw_main = RS[(RS.variant == "degree-preserving") & (RS.statistic == "gini_eff")].iloc[0]
h2_ok = bool(rw_main.p_upper < .05)
h3_ok = bool((H3.p_information_matched < .05).any())
h4_ok = bool(KEY["dibi"]["spearman_DIBI_vs_naive"] < .9 or KEY["dibi"]["jaccard_top10_DIBI_vs_naive"] < .6)
SP["p_pos_bh"] = bh(SP.p_pos); h5_clust = bool(SP.p_pos_bh.min() < .05);   # BH across the 8 weight definitions
pw05 = PW[(PW.true_rho == .5) & (PW.test_k == 5)].detection_rate.iloc[0]
VERD = {"H1' (out-of-scope value is more concentrated than a random same-size station set; Holm-adjusted)": h1_ok,
        "H2' (observed concentration is unusual vs degree-preserving rewired ensembles)": h2_ok,
        "H3' (Stage B/C agreement exceeds an information-matched null)": h3_ok,
        "H4' (DIBI adds information beyond naive ranking)": h4_ok,
        "H5' (global spatial clustering detected)": h5_clust}
KEY["verdicts"] = VERD; KEY["power_at_rho_0.5_k5"] = float(pw05)
(OUT / "manuscript_numbers.json").write_text(json.dumps(KEY, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
md = f"""# Re-analysis summary (auto-generated; seed {SEED}{', QUICK MODE - do not cite' if QUICK else ''})

## Setting
- {N} stations, {len(EDG)} service edges, {L} lines, {len(tt)} trains. **Timetable reference year: UNKNOWN (author must state).**
- Design space: {nS} line subsets. Reference scope (top-{M_REF} lines by train-km): {KEY['reference_lines']}; original four lines: {KEY['original_four_present']}; identical: {KEY['reference_equals_original']}.

## Key numbers
- OCG reference/original: {KEY['ocg']['reference']:.3f} / {KEY['ocg']['original_four']:.3f}; baselines (1 - mean w): {KEY['ocg']['reference_baseline']:.3f} / {KEY['ocg']['original_baseline']:.3f}; effects {KEY['ocg']['reference_effect']:+.3f} / {KEY['ocg']['original_effect']:+.3f}.
- Pool size (reference): {n}. Primary indicator ({PRIM}) Gini: {cs.loc[PRIM, 'gini_pool']:.3f} (in-scope {cs.loc[PRIM, 'gini_inscope']:.3f}); random-pool p={cs.loc[PRIM, 'p_randompool']:.4f}; Dirichlet p={cs.loc[PRIM, 'p_dirichlet']:.4f}.
- DIBI top-5: {KEY['dibi']['top5_DIBI']}; invariant blind spots: {KEY['dibi']['invariant_blind_spots']}; Spearman(DIBI, naive)={KEY['dibi']['spearman_DIBI_vs_naive']:.3f}.
- Stage C (K={K_MAIN}, lambda={LAMBDA}): greedy {KEY['stageC']['greedy_set']}; exact {KEY['stageC']['exact_set']}; greedy/exact = {KEY['stageC']['greedy_over_exact']}; monotone violations {viol_mono}, submodular violations {viol_sub}.
- H3 overlaps / information-matched p: {[(r['feature_set'].split(' (')[0], r['overlap'], r['p_information_matched']) for r in KEY['h3']]}.
- Moran I range (k=3..8): {KEY['moran']['I_range_knn']} vs E[I]={EI:.4f}; p range {KEY['moran']['p_two_range_knn']}; power at rho=0.5 (k=5) = {pw05:.2f}.
- Rewiring (degree-preserving) Gini: observed {rw_main.observed:.3f} vs ensemble {rw_main.ensemble_mean:.3f} +/- {rw_main.ensemble_sd:.3f}; p={rw_main.p_upper:.4f} (n={int(rw_main.n_rewired)}).
- Specification curve: {KEY['spec_curve']['share_p_below_05']:.0%} of {KEY['spec_curve']['n_specs']} specifications have p<.05 vs random-pool null.

## Recalibrated hypotheses (data-driven verdicts)
""" + "\n".join(f"- {k}: **{'SUPPORTED' if v else 'NOT SUPPORTED'}**" if not k.startswith("H5") else f"- {k}: **{'DETECTED' if v else 'NOT DETECTED (interpret with power above; never as proof of dispersion)'}**" for k, v in VERD.items()) + """

## Wording rules for the manuscript
- Say "train-service connection network", never "physical topology".
- Say "no detected global spatial autocorrelation", never "dispersion".
- Say "agreement between partially information-sharing procedures", never "independent validation".
- Say "coverage of the composite objective", never "% of structural risk".
- K=5 is a cardinality limit, not a budget (unless the cost-weighted version is used).
- Legacy 'cascade impact' = single-node removal disconnection + simplified overload redistribution (see F01B).
"""
(OUT / "results_summary.md").write_text(md, encoding="utf-8")
log(md); log(f"\nDONE. Tables: {TAB}\nFigures: {FIG}\nSummary: {OUT/'results_summary.md'}")
