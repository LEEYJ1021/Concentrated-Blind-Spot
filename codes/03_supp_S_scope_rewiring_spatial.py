# -*- coding: utf-8 -*-
"""
SUPPLEMENT S (run AFTER rail_reanalysis_full.py).  QUICK=1 for a smoke test (never cite QUICK output).
Fixes: (1) manuscript scope (w_original, 34-station pool) promoted to PRIMARY; (2) rewiring NaN + tie artefacts;
(3) Moran k<=n-2, row-standardised weights, power / minimum detectable rho; (4) Stage C with K = fraction of pool,
tie-aware overlap, exact hypergeometric + information-matched nulls; (5) extracts of existing tables.
Everything is computed dynamically; the only fixed choices are listed in CONFIG and are reported in the output JSON.
"""
import os, json, math, warnings
from itertools import combinations, islice
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, norm, hypergeom
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ================================================================ CONFIG
ROOT = Path(os.environ.get("RAIL_ROOT", "/home/yjlee/Research/AI_Rail_OM"))
B, R1 = ROOT / "Data_bundle", ROOT / "Reanalysis_v1"
OUT = R1 / "supp"; TAB, FIG = OUT / "tables", OUT / "figures"
for d in (OUT, TAB, FIG): d.mkdir(parents=True, exist_ok=True)
QUICK = os.environ.get("QUICK", "0") == "1"
SEED = 20261010; rng = np.random.default_rng(SEED)
N_NULL, N_MORAN = (300, 199) if QUICK else (10000, 9999)
N_REWIRE, N_POWER = (40, 30) if QUICK else (1000, 300)
N_H3, N_RAND, N_TIE = (2000, 500, 50) if QUICK else (50000, 5000, 200)
EXACT_CAP = 300_000 if QUICK else 6_000_000
TAU, LAMBDA, K_FRACS, MIN_K = 0.5, 0.30, (0.10, 0.15, 0.20), 3
FAM = ["eff_loss_dist", "betw_dist", "service_km"]                     # pre-specified confirmatory family
IND = ["eff_loss_dist", "eff_loss_hop", "topo", "betw_dist", "betw_unw", "close_dist", "eig_unw", "deg", "service_km", "overload_a05"]
ORIG_FOUR = ["경부선", "충북선", "영동선", "중앙선"]
KEY = {"seed": SEED, "quick_mode": QUICK, "config": {"tau": TAU, "lambda": LAMBDA, "K_fractions": K_FRACS, "family": FAM}}

def log(*a): print(*a, flush=True)
def hdr(t): log("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)
def rd(p): return pd.read_csv(p, encoding="utf-8-sig")
def savet(df, nm): df.to_csv(TAB / f"{nm}.csv", index=False, encoding="utf-8-sig")
def savefig(fig, nm): fig.tight_layout(); fig.savefig(FIG / f"{nm}.png", dpi=300); fig.savefig(FIG / f"{nm}.pdf"); plt.close(fig)
def mc_p(c, n): return (c + 1) / (n + 1)
def mc_se(p, n): return math.sqrt(max(p * (1 - p), 1e-12) / n)
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj
def bh(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps); adj = np.empty(m); prev = 1.0
    for r in range(m - 1, -1, -1): i = o[r]; prev = min(prev, ps[i] * m / (r + 1)); adj[i] = prev
    return adj
def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x); s = x.sum()
    if n == 0 or s <= 0: return np.nan
    i = np.arange(1, n + 1); return 2 * (i * x).sum() / (n * s) - (n + 1) / n
def gini_rows(X):
    X = np.sort(X, axis=1); n = X.shape[1]; i = np.arange(1, n + 1); s = X.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"): return 2 * (X * i).sum(1) / (n * s) - (n + 1) / n
def top_share(x, k):
    x = np.sort(np.asarray(x, float))[::-1]; return x[:k].sum() / x.sum() if x.sum() > 0 else np.nan
def top_idx(v, k, pool=None, rg=None):                                  # rg given -> random tie-break (removes index-order artefact)
    v = np.asarray(v, float); pool = np.arange(len(v)) if pool is None else np.asarray(pool)
    tb = rg.random(len(pool)) if rg is not None else pool.astype(float)
    return pool[np.lexsort((tb, -v[pool]))[:k]]
def pct(s): return pd.Series(s).rank(pct=True).to_numpy()
def haversine(lat, lon):
    la, lo = np.radians(lat), np.radians(lon); dla = la[:, None] - la[None, :]; dlo = lo[:, None] - lo[None, :]
    h = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))

# ================================================================ LOAD
hdr("[S0] LOAD (indicators from T02; scopes rebuilt)")
master = rd(B / "01_station_master.csv"); tt = rd(B / "03a_timetable_trains.csv")
nodes = master.station_kor.tolist(); N = len(nodes); ix = {n: i for i, n in enumerate(nodes)}
eng = master.station_eng.fillna(master.station_kor).tolist()
lat, lon = master.latitude.to_numpy(float), master.longitude.to_numpy(float); Dkm = haversine(lat, lon)
V = rd(R1 / "tables" / "T02_station_indicators.csv").set_index("station_kor").reindex(nodes)
pairs = {tuple(sorted((ix[o], ix[d]))) for o, d in zip(tt.origin, tt.dest)}; edges0 = sorted(pairs)
A_hop = np.zeros((N, N))
for a, b in edges0: A_hop[a, b] = A_hop[b, a] = 1.0
tt["n_days"] = tt["n_days"].astype(float)
long = pd.concat([tt[["origin", "main_line_kor", "n_days"]].rename(columns={"origin": "st"}),
                  tt[["dest", "main_line_kor", "n_days"]].rename(columns={"dest": "st"})])
TD = long.pivot_table(index="st", columns="main_line_kor", values="n_days", aggfunc="sum", fill_value=0).reindex(index=nodes).fillna(0)
share = TD.div(TD.sum(1).replace(0, np.nan), axis=0).fillna(0); lines = list(share.columns); S_sh = share.to_numpy()
scopes = {}
for c, nm in (("w_original", "manuscript line-tag weights"), ("w_branch_to_trunk", "branch->trunk tag weights")):
    if c in master: scopes[nm] = master[c].fillna(0).to_numpy(float)
orig = [l for l in ORIG_FOUR if l in lines]; scopes["original four, service share"] = share[orig].sum(1).to_numpy()
tkm = (tt.n_days * tt.distance_km).groupby(tt.main_line_kor).sum(); ref = tkm.sort_values(ascending=False).index[:len(orig)].tolist()
scopes[f"top-{len(orig)} by train-km, service share"] = share[ref].sum(1).to_numpy()
PRIMARY = list(scopes)[0]; KEY["primary_scope"] = PRIMARY
KEY["pool_sizes"] = {k: int((w < TAU).sum()) for k, w in scopes.items()}; log(json.dumps(KEY["pool_sizes"], indent=1)); log("PRIMARY =", PRIMARY)

# ================================================================ S1 CONCENTRATION (all scopes; random-pool / Dirichlet / in-vs-out contrast / MDE)
hdr("[S1] Concentration on the manuscript's own scope + contrast with in-scope Gini")
def conc(snm, w):
    pool = np.where(w < TAU)[0]; n = len(pool); ins = np.setdiff1d(np.arange(N), pool)
    if n < 4 or len(ins) < 4: return None
    perm = np.argsort(rng.random((N_NULL, N)), axis=1); P_, C_ = perm[:, :n], perm[:, n:]; rows = []
    for k in IND:
        v = V[k].to_numpy(float); x = v[pool]; g, gi = gini(x), gini(v[ins]); diff = g - gi
        g_rp, g_cp = gini_rows(v[P_]), gini_rows(v[C_]); dn = g_rp - g_cp; ok = np.isfinite(g_rp) & np.isfinite(dn)
        g_dir = gini_rows(rng.dirichlet(np.ones(n), N_NULL)); okd = np.isfinite(g)
        p_rp = mc_p(int((g_rp[ok] >= g).sum()), int(ok.sum())) if okd else np.nan
        p_df = mc_p(int((dn[ok] >= diff).sum()), int(ok.sum())) if okd and np.isfinite(gi) else np.nan
        p_di = mc_p(int((g_dir >= g).sum()), N_NULL) if okd else np.nan
        sd = np.nanstd(g_rp); rows.append({"scope": snm, "indicator": k, "n_pool": n, "n_zero_in_pool": int((x == 0).sum()), "gini_pool": g, "gini_inscope": gi,
            "gini_diff_out_minus_in": diff, "top3_share_pool": top_share(x, 3), "null_randompool_mean": np.nanmean(g_rp), "null_randompool_sd": sd,
            "effect_vs_randompool": g - np.nanmean(g_rp), "p_randompool": p_rp, "p_randompool_mcse": mc_se(p_rp, N_NULL) if okd else np.nan,
            "p_diff_label_permutation": p_df, "p_dirichlet_legacy": p_di, "MDE_gini_80pct_power": (norm.ppf(.95) + norm.ppf(.8)) * sd})
    D = pd.DataFrame(rows)
    for col in ("p_randompool", "p_diff_label_permutation"):
        m_ = D.indicator.isin(FAM) & D[col].notna(); D[col + "_holm"] = np.nan; D.loc[m_, col + "_holm"] = holm(D.loc[m_, col])
    return D
CT = pd.concat([d for d in (conc(s, w) for s, w in scopes.items()) if d is not None]); savet(CT, "S1_concentration_all_scopes")
log(CT[CT.scope == PRIMARY].round(4).drop(columns=["scope"]).to_string(index=False))
KEY["concentration_primary"] = CT[CT.scope == PRIMARY].set_index("indicator").round(5).to_dict("index")

# ================================================================ S2 OCG vs permutation baseline + design-space rank
hdr("[S2] OCG: exact baseline 1 - mean(w), permutation z/p, rank among all line subsets of equal size")
def design_rank(v, m, w):
    v = np.asarray(v, float); cv = np.array([(S_sh[:, list(c)].sum(1) * v).sum() / v.sum() for c in combinations(range(len(lines)), m)]); c0 = (w * v).sum() / v.sum()
    return int((cv > c0 + 1e-12).sum() + 1), len(cv), float(cv.max()), float(np.median(cv))
oc = []
for snm, w in scopes.items():
    for k in FAM:
        v = V[k].to_numpy(float); a = 1 - w; obs = (a * v).sum() / v.sum(); Pm = np.argsort(rng.random((N_NULL, N)), axis=1); op = (v[Pm] * a).sum(1) / v.sum()
        row = {"scope": snm, "indicator": k, "OCG": obs, "baseline_1_minus_mean_w": a.mean(), "effect": obs - a.mean(), "perm_sd": op.std(),
               "z": (obs - op.mean()) / op.std(), "p_upper": mc_p(int((op >= obs).sum()), N_NULL), "p_lower": mc_p(int((op <= obs).sum()), N_NULL)}
        if "service share" in snm:
            r_, nd, cm, cmed = design_rank(v, len(orig), w); row.update({"coverage_rank_among_designs": r_, "n_designs": nd, "best_possible_coverage": cm, "median_design_coverage": cmed, "own_coverage": 1 - obs})
        oc.append(row)
OC = pd.DataFrame(oc); savet(OC, "S2_OCG_baseline_rank"); log(OC.round(4).to_string(index=False)); KEY["ocg"] = OC.round(5).to_dict("records")

# ================================================================ S3 STAGE C on the primary pool (K = fraction of pool)
hdr("[S3] Stage C on the primary pool: tie-aware overlap, exact hypergeometric, information-matched null, random-set percentile")
pool = np.where(scopes[PRIMARY] < TAU)[0]; n = len(pool); pe = [eng[i] for i in pool]
FEATS = {"Full (closeness, eigenvector, service, betweenness)": ["close_dist", "eig_unw", "service_km", "betw_dist"],
         "Service-augmented (closeness, eigenvector, service)": ["close_dist", "eig_unw", "service_km"], "Distinct only (closeness, eigenvector)": ["close_dist", "eig_unw"]}
Fp = {f: pct(V[f]) for f in {x for v in FEATS.values() for x in v}}
nb = (A_hop + np.eye(N)) > 0; J = (nb[:, None, :] & nb[None, :, :]).sum(2) / np.maximum((nb[:, None, :] | nb[None, :, :]).sum(2), 1)
def smat(pidx, lam): S = lam * J[np.ix_(pidx, pidx)]; np.fill_diagonal(S, 1.0); return S
def fval(S, u, sel): return float((S[list(sel)].max(0) * u).sum()) if len(sel) else 0.0
def greedy(S, u, K):
    cur = np.zeros(len(u)); sel = []; gains = []
    for _ in range(min(K, len(u))):
        g = (np.maximum(cur[None, :], S) * u).sum(1) - (cur * u).sum(); g[sel] = -np.inf; j = int(np.argmax(g)); sel.append(j); gains.append(float(g[j])); cur = np.maximum(cur, S[j])
    return sel, gains
def exact(S, u, K):
    if math.comb(len(u), K) > EXACT_CAP: return None, None
    best, bs = -1.0, None; it = combinations(range(len(u)), K)
    while True:
        ch = np.array(list(islice(it, 20000)))
        if len(ch) == 0: break
        vals = (S[ch].max(1) * u).sum(1); j = int(np.argmax(vals))
        if vals[j] > best: best, bs = float(vals[j]), tuple(int(t) for t in ch[j])
    return bs, best
S_c = smat(pool, LAMBDA); b_scores = V.eff_loss_dist.to_numpy(float)[pool]
Ks = sorted({max(MIN_K, int(round(f * n))) for f in K_FRACS}); Ks = [k for k in Ks if k < n]; rows = []
for K in Ks:
    for nm, feats in FEATS.items():
        u = np.mean([Fp[f] for f in feats], axis=0)[pool]; tot = u.sum(); sg, gains = greedy(S_c, u, K); se, fe = exact(S_c, u, K); fg = fval(S_c, u, sg)
        rv = np.array([fval(S_c, u, rng.choice(n, K, replace=False)) for _ in range(N_RAND)])
        tops = [top_idx(b_scores, K, rg=rng) for _ in range(N_TIE)]; ovs = np.array([len(set(t) & set(sg)) for t in tops])
        pu = np.array([float(hypergeom.sf(o - 1, n, K, K)) if o > 0 else 1.0 for o in ovs]); ov_med = int(np.median(ovs))
        rho = spearmanr(b_scores, u)[0]; r_ = pd.Series(b_scores).rank().to_numpy(); zb = norm.ppf((r_ - 0.5) / n)
        Zc = rho * zb + math.sqrt(max(1 - rho ** 2, 0)) * rng.standard_normal((N_H3, n)); tp = np.argsort(-Zc, axis=1)[:, :K]
        mb = np.zeros(n, bool); mb[tops[0]] = True; pim = mc_p(int((mb[tp].sum(1) >= ov_med).sum()), N_H3)
        rows.append({"K": K, "K_over_pool": K / n, "feature_set": nm, "greedy_set": "|".join(pe[i] for i in sg), "exact_set": "|".join(pe[i] for i in se) if se else "",
                     "greedy_coverage": fg / tot, "greedy_over_exact": fg / fe if fe else np.nan, "random_mean_coverage": rv.mean() / tot,
                     "greedy_percentile_vs_random_sets": float((rv < fg).mean()), "StageB_topK_objective_over_greedy": fval(S_c, u, tops[0]) / fg,
                     "overlap_mean_tiebreak": ovs.mean(), "overlap_median": ov_med, "expected_overlap_uniform": K * K / n, "p_uniform_exact_mean": pu.mean(),
                     "spearman_B_vs_composite": rho, "p_information_matched": pim, "p_info_matched_mcse": mc_se(pim, N_H3)})
SC = pd.DataFrame(rows); savet(SC, "S3_stageC_primary_pool"); log(SC.drop(columns=["greedy_set", "exact_set"]).round(4).to_string(index=False))
KEY["stageC"] = {"pool_n": n, "Ks": Ks, "rows": SC.round(5).to_dict("records")}

# ================================================================ S4 REWIRING (NaN-safe, tie-safe, consistent statistic)
hdr("[S4] Rewiring ensemble: valid-draw bookkeeping, tie-randomised overlap, same population as primary pool")
def eff_loss_hop(A):
    m = A.shape[0]; D0 = shortest_path(csr_matrix(A), method="D", directed=False, unweighted=True)
    inv0 = np.where(np.isfinite(D0) & (D0 > 0), 1.0 / np.where(D0 > 0, D0, 1), 0.0); loss = np.zeros(m)
    for v in range(m):
        idx = np.delete(np.arange(m), v); D = shortest_path(csr_matrix(A[np.ix_(idx, idx)]), method="D", directed=False, unweighted=True)
        inv = np.where(np.isfinite(D) & (D > 0), 1.0 / np.where(D > 0, D, 1), 0.0); base = inv0[np.ix_(idx, idx)].sum(); loss[v] = 1 - inv.sum() / base if base > 0 else 0.0
    return loss
obs = V.eff_loss_hop.to_numpy(float); chk = float(np.abs(eff_loss_hop(A_hop) - obs).max()); KEY["rewire_reproduction_max_abs_diff"] = chk; log(f"reproduction check vs T02 eff_loss_hop: {chk:.2e}")
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
        if tol is not None and abs((Dkm[a, d] + Dkm[c, b]) - (Dkm[a, b] + Dkm[c, d])) > tol * (Dkm[a, b] + Dkm[c, d]): continue
        o1, o2 = E_[i], E_[j]
        for x, y in (o1, o2): adj[x].discard(y); adj[y].discard(x)
        for x, y in (n1, n2): adj[x].add(y); adj[y].add(x)
        if not bfs_conn(adj):
            for x, y in (n1, n2): adj[x].discard(y); adj[y].discard(x)
            for x, y in (o1, o2): adj[x].add(y); adj[y].add(x)
            continue
        S_.discard(o1); S_.discard(o2); S_.add(n1); S_.add(n2); E_[i], E_[j] = n1, n2; done += 1
    return E_, done
def rw_stats(ef, rg):
    xp = ef[pool]; return {"gini_pool": gini(xp) if xp.sum() > 0 else np.nan, "gini_all53": gini(ef) if ef.sum() > 0 else np.nan, "top3_all53": top_share(ef, 3),
        "overlap5": len(set(top_idx(ef, 5, pool, rg)) & set(top_idx(obs, 5, pool, rg))), "n_nonzero_pool": int((xp > 1e-12).sum())}
OBS = rw_stats(obs, rng); rw = []
for variant, tol in (("degree-preserving", None), ("degree-preserving + length-constrained 25%", 0.25)):
    for _ in range(N_REWIRE):
        Er, dn = rewire(edges0, rng, 5 * len(edges0), tol); A = np.zeros((N, N))
        for a, b in Er: A[a, b] = A[b, a] = 1.0
        rw.append({"variant": variant, "swaps_done": dn, **rw_stats(eff_loss_hop(A), rng)})
RW = pd.DataFrame(rw); savet(RW, "S4_rewiring_realizations"); rs = []
for variant, g in RW.groupby("variant"):
    for st in ("gini_pool", "gini_all53", "top3_all53"):
        x = g[st].to_numpy(float); ok = np.isfinite(x); nv = int(ok.sum()); o = OBS[st]
        p = mc_p(int((x[ok] >= o).sum()), nv) if nv and np.isfinite(o) else np.nan
        rs.append({"variant": variant, "statistic": st, "observed": o, "ensemble_mean": np.nanmean(x) if nv else np.nan, "ensemble_sd": np.nanstd(x) if nv else np.nan, "n_valid": nv,
                   "n_degenerate": int(len(x) - nv), "percentile_of_observed": float((x[ok] < o).mean()) if nv else np.nan, "p_upper": p, "p_mcse": mc_se(p, nv) if nv else np.nan, "mean_swaps": g.swaps_done.mean()})
    rs.append({"variant": variant, "statistic": "overlap5 (tie-randomised)", "observed": np.nan, "ensemble_mean": g.overlap5.mean(), "ensemble_sd": g.overlap5.std(), "n_valid": len(g),
               "expected_random": 25 / n})
RS = pd.DataFrame(rs); savet(RS, "S4_rewiring_summary"); log(RS.round(4).to_string(index=False)); KEY["rewiring"] = RS.round(5).to_dict("records")

# ================================================================ S5 SPATIAL
hdr("[S5] Moran's I on the primary pool: valid k only, row-standardised weights, local Gi*, power / minimum detectable rho")
vp = V.eff_loss_dist.to_numpy(float)[pool]; Dp = Dkm[np.ix_(pool, pool)]; EI = -1 / (n - 1)
def rowstd(W): s = W.sum(1, keepdims=True); return np.divide(W, s, out=np.zeros_like(W), where=s > 0)
def knn_W(D, k):
    m = len(D); W = np.zeros((m, m)); DD = D + np.diag(np.full(m, np.inf))
    for i in range(m): W[i, np.argsort(DD[i], kind="stable")[:k]] = 1
    return rowstd(W)
def moran(z, W, P, rg):
    m = len(z); zc = z - z.mean(); S0 = W.sum(); den = (zc ** 2).sum()
    if S0 == 0 or den == 0: return None
    I = (m / S0) * (zc @ W @ zc) / den; Zp = zc[np.argsort(rg.random((P, m)), axis=1)]; Ip = (m / S0) * ((Zp @ W) * Zp).sum(1) / den; sd = Ip.std()
    return {"I": I, "E_I": -1 / (m - 1), "perm_mean": Ip.mean(), "z": (I - Ip.mean()) / sd if sd > 0 else np.nan,
            "p_two": mc_p(int((np.abs(Ip - Ip.mean()) >= abs(I - Ip.mean())).sum()), P), "p_pos": mc_p(int((Ip >= I).sum()), P)}
kmax = min(8, n - 2); WS = {f"kNN k={k}": knn_W(Dp, k) for k in range(3, kmax + 1)}
Dinv = np.where(Dp > 0, 1 / np.where(Dp > 0, Dp, 1), 0.0); WS["inverse distance"] = rowstd(Dinv)
WS["service adjacency (pool-pool)"] = rowstd(A_hop[np.ix_(pool, pool)])
W2 = ((A_hop @ A_hop)[np.ix_(pool, pool)] > 0).astype(float); np.fill_diagonal(W2, 0); WS["share a service neighbour"] = rowstd(W2)
sp = []
for nm, W in WS.items():
    r = moran(vp, W, N_MORAN, rng)
    if r: sp.append({"weights": nm, "n": n, "n_isolated": int((W.sum(1) == 0).sum()), **r})
SP = pd.DataFrame(sp); SP["p_pos_bh"] = bh(SP.p_pos); savet(SP, "S5_moran"); log(SP.round(4).to_string(index=False))
Wg = knn_W(Dp, min(5, n - 2)); Wb = (Wg > 0).astype(float) + np.eye(n); xbar, sx = vp.mean(), vp.std(); sw, sw2 = Wb.sum(1), (Wb ** 2).sum(1)
gi_z = (Wb @ vp - xbar * sw) / (sx * np.sqrt((n * sw2 - sw ** 2) / (n - 1))) if sx > 0 else np.full(n, np.nan); gp = []
for i in range(n):
    nbi = np.where(Wb[i] > 0)[0]; oth = nbi[nbi != i]; pool_o = np.delete(vp, i); pr = np.argsort(rng.random((N_MORAN // 10 + 99, n - 1)), axis=1)[:, :len(oth)]
    gp.append(mc_p(int(((pool_o[pr].sum(1) + vp[i]) >= vp[nbi].sum()).sum()), len(pr)))
LI = pd.DataFrame({"station": pe, "value": vp, "Gi_star_z": gi_z, "p_hot": gp}); LI["p_hot_bh"] = bh(LI.p_hot); savet(LI, "S5_local_Gi"); KEY["gi_star"] = {"n_hot_raw": int((LI.p_hot < .05).sum()), "n_hot_BH": int((LI.p_hot_bh < .05).sum())}
pw = []; sv = np.sort(vp); rhos = (0.0, 0.3, 0.5, 0.7, 0.9); Linv = {r_: np.linalg.inv(np.eye(n) - r_ * Wg) for r_ in rhos}
for r_ in rhos:
    for kt in sorted({min(5, n - 2), kmax}):
        Wt = knn_W(Dp, kt); hit = 0
        for _ in range(N_POWER):
            y = Linv[r_] @ rng.standard_normal(n); y = sv[pd.Series(y).rank(method="first").astype(int).to_numpy() - 1]; hit += moran(y, Wt, 499, rng)["p_pos"] <= 0.05
        pw.append({"true_rho": r_, "test_k": kt, "detection_rate": hit / N_POWER})
PW = pd.DataFrame(pw); savet(PW, "S5_power"); log(PW.round(3).to_string(index=False))
KEY["moran"] = {"n": n, "E_I": EI, "k_valid": list(range(3, kmax + 1)), "table": SP.round(5).to_dict("records")}
KEY["min_detectable_rho"] = {int(k): (float(g[g.detection_rate >= .8].true_rho.min()) if (g.detection_rate >= .8).any() else None) for k, g in PW.groupby("test_k")}

# ================================================================ S6 EXTRACTS OF EXISTING TABLES + candidate stability across specifications
hdr("[S6] Existing tables to inspect / share (T03, T07, T09, T22, T24, T25)")
def show(nm, q=None, n_=12):
    p = R1 / "tables" / nm
    if not p.exists(): log(f"-- {nm}: not found"); return None
    d = rd(p); d = d.query(q) if q else d; log(f"-- {nm} ({len(d)} rows)"); log(d.head(n_).round(4).to_string(index=False)); return d
show("T07_pareto_frontier.csv"); show("T09_corridor_vs_station.csv"); show("T22_exogenous_validation.csv"); show("T24_temporal_stability.csv")
spc = show("T25_specification_curve.csv", None, 0)
if spc is not None:
    log("share of specs with p<.05 by scope:"); log(spc.groupby("scope").p_randompool.apply(lambda s: (s < .05).mean()).round(3).to_string())
    sub = spc[spc.scope == PRIMARY]; fr = (pd.Series("|".join(sub.top5).split("|")).value_counts() / max(len(sub), 1)).head(10).round(3)
    log(f"candidate frequency across {len(sub)} specifications (scope = {PRIMARY}):"); log(fr.to_string()); KEY["spec_candidate_frequency_primary"] = fr.to_dict()

# ================================================================ S7 FIGURES + SUMMARY
c1 = CT[CT.scope == PRIMARY].set_index("indicator")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
ix_ = np.arange(len(c1)); ax[0].bar(ix_ - .2, c1.p_randompool, .4, label="random pool of equal size"); ax[0].bar(ix_ + .2, c1.p_dirichlet_legacy, .4, label="Dirichlet(1) (legacy)")
ax[0].set_yscale("log"); ax[0].axhline(.05, color="k", ls=":"); ax[0].set_xticks(ix_); ax[0].set_xticklabels(c1.index, rotation=60, ha="right", fontsize=6); ax[0].set_ylabel("p-value"); ax[0].set_title(f"A. Null-model dependence (n_pool={n})"); ax[0].legend(fontsize=6)
for variant in RW.variant.unique():
    x = RW[RW.variant == variant].gini_all53.dropna(); ax[1].hist(x, bins=30, alpha=.55, label=variant)
if np.isfinite(OBS["gini_all53"]): ax[1].axvline(OBS["gini_all53"], color="r", lw=2, label="observed")
ax[1].set_xlabel("Gini of efficiency loss, all 53 stations"); ax[1].set_title("B. Rewiring (valid draws only)"); ax[1].legend(fontsize=6)
for kt, g in PW.groupby("test_k"): ax[2].plot(g.true_rho, g.detection_rate, "o-", label=f"test kNN k={kt}")
ax[2].axhline(.8, color="k", ls=":"); ax[2].set_xlabel("injected spatial dependence rho"); ax[2].set_ylabel("detection rate"); ax[2].set_title(f"C. Power of Moran's I at n = {n}"); ax[2].legend(fontsize=6)
savefig(fig, "FS1_null_rewiring_power")
h1 = bool((c1.loc[FAM, "p_diff_label_permutation_holm"] < .05).any()); rw0 = RS[(RS.variant == "degree-preserving") & (RS.statistic == "gini_all53")].iloc[0]
KEY["verdicts"] = {"H1 (out-of-scope MORE concentrated than in-scope; Holm over family)": h1,
                   "H2 (observed Gini above rewired ensemble, valid draws; p<.05)": bool(np.isfinite(rw0.p_upper) and rw0.p_upper < .05),
                   "H3 (overlap beats information-matched null at any K/feature set)": bool((SC.p_information_matched < .05).any()),
                   "H5 (global clustering, BH over weights)": bool(SP.p_pos_bh.min() < .05)}
(OUT / "supp_numbers.json").write_text(json.dumps(KEY, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
md = f"""# Supplement summary (seed {SEED}{', QUICK - do not cite' if QUICK else ''})
Primary scope: {PRIMARY}; pool n = {n}. Verdicts: {json.dumps(KEY['verdicts'], ensure_ascii=False)}
Minimum detectable rho (power>=0.8): {KEY['min_detectable_rho']}. Gi* hot spots (BH): {KEY['gi_star']['n_hot_BH']}.
## Rules fixed BEFORE looking at results
- Report every verdict above, including nulls; do not choose a framing because it yields positive results.
- Non-significant Moran's I is reported as 'not detected', always with the power table.
- Overlap is 'agreement between information-sharing procedures'; coverage is 'composite-objective coverage', not '% of risk'.
"""
(OUT / "supp_summary.md").write_text(md, encoding="utf-8"); log(md); log(f"DONE -> {OUT}")
