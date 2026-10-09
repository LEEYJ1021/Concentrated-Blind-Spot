# -*- coding: utf-8 -*-
"""
make_figures.py (v4) - journal-grade figures for the scope-audit manuscript (Fig. 1-8, S1-S8); grayscale by default
=====================================================================================================================
Run :  python make_figures.py                 (all figures)
       FIGS=03,05,S2 python make_figures.py   (selected figures; IDs follow the MANUSCRIPT numbering, see below)
       QUICK=1 python make_figures.py         (smoke test: fewer simulations, 150 dpi; never submit QUICK output)
       COLOR=1 python make_figures.py         (semantic colour: conventional = vermillion, calibrated = blue; Fig. 3, 4a, S4)
       EDGES=core python make_figures.py      (Fig. 2: draw only service edges touching a core A/B station; default = all, light)
       FIG8C=panel python make_figures.py     (Fig. 8: keep the exogenous panel c; default = table file Table_Fig08c_exogenous.csv)
       S3_MAP=0 python make_figures.py        (Fig. S3: size screen only; default = size screen + Gi* map = the 'local' file)
       DELTA_BAND=0 python make_figures.py    (Fig. 7a without the +/-delta tolerance band)
Input  : <ROOT>/Data_bundle/*.csv  and  <ROOT>/Reanalysis_v1/{tables,supp,supp_c,supp_d,supp_e}/
Output : <ROOT>/Reanalysis_v1/manuscript_figures/{FigNN.pdf, .png}, figure_data/FigNN_*.csv, figure_manifest.csv,
         figure_captions_v4.md (corrected captions + body sentences, numbers filled from the data)

v4 changes (from the guideline list 6.1):
  [1] Fig. 7  a3 footer no longer cut: footer split into short lines, taller canvas, all text artists forced into the tight bbox.
  [2] Fig. 8  b axis description split into three short lines (was wider than the axes), larger bottom margin.
  [3] Stage C (manuscript Fig. S6, old file S5) panel c x-label wrapped, legend moved below the two-line label.
  [4] Dependence matrix (manuscript Fig. S7, old file S6) x-label wrapped under the matrix; empty first row/last column dropped.
  [5] Fig. 7  figure header and caption state 'hop-based (unweighted) efficiency loss' (other figures are distance-based).
  [6] Fig. 7c shades the interval (previous grid value, first grid value with power >= 0.8]; caption says the true minimum lies inside it.
  [7] Fig. 4a legend 'Top-11 leakage (20% of N)'; caption defines K = 11.
  [8] Fig. 5  outside-band effects are detected in code, ringed in the plot, named in the figure text, the caption and a CSV; reading guide no longer overflows.
  [9] Fig. 5a diamonds labelled 'normal approximation'; guide and caption make Fig. 4c simulated power the primary basis.
 [10] Fig. 6  box says 'service-share scope only'; legend box moved above the frontier; caption + body sentence generated.
 [11] Fig. S1 stations inside the line-tag scope are marked (asterisk + note).
 [12] Supplement numbering now follows the manuscript: S3 size screen (+Gi*), S4 in/out Gini, S5 specification curve, S6 Stage C,
      S7 dependence matrix, S8 exogenous/temporal. Old v3 files with other numbers are reported at the end of the run.
  extra: 'legacy' wording replaced by 'conventional' (guideline 0.4); audit counts text boxes beyond the canvas.
"""
import os, re, json, math, traceback, urllib.request, warnings
from pathlib import Path
from statistics import NormalDist
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
warnings.filterwarnings("ignore")

# ================================================================= CONFIG
ROOT = Path(os.environ.get("RAIL_ROOT", "."))
B, R1 = ROOT / "Data_bundle", ROOT / "Reanalysis_v1"
OUT = R1 / "manuscript_figures"; DATA = OUT / "figure_data"; GEO = B / "geo"
for d in (OUT, DATA, GEO): d.mkdir(parents=True, exist_ok=True)
QUICK = os.environ.get("QUICK", "0") == "1"; SAVE_TIFF = os.environ.get("TIFF", "0") == "1"
SHOW_BAND = os.environ.get("DELTA_BAND", "1") == "1"
COLOR = os.environ.get("COLOR", "0") == "1"; EDGES = os.environ.get("EDGES", "all"); FIG8C = os.environ.get("FIG8C", "table")
S3_MAP = os.environ.get("S3_MAP", "1") == "1"
SEED = 20261014; R_NULL = 2000 if QUICK else 20000; R_BAND = 500 if QUICK else 5000
DPI = 150 if QUICK else 600
ALPHA, TAU = 0.05, 0.5
MM = 1 / 25.4
EXT = (125.9, 129.9, 34.2, 38.4)
GEO_URLS = ["https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea_provinces_geo_simple.json",
            "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea_provinces_geo.json"]
ORIG_FOUR = ["경부선", "충북선", "영동선", "중앙선"]
FAM = ["eff_loss_dist", "betw_dist", "service_km"]
POWER_REF = "Fig. 4c"            # cross-reference used in the Fig. 5 guide; keep equal to the final numbering
CONV = "conventional"            # wording for the Dirichlet(1) reference (guideline 0.4: no 'legacy' in the manuscript)

K, G1, G2, G3, G4, W = "#000000", "#3A3A3A", "#7A7A7A", "#B4B4B4", "#E4E4E4", "#FFFFFF"
LEG = "#D55E00" if COLOR else K          # conventional / mis-calibrated reference
CAL = "#0072B2" if COLOR else K          # calibrated reference
CALFILL = "#9CC3DD" if COLOR else G3
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.0, "lines.markersize": 4, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 150, "hatch.linewidth": 0.5, "image.cmap": "Greys",
    "axes.prop_cycle": matplotlib.cycler(color=[K, G2, G3])})

NAMES = {"BusanNewPort": "Busan New Port", "JecheonYard": "Jecheon Yard", "ShingwangyangPort": "Shingwangyang-hang",
         "Shingwangyang-hang": "Shingwangyang-hang"}
STAGEC = ["Obong", "Busan New Port", "Goedong", "Susaek", "Shingwangyang-hang"]
IND_LAB = {"eff_loss_dist": "Efficiency loss (distance)", "eff_loss_hop": "Efficiency loss (hops)", "topo": "Disconnection",
           "betw_dist": "Betweenness (distance)", "betw_unw": "Betweenness (unweighted)", "close_dist": "Closeness",
           "eig_unw": "Eigenvector", "deg": "Degree", "service_km": "Service train-km", "overload_a05": "Cascade (overload, 0.5)"}
IND_SHORT = {"eff_loss_dist": "EffLoss-d", "eff_loss_hop": "EffLoss-h", "topo": "Discon.", "betw_dist": "Betw-d", "betw_unw": "Betw-u",
             "close_dist": "Close", "eig_unw": "Eigen", "service_km": "Svc-km", "deg": "Degree"}
OFFS = {"Goedong": (-6, 4, "right"), "Busan New Port": (-6, -10, "right"), "Donghae": (-6, 4, "right")}   # label offsets (dx, dy, ha)
OLD_V3_FILES = ["FigS3_in_vs_out_gini", "FigS4_specification_curve", "FigS5_stageC_diagnostics", "FigS6_dependence_matrix",
                "FigS7_exogenous_temporal", "FigS8_size_screen", "FigS8_size_screen_local"]
C = {}; FAILED = []; MANIFEST = []; CAPS = {}

# ================================================================= HELPERS
def log(*a): print(*a, flush=True)
def rd(p):
    p = Path(p)
    if not p.exists(): raise FileNotFoundError(f"missing input: {p}")
    return pd.read_csv(p, encoding="utf-8-sig")
def tab(rel): return rd(R1 / rel)
def jload(rel):
    p = R1 / rel; return json.loads(p.read_text("utf-8")) if p.exists() else {}
def src(df, name): df.to_csv(DATA / f"{name}.csv", index=False, encoding="utf-8-sig")
def nm(s): s = str(s); return NAMES.get(s, re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s))
def lab(x): return IND_LAB.get(x, x)
def stat_lab(s):
    m = re.match(r"top(\d+)_leakage", s)
    if m:
        k = int(m.group(1)); return f"Top-{k} leakage" + (" (20% of N)" if k == 11 else "")
    return {"pool_share (OCG)": "OCG (pool share)", "gini_pool": "Pool Gini", "gini_diff_out_minus_in": "Gini difference (out - in)",
            "pool_mean_rank": "Pool mean rank"}.get(s, s)
def short_scope(n):
    """map every scope name used in the tables to one of 4 canonical labels (order of tests matters)."""
    l = str(n).lower()
    if "branch" in l: return "Branch-to-trunk tag weights"
    if "manuscript" in l or "line-tag" in l: return "Manuscript line-tag weights"
    if "original four" in l: return "Original four lines (service share)"
    if "top-4" in l or l.startswith("top-") or "reference" in l: return "Top-4 lines by train-km"
    return str(n)
def fmt_p(p, R):
    if not np.isfinite(p): return "p = n/a"
    return f"p < {1 / R:.0e}".replace("e-0", "e-") if p <= 1.5 / (R + 1) else f"p = {p:.3f}"
def panel(ax, t, y=1.03): ax.text(-0.015, y, t, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")
def gini_rows(X):
    X = np.sort(X, axis=1); n = X.shape[1]; i = np.arange(1, n + 1); s = X.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"): return 2 * (X * i).sum(1) / (n * s) - (n + 1) / n
def gini(x): return float(gini_rows(np.asarray(x, float)[None, :])[0])
def top_share(x, k): x = np.sort(np.asarray(x, float))[::-1]; return float(x[:k].sum() / x.sum())
def n80(x): x = np.sort(np.asarray(x, float))[::-1]; return int(np.searchsorted(np.cumsum(x) / x.sum(), 0.8) + 1)

def audit(fig, name):
    """text-size / overlap / canvas / width audit (warnings only); returns width in mm after tight crop."""
    wmm = np.nan
    try:
        fig.canvas.draw(); r = fig.canvas.get_renderer(); items = []; small = 0
        for t in fig.findobj(mtext.Text):
            s = t.get_text().strip()
            if not t.get_visible() or not s: continue
            if t.get_fontsize() < 6.5 - 1e-6: small += 1
            try: items.append((s, t.get_window_extent(r)))
            except Exception: pass
        ov = [(items[i][0][:20], items[j][0][:20]) for i in range(len(items)) for j in range(i + 1, len(items)) if items[i][1].overlaps(items[j][1])]
        fb = fig.bbox
        beyond = [s[:24] for s, bb in items if bb.x1 > fb.x1 + 2 or bb.x0 < fb.x0 - 2 or bb.y0 < fb.y0 - 2 or bb.y1 > fb.y1 + 2]
        wmm = fig.get_tightbbox(r).width * 25.4
        log(f"   audit {name}: {len(items)} texts | below 6.5 pt: {small} | overlapping boxes: {len(ov)} {ov[:3] if ov else ''} | beyond canvas (kept by tight crop): {len(beyond)} {beyond[:3] if beyond else ''} | width after tight crop: {wmm:.0f} mm" + ("  <-- >190 mm" if wmm > 190.5 else ""))
        MANIFEST.append({"figure": name, "width_mm_after_crop": round(float(wmm), 1), "texts_below_6.5pt": small, "overlapping_text_boxes": len(ov),
                         "texts_beyond_canvas": len(beyond), "over_190mm": bool(wmm > 190.5)})
    except Exception as e: log("   audit skipped:", e)
    return wmm
def save(fig, name):
    extra = [t for t in fig.findobj(mtext.Text) if t.get_text().strip()]          # force every text (x-labels, footers) into the tight bbox
    kw = dict(bbox_inches="tight", pad_inches=0.06, bbox_extra_artists=extra)
    fig.savefig(OUT / f"{name}.pdf", **kw); fig.savefig(OUT / f"{name}.png", dpi=DPI, **kw)
    if SAVE_TIFF: fig.savefig(OUT / f"{name}.tiff", dpi=DPI, pil_kwargs={"compression": "tiff_lzw"}, **kw)
    audit(fig, name); plt.close(fig); log(f"   saved {name}")
def write_captions():
    if not CAPS: return
    L = ["# Corrected captions and body sentences (auto-generated by make_figures.py v4; numbers come from the data)\n"]
    for k, v in CAPS.items(): L.append(f"**{k}.** {v}\n")
    (OUT / "figure_captions_v4.md").write_text("\n".join(L), "utf-8"); log("   captions written: figure_captions_v4.md")

# ================================================================= DATA + MAP
def load_geo():
    cache = GEO / "skorea-provinces-geo.json"; cands = [os.environ.get("KOREA_GEOJSON"), cache]
    gj = None
    for c in cands:
        if c and Path(c).exists():
            try: gj = json.loads(Path(c).read_text("utf-8")); break
            except Exception: pass
    if gj is None:
        for u in GEO_URLS:
            try:
                raw = urllib.request.urlopen(u, timeout=25).read().decode("utf-8"); gj = json.loads(raw); cache.write_text(raw, "utf-8"); break
            except Exception as e: log("   map download failed:", u.split('/')[-1], type(e).__name__)
    if gj is None: log("   WARNING: no Korea map available - drawing without coastline"); return []
    polys = []
    for f in gj["features"]:
        g = f["geometry"]; rings = [g["coordinates"][0]] if g["type"] == "Polygon" else [p[0] for p in g["coordinates"]] if g["type"] == "MultiPolygon" else []
        polys += [np.asarray(r, float)[:, :2] for r in rings]
    if polys and np.abs(np.concatenate(polys)).max() > 400: log("   WARNING: map is not in WGS84 degrees - ignored"); return []
    return polys
def build_scopes(master, tt, nodes):
    tt = tt.copy(); tt["n_days"] = tt.n_days.astype(float)
    lg = pd.concat([tt[["origin", "main_line_kor", "n_days"]].rename(columns={"origin": "st"}), tt[["dest", "main_line_kor", "n_days"]].rename(columns={"dest": "st"})])
    TD = lg.pivot_table(index="st", columns="main_line_kor", values="n_days", aggfunc="sum", fill_value=0).reindex(index=nodes).fillna(0)
    sh = TD.div(TD.sum(1).replace(0, np.nan), axis=0).fillna(0); orig = [l for l in ORIG_FOUR if l in sh.columns]
    ref = (tt.n_days * tt.distance_km).groupby(tt.main_line_kor).sum().sort_values(ascending=False).index[:len(orig)].tolist(); S = {}
    if "w_original" in master: S["manuscript line-tag weights"] = master.w_original.fillna(0).to_numpy(float)
    if "w_branch_to_trunk" in master: S["branch->trunk tag weights"] = master.w_branch_to_trunk.fillna(0).to_numpy(float)
    S["original four, service share"] = sh[orig].sum(1).to_numpy(); S[f"top-{len(orig)} by train-km, service share"] = sh[ref].sum(1).to_numpy()
    return S
def find_scope(name):
    key = short_scope(name)
    for k, w in C["scopes"].items():
        if short_scope(k) == key: return w
    raise KeyError(name)
def check_names():
    """station-name integrity: the figures join tables by English station name, so duplicates / spelling mismatches silently mis-colour markers."""
    eng = [nm(e) for e in C["eng"]]; vc = pd.Series(eng).value_counts(); dup = vc[vc > 1]
    if len(dup): log(f"   WARNING duplicate station names (markers/tiers will be applied to ALL of them): {dup.to_dict()}")
    try:
        t3 = tab("supp_c/tables/C3_candidate_tiers.csv"); miss = [s for s in t3.station if nm(s) not in eng]
        if miss: log(f"   WARNING tier stations without a master match: {miss}")
    except Exception as e: log("   (tier check skipped:", e, ")")
    try:
        spc = tab("tables/T25_specification_curve.csv"); sn = [s for s in spc.scope.unique() if "line-tag" in s]
        if sn:
            w = C["scopes"][C["prim"]]; pool = {eng[i] for i in np.where(w < TAU)[0]}; names = {nm(x) for t in spc[spc.scope == sn[0]].top5 for x in str(t).split("|")}
            bad = sorted(names - pool)
            if bad: log(f"   WARNING top-5 stations in the specification curve that are NOT in the primary pool (mapping conflict?): {bad}")
            else: log("   name check OK: every specification-curve top-5 station lies in the primary pool")
    except Exception as e: log("   (spec-curve pool check skipped:", e, ")")
def prep():
    master = rd(B / "01_station_master.csv"); tt = rd(B / "03a_timetable_trains.csv"); nodes = master.station_kor.tolist()
    eng = master.station_eng.fillna(master.station_kor).tolist(); ix = {n: i for i, n in enumerate(nodes)}
    V = tab("tables/T02_station_indicators.csv").set_index("station_kor").reindex(nodes)
    pairs = sorted({tuple(sorted((ix[o], ix[d]))) for o, d in zip(tt.origin, tt.dest)})
    C.update(master=master, tt=tt, nodes=nodes, eng=eng, V=V, ix=ix, pairs=pairs, lon=master.longitude.to_numpy(float), lat=master.latitude.to_numpy(float),
             N=len(nodes), scopes=build_scopes(master, tt, nodes), polys=load_geo(), rng=np.random.default_rng(SEED), eng_ix={e: i for i, e in enumerate(eng)})
    C["prim"] = list(C["scopes"])[0]; log(f"   stations={C['N']}, edges={len(pairs)}, scopes={list(C['scopes'])}, map polygons={len(C['polys'])}")
    check_names()
def draw_map(ax):
    if C["polys"]: ax.add_collection(PolyCollection(C["polys"], facecolors="#F3F3F3", edgecolors="#8F8F8F", linewidths=0.3, zorder=0))
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3]); ax.set_aspect(1 / math.cos(math.radians(36)))
    ax.set_xticks([126, 127, 128, 129]); ax.set_xticklabels(["126°E", "127°E", "128°E", "129°E"], fontsize=6.5)
    ax.set_yticks([35, 36, 37, 38]); ax.set_yticklabels(["35°N", "36°N", "37°N", "38°N"], fontsize=6.5)
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(0.5)
def halo(): return [pe.withStroke(linewidth=1.8, foreground="white")]

# ================================================================= FIG 1  design flow
def box(ax, x, y, w, h, title, lines, fc=W, ec=K, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=7.5, fontweight="bold")
    ax.text(x + w / 2, y + h - 9.5, "\n".join(lines), ha="center", va="top", fontsize=6.8, linespacing=1.4)
YEAR_COL = re.compile(r"year|yr|date|dt|effective|valid|start|end|period|issue|version|연도|년도|일자|시행|적용|기준|일시", re.I)
def timetable_year(tt):
    """(label, source). Priority: env TIMETABLE_YEAR > unambiguous single year in date-like columns of the timetable file > None (placeholder kept)."""
    ov = os.environ.get("TIMETABLE_YEAR")
    if ov and ov.strip().lower() == "snapshot": return "hand-entered\nsingle snapshot", "env TIMETABLE_YEAR=snapshot (reference year not recorded)"
    if ov: return ov.strip(), "env TIMETABLE_YEAR"
    ys = set()
    for c in tt.columns:
        if YEAR_COL.search(str(c)): ys |= {int(y) for y in re.findall(r"(?<!\d)(20\d{2})(?!\d)", " ".join(tt[c].dropna().astype(str).head(20000)))}
    if len(ys) == 1: return str(next(iter(ys))), "timetable column"
    log(f"   WARNING timetable year not determined (found: {sorted(ys) or 'none'}); run check_timetable_year.py, then set TIMETABLE_YEAR=...")
    return None, "undetermined"
def fig01():
    nm_ = jload("manuscript_numbers.json"); sn = jload("supp/supp_numbers.json"); ps = sn.get("pool_sizes", {})
    pools = " / ".join(str(v) for v in ps.values()) if ps else f"{' / '.join(str(int((w < TAU).sum())) for w in C['scopes'].values())}"
    fig = plt.figure(figsize=(190 * MM, 80 * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-1.5, 191.5); ax.set_ylim(0, 80); ax.axis("off")
    bw, gap, y0, bh = 32, 7.5, 44, 34; xs = [i * (bw + gap) for i in range(5)]
    ty, ty_src = timetable_year(C["tt"]); log(f"   timetable year: {ty} ({ty_src})")
    nt = nm_.get("n_trains", len(C["tt"])); ns = nm_.get("n_stations", C["N"]); ne = nm_.get("n_edges_undirected", len(C["pairs"]))
    box(ax, xs[0], y0, bw, bh, "1  Data", [f"{nt} trains", f"→ {ns}-station", "train-service network", f"({ne} undirected edges)", (f"timetable: {ty}" if ty and "\n" in ty else f"timetable year: {ty or '[state]'}")])
    box(ax, xs[1], y0, bw, bh, "2  Scope definitions", ["4 definitions of", "'designated' lines", "stations outside scope:", pools])
    box(ax, xs[2], y0, bw, bh, "3  Audit statistics", ["OCG (pool share)", "pool Gini", "Gini(out) - Gini(in)", "top-K leakage"])
    box(ax, xs[3], y0, bw, bh, "4  Reference null", ["scope-matched", "random pool", "(calibrated)", f"vs Dirichlet(1) ({CONV})"])
    box(ax, xs[4], y0, bw, bh, "5  Two questions", ["calibration:", "size, power,", "detectability", "→ Korean application"], fc=G4)
    for i in range(4): ax.annotate("", xy=(xs[i + 1] - 0.4, y0 + bh / 2), xytext=(xs[i] + bw + 0.4, y0 + bh / 2), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=K))
    xl, xr = xs[2], xs[3] + bw; yb = 40; cw = (xr - xl - 2 * 2) / 3; cxs = [xl + i * (cw + 2) + cw / 2 for i in range(3)]
    b3, b4 = xs[2] + bw / 2, xs[3] + bw / 2
    for xb in (b3, b4): ax.plot([xb, xb], [y0, yb], color=K, lw=0.6)
    ax.plot([min(cxs + [b3]), max(cxs + [b4])], [yb, yb], color=K, lw=0.6)
    for i, (h, t) in enumerate([("HA", "size"), ("HB", "power"), ("HG", "detectability")]):
        x = xl + i * (cw + 2); ax.add_patch(FancyBboxPatch((x, 29.5), cw, 9, boxstyle="round,pad=0,rounding_size=1", fc=G1, ec=K, lw=0.6))
        ax.text(x + cw / 2, 34, f"{h}\n{t}", ha="center", va="center", color=W, fontsize=6.8, fontweight="bold", linespacing=1.25)
        ax.plot([cxs[i]] * 2, [yb, 38.5], color=K, lw=0.6)
    b5 = xs[4] + bw / 2; c1, c2 = xs[4] + 7.5, xs[4] + 24.5
    ax.plot([b5, b5], [y0, 41], color=K, lw=0.6); ax.plot([c1, c2], [41, 41], color=K, lw=0.6)
    for xc in (c1, c2): ax.plot([xc, xc], [41, 40], color=K, lw=0.6)
    for i, (h, t) in enumerate([("HC", "scope"), ("HD", "typicality"), ("HE", "core"), ("HF", "clustering")]):
        x = xs[4] + (i % 2) * 17; y = 31.5 - (i // 2) * 10.5
        ax.add_patch(FancyBboxPatch((x, y), 15, 8.5, boxstyle="round,pad=0,rounding_size=1", fc=W, ec=K, lw=0.8))
        ax.text(x + 7.5, y + 4.25, f"{h}\n{t}", ha="center", va="center", fontsize=6.8, fontweight="bold", linespacing=1.25)
    ax.add_patch(FancyBboxPatch((0, 10), 6, 4, boxstyle="round,pad=0,rounding_size=0.8", fc=G1, ec=K, lw=0.6)); ax.text(8, 12, "Calibration hypotheses (simulation, independent of the Korean data)", va="center", fontsize=7)
    ax.add_patch(FancyBboxPatch((0, 3.5), 6, 4, boxstyle="round,pad=0,rounding_size=0.8", fc=W, ec=K, lw=0.8)); ax.text(8, 5.5, "Application hypotheses (Korean freight-rail service network)", va="center", fontsize=7)
    src(pd.DataFrame({"n_trains": [nt], "n_stations": [ns], "n_edges": [ne], "pool_sizes": [pools]}), "Fig01_source"); save(fig, "Fig01_design")

# ================================================================= FIG 2  Korea maps x 4 scope definitions
def fig02():
    V = C["V"]; v = V.eff_loss_dist.to_numpy(float); vmax = v.max(); lon, lat, eng = C["lon"], C["lat"], C["eng"]
    smin, smax = 10, 180; size = lambda x: smin + (smax - smin) * np.asarray(x) / vmax
    tiers = {}
    try:
        t3 = tab("supp_c/tables/C3_candidate_tiers.csv"); tiers = {r.station: r.tier[0] for r in t3.itertuples()}
    except Exception as e: log("   (no C3 tiers:", e, ")")
    tier_of = lambda i: tiers.get(nm(eng[i]), tiers.get(eng[i]))
    fig = plt.figure(figsize=(190 * MM, 226 * MM)); gs = GridSpec(3, 2, height_ratios=[1, 1, 0.13], hspace=0.07, wspace=0.03, left=0.065, right=0.99, top=0.975, bottom=0.005)
    pairs = C["pairs"]
    if EDGES == "core": pairs = [(a, b) for a, b in pairs if tier_of(a) in ("A", "B") or tier_of(b) in ("A", "B")]
    segs = [[(lon[a], lat[a]), (lon[b], lat[b])] for a, b in pairs]; rows = []
    for k, (sname, w) in enumerate(C["scopes"].items()):
        ax = fig.add_subplot(gs[k // 2, k % 2]); draw_map(ax); out = w < TAU
        ax.add_collection(LineCollection(segs, colors="#C4C4C4", linewidths=0.3, zorder=1))
        ax.scatter(lon[~out], lat[~out], s=size(v[~out]), c="#4D4D4D", edgecolors=K, linewidths=0.5, zorder=3)
        ax.scatter(lon[out], lat[out], s=size(v[out]), c=W, edgecolors=K, linewidths=0.6, zorder=3)
        for i in range(C["N"]):
            t = tier_of(i)
            if t in ("A", "B"):
                ax.scatter([lon[i]], [lat[i]], s=size(v[i]) + (95 if t == "A" else 70), facecolors="none", edgecolors=K, linewidths=1.7 if t == "A" else 0.7, zorder=4)
                if k == 0:
                    dx, dy, ha = OFFS.get(nm(eng[i]), (6, 4, "left"))
                    ax.annotate(nm(eng[i]), (lon[i], lat[i]), xytext=(dx, dy), textcoords="offset points", fontsize=6.8, ha=ha, fontweight="bold" if t == "A" else "normal", path_effects=halo(), zorder=6)
        ax.set_title(f"{short_scope(sname)}: {int(out.sum())} stations outside scope", fontsize=7.5, pad=3, loc="left")
        panel(ax, "abcd"[k])
        if k % 2: ax.set_yticklabels([])
        if k < 2: ax.set_xticklabels([])
        if k == 0: ax.text(0.02, 0.02, f"{int((v[out] == 0).sum())} of {int(out.sum())} outside stations\nhave zero value", transform=ax.transAxes, fontsize=6.8, va="bottom", bbox=dict(fc=W, ec=G3, lw=0.4, pad=2))
        rows += [{"scope": sname, "station": eng[i], "lon": lon[i], "lat": lat[i], "eff_loss_dist": v[i], "w": w[i], "outside": bool(out[i]), "tier": tier_of(i) or ""} for i in range(C["N"])]
    axl = fig.add_subplot(gs[2, :]); axl.axis("off"); nz = v[v > 0]; sz = [0, float(np.median(nz)), vmax]
    H = [Line2D([], [], marker="o", ls="", mfc=W, mec=K, mew=0.7, ms=6, label="Outside scope (w < 0.5)"), Line2D([], [], marker="o", ls="", mfc="#4D4D4D", mec=K, mew=0.5, ms=6, label="Inside scope"),
         Line2D([], [], marker="o", ls="", mfc="none", mec=K, mew=1.7, ms=9, label="Core A (top-5 in >= 80% of specifications)"), Line2D([], [], marker="o", ls="", mfc="none", mec=K, mew=0.7, ms=9, label="Core B (30-80%)"),
         Line2D([], [], color="#A8A8A8", lw=0.8, label="Train service connection (not track)" + (" - core-adjacent only" if EDGES == "core" else ""))]
    H += [Line2D([], [], marker="o", ls="", mfc=W, mec=K, mew=0.6, ms=math.sqrt(size(s)), label=f"Efficiency loss = {s:.3f}") for s in sz]
    axl.legend(handles=H, loc="center", ncol=4, fontsize=7, columnspacing=1.2, handletextpad=0.4, labelspacing=0.7)
    src(pd.DataFrame(rows), "Fig02_source"); save(fig, "Fig02_maps")

# ================================================================= FIG 3  calibration of the reference distribution
def fig03():
    rng = C["rng"]; V = C["V"]; w = C["scopes"][C["prim"]]; pool = np.where(w < TAU)[0]; n = len(pool); N = C["N"]; v = V.betw_dist.to_numpy(float)
    g = gini(v[pool]); gd = gini_rows(rng.dirichlet(np.ones(n), R_NULL)); gr = gini_rows(v[np.argsort(rng.random((R_NULL, N)), axis=1)[:, :n]])
    gd, gr = gd[np.isfinite(gd)], gr[np.isfinite(gr)]; pd_ = (1 + (gd >= g).sum()) / (len(gd) + 1); pr_ = (1 + (gr >= g).sum()) / (len(gr) + 1)
    fig = plt.figure(figsize=(190 * MM, 125 * MM)); gs = GridSpec(2, 2, height_ratios=[1.05, 1], hspace=0.55, wspace=0.42, left=0.075, right=0.985, top=0.95, bottom=0.09)
    ax = fig.add_subplot(gs[0, 0]); bins = np.linspace(min(gd.min(), gr.min()) - .01, max(gd.max(), gr.max(), g) + .01, 60)
    ax.hist(gr, bins=bins, density=True, histtype="stepfilled", fc=CALFILL, ec=CAL, lw=0.5, label="Scope-matched random pool")
    ax.hist(gd, bins=bins, density=True, histtype="stepfilled", fc="none", ec=LEG, hatch="////", lw=0.5, label=f"Dirichlet(1) ({CONV})")
    ax.axvline(g, color=K, lw=1.8); ymax = ax.get_ylim()[1]; ax.set_ylim(0, ymax * 2.3)
    ya = ymax * 1.2; ax.annotate("", xy=(gr.mean(), ya), xytext=(gd.mean(), ya), arrowprops=dict(arrowstyle="<->", lw=0.7)); ax.text((gr.mean() + gd.mean()) / 2, ya * 1.03, f"mean shift = {gr.mean() - gd.mean():.2f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xlabel(f"Gini of out-of-scope {lab('betw_dist').lower()} (n = {n})"); ax.set_ylabel("Density")
    H, _ = ax.get_legend_handles_labels(); H += [Line2D([], [], color=K, lw=1.8, label=f"Observed Gini = {g:.3f}"), Line2D([], [], lw=0, label=f"Dirichlet: {fmt_p(pd_, len(gd))}"), Line2D([], [], lw=0, label=f"Random pool: {fmt_p(pr_, len(gr))}")]
    ax.legend(handles=H, loc="upper left", fontsize=6.8, labelspacing=0.3, handlelength=1.8); panel(ax, "a")
    c1 = tab("supp_c/tables/C1_size_power.csv"); z = c1[c1.planted_m == 0].sort_values("reject_dirichlet").reset_index(drop=True); y = np.arange(len(z))
    ax = fig.add_subplot(gs[0, 1]); ax.axvspan(ALPHA - 2 * z.mcse_randompool.mean(), ALPHA + 2 * z.mcse_randompool.mean(), color=G4, zorder=0); ax.axvline(ALPHA, color=G2, ls=":", lw=0.8)
    ax.errorbar(z.reject_dirichlet, y, xerr=2 * z.mcse_dirichlet, fmt="none", ecolor=K, elinewidth=0.6, capsize=1.5); ax.errorbar(z.reject_randompool, y, xerr=2 * z.mcse_randompool, fmt="none", ecolor=K, elinewidth=0.6, capsize=1.5)
    ax.plot(z.reject_dirichlet, y, "o", mfc=W, mec=LEG, mew=1.1, ms=5.5, ls="", label=f"Dirichlet(1) ({CONV})"); ax.plot(z.reject_randompool, y, "o", mfc=CAL, mec=CAL, ms=5, ls="", label="Scope-matched random pool")
    ax.set_yticks(y); ax.set_yticklabels([lab(i) for i in z.indicator]); ax.set_xlim(-0.04, 1.02); ax.set_ylim(-0.7, len(z) + 0.9)
    ax.set_xlabel("Rejection rate under an irrelevant (random) scope"); ax.legend(loc="upper center", ncol=1, fontsize=6.8, bbox_to_anchor=(0.62, 1.04)); panel(ax, "b")
    c4 = tab("supp_c/tables/C4_synthetic_size.csv"); cols = ["BA-53", "WS-53", "ER-53", "BA-100", "WS-100", "ER-100"]; rws = ["degree", "betweenness", "closeness"]
    M = np.full((3, 6), np.nan); lo = M.copy(); hi = M.copy()
    for r in c4.itertuples():
        key = f"{r.topology}-{r.n}"
        if key in cols and r.indicator in rws: M[rws.index(r.indicator), cols.index(key)] = r.dirichlet_size_mean; lo[rws.index(r.indicator), cols.index(key)] = r.dirichlet_size_min; hi[rws.index(r.indicator), cols.index(key)] = r.dirichlet_size_max
    ax = fig.add_subplot(gs[1, :]); ax.imshow(M, cmap="Greys", vmin=-0.12, vmax=1, aspect="auto")   # vmin<0: zero cells get a light gray fill, not white
    for i in range(3):
        for j in range(6):
            if np.isnan(M[i, j]): continue
            ax.text(j, i, f"{M[i, j]:.2f}\n[{lo[i, j]:.2f}-{hi[i, j]:.2f}]", ha="center", va="center", fontsize=6.8, color=W if M[i, j] > 0.5 else K, fontweight="bold" if M[i, j] > 0.10 else "normal", linespacing=1.3)
    ax.set_xticks(range(6)); ax.set_xticklabels([c.replace("-", ", N = ") for c in cols]); ax.set_yticks(range(3)); ax.set_yticklabels([r.capitalize() for r in rws])
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True); ax.set_yticks(np.arange(-.5, 3, 1), minor=True); ax.grid(which="minor", color=G2, lw=0.6); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlabel("Synthetic topology (BA: Barabasi-Albert, WS: Watts-Strogatz, ER: Erdos-Renyi)\ncell = mean [min-max] Dirichlet rejection rate across networks"); panel(ax, "c")
    src(z, "Fig03b_source"); src(c4, "Fig03c_source"); src(pd.DataFrame({"gini_obs": [g], "p_dirichlet": [pd_], "p_randompool": [pr_], "null_mean_dirichlet": [gd.mean()], "null_mean_randompool": [gr.mean()]}), "Fig03a_source"); save(fig, "Fig03_calibration")

# ================================================================= FIG 4  power and detectability
def fig04():
    d1 = tab("supp_d/tables/D1_size_power_statistics.csv"); e2 = tab("supp_e/tables/E2_leakage_floor.csv"); e3 = tab("supp_e/tables/E3_detectability_map.csv")
    fig = plt.figure(figsize=(190 * MM, 150 * MM)); gs = GridSpec(2, 2, hspace=0.55, wspace=0.3, left=0.075, right=0.985, top=0.95, bottom=0.12)
    sty = {"pool_share (OCG)": (CAL, "-", "o", CAL), "gini_pool": (LEG, "--", "s", W), "gini_diff_out_minus_in": (G2, "-", "^", G2), "pool_mean_rank": (G2, "--", "v", W),
           "top11_leakage": (G3, "-", "D", G3), "top5_leakage": (G3, "--", "X", G3)}
    ax = fig.add_subplot(gs[0, 0]); ms = sorted(d1.planted_m.unique()); ax.axhline(ALPHA, color=G2, ls=":", lw=0.8); ax.axhline(0.8, color=K, ls=":", lw=0.8)
    for s in d1.statistic.unique():
        g = d1[d1.statistic == s].groupby("planted_m").reject; mean = g.mean().reindex(ms); c, ls, mk, fc = sty.get(s, (G2, "-", "o", G2))
        if s in ("pool_share (OCG)", "gini_pool"): ax.fill_between(ms, g.min().reindex(ms), g.max().reindex(ms), color=G4, lw=0, zorder=0)
        ax.plot(ms, mean, color=c, ls=ls, marker=mk, mfc=fc, mec=c, ms=4.5, lw=1.1, label=stat_lab(s))
    ax.set_xlabel("Planted blind spots m"); ax.set_ylabel("Mean rejection rate"); ax.set_ylim(0, 1.3); ax.set_yticks([0, .2, .4, .6, .8, 1.0]); ax.set_xticks(ms)
    ax.legend(loc="upper left", ncol=2, fontsize=6.5, labelspacing=0.3, columnspacing=0.8, handlelength=2.0)
    ax.text(max(ms), 0.82, "0.8", fontsize=6.5, ha="right", va="bottom"); ax.text(max(ms), ALPHA + .015, "0.05", fontsize=6.5, ha="right", va="bottom")
    if "top5_leakage" in d1.statistic.values: ax.text(0.02, 0.5, "Top-5 leakage: minimum attainable p > 0.05", transform=ax.transAxes, fontsize=6.5, va="center", color=G1)
    panel(ax, "a")
    ax = fig.add_subplot(gs[0, 1]); gp = d1[d1.statistic == "gini_pool"].copy(); gp["delta"] = gp.mean_stat_planted - gp.mean_stat_null; mk = ["o", "s", "^", "v", "D", "P"]; lsx = ["-", "--", "-.", ":", "-"]
    for i, (k, g) in enumerate(gp.groupby("indicator")): ax.plot(g.planted_m, g.delta, color=[K, G1, G2, G1, G2][i % 5], ls=lsx[i % 5], marker=mk[i % 6], mfc=W if i % 2 else [K, G1, G2, G1, G2][i % 5], ms=4.5, label=lab(k))
    ax.axhline(0, color=K, lw=0.6); ax.set_xlabel("Planted blind spots m"); ax.set_ylabel("Mean pool Gini minus null mean"); ax.set_xticks(ms)
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_ - 0.12 * (hi_ - lo_), hi_ + 0.45 * (hi_ - lo_)); ax.legend(loc="upper left", ncol=2, fontsize=6.5, labelspacing=0.3, columnspacing=1.0)
    j = gp.loc[gp[gp.planted_m == max(ms)].delta.idxmin()]
    ax.annotate("Non-monotone: rises for m <= 3;\nfalls at m = 5 for some indicators", xy=(max(ms), j.delta), xytext=(0.12, 0.05), textcoords="axes fraction", fontsize=6.8, arrowprops=dict(arrowstyle="->", lw=0.6), va="bottom"); panel(ax, "b")
    ax = fig.add_subplot(gs[1, 0]); ind = "betw_dist" if "betw_dist" in e3.indicator.values else e3.indicator.iloc[0]; e = e3[e3.indicator == ind]; Ns = sorted(e.N.unique()); pfs = sorted(e.planted_fraction.unique())
    colr = [K, "#5A5A5A", "#9A9A9A"]; mkr = ["o", "s", "^"]
    for i, Nn in enumerate(Ns):
        for pf in pfs:
            g = e[(e.N == Nn) & (e.planted_fraction == pf)].sort_values("f_out"); ax.plot(g.f_out, g.power, color=colr[i % 3], ls="-" if pf == max(pfs) else "--", marker=mkr[i % 3], ms=4, lw=1.1, mfc=colr[i % 3] if pf == max(pfs) else W)
    ax.axhline(0.8, color=K, ls=":", lw=0.8); kf = e[(e.N == min(Ns)) & (e.planted_fraction == max(pfs))]; kf = kf.iloc[(kf.f_out - 0.64).abs().argsort()].iloc[0]
    ax.axvline(kf.f_out, color=G2, ls="--", lw=0.7); ax.plot([kf.f_out], [kf.power], marker="*", ms=11, mfc=W, mec=K, mew=1.0, ls="", zorder=5)
    ax.annotate(f"Korea: power = {kf.power:.2f}\n(N = {int(min(Ns))}, {int(100 * max(pfs))}% planted)", xy=(kf.f_out, kf.power), xytext=(kf.f_out - 0.04, 0.05), ha="right", va="bottom", fontsize=6.8, arrowprops=dict(arrowstyle="->", lw=0.6))
    H = [Line2D([], [], color=colr[i % 3], marker=mkr[i % 3], ms=4, label=f"N = {Nn}") for i, Nn in enumerate(Ns)] + [Line2D([], [], color=K, ls="-", label=f"{int(100 * max(pfs))}% planted"), Line2D([], [], color=K, ls="--", label=f"{int(100 * min(pfs))}% planted")]
    ax.legend(handles=H, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=5, fontsize=6.5, columnspacing=1.0)
    ax.set_xlabel("Out-of-scope share"); ax.set_ylabel(f"Power of OCG test ({lab(ind).lower()})"); ax.set_ylim(0, 1.03); panel(ax, "c")
    ax = fig.add_subplot(gs[1, 1]); N0 = min(e2.N); g = e2[e2.N == N0].sort_values("f_out"); kk = g["K_min_for_p<=0.05"]; ax.plot(g.f_out, kk, color=K, marker="o", ms=4.5, lw=1.1)
    for x, yv in zip(g.f_out, kk): ax.text(x, yv + 0.5, f"{int(yv)}", ha="center", fontsize=6.8)
    ax.axhline(5, color=G2, ls=":", lw=0.8); ax.text(g.f_out.min(), 5.3, "K = 5", fontsize=6.8, va="bottom")
    r = g.iloc[(g.f_out - 0.64).abs().argsort()].iloc[0]; ax.text(0.03, 0.95, f"At {r.f_out:.2f} outside (N = {int(N0)}),\ntop-{int(r['K_min_for_p<=0.05'])} must all be outside\nfor p <= 0.05", transform=ax.transAxes, va="top", fontsize=6.8, bbox=dict(fc=W, ec=G3, lw=0.4, pad=2))
    ax.set_xlabel("Out-of-scope share"); ax.set_ylabel("Smallest K with attainable p <= 0.05"); ax.set_ylim(0, kk.max() * 1.15); panel(ax, "d")
    CAPS["Fig. 4"] = (f"Power and detectability (H2, H3). (a) Rejection rate versus the number m of planted blind spots (top-m stations forced outside scope), mean over indicators; bands are min-max for OCG and pool Gini. "
                      f"Top-11 leakage uses K = 11, about 20% of the N = {int(N0)} stations; top-5 leakage cannot reach p <= 0.05 (minimum attainable p = C(34,5)/C(53,5) = 0.097 when 34 of 53 stations are outside scope). "
                      f"(b) Mean pool Gini minus null mean versus m. (c) Simulated power of the OCG test ({lab(ind).lower()}) by out-of-scope share, N and planted share; the star marks the Korean configuration "
                      f"(power = {kf.power:.2f}). This simulated power is the primary basis for every detectability statement in the paper. (d) Smallest K for which top-K leakage can reach p <= 0.05 (N = {int(N0)}). "
                      "Planting is the most favourable case, so power is an upper bound. [Add: simulations per cell, null draws, seeds.]")
    src(d1, "Fig04ab_source"); src(e, "Fig04c_source"); src(g, "Fig04d_source"); save(fig, "Fig04_power_detectability")

# ================================================================= FIG 5  Korean application
def fig05():
    rng = C["rng"]; V = C["V"]; s2 = tab("supp/tables/S2_OCG_baseline_rank.csv"); d2 = tab("supp_d/tables/D2_korea_audit_statistics.csv"); d2 = d2[d2.statistic == "gini_diff_out_minus_in"]; ps = jload("supp/supp_numbers.json").get("pool_sizes", {})
    fig = plt.figure(figsize=(190 * MM, 140 * MM)); gs = GridSpec(2, 2, height_ratios=[1, 0.3], hspace=0.25, wspace=0.28, left=0.2, right=0.985, top=0.96, bottom=0.015)
    ax = fig.add_subplot(gs[0, 0]); y = 0; yt, yl = [], []; rows = []; K_ = 2.4864; segs = []
    for sname in s2.scope.unique():
        w = find_scope(sname); sub = s2[s2.scope == sname]; y_top = y; npool = ps.get(sname, int((w < TAU).sum()))
        for ind in FAM:
            r = sub[sub.indicator == ind]
            if r.empty: continue
            r = r.iloc[0]; v = V[ind].to_numpy(float); perm = np.argsort(rng.random((R_BAND, len(v))), axis=1); T = (v[perm] * (1 - w)).sum(1) / v.sum() - (1 - w).mean(); lo, hi = np.percentile(T, [2.5, 97.5])
            inside = bool(lo <= r.effect <= hi)
            ax.plot([lo, hi], [y, y], color=G3, lw=5, solid_capstyle="butt", zorder=1); ax.plot([r.effect], [y], "o", color=K, ms=5, zorder=3)
            if not inside: ax.plot([r.effect], [y], "o", mfc="none", mec=K, ms=10, mew=0.9, zorder=4)
            mde = K_ * r.perm_sd; unreach = r.baseline_1_minus_mean_w + mde > 1; ax.plot([mde], [y], "D", mfc=W if unreach else K, mec=K, ms=5, zorder=3)
            yt.append(y); yl.append(lab(ind)); rows.append({"panel": "a", "scope": sname, "indicator": ind, "effect": r.effect, "null_lo": lo, "null_hi": hi, "inside_band": inside, "MDE": mde, "MDE_exceeds_OCG_ceiling": unreach}); y -= 1
        segs.append((y + 0.55, y_top + 0.45)); ax.text(-0.6, y_top + 0.8, f"{short_scope(sname)} (n = {npool})", fontsize=6.8, fontweight="bold", va="center"); y -= 1.0
    for lo_, hi_ in segs: ax.vlines(0, lo_, hi_, color=K, lw=0.7, zorder=2)      # zero line only beside the rows, never through the group titles
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.8); ax.set_xlim(-0.6, 0.6); ax.set_ylim(y + 0.5, 1.5)
    ax.set_xlabel("OCG minus permutation baseline (1 - mean w)"); panel(ax, "a")
    ax = fig.add_subplot(gs[0, 1]); y = 0; yt, yl = [], []; rows2 = []; sc = list(d2.scope.unique()); segs = []
    for sname in sc:
        w = find_scope(sname); sub = d2[d2.scope == sname]; y_top = y
        for ind in FAM:
            r = sub[sub.indicator == ind]
            if r.empty: continue
            r = r.iloc[0]; v = V[ind].to_numpy(float); pool = np.where(w < TAU)[0]; n = len(pool); perm = np.argsort(rng.random((R_BAND, len(v))), axis=1)
            nl = gini_rows(v[perm[:, :n]]) - gini_rows(v[perm[:, n:]]); nl = nl[np.isfinite(nl)]; lo, hi = np.percentile(nl - nl.mean(), [2.5, 97.5])
            eff = r.observed - r.null_mean; inside = bool(lo <= eff <= hi)
            ax.plot([lo, hi], [y, y], color=G3, lw=5, solid_capstyle="butt", zorder=1); ax.plot([eff], [y], "o", color=K, ms=5, zorder=3)
            if not inside: ax.plot([eff], [y], "o", mfc="none", mec=K, ms=10, mew=0.9, zorder=4)
            ph = r.p_holm_family; ax.text(0.46, y, f"p = {r.p_upper:.2f}" + (f" ({ph:.2f})" if np.isfinite(ph) else ""), fontsize=6.5, va="center")
            yt.append(y); yl.append(lab(ind)); rows2.append({"panel": "b", "scope": sname, "indicator": ind, "effect": eff, "null_lo": lo, "null_hi": hi, "inside_band": inside, "p": r.p_upper, "p_holm": ph}); y -= 1
        segs.append((y + 0.55, y_top + 0.45)); ax.text(-0.4, y_top + 0.8, short_scope(sname), fontsize=6.8, fontweight="bold", va="center"); y -= 1.0
    for lo_, hi_ in segs: ax.vlines(0, lo_, hi_, color=K, lw=0.7, zorder=2)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.8); ax.set_xlim(-0.4, 0.82); ax.spines["bottom"].set_bounds(-0.4, 0.4); ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4]); ax.set_ylim(y + 0.5, 1.5)
    ax.text(0.46, 1.1, "p (Holm)", fontsize=6.5, fontweight="bold"); ax.set_xlabel("Gini(out) - Gini(in), minus null mean"); panel(ax, "b")
    allr = rows + rows2; n_tot = len(allr); exc = [r for r in allr if not r["inside_band"]]; n_in = n_tot - len(exc)
    def exc_txt(r):
        side = "below" if r["effect"] < r["null_lo"] else "above"
        return f"panel {r['panel']}: {short_scope(r['scope'])}, {lab(r['indicator']).lower()}, effect {r['effect']:+.3f} ({side} the band)"
    exc_line = ("Outside the band (ring): " + "\n".join(exc_txt(r) for r in exc[:3]) + (f"\n(+{len(exc) - 3} more, see Fig05_exceptions.csv)" if len(exc) > 3 else "")) if exc else "No effect lies outside the band."
    log(f"   Fig. 5: {n_in} of {n_tot} effects inside the 95% band; exceptions: {[exc_txt(r) for r in exc]}")
    pd.DataFrame(exc).to_csv(DATA / "Fig05_exceptions.csv", index=False, encoding="utf-8-sig")
    axn = fig.add_subplot(gs[1, :]); axn.axis("off"); axn.set_xlim(0, 1); axn.set_ylim(0, 1); x0 = -0.245       # start at the figure's left margin: text is wider than the axes
    axn.text(x0, 1.0, "Reading guide", fontweight="bold", fontsize=7.5, va="top")
    axn.text(x0, 0.86, "Gray bar: 95% interval of the scope-matched random-pool null.  Black dot: observed effect.  Ring: effect outside the band.\n"
                       "Diamond (panel a): approximate minimum detectable effect (80% power, normal approximation; open = exceeds OCG <= 1).\n"
                       f"The simulated power in {POWER_REF} is the primary basis for detectability; the diamonds are a rough guide only.", fontsize=6.8, va="top", linespacing=1.4)
    axn.text(x0, 0.40, f"{n_in} of {n_tot} observed effects lie inside the 95% null band. Inside the band means 'undetermined', not 'absent'.\n{exc_line}\n"
                       "Eigenvector (exploratory, outside the confirmatory family) is omitted here.", fontsize=6.8, va="top", linespacing=1.4)
    ex0 = exc[0] if exc else None
    CAPS["Fig. 5"] = ("Korean audit under calibrated references. (a) OCG minus the permutation baseline 1 - w̄ (dots) with the 95% interval of the scope-matched null (grey) and the approximate 80%-power minimum detectable effect "
                      "(diamonds; normal approximation; open: exceeds the OCG <= 1 limit). The simulated power in Fig. 4c is the primary basis for detectability; the diamonds are a rough guide. "
                      f"(b) Gini(out) - Gini(in) minus the null mean, with p (Holm-adjusted within the three confirmatory indicators). {n_in} of {n_tot} effects lie inside the 95% band; "
                      + (("outside (ringed): " + "; ".join(exc_txt(r) for r in exc) + ". ") if exc else "none lies outside. ") + "Inside the band means undetermined, not absent (see Fig. 4c).")
    if ex0 is not None:
        CAPS["Fig. 5 - body sentence"] = (f"Of the {n_tot} observed effects, {n_in} lie within the 95% null band; the exception is {short_scope(ex0['scope'])}, {lab(ex0['indicator']).lower()} "
                                          f"({'Gini difference' if ex0['panel'] == 'b' else 'OCG effect'} {ex0['effect']:+.3f}, {'below' if ex0['effect'] < ex0['null_lo'] else 'above'} the band) [state direction and interpretation].")
    src(pd.DataFrame(rows), "Fig05a_source"); src(pd.DataFrame(rows2), "Fig05b_source"); save(fig, "Fig05_korean_application")

# ================================================================= FIG 6  design space
def fig06():
    t3 = tab("tables/T03_design_space_OCG.csv"); t7 = tab("tables/T07_pareto_frontier.csv"); t9 = tab("tables/T09_corridor_vs_station.csv"); rng = C["rng"]
    fig = plt.figure(figsize=(190 * MM, 92 * MM)); gs = GridSpec(1, 2, wspace=0.55, left=0.07, right=0.985, top=0.93, bottom=0.17, width_ratios=[1.05, 1])
    ax = fig.add_subplot(gs[0, 0]); ax.scatter(t3["size"] + rng.uniform(-0.15, 0.15, len(t3)), t3.coverage_value, s=5, c=G3, lw=0, zorder=1, label="All designs")
    ax.plot(t7.m, t7.best_coverage, "-", color=K, lw=1.2, label="Best design (exhaustive)"); ax.plot(t7.m, t7.median_coverage, ":", color=K, lw=1.1, label="Median design")
    ro = t7[t7.original_coverage.notna()].iloc[0]; rr = t7[t7.reference_coverage.notna()].iloc[0]; sub = t3[t3["size"] == int(ro.m)]; rank = int((sub.coverage_value > ro.original_coverage + 1e-12).sum() + 1); gap = ro.best_coverage - ro.original_coverage
    ax.plot([ro.m], [ro.original_coverage], "D", mfc=W, mec=K, ms=7, mew=1.2, ls="", label="Original four lines", zorder=5); ax.plot([rr.m], [rr.reference_coverage], "*", mfc=K, mec=K, ms=11, ls="", label="Top-4 by train-km", zorder=5)
    ax.set_ylim(0, 1.32); ax.set_yticks([0, .2, .4, .6, .8, 1.0])          # headroom above the frontier for the note box
    ax.text(0.03, 0.985, f"Service-share scope only: original four lines rank\n{rank}th of {len(sub)} four-line designs; frontier gap = {gap:.3f}\n(not defined for the line-tag pool)", transform=ax.transAxes, fontsize=6.5, va="top", bbox=dict(fc=W, ec=G3, lw=0.4, pad=2), zorder=6)
    ax.set_xlabel("Number of designated lines m"); ax.set_ylabel("Covered structural value (efficiency loss)"); ax.set_xticks(sorted(t3["size"].unique())); ax.legend(loc="lower right", fontsize=6.5, labelspacing=0.3); panel(ax, "a")
    ax = fig.add_subplot(gs[0, 1]); t9 = t9[t9.newly_covered_stations > 0].sort_values("corridor_gain"); y = np.arange(len(t9)); xm = max(t9.corridor_gain.max(), t9.station_level_gain_same_count.max())
    for yy, r in zip(y, t9.itertuples()): ax.plot([r.station_level_gain_same_count, r.corridor_gain], [yy, yy], color=G2, lw=1.0, zorder=1)
    ax.plot(t9.corridor_gain, y, "o", color=K, ms=6, label="Corridor expansion", zorder=3); ax.plot(t9.station_level_gain_same_count, y, "o", mfc=W, mec=K, ms=6, mew=1.1, ls="", label="Station-level (same number of stations)", zorder=3)
    for yy, r in zip(y, t9.itertuples()): ax.text(xm * 1.12, yy, f"ratio {r.ratio_station_over_corridor:.2f}", fontsize=6.8, va="center")
    ax.text(xm * 1.12, len(t9) - 0.45, "station / corridor", fontsize=6.5, fontweight="bold"); ax.set_yticks(y); ax.set_yticklabels([f"+{r.added_line} ({int(r.newly_covered_stations)} st.)" for r in t9.itertuples()])
    ax.set_xlim(0, xm * 1.75); ax.spines["bottom"].set_bounds(0, xm * 1.05); ax.set_xticks(np.arange(0, xm * 1.05 + 1e-9, 0.02)); ax.set_xlabel("Gain in covered value")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1, fontsize=6.5); ax.set_ylim(-0.6, len(t9) + 0.2); panel(ax, "b")
    CAPS["Fig. 6"] = (f"Position of the scope in the design space (service-share weights only). (a) Covered efficiency loss of all line subsets by number m of designated lines (grey: all designs, black line: best, dotted: median). "
                      f"The rank of the original four lines ({rank}th of {len(sub)} four-line designs; frontier gap {gap:.3f}) applies to the service-share scope only and is not defined for the 34-station line-tag pool. "
                      "(b) Gain from adding one line versus adding the same number of stations individually; ratio = station-level / corridor gain.")
    CAPS["Fig. 6 - body sentence"] = (f"Under service-share weights, the original four lines rank {rank}th of {len(sub)} four-line designs on covered efficiency loss (frontier gap {gap:.3f}); "
                                      "this ranking is defined only for the service-share scope and is not available for the line-tag pool.")
    src(t7, "Fig06a_source"); src(t9, "Fig06b_source"); save(fig, "Fig06_design_space")

# ================================================================= FIG 7  typicality and spatial pattern
def fig07():
    rr = tab("supp/tables/S4_rewiring_realizations.csv"); c2 = tab("supp_c/tables/C2_typicality.csv"); mo = tab("supp/tables/S5_moran.csv"); pw = tab("supp/tables/S5_power.csv"); delta = jload("supp_c/c_numbers.json").get("config", {}).get("delta", 0.05)
    fig = plt.figure(figsize=(190 * MM, 165 * MM))
    gs_t = GridSpec(1, 3, left=0.075, right=0.985, top=0.90, bottom=0.63, wspace=0.55)
    gs_b = GridSpec(1, 2, left=0.21, right=0.985, top=0.42, bottom=0.18, wspace=0.35, width_ratios=[2.0, 1])
    fig.text(0.5, 0.995, "a1-a3: hop-based (unweighted) efficiency loss; observed value vs 1,000 rewirings per variant", ha="center", va="top", fontsize=7.5, fontweight="bold")
    stats = [("gini_all53", "Gini, all 53 stations"), ("gini_pool", "Gini, out-of-scope pool"), ("top3_all53", "Top-3 share, all 53")]; vars_ = list(rr.variant.unique())[:2]
    for k, (st, ttl) in enumerate(stats):
        ax = fig.add_subplot(gs_t[0, k]); data = [rr[rr.variant == v][st].dropna().to_numpy(float) for v in vars_]; c = c2[c2.statistic == st]
        mean0 = float(c.ensemble_mean.iloc[0]) if len(c) else float(np.mean(data[0]))
        if SHOW_BAND: ax.axhspan(mean0 - delta, mean0 + delta, color="#EFEFEF", zorder=0)
        vp = ax.violinplot(data, positions=range(len(data)), widths=0.8, showextrema=False)
        for i, b in enumerate(vp["bodies"]): b.set_facecolor(G4 if i == 0 else W); b.set_edgecolor(K); b.set_linewidth(0.6); b.set_alpha(1); b.set_hatch("" if i == 0 else "////")
        for i, d in enumerate(data):
            lo, hi = np.percentile(d, [2.5, 97.5]); ax.plot([i, i], [lo, hi], color=K, lw=1.3); ax.plot([i - .08, i + .08], [lo, lo], color=K, lw=1.3); ax.plot([i - .08, i + .08], [hi, hi], color=K, lw=1.3); ax.plot([i], [np.median(d)], "o", mfc=W, mec=K, ms=3.5, zorder=4)
        allv = np.concatenate(data); ymin, ymax = allv.min(), allv.max(); xl_ = "violin: realizations; bar: 95% PI;\ncircle: median"
        if len(c):
            ob = float(c.observed.iloc[0]); ins = bool(c.inside_95PI.all()); wd = bool(c.within_delta.all()) if "within_delta" in c else None; ax.axhline(ob, color=K, ls="--", lw=1.0)
            ymin, ymax = min(ymin, ob), max(ymax, ob)
            xl_ = f"dashed line: observed = {ob:.3f}\ninside 95% PI: {'yes' if ins else 'no'}" + (f"\nwithin ±{delta:g} of mean: {'yes' if wd else 'no'}" if wd is not None else "")
        if SHOW_BAND: xl_ += f"\nshaded band: mean ± {delta:g}"
        pad = 0.04 * (ymax - ymin); ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xticks(range(len(vars_))); ax.set_xticklabels(["Degree-\npreserving", "+ length\nconstraint"][:len(vars_)], fontsize=6.8); ax.set_ylabel(ttl, fontsize=7.5); ax.set_xlim(-0.6, len(vars_) - 0.4)
        ax.set_xlabel(xl_, fontsize=6.5, labelpad=3); panel(ax, f"a{k + 1}")
    ax = fig.add_subplot(gs_b[0, 0]); mo = mo.reset_index(drop=True); y = -np.arange(len(mo)); ax.axvspan(-1.96, 1.96, color=G4, zorder=0); ax.axvline(0, color=K, lw=0.6)
    ax.plot(mo.z, y, "o", color=K, ms=5, zorder=3)
    for yy, r in zip(y, mo.itertuples()): ax.text(2.7, yy, f"I = {r.I:.3f}  (E[I] = {r.E_I:.3f})", fontsize=6.5, va="center")
    ax.set_yticks(y); ax.set_yticklabels([w.replace("service adjacency", "Service adjacency").replace("share a service neighbour", "Shared service neighbour").replace("inverse distance", "Inverse distance") for w in mo.weights], fontsize=6.8)
    ax.set_xlim(-3.4, 6.6); ax.spines["bottom"].set_bounds(-3, 2.5); ax.set_xticks([-3, -2, -1, 0, 1, 2]); ax.set_xlabel("Moran's I standardised against the permutation null (z); shaded: ±1.96"); panel(ax, "b")
    ax = fig.add_subplot(gs_b[0, 1]); ks = sorted(pw.test_k.unique()); sty = [("-", "o", K), ("--", "s", G2)]
    for i, kt in enumerate(ks): g = pw[pw.test_k == kt].sort_values("true_rho"); ax.plot(g.true_rho, g.detection_rate, ls=sty[i % 2][0], marker=sty[i % 2][1], color=sty[i % 2][2], mfc=K if i == 0 else W, mec=sty[i % 2][2], ms=4.5, label=f"kNN, k = {int(kt)}")
    ax.axhline(0.8, color=K, ls=":", lw=0.8); ax.axhline(ALPHA, color=G2, ls=":", lw=0.8); ok = pw[pw.detection_rate >= 0.8]; H, _ = ax.get_legend_handles_labels()
    r0 = prev = None; grid = sorted(pw.true_rho.unique()); gridstr = ", ".join(f"{x:g}" for x in grid)
    if len(ok):
        r0 = float(ok.true_rho.min()); lower = [x for x in grid if x < r0]; prev = float(max(lower)) if lower else None
        if prev is not None: ax.axvspan(prev, r0, color=G4, zorder=0)              # the true minimum detectable rho lies somewhere in this interval
        ax.axvline(r0, color=G2, ls="--", lw=0.7)
        H = H + [Line2D([], [], color=G2, ls="--", lw=0.7, label=f"first grid value with power ≥ 0.8: {r0:g}")]
        if prev is not None: H = H + [Rectangle((0, 0), 1, 1, fc=G4, ec="none", label=f"minimum detectable ρ in ({prev:g}, {r0:g}]\n(grid: {gridstr})")]
    ax.set_xlim(-0.05, pw.true_rho.max() + 0.08); ax.set_xlabel("Injected spatial dependence rho"); ax.set_ylabel("Detection rate"); ax.set_ylim(0, 1.02)
    ax.legend(handles=H, loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=6.5, ncol=1); panel(ax, "c")
    rho_txt = (f"The first grid value with power >= 0.8 is {r0:g} (grid: {gridstr}), so the minimum detectable rho lies between {prev:g} and {r0:g}, not at {r0:g}." if (r0 is not None and prev is not None) else "[state the first grid value with power >= 0.8]")
    CAPS["Fig. 7"] = ("Typicality and spatial pattern (H5, H6). (a1-a3) Statistics of the hop-based (unweighted) efficiency loss (all other figures use distance-weighted values; hops are used because rewired edges have no defined length): "
                      f"observed value (dashed) versus 1,000 degree-preserving rewirings (left) and 1,000 length-constrained rewirings (right); error bars: 95% prediction interval; shaded band: ensemble mean ± {delta:g}. "
                      "(b) Moran's I (z against the permutation null; n = 34; row-standardised weights; 9,999 permutations); E[I] = -0.030. "
                      f"(c) Detection rate of Moran's I under injected spatial dependence rho (SAR parameter; 300 simulations per cell). {rho_txt}")
    src(c2, "Fig07a_source"); src(mo, "Fig07b_source"); src(pw, "Fig07c_source"); save(fig, "Fig07_typicality_spatial")

# ================================================================= FIG 8  invariant candidates and external consistency
def fig08():
    t3 = tab("supp_c/tables/C3_candidate_tiers.csv"); spc = tab("tables/T25_specification_curve.csv"); ex = tab("supp_c/tables/C3_core_exogenous.csv"); panel_c = FIG8C == "panel"
    if panel_c: fig = plt.figure(figsize=(190 * MM, 145 * MM)); gs = GridSpec(2, 2, height_ratios=[1.25, 0.8], width_ratios=[1.0, 1.45], hspace=0.8, wspace=0.55, left=0.16, right=0.985, top=0.92, bottom=0.12)
    else: fig = plt.figure(figsize=(190 * MM, 90 * MM)); gs = GridSpec(1, 2, width_ratios=[1.0, 1.45], wspace=0.55, left=0.16, right=0.985, top=0.80, bottom=0.30)
    ax = fig.add_subplot(gs[0, 0]); t = t3.sort_values("frequency", ascending=False).head(10).reset_index(drop=True); y = np.arange(len(t))[::-1]; fc = {"A": K, "B": G2, "C": W}
    for yy, r in zip(y, t.itertuples()): ax.barh(yy, r.frequency, color=fc[r.tier[0]], edgecolor=K, lw=0.6, hatch="...." if r.tier[0] == "C" else "", height=0.7)
    for yy, r in zip(y, t.itertuples()):
        if nm(r.station) in STAGEC: ax.plot([r.frequency + 0.06], [yy], marker="*", ms=8, color=K, ls="", clip_on=False)      # same symbol as in the legend
    ax.axvline(0.8, color=K, ls="--", lw=0.7); ax.axvline(0.3, color=K, ls=":", lw=0.7); ax.set_yticks(y); ax.set_yticklabels([nm(s) for s in t.station])
    ax.set_xlim(0, 1.75); ax.spines["bottom"].set_bounds(0, 1.0); ax.set_xticks([0, .25, .5, .75, 1.0]); ax.set_xlabel("Share of specifications\nwith top-5 membership", loc="left")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=K, ec=K, label="Core A (>= 0.8)"), Rectangle((0, 0), 1, 1, fc=G2, ec=K, label="B (0.3-0.8)"), Rectangle((0, 0), 1, 1, fc=W, ec=K, hatch="....", label="C (< 0.3)"), Line2D([], [], marker="*", ls="", color=K, ms=8, label="Original Stage-C pick")],
              loc="lower right", fontsize=6.5, labelspacing=0.3, frameon=True, facecolor=W, edgecolor="none", framealpha=1); panel(ax, "a")
    ax = fig.add_subplot(gs[0, 1]); scn = [s for s in spc.scope.unique() if "line-tag" in s][0]; sub = spc[spc.scope == scn].copy(); sub["ord"] = sub.indicator.map({k: i for i, k in enumerate(IND_SHORT)}).fillna(99); sub = sub.sort_values(["ord", "tau"]).reset_index(drop=True)
    tier = {nm(r.station): r.tier[0] for r in t3.itertuples()}; rows = [nm(s) for s in t.station[:8]]; M = np.zeros((len(rows), len(sub))); R = np.full(M.shape, np.nan)
    for j, r in enumerate(sub.itertuples()):
        for rank, s in enumerate(str(r.top5).split("|"), 1):
            if nm(s) in rows: i = rows.index(nm(s)); M[i, j] = {"C": 1, "B": 2, "A": 3}[tier.get(nm(s), "C")]; R[i, j] = rank
    ax.imshow(M, cmap=ListedColormap([W, "#D2D2D2", "#808080", K]), vmin=0, vmax=3, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] > 0: ax.text(j, i, f"{int(R[i, j])}", ha="center", va="center", fontsize=6.5, color=W if M[i, j] >= 2 else K, fontweight="bold")
    ng = int(sub.tau.nunique()); inds = list(dict.fromkeys(sub.indicator))
    for g_, ind in enumerate(inds):
        if g_: ax.axvline(g_ * ng - 0.5, color=K, lw=0.7)
        ax.text(g_ * ng + (ng - 1) / 2, -0.75 - 0.95 * (g_ % 2), IND_SHORT.get(ind, ind), ha="center", va="bottom", fontsize=6.5, fontweight="bold")   # staggered: neighbours never touch
    ax.set_xticks(range(len(sub))); ax.set_xticklabels([f"{x:.2f}".lstrip("0") for x in sub.tau], fontsize=6.5, rotation=90); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_xticks(np.arange(-.5, len(sub), 1), minor=True); ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True); ax.grid(which="minor", color=G3, lw=0.4); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlabel("Coverage threshold τ within each indicator\nnumber = rank within top-5\nindicators partly redundant (e.g. EffLoss-d, Betw-d)", fontsize=7); panel(ax, "b", y=1.30)
    core = [x for x in t3[t3.tier.str.startswith("A")].station]; cn = [nm(s) for s in core]; rows_c = []
    for r in ex.itertuples():
        pr = {}
        if isinstance(r.core_percentiles, str): pr = {nm(p.split(":")[0]): float(p.split(":")[1]) for p in r.core_percentiles.split("|")}
        rows_c.append({"exogenous": r.exogenous.replace("_", " "), "n_pool_with_data": r.n_pool_with_data, "core_with_data": r.core_with_data, "core_missing": r.core_missing if isinstance(r.core_missing, str) else "",
                       **{f"percentile_{s}": pr.get(s, np.nan) for s in cn}, "p_one_sided": r.p_one_sided, "p_holm": r.p_holm})
    CAPS["Fig. 8"] = ("Specification-invariant candidates (H7). (a) Share of 24 specifications in which a station enters the top five of the out-of-scope pool; stars: stations selected by the original Stage C. "
                      "(b) Station rank (1-5) in each specification, by indicator and coverage threshold tau; indicators are partly redundant. Core status reflects robustness across analytic variants, not external validity.")
    if not panel_c:
        TB = pd.DataFrame(rows_c); TB.round(3).to_csv(OUT / "Table_Fig08c_exogenous.csv", index=False, encoding="utf-8-sig"); src(TB, "Fig08c_source"); log("   Fig. 8c written as table: Table_Fig08c_exogenous.csv")
        src(t3, "Fig08a_source"); src(sub, "Fig08b_source"); save(fig, "Fig08_core_candidates"); return
    ax = fig.add_subplot(gs[1, :]); y = 0; yt, yl = [], []; rows = []
    for r, rc in zip(ex.itertuples(), rows_c):
        for i, s in enumerate(cn):
            p_ = rc.get(f"percentile_{s}", np.nan)
            if np.isfinite(p_): ax.plot([p_], [y], marker=["o", "s", "^"][i % 3], mfc=[K, W, G2][i % 3], mec=K, ms=6, ls="", zorder=3); rows.append({"exogenous": r.exogenous, "station": s, "percentile": p_})
        ax.text(1.04, y, f"Holm p = {r.p_holm:.2f}" if np.isfinite(r.p_holm) else "n/a", fontsize=6.5, va="center"); yt.append(y); yl.append(r.exogenous.replace("_", " ")); y -= 1
    ax.axvline(0.5, color=K, ls=":", lw=0.8); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.8); ax.set_xlim(0, 1.25); ax.spines["bottom"].set_bounds(0, 1); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_ylim(y + 0.4, 0.8)
    miss = sorted({m for r in ex.itertuples() if isinstance(r.core_missing, str) for m in r.core_missing.split("|") if m})
    H = [Line2D([], [], marker=["o", "s", "^"][i % 3], mfc=[K, W, G2][i % 3], mec=K, ls="", ms=6, label=s) for i, s in enumerate(cn) if s not in [nm(m) for m in miss]]
    ax.legend(handles=H, loc="lower left", ncol=3, fontsize=6.5, bbox_to_anchor=(0.0, 1.0))
    ax.set_xlabel("Percentile within the out-of-scope pool (stations with data)" + (f"\nno exogenous data for: {', '.join(nm(m) for m in miss)}" if miss else "")); panel(ax, "c")
    src(t3, "Fig08a_source"); src(sub, "Fig08b_source"); src(pd.DataFrame(rows), "Fig08c_source"); save(fig, "Fig08_core_candidates")

# ================================================================= SUPPLEMENTARY (manuscript numbering)
def figS1():
    V = C["V"].copy(); V["st"] = [nm(s) for s in C["eng"]]; V["inside"] = C["scopes"][C["prim"]] >= TAU
    t = V.sort_values("overload_a05", ascending=False).head(15).iloc[::-1].copy(); t["lab"] = [f"{s}*" if ins else s for s, ins in zip(t.st, t.inside)]
    fig, ax = plt.subplots(figsize=(120 * MM, 85 * MM)); fig.subplots_adjust(left=0.27, right=0.97, top=0.95, bottom=0.14)
    ax.barh(t.lab, t.topo_alpha_inf, color="#4D4D4D", ec=K, lw=0.5, label="Topological disconnection only (alpha = 1e6, numerically infinite)")
    ax.barh(t.lab, t.overload_extra_a05, left=t.topo_alpha_inf, color=W, ec=K, hatch="////", lw=0.5, label="Additional overload cascade (alpha = 0.5)")
    ax.text(0.98, 0.36, "* inside the line-tag scope (w >= 0.5);\nunmarked: outside scope (pool).\nBars reaching 1.0 are saturation of the original\noverload rule (floor load 1e-4 for zero-value\nstations), not a measured network-wide collapse.", transform=ax.transAxes, ha="right", va="center", fontsize=6.5, linespacing=1.3)
    ax.set_xlabel("Fraction of stations failed after single-station removal"); ax.legend(loc="lower right", fontsize=6.5)
    CAPS["Fig. S1"] = ("Decomposition of the single-station-removal 'cascade impact' of the 15 highest-impact stations into topological disconnection (alpha = 10^6) and additional overload cascade (alpha = 0.5). "
                       "Asterisk: station inside the line-tag scope (w >= 0.5); unmarked stations lie in the out-of-scope pool. Bars at 1.0 reflect saturation of the overload rule (floor load 10^-4 for zero-value stations), not a measured network-wide collapse.")
    src(t[["st", "inside", "topo_alpha_inf", "overload_extra_a05", "overload_a05"]], "FigS1_source"); save(fig, "FigS1_cascade_decomposition")
def figS2():
    V = C["V"]; pool = np.where(C["scopes"][C["prim"]] < TAU)[0]; n = len(pool)
    spec = [("betw_unw", "Betweenness (unweighted)"), ("overload_a05", "Cascade impact (overload)"), ("deg", "Degree"), ("eff_loss_dist", "Efficiency loss (distance)")]
    styles = [(K, "-", "o"), (G1, "--", "s"), (G2, ":", "^"), (G2, "-.", "D")]
    spec = [s for s in spec if s[0] in V.columns]; fig = plt.figure(figsize=(190 * MM, 80 * MM)); gs = GridSpec(1, 2, wspace=0.28, left=0.07, right=0.99, top=0.92, bottom=0.17); rows = []
    a1, a2 = (fig.add_subplot(gs[0, i]) for i in range(2)); a1.plot([0, 1], [0, 1], color=G3, lw=0.8)
    for (key, label), (c, ls, mk) in zip(spec, styles):
        x = V[key].to_numpy(float)[pool]; xs = np.sort(x); cum = np.r_[0, np.cumsum(xs) / xs.sum()]; mf = c if mk in "oD" else W
        a1.plot(np.arange(n + 1) / n, cum, color=c, ls=ls, lw=1.0, marker=mk, ms=2.6, mfc=mf, mec=c, mew=0.5, label=label)          # every station drawn: conventional Lorenz polyline
        d = np.sort(x)[::-1]; cd = np.r_[0, np.cumsum(d) / d.sum()]; a2.plot(np.arange(n + 1), cd, color=c, ls=ls, lw=1.0, marker=mk, ms=2.6, mfc=mf, mec=c, mew=0.5)
        assert abs(cd[3] - top_share(x, 3)) < 1e-12; rows.append({"indicator": label, "gini": round(gini(x), 3), "top3_share_pct": round(100 * top_share(x, 3), 1), "n_for_80pct": n80(x), "zero_values": int((x == 0).sum())})
    a1.set_xlabel("Cumulative share of stations (ascending)"); a1.set_ylabel("Cumulative share of value"); a1.legend(loc="upper left", fontsize=6.5); panel(a1, "a")
    a2.axvline(3, color=K, ls=":", lw=0.7); a2.set_xlabel("Number of top stations (descending)"); a2.set_ylabel("Cumulative share of value"); a2.set_xlim(0, n); panel(a2, "b")
    TBL = pd.DataFrame(rows); TBL.to_csv(OUT / "TableS2_concentration.csv", index=False, encoding="utf-8-sig"); log("   S2 summary table written: TableS2_concentration.csv")
    src(TBL, "FigS2_source"); save(fig, "FigS2_lorenz_topshare")
def figS3():
    """size screen (a) + Getis-Ord Gi* map (b) = the former 'FigS8_size_screen_local'; S3_MAP=0 gives panel a only."""
    e1 = tab("supp_e/tables/E1_size_screen.csv"); gi = tab("supp/tables/S5_local_Gi.csv"); n_hot = int((gi.p_hot_bh < ALPHA).sum())
    if S3_MAP: fig = plt.figure(figsize=(190 * MM, 125 * MM)); gs = GridSpec(1, 2, width_ratios=[1.15, 1], wspace=0.28, left=0.17, right=0.985, top=0.95, bottom=0.27)
    else: fig = plt.figure(figsize=(130 * MM, 110 * MM)); gs = GridSpec(1, 1, left=0.3, right=0.985, top=0.95, bottom=0.3)
    ax = fig.add_subplot(gs[0, 0]); stats = list(dict.fromkeys(e1.statistic)); inds = list(dict.fromkeys(e1.indicator)); mk = ["o", "s", "^", "v", "D", "P"]; zb = NormalDist().inv_cdf(1 - ALPHA / len(e1))
    ax.axvspan(ALPHA - zb * e1.mcse.mean(), ALPHA + zb * e1.mcse.mean(), color=G4, zorder=0); ax.axvline(ALPHA, color=K, ls=":", lw=0.8)
    for i, s in enumerate(stats):
        for j, ind in enumerate(inds):
            r = e1[(e1.statistic == s) & (e1.indicator == ind)]
            if r.empty: continue
            r = r.iloc[0]; yy = -i + (j - (len(inds) - 1) / 2) * 0.13; ax.errorbar(r.reject, yy, xerr=2 * r.mcse, fmt="none", ecolor=G2, elinewidth=0.5, capsize=1); ax.plot([r.reject], [yy], mk[j % 6], mfc=W if j % 2 else K, mec=K, ms=4.5)
    ax.set_yticks([-i for i in range(len(stats))]); ax.set_yticklabels([stat_lab(s) for s in stats], fontsize=6.8); ax.set_xlabel("Rejection rate under random scope (m = 0)"); ax.set_xlim(-0.005, max(0.09, e1.reject.max() * 1.15))
    H = [Line2D([], [], marker=mk[j % 6], mfc=W if j % 2 else K, mec=K, ls="", ms=4.5, label=lab(ind)) for j, ind in enumerate(inds)] + [Line2D([], [], color=K, ls=":", lw=0.8, label="alpha = 0.05"), Rectangle((0, 0), 1, 1, fc=G4, ec="none", label="Bonferroni bound band")]
    ax.legend(handles=H, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6.5, columnspacing=1.0); panel(ax, "a")
    if not S3_MAP:
        ax.text(0.99, 0.02, f"Local Gi* hot spots (BH-adjusted): {n_hot}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, bbox=dict(fc=W, ec=G3, lw=0.4, pad=2))
        src(e1, "FigS3a_source"); save(fig, "FigS3_size_screen"); return
    ax = fig.add_subplot(gs[0, 1]); draw_map(ax); idx = [C["eng_ix"].get(s) for s in gi.station]; ok = [i is not None for i in idx]; gi = gi[ok].copy(); idx = [i for i in idx if i is not None]; lon, lat = C["lon"][idx], C["lat"][idx]
    vmax = max(gi.value.max(), 1e-9); zc = np.clip(gi.Gi_star_z.to_numpy(float), -3, 3); sc = ax.scatter(lon, lat, s=10 + 150 * gi.value / vmax, c=zc, cmap="Greys", vmin=-3, vmax=3, edgecolors=K, linewidths=0.5, zorder=3)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.08, aspect=30); cb.set_label("Getis-Ord Gi* z", fontsize=7); cb.ax.tick_params(labelsize=6.5); cb.outline.set_linewidth(0.4)
    ax.text(0.03, 0.03, f"Significant hot spots (BH): {n_hot}", transform=ax.transAxes, fontsize=6.8, bbox=dict(fc=W, ec=G3, lw=0.4, pad=2)); panel(ax, "b")
    CAPS["Fig. S3"] = ("(a) Empirical size (m = 0) of six audit statistics for five indicators; error bars ±2 MCSE; band: Bonferroni bound over 30 cells. "
                       f"(b) Getis-Ord Gi* (k = 5) for the out-of-scope pool; marker area is proportional to the station value; {n_hot} hot spots survive BH adjustment.")
    src(e1, "FigS3a_source"); src(gi, "FigS3b_source"); save(fig, "FigS3_size_screen_local")
def figS4():
    s1 = tab("supp/tables/S1_concentration_all_scopes.csv"); s1 = s1[s1.scope == s1.scope.iloc[0]].copy(); s1["d"] = s1.gini_pool - s1.gini_inscope; s1 = s1.sort_values("d").reset_index(drop=True); y = np.arange(len(s1)); FL = 1e-4
    fig = plt.figure(figsize=(190 * MM, 105 * MM)); gs = GridSpec(1, 2, wspace=0.08, left=0.2, right=0.985, top=0.74, bottom=0.12)
    ax = fig.add_subplot(gs[0, 0])
    for yy, r in zip(y, s1.itertuples()): ax.plot([r.gini_inscope, r.gini_pool], [yy, yy], color=G2, lw=1.0)
    ax.plot(s1.gini_inscope, y, "o", mfc=W, mec=K, ms=5.5, ls="", label="In scope"); ax.plot(s1.gini_pool, y, "o", color=K, ms=5.5, ls="", label="Out of scope"); ax.set_yticks(y); ax.set_yticklabels([lab(i) for i in s1.indicator]); ax.set_xlabel("Gini coefficient")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=6.5); panel(ax, "a", y=1.22)
    ax = fig.add_subplot(gs[0, 1], sharey=ax); plt.setp(ax.get_yticklabels(), visible=False); ax.tick_params(left=False)
    for col, fc, mec, la in (("p_randompool", CAL, CAL, "Random pool"), ("p_dirichlet_legacy", W, LEG, f"Dirichlet(1) ({CONV})"), ("p_diff_label_permutation", G2, K, "Label permutation (out - in)")):
        p = s1[col].to_numpy(float); fl = p <= FL; mk = "s" if "permutation" in col else "o"
        ax.plot(np.clip(p[~fl], FL, None), y[~fl], mk, mfc=fc, mec=mec, ms=5, ls="", label=la)
        if fl.any(): ax.plot(np.full(fl.sum(), FL), y[fl], "<", mfc=fc, mec=mec, ms=6, ls="")        # arrow marker: value at or below the floor
    ax.plot([], [], "<", mfc=W, mec=K, ms=6, ls="", label="p <= 1e-4 (floor)")
    ax.set_xscale("log"); ax.set_xlim(3e-5, 1.5); ax.axvline(ALPHA, color=K, ls=":", lw=0.8); ax.set_xlabel("p-value (floor 1e-4)"); ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, fontsize=6.5, labelspacing=0.3); panel(ax, "b", y=1.22)
    src(s1, "FigS4_source"); save(fig, "FigS4_in_vs_out_gini")
def figS5():
    spc = tab("tables/T25_specification_curve.csv").sort_values("effect_vs_randompool").reset_index(drop=True); x = spc.index.to_numpy(); sig = spc.p_randompool < ALPHA
    spc["scope_lab"] = spc.scope.map(short_scope)
    scopes = list(dict.fromkeys(spc.scope_lab)); taus = sorted(spc.tau.unique()); inds = list(dict.fromkeys(spc.indicator))
    rowsel = [("scope_lab", s, s) for s in scopes] + [("tau", t, f"tau = {t:g}") for t in taus] + [("indicator", i, lab(i)) for i in inds]
    fig = plt.figure(figsize=(190 * MM, 152 * MM)); gs = GridSpec(2, 1, height_ratios=[1.6, 2.3], hspace=0.16, left=0.27, right=0.985, top=0.95, bottom=0.13); a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1], sharex=a1)
    a1.scatter(x[~sig], spc.effect_vs_randompool[~sig], s=14, facecolors=W, edgecolors=G1, linewidths=0.6, label=f"p >= {ALPHA}"); a1.scatter(x[sig], spc.effect_vs_randompool[sig], s=16, c=K, label=f"p < {ALPHA}"); a1.axhline(0, color=K, lw=0.6)
    a1.set_ylabel("Gini(out-of-scope) minus\nrandom-pool null mean"); a1.legend(loc="upper left", fontsize=6.5); plt.setp(a1.get_xticklabels(), visible=False)
    pct = "\n".join(f"{s}: {100 * (spc[spc.scope_lab == s].p_randompool < ALPHA).mean():.1f}%" for s in scopes)
    a1.text(0.99, 0.04, "Share of specifications with p < 0.05\n" + pct, transform=a1.transAxes, ha="right", va="bottom", fontsize=6.5, linespacing=1.3, bbox=dict(fc=W, ec=G3, lw=0.4, pad=2)); panel(a1, "a")
    for r, (col, val, _) in enumerate(rowsel): hit = spc.index[spc[col] == val]; a2.scatter(hit, np.full(len(hit), r), marker="|", s=40, c=K, lw=0.8)
    a2.set_yticks(range(len(rowsel))); a2.set_yticklabels([t_ for _, _, t_ in rowsel], fontsize=6.5); a2.invert_yaxis()
    for b_ in (len(scopes) - 0.5, len(scopes) + len(taus) - 0.5): a2.axhline(b_, color=G3, lw=0.5)
    a2.set_xlabel("Specifications sorted by effect size"); panel(a2, "b")
    fig.text(0.27, 0.005, "Specifications are not independent (several indicators are strongly correlated) and pool sizes differ across scope definitions,\nso effects are not directly comparable across rows; the binomial share is descriptive only.", fontsize=6.5, va="bottom", linespacing=1.3)
    src(spc, "FigS5_source"); save(fig, "FigS5_specification_curve")
def figS6():
    """Stage C diagnostics (old file S5). Panel c x-label wrapped to two lines; legend moved below it."""
    s3 = tab("supp/tables/S3_stageC_primary_pool.csv"); s3["fs"] = s3.feature_set.str.split(" \\(").str[0]; fss = list(dict.fromkeys(s3.fs)); Ks = sorted(s3.K.unique()); sty = [("-", "o", K, K), ("--", "s", K, W), (":", "^", G2, G2)]
    ab = lambda f: {"Service-augmented": "Svc-aug."}.get(f.split(" ")[0], f.split(" ")[0])
    fig = plt.figure(figsize=(190 * MM, 95 * MM)); gs = GridSpec(1, 3, wspace=0.55, left=0.06, right=0.99, top=0.92, bottom=0.27, width_ratios=[1, 1, 1.3]); ax = fig.add_subplot(gs[0, 0])
    for i, f in enumerate(fss): g = s3[s3.fs == f].sort_values("K"); ax.plot(g.K, 100 * g.greedy_coverage, ls=sty[i % 3][0], marker=sty[i % 3][1], color=sty[i % 3][2], mfc=sty[i % 3][3], ms=4.5, label=f"Greedy: {f}")
    g = s3.groupby("K").random_mean_coverage.mean(); ax.plot(g.index, 100 * g.values, color=G3, ls="-", lw=2, label="Random sets (mean)"); ax.set_xticks(Ks); ax.set_xlabel("K (cardinality limit)"); ax.set_ylabel("Objective (% of pool total)")
    ymin_, ymax_ = ax.get_ylim(); ax.set_ylim(ymin_, ymax_ + 0.45 * (ymax_ - ymin_)); ax.legend(fontsize=6.5, loc="upper left", labelspacing=0.3); panel(ax, "a")
    ax = fig.add_subplot(gs[0, 1]); wd = 0.8 / len(fss)
    for i, f in enumerate(fss): g = s3[s3.fs == f].sort_values("K"); ax.bar(np.arange(len(Ks)) + i * wd, g.StageB_topK_objective_over_greedy, wd, color=[K, W, G3][i % 3], ec=K, lw=0.5, hatch=["", "////", ""][i % 3], label=f)
    ax.axhline(1, color=K, ls=":", lw=0.7); ax.set_xticks(np.arange(len(Ks)) + wd * (len(fss) - 1) / 2); ax.set_xticklabels([f"K = {int(k)}" for k in Ks]); ax.set_ylim(0, 1.5); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.set_ylabel("Diagnostic top-K objective / greedy objective"); ax.legend(loc="upper right", fontsize=6.5, labelspacing=0.3); panel(ax, "b")
    ax = fig.add_subplot(gs[0, 2]); s3 = s3.sort_values(["K", "fs"]).reset_index(drop=True); y = -np.arange(len(s3)); ax.axvline(ALPHA, color=K, ls=":", lw=0.8)
    ax.plot(s3.p_uniform_exact_mean, y, "o", mfc=W, mec=K, ms=5, ls="", label="Uniform null (exact)"); ax.plot(s3.p_information_matched, y, "o", color=K, ms=5, ls="", label="Information-matched null")
    ax.set_xscale("log"); ax.set_xlim(5e-3, 1.5); ax.set_yticks(y); ax.set_yticklabels([f"K = {int(r.K)} | {ab(r.fs)}" for r in s3.itertuples()], fontsize=6.5); ax.set_xlabel("p-value of overlap\n(diagnostic top-K vs greedy set)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), fontsize=6.5, ncol=1); panel(ax, "c")
    CAPS["Fig. S6"] = ("Stage C diagnostics. (a) Greedy objective (equal to the exact optimum) by K. (b) Objective of the Stage B top-K relative to greedy. "
                       "(c) Overlap p under the uniform (exact) and information-matched nulls. 'Distinct' uses closeness and eigenvector only; 'Svc-aug.' = service-augmented feature set.")
    src(s3, "FigS6_source"); save(fig, "FigS6_stageC_diagnostics")
def figS7():
    """Dependence matrix (old file S6). Empty first row / last column dropped; x-label wrapped under the matrix."""
    V = C["V"]; pool = np.where(C["scopes"][C["prim"]] < TAU)[0]; cols = [c for c in ["eff_loss_dist", "betw_dist", "overload_a05", "close_dist", "eig_unw", "service_km", "deg"] if c in V.columns]
    M = V.iloc[pool][cols].corr(method="spearman").to_numpy(); n = len(cols); fig, ax = plt.subplots(figsize=(110 * MM, 108 * MM)); fig.subplots_adjust(left=0.33, right=0.98, top=0.68, bottom=0.11)
    sub = M[1:, :-1]; m = n - 1; mask = np.triu(np.ones((m, m)), 1) == 1; A = np.where(mask, np.nan, np.abs(sub)); ax.imshow(A, cmap="Greys", vmin=0, vmax=1)      # lower triangle only; diagonal/empty row+column omitted
    for i in range(m):
        for j in range(i + 1): ax.text(j, i, f"{sub[i, j]:.2f}", ha="center", va="center", fontsize=6.8, color=W if abs(sub[i, j]) > 0.55 else K)
    ax.set_xticks(range(m)); ax.set_xticklabels([lab(c) for c in cols[:-1]], rotation=60, ha="left", fontsize=6.8); ax.xaxis.tick_top(); ax.set_yticks(range(m)); ax.set_yticklabels([lab(c) for c in cols[1:]], fontsize=6.8)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlabel(f"Spearman correlation, out-of-scope pool (n = {len(pool)})\nshade = |rho|; diagonal omitted", fontsize=7)
    CAPS["Fig. S7"] = (f"Spearman dependence among indicators in the out-of-scope pool (n = {len(pool)}); lower triangle, diagonal omitted. "
                       f"The diagnostic indicator (distance-weighted efficiency loss) correlates {M[1, 0]:.2f} with distance-weighted betweenness.")
    src(pd.DataFrame(M, index=cols, columns=cols).reset_index(), "FigS7_source"); save(fig, "FigS7_dependence_matrix")
def figS8():
    """Exogenous consistency + quarterly stability (old file S7). Footer split into short lines, larger bottom margin."""
    vx = tab("tables/T22_exogenous_validation.csv").sort_values("spearman").reset_index(drop=True); fig = plt.figure(figsize=(190 * MM, 128 * MM)); gs = GridSpec(2, 2, width_ratios=[1.35, 1], hspace=0.45, wspace=0.3, left=0.3, right=0.985, top=0.94, bottom=0.16)
    ax = fig.add_subplot(gs[:, 0]); y = np.arange(len(vx)); sig = (vx.ci_lo > 0) | (vx.ci_hi < 0); ax.axvline(0, color=K, lw=0.6)
    for yy, r, s in zip(y, vx.itertuples(), sig): ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=K, lw=0.9); ax.plot([r.spearman], [yy], "o", mfc=K if s else W, mec=K, ms=5)
    sh = lambda k: IND_SHORT.get(k, lab(k))
    ax.set_yticks(y); ax.set_yticklabels([f"{sh(r.structural_or_service)} vs {r.exogenous.replace('_', ' ')} (n={r.n_stations_with_data})" for r in vx.itertuples()], fontsize=6.5)
    ax.set_xlabel(f"Spearman rho (bootstrap 95% CI); filled: CI excludes 0\n{len(vx)} unadjusted intervals\n(no multiplicity correction; exploratory)"); panel(ax, "a")
    rsf = B / "05e_reservations_slim.csv"; got = False
    if rsf.exists():
        try:
            rs = rd(rsf); rs["q"] = pd.to_datetime(rs.rsv_date).dt.to_period("Q").astype(str); qs = sorted(rs.q.unique()); nodes = C["nodes"]
            cnt = {q: pd.concat([rs[rs.q == q].dep_station, rs[rs.q == q].arr_station]).value_counts().reindex(nodes).fillna(0) for q in qs}; rows = []
            for k, (q1, q2) in enumerate(list(zip(qs[:-1], qs[1:]))[:2]):
                ax = fig.add_subplot(gs[k, 1]); both = (cnt[q1] > 0) & (cnt[q2] > 0); a, b = cnt[q1][both].rank(), cnt[q2][both].rank(); r_ = a.corr(b); ax.plot([0, both.sum()], [0, both.sum()], color=G3, lw=0.8); ax.scatter(a, b, s=12, c=K, lw=0)
                ax.text(0.04, 0.96, f"rho = {r_:.2f}  (n = {int(both.sum())})", transform=ax.transAxes, va="top", fontsize=6.8); ax.set_xlabel(f"Rank of reservation flow, {q1}"); ax.set_ylabel(f"Rank, {q2}"); panel(ax, "bc"[k]); rows.append({"q1": q1, "q2": q2, "n": int(both.sum()), "spearman": r_})
            src(pd.DataFrame(rows), "FigS8b_source"); got = True
        except Exception as e: log("   (reservation scatter skipped:", e, ")")
    if not got:
        t24 = tab("tables/T24_temporal_stability.csv"); ax = fig.add_subplot(gs[0, 1]); ax.bar([f"{a}→{b}" for a, b in zip(t24.quarter_1, t24.quarter_2)], t24.spearman, color=G2, ec=K, lw=0.5); ax.set_ylim(0, 1); ax.set_ylabel("Spearman rho"); panel(ax, "b")
    CAPS["Fig. S8"] = (f"(a) Spearman correlations (bootstrap 95% CI, 2,000 resamples) between structural or service indicators and exogenous operational data; {len(vx)} unadjusted intervals, exploratory. "
                       "(b, c) Rank stability of reservation flows across consecutive 2025 quarters.")
    src(vx, "FigS8a_source"); save(fig, "FigS8_exogenous_temporal")

FIGS = {"01": fig01, "02": fig02, "03": fig03, "04": fig04, "05": fig05, "06": fig06, "07": fig07, "08": fig08,
        "S1": figS1, "S2": figS2, "S3": figS3, "S4": figS4, "S5": figS5, "S6": figS6, "S7": figS7, "S8": figS8}
if __name__ == "__main__":
    sel = os.environ.get("FIGS"); ids = [s.strip() for s in sel.split(",")] if sel else list(FIGS)
    log(f"ROOT={ROOT} QUICK={QUICK} COLOR={COLOR} EDGES={EDGES} FIG8C={FIG8C} S3_MAP={S3_MAP} figures={ids}"); prep()
    for i in ids:
        log(f"-- Fig {i}")
        try: FIGS[i]()
        except Exception: FAILED.append(i); traceback.print_exc()
    if MANIFEST: pd.DataFrame(MANIFEST).to_csv(OUT / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    write_captions()
    stale = [f"{n}.{e}" for n in OLD_V3_FILES for e in ("png", "pdf") if (OUT / f"{n}.{e}").exists()]
    if stale: log("\nWARNING: files from the v3 supplement numbering exist and will NOT match the manuscript - delete before submission:\n  " + "\n  ".join(stale))
    log("\nDONE ->", OUT, "| failed:", FAILED if FAILED else "none")
