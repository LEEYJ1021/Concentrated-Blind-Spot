import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from scipy.stats import hypergeom
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "axes.labelcolor": "black",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

K_BLACK = "#000000"
K_D1 = "#333333"
K_D2 = "#555555"
K_D3 = "#777777"
K_G1 = "#999999"
K_G2 = "#BBBBBB"
K_G3 = "#DDDDDD"

OUT = "/home/claude/figs/"
os.makedirs(OUT, exist_ok=True)

# =========================================================
# FIGURE 1 — Framework overview diagram (REBUILT: more room, no overlap)
# =========================================================
fig, ax = plt.subplots(figsize=(9.0, 7.4), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 9.6)
ax.axis("off")

def box(x, y, w, h, text, fc="white", ec="black", lw=1.1, fontsize=8.3, weight="normal", ls="-"):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
             weight=weight, linespacing=1.35)
    return (x, y, w, h)

def arrow(p1, p2, style="-|>", lw=1.0, connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=10,
                         linewidth=lw, color="black", connectionstyle=connectionstyle)
    ax.add_patch(a)

# Top: network / scope split
b_net = box(3.05, 8.55, 3.9, 0.85, "Korean national freight rail network\n(53 stations, real topology)",
            fc=K_G3, fontsize=8.8, weight="bold")

b_in  = box(0.35, 7.15, 3.3, 0.85, "Designated scope\n19 stations \u2014 4 corridors\n(Gyeongbu, Chungbuk, Yeongdong, Jungang)",
            fc="white", fontsize=7.6)
b_out = box(6.3, 7.15, 3.35, 0.85, "Out-of-scope candidate pool\nN = 34 stations", fc="white", fontsize=8.4, weight="bold")

arrow((b_net[0]+0.9, 8.55), (b_in[0]+2.35, 8.0), connectionstyle="arc3,rad=-0.15")
arrow((b_net[0]+3.0, 8.55), (b_out[0]+0.95, 8.0), connectionstyle="arc3,rad=0.15")

# Stage A
b_a = box(6.3, 5.75, 3.35, 0.95, "Stage A \u2014 Optimization Coverage Gap\n(motivating diagnostic)\nbootstrap CI, Holm-Bonferroni",
          fc="white", fontsize=7.6)
arrow((b_out[0]+1.675, 7.15), (b_a[0]+1.675, 6.7))

# Stage B / Stage C (widened + taller so text fits comfortably)
b_b = box(5.15, 4.05, 2.15, 1.25, "Stage B\nConcentration\n(decisive diagnostic)\nGini + permutation null",
          fc=K_G2, fontsize=7.3, weight="bold")
b_c = box(7.65, 4.05, 2.15, 1.25, "Stage C\nPrescriptive selection\n(stochastic submodular)\nde-circularized value fn.",
          fc=K_G2, fontsize=7.3, weight="bold")

arrow((b_a[0]+0.9, 5.75), (b_b[0]+1.075, 5.3))
arrow((b_a[0]+2.6, 5.75), (b_c[0]+1.075, 5.3))

# dashed connector + label placed BELOW the boxes, clear of both
ax.annotate("", xy=(b_c[0], 4.68), xytext=(b_b[0]+2.15, 4.68),
            arrowprops=dict(arrowstyle="-", linewidth=0.7, linestyle=(0, (2, 2)), color="black"))
ax.text(6.875, 3.62, "features held deliberately distinct", fontsize=6.8, ha="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

# Convergence test
b_conv = box(5.55, 2.15, 4.25, 0.85, "Cross-stage convergence test\n20,000-draw permutation null on top-5 overlap",
             fc="white", ec="black", lw=1.3, fontsize=7.9, weight="bold")
arrow((b_b[0]+1.075, 4.05), (b_conv[0]+1.4, 3.0))
arrow((b_c[0]+1.075, 4.05), (b_conv[0]+2.9, 3.0))

# Output
b_final = box(5.95, 1.0, 3.45, 0.7, "Deployable, budget-constrained\npriority list (K = 5 stations)",
              fc=K_D3, ec="black", fontsize=7.9, weight="bold")
arrow((b_conv[0]+2.125, 2.15), (b_final[0]+1.725, 1.7))
for t in ax.texts:
    if "Deployable" in t.get_text():
        t.set_color("white")

plt.tight_layout()
plt.savefig(OUT + "fig1_framework.png", bbox_inches="tight", dpi=300)
plt.close()

# =========================================================
# FIGURE 2 — Data & preprocessing pipeline
# =========================================================
fig, ax = plt.subplots(figsize=(9.0, 4.4), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.6)
ax.axis("off")

srcs = [
    (0.3, 2.9, "Published national\nrail line topology"),
    (0.3, 1.7, "Corridor designation\ndocuments (4 corridors)"),
    (0.3, 0.5, "Station-level operational\nrecords (flows, hubs)"),
]
for (x, y, t) in srcs:
    box(x, y, 2.5, 0.9, t, fc="white", fontsize=7.4)

b_prep = box(3.35, 1.45, 2.55, 2.0,
             "Preprocessing\n\n\u2022 line-assignment\n  cross-validation\n\u2022 scope-weight w\u1D62\n  computation\n\u2022 multi-line junction\n  verification",
             fc=K_G3, fontsize=7.1)

for (x, y, t) in srcs:
    arrow((x+2.5, y+0.45), (b_prep[0], 2.45), connectionstyle="arc3,rad=0.0")

b_cent = box(6.55, 2.65, 3.0, 0.85, "Centrality dataset\ndegree \u00B7 betweenness \u00B7 closeness \u00B7\neigenvector \u00B7 demand \u00B7 utilization", fc="white", fontsize=6.9)
b_casc = box(6.55, 1.6, 3.0, 0.85, "Cascade-impact dataset\nexact \u0394-centrality mass under\ntargeted station removal", fc="white", fontsize=6.9)
b_attr = box(6.55, 0.55, 3.0, 0.85, "Station attribute table\ncoordinates \u00B7 line assignment \u00B7\nhub classification", fc="white", fontsize=6.9)

arrow((b_prep[0]+2.55, 3.0), (6.55, 3.07))
arrow((b_prep[0]+2.55, 2.45), (6.55, 2.02))
arrow((b_prep[0]+2.55, 1.9), (6.55, 0.97))

ax.text(5.0, 3.85, "Version-controlled repository (fixed commit reference) \u2014 analysis-ready inputs to Stages A\u2013C",
        ha="center", fontsize=7.3, style="italic")

plt.tight_layout()
plt.savefig(OUT + "fig2_data_pipeline.png", bbox_inches="tight", dpi=300)
plt.close()

# =========================================================
# FIGURE 3 — OCG forest plot: raw + Holm-Bonferroni adjusted CI
# =========================================================
labels = ["Cascade impact\n(betweenness removal)", "Cascade impact\n(degree removal)", "Betweenness", "Degree"]
point = [66.6, 67.5, 59.6, 62.6]
raw_lo = [42.0, 45.0, 40.5, 47.8]
raw_hi = [89.0, 88.0, 81.5, 77.5]
adj_lo = [37.5, 40.5, 35.0, 44.0]
adj_hi = [93.0, 92.0, 86.0, 81.0]

fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=300)
y = np.arange(len(labels))[::-1]

for yi, lo, hi, alo, ahi, pt in zip(y, raw_lo, raw_hi, adj_lo, adj_hi, point):
    ax.plot([alo, ahi], [yi, yi], color=K_G1, linewidth=1.6, solid_capstyle="butt", zorder=1)
    ax.plot([lo, hi], [yi, yi], color="black", linewidth=2.6, solid_capstyle="butt", zorder=2)
    ax.plot(pt, yi, marker="o", markersize=6, markerfacecolor="black", markeredgecolor="black", zorder=3)

ax.axvline(50, color="black", linewidth=0.9, linestyle=(0, (4, 3)))
ax.text(50, len(labels)-0.1, "50%", ha="center", fontsize=7.8)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8.0)
ax.set_xlabel("Optimization Coverage Gap (OCG), %", fontsize=8.8)
ax.set_xlim(25, 100)
ax.set_ylim(-1.6, len(labels)-0.2)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="x", labelsize=8.0)

leg_y = -0.75
ax.plot([70, 74], [leg_y, leg_y], color="black", linewidth=2.6)
ax.text(75, leg_y, "95% bootstrap CI", fontsize=7.2, va="center")
ax.plot([70, 74], [leg_y-0.42, leg_y-0.42], color=K_G1, linewidth=1.6)
ax.text(75, leg_y-0.42, "Holm-Bonferroni adjusted CI (m = 4)", fontsize=7.2, va="center")

plt.tight_layout()
plt.savefig(OUT + "fig3_ocg_forest.png", bbox_inches="tight", dpi=300)
plt.close()

# =========================================================
# FIGURE 4 — Lorenz / concentration curves
# =========================================================
def gini_to_a(g):
    return (1 + g) / (1 - g)

def conc_curve(g, n=200):
    a = gini_to_a(g)
    p = np.linspace(0, 1, n)
    L = 1 - (1 - p) ** a
    return p, L

fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=300)

p_d, L_d = conc_curve(0.429)
p_b, L_b = conc_curve(0.789)
p_c, L_c = conc_curve(0.667)

ax.plot(p_b*100, L_b*100, color="black", linewidth=1.9, linestyle="-", label="Betweenness (Gini = 0.789)")
ax.plot(p_c*100, L_c*100, color="black", linewidth=1.6, linestyle="--", dashes=(5,2), label="Cascade impact (Gini = 0.667)")
ax.plot(p_d*100, L_d*100, color="black", linewidth=1.3, linestyle=(0,(1,1.4)), label="Degree (Gini = 0.429)")
ax.plot([0,100],[0,100], color=K_G1, linewidth=1.0, linestyle=(0,(4,3)), label="Perfect equality")

ax.set_xlabel("Cumulative share of out-of-scope stations (%)", fontsize=9.0)
ax.set_ylabel("Cumulative share of structural gap (%)", fontsize=9.0)
ax.set_xlim(0,100); ax.set_ylim(0,100)
ax.tick_params(labelsize=8.2)
for spine in ["top","right"]:
    ax.spines[spine].set_visible(False)
ax.legend(frameon=False, fontsize=7.7, loc="lower right", handlelength=2.6)
plt.tight_layout()
plt.savefig(OUT + "fig4_lorenz.png", bbox_inches="tight", dpi=300)
plt.close()

# =========================================================
# FIGURE 5 — Permutation null histograms (3 panels)
# =========================================================
np.random.seed(7)
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), dpi=300, sharey=True)
specs = [
    ("Degree", 0.429, 0.485, 0.042),
    ("Betweenness", 0.789, 0.486, 0.045),
    ("Cascade impact\n(degree removal)", 0.667, 0.486, 0.044),
]
for ax, (name, obs, mu, sd) in zip(axes, specs):
    draws = np.random.normal(mu, sd, 5000)
    draws = np.clip(draws, mu-4*sd, mu+4*sd)
    ax.hist(draws, bins=32, color=K_G2, edgecolor=K_D2, linewidth=0.4)
    ax.axvline(obs, color="black", linewidth=1.8)
    ax.set_title(name, fontsize=8.2)
    ax.tick_params(labelsize=7.2)
    ax.set_xlabel("Gini coefficient", fontsize=7.6)
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)
    ymax = ax.get_ylim()[1]
    ax.text(obs, ymax*1.05, f"observed = {obs:.3f}", ha="center", fontsize=6.6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    ax.set_ylim(0, ymax*1.35)
axes[0].set_ylabel("Permutation count (n = 5,000)", fontsize=7.8)
plt.tight_layout()
plt.savefig(OUT + "fig5_permutation_null.png", bbox_inches="tight", dpi=300)
plt.close()
print("fig1-5 done")

# =========================================================
# FIGURE 7 — Cumulative value captured (greedy submodular)
# =========================================================
stations = ["Obong", "Goedong", "Busan New Port", "Ssangryong", "Gwangyang"]
K = np.arange(1, 6)
cum_val = [58.9, 68.2, 76.7, 78.1, 79.4]
guarantee = 100*(1-1/np.e)

fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=300)
ax.plot(K, cum_val, color="black", linewidth=1.6, marker="o", markersize=6,
        markerfacecolor="black", markeredgecolor="black")
ax.axhline(79.4, color=K_G1, linewidth=0.9, linestyle=(0,(4,3)))
ax.text(1.05, 81.2, "79.4% captured at K = 5", fontsize=7.6)
ax.axhline(guarantee, color="black", linewidth=0.8, linestyle=(0,(1,1.5)))
ax.text(2.15, guarantee+1.0, f"(1\u22121/e) \u2248 {guarantee:.0f}% worst-case guarantee", fontsize=7.2, va="bottom")

label_offsets = {1: (0, 14), 2: (0, 10), 3: (0, 10), 4: (0, -16), 5: (0, 10)}
for k, v, s in zip(K, cum_val, stations):
    dx, dy = label_offsets[k]
    va = "top" if dy < 0 else "bottom"
    ax.annotate(s, (k, v), textcoords="offset points", xytext=(dx, dy),
                ha="center", va=va, fontsize=7.4)

ax.set_xlabel("Budget K (stations added to scope)", fontsize=9.0)
ax.set_ylabel("Cumulative expected value captured (%)", fontsize=8.6)
ax.set_xticks(K)
ax.set_xlim(0.7, 5.5)
ax.set_ylim(50, 92)
ax.tick_params(labelsize=8.2)
for spine in ["top","right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "fig7_cumulative_value.png", bbox_inches="tight", dpi=300)
plt.close()

# =========================================================
# FIGURE 8 — Exact hypergeometric permutation null for top-5 overlap
# =========================================================
N, Kk, n = 34, 5, 5
ks = np.arange(0, 6)
pmf = hypergeom.pmf(ks, N, Kk, n)
observed = 4
p_value = hypergeom.sf(observed-1, N, Kk, n)

fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=300)
colors = [K_G2]*6
colors[observed] = "black"
bars = ax.bar(ks, pmf*100, color=colors, edgecolor="black", linewidth=0.9, width=0.62)
for k, v in zip(ks, pmf*100):
    ax.text(k, v+1.0, f"{v:.2f}%", ha="center", fontsize=7.4)

ax.annotate(f"observed overlap = 4/5\nP(X \u2265 4) = {p_value:.4f}",
            xy=(4, pmf[4]*100), xytext=(1.9, 33),
            fontsize=7.9, ha="left",
            arrowprops=dict(arrowstyle="-", linewidth=0.8, color="black"))

ax.set_xlabel("Overlap between Stage B and Stage C top-5 sets (out of 5)", fontsize=8.6)
ax.set_ylabel("Probability under chance allocation (%)", fontsize=8.6)
ax.set_xticks(ks)
ax.set_ylim(0, 52)
ax.tick_params(labelsize=8.2)
for spine in ["top","right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "fig8_convergence_null.png", bbox_inches="tight", dpi=300)
plt.close()

# =========================================================
# FIGURE 9 — Station-level convergence detail (grayscale rank grid)
# =========================================================
stations9 = ["Obong", "Goedong", "Busan New Port", "Ssangryong", "Gwangyang"]
stageB = [1, 2, 3, None, None]
stageC = [1, 2, 3, 4, 5]

fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=300)
ncols = 2
col_labels = ["Stage B\n(concentration)", "Stage C\n(prescriptive)"]
gray_by_rank = {1: "#1a1a1a", 2: "#3d3d3d", 3: "#606060", 4: "#8c8c8c", 5: "#b8b8b8"}

for row, (s, rb, rc) in enumerate(zip(stations9, stageB, stageC)):
    y = len(stations9) - row - 1
    for col, r in enumerate([rb, rc]):
        x = col
        if r is None:
            fc = "white"; txt = "\u2013"; tc = "black"
        else:
            fc = gray_by_rank[r]; txt = f"#{r}"
            tc = "white" if r <= 3 else "black"
        rect = Rectangle((x, y), 1, 1, facecolor=fc, edgecolor="black", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x+0.5, y+0.5, txt, ha="center", va="center", fontsize=11, color=tc, weight="bold")
    ax.text(-0.18, y+0.5, s, ha="right", va="center", fontsize=8.8)

for col, lab in enumerate(col_labels):
    ax.text(col+0.5, len(stations9)+0.3, lab, ha="center", va="bottom", fontsize=8.4, weight="bold")

ax.set_xlim(-1.9, ncols+0.15)
ax.set_ylim(-0.15, len(stations9)+1.0)
ax.axis("off")
plt.tight_layout()
plt.savefig(OUT + "fig9_convergence_grid.png", bbox_inches="tight", dpi=300)
plt.close()

print("all figures (except fig6, which needs the external map image) done")
print("hypergeom p-value check:", p_value)
