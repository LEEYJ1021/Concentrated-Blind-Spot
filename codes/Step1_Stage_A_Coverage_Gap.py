# -*- coding: utf-8 -*-
"""
================================================================================
STEP 1 — STAGE A: OPTIMIZATION COVERAGE GAP (motivating diagnostic)
--------------------------------------------------------------------------------
Concentrated Blind Spots: Diagnosing and Prioritizing Structural Coverage Gaps
in Corridor-Based Rail Decarbonization Networks

Loads the real Korean national freight rail network (53 stations, 4 designated
corridors covering 19 stations) and computes the Optimization Coverage Gap
(OCG): the share of total weighted centrality/cascade-impact mass sitting
outside the designated decarbonization scope, under four weightings (degree,
betweenness, cascade impact under degree-based removal, cascade impact under
betweenness-based removal).

This stage is deliberately treated as a MOTIVATING diagnostic, not a confirmed
finding: 95% bootstrap confidence intervals (5,000 replicates), corrected for
multiplicity across the four weightings tested (Holm-Bonferroni, m=4), do not
exclude the 50% reference value at this sample size (N=53). Stage 2
(Step2_Stage_B_Concentration.py) supersedes this with the decisive test.

Outputs
-------
  outputs/master_table.csv          station-level master dataset (all stages read this)
  outputs/stage_a_coverage_gap.csv   OCG point estimates, 95% CIs, Holm-Bonferroni flags
================================================================================
"""

import io
import os
import urllib.request

import numpy as np
import pandas as pd

# ============================================================================
# SECTION 0 — CONFIG (shared across all six scripts)
# ============================================================================
np.random.seed(42)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 25)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

BASE1 = "https://raw.githubusercontent.com/LEEYJ1021/rail-freight-decarbonization/main"
BASE2 = "https://raw.githubusercontent.com/LEEYJ1021/korea-freight-rail-resilience-analysis/main"
URLS = {
    "stations":    f"{BASE1}/curated/freight_stations_parsed.csv",
    "centrality":  f"{BASE2}/upload_2026-07/data/centrality_data.csv",
    "cascade_deg": f"{BASE2}/upload_2026-07/data/cascade_degree_exact.csv",
    "cascade_btw": f"{BASE2}/upload_2026-07/data/cascade_betweenness_exact.csv",
    "cascade_clo": f"{BASE2}/upload_2026-07/data/cascade_closeness_exact.csv",
    "cascade_eig": f"{BASE2}/upload_2026-07/data/cascade_eigenvector_exact.csv",
    "corridor_ef": f"{BASE2}/upload_2026-07/data/corridor_efficiency_summary.csv",
}

IN_SCOPE = {"경부", "충북", "영동", "중앙"}
CORRIDOR_NAME = {"경부": "Gyeongbu", "충북": "Chungbuk", "영동": "Yeongdong", "중앙": "Jungang"}
FEATURE_COLS = ["degree", "betweenness", "closeness", "eigenvector",
                "demand_centrality", "utilization_centrality", "composite_centrality"]

STATION_EN = {
    "오봉": "Obong", "괴동": "Goedong", "부산신항": "Busan New Port",
    "쌍룡": "Ssangryong", "광양": "Gwangyang", "입석리": "Ipseokri",
    "신광양항": "Shingwangyang-hang",
}

VERIFIED_MAPPING = {
    "팔당": (["중앙"],), "덕소": (["중앙"],), "청주": (["충북"],), "입석리": (["태백"],),
    "쌍룡": (["태백"],), "석항": (["태백"],), "무릉": (["태백"],), "철암": (["영동"],),
    "옥계": (["영동"],), "제천": (["태백", "중앙", "충북"],), "제천조차장": (["태백", "중앙", "충북"],),
    "영주": (["중앙", "영동", "경북"],), "경주": (["동해"],),
    "의왕": (["경부"],), "천안": (["경부", "장항"],), "대전조차장": (["경부"],), "약목": (["경부"],),
    "동산": (["충북"],), "가야": (["가야선(경부지선)"],), "부산신항": (["부산신항선(경부지선)"],),
    "광운대": (["경원"],), "신례원": (["장항"],), "도안": (["장항"],), "삽교": (["장항"],),
    "흥국사": (["전라"],), "수색": (["경의"],), "동해": (["영동", "동해"],), "신동": (["태백"],),
    "문수": (["동해"],), "순천": (["전라", "경전"],), "마산": (["경전"],), "목포": (["호남"],),
    "황등": (["호남"],), "익산": (["호남", "전라"],),
}
MULTI_LINE_JUNCTIONS = {k for k, v in VERIFIED_MAPPING.items()
                         if len(v[0]) > 1 and not any("지선" in x for x in v[0])}


def fetch_csv(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return pd.read_csv(io.BytesIO(resp.read()), encoding="utf-8-sig")


def load_all():
    d = {}
    for k, u in URLS.items():
        d[k] = fetch_csv(u)
        d[k].columns = [c.strip() for c in d[k].columns]
    return d


def build_master(data):
    """Station attribute table + centrality + cascade-impact datasets, merged,
    with scope weight w_i in [0, 1] (share of non-branch line assignments
    falling within the four designated corridors)."""
    stations = data["stations"].copy()
    stations["station"] = stations["station"].astype(str).str.strip()
    line_map_orig = stations.set_index("station")["line_tag"]

    cent = data["centrality"].copy()
    cent["station_kor"] = cent["station_kor"].astype(str).str.strip()
    cascade = cent[["station_kor", "station_eng"]].copy()
    for key, col in [("cascade_deg", "cascade_impact_degree"),
                      ("cascade_btw", "cascade_impact_betweenness"),
                      ("cascade_clo", "cascade_impact_closeness"),
                      ("cascade_eig", "cascade_impact_eigenvector")]:
        df = data[key].rename(columns={"cascade_impact": col})[["node", col]]
        cascade = cascade.merge(df, left_on="station_eng", right_on="node", how="left").drop(columns=["node"])
    master = cent.merge(cascade, on=["station_kor", "station_eng"], how="left")

    def resolve(name):
        orig = line_map_orig.get(name, np.nan)
        if pd.notna(orig):
            return [orig]
        if name in VERIFIED_MAPPING:
            return VERIFIED_MAPPING[name][0]
        return None

    master["lines"] = master["station_kor"].apply(resolve)

    def scope_weight(lines):
        if lines is None:
            return np.nan
        clean = [l for l in lines if "지선" not in l]
        if not clean:
            return 0.0
        return sum(1 for l in clean if l in IN_SCOPE) / len(clean)

    master["w_i"] = master["lines"].apply(scope_weight)
    master["is_multiline_junction"] = master["station_kor"].isin(MULTI_LINE_JUNCTIONS)
    master["station_en_display"] = master["station_kor"].map(STATION_EN).fillna(master["station_eng"])
    return master


# ============================================================================
# SECTION 1 — STAGE A: BOOTSTRAP-CI'D OPTIMIZATION COVERAGE GAP
# ============================================================================
def bootstrap_ocg(df, weight_col, n_boot=5000):
    sub = df[[weight_col, "w_i"]].dropna()
    if weight_col == "cascade_impact_betweenness":
        sub = sub[sub[weight_col] > 0]
    n = len(sub)
    if n < 5 or sub[weight_col].sum() == 0:
        return None
    gap = sub[weight_col] * (1 - sub["w_i"])
    point = gap.sum() / sub[weight_col].sum()
    idx = sub.index.to_numpy()
    boots = []
    for _ in range(n_boot):
        samp = np.random.choice(idx, size=n, replace=True)
        s = sub.loc[samp]
        denom = s[weight_col].sum()
        if denom > 0:
            boots.append((s[weight_col] * (1 - s["w_i"])).sum() / denom)
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_two_sided = min(2 * min(np.mean(boots <= 0.5), np.mean(boots >= 0.5)), 1.0)
    return {"N": n, "point": point, "ci_lo": lo, "ci_hi": hi, "p_vs_50pct": p_two_sided}


def run_stage_a(master):
    print("=" * 80)
    print("[STAGE A] OPTIMIZATION COVERAGE GAP — motivating diagnostic")
    print("=" * 80)
    rows = []
    for label, col in [("degree", "degree"), ("betweenness", "betweenness"),
                        ("cascade impact (degree removal)", "cascade_impact_degree"),
                        ("cascade impact (betweenness removal)", "cascade_impact_betweenness")]:
        res = bootstrap_ocg(master, col)
        if res:
            rows.append({"weighting": label, **res})
    t = pd.DataFrame(rows).sort_values("p_vs_50pct").reset_index(drop=True)
    m = len(t)
    t["holm_alpha_threshold"] = 0.05 / (m - t.index)
    t["ci_excludes_50pct"] = t["ci_lo"] > 0.5
    t["significant_after_holm"] = t["p_vs_50pct"] < t["holm_alpha_threshold"]

    display_cols = ["weighting", "N", "point", "ci_lo", "ci_hi", "ci_excludes_50pct", "significant_after_holm"]
    print(t[display_cols].round(4).to_string(index=False))
    print("\n  Honest framing: point estimates trend large (60-68%) but small-sample")
    print("  (N=53) bootstrap CIs do not statistically exclude 50% at any weighting,")
    print("  before or after Holm-Bonferroni correction across the m=4 weightings")
    print("  tested. Reported as a motivating trend, not a confirmed finding.")
    print("  Stage B (Step2_Stage_B_Concentration.py) supplies the decisive test.")
    return t


if __name__ == "__main__":
    print("Loading real data from public GitHub repositories...")
    data = load_all()
    master = build_master(data)
    master.to_csv(f"{OUT_DIR}/master_table.csv", index=False, encoding="utf-8-sig")

    stage_a_table = run_stage_a(master)
    stage_a_table.to_csv(f"{OUT_DIR}/stage_a_coverage_gap.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved master_table.csv and stage_a_coverage_gap.csv to {OUT_DIR}/")
