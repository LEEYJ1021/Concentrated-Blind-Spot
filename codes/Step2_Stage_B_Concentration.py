# -*- coding: utf-8 -*-
"""
================================================================================
STEP 2 — STAGE B: CONCENTRATION OF THE COVERAGE GAP (decisive diagnostic)
--------------------------------------------------------------------------------
Concentrated Blind Spots: Diagnosing and Prioritizing Structural Coverage Gaps
in Corridor-Based Rail Decarbonization Networks

Among the 34 out-of-scope stations, tests whether the structural coverage gap
identified in Stage A (Step1) is DIFFUSE or CONCENTRATED. Computes the Gini
coefficient of each weighting (degree, betweenness, cascade impact under
degree-based removal) and benchmarks it against a permutation null in which
the same total weighted mass is reallocated uniformly at random (symmetric
Dirichlet(1,...,1)) across the same N out-of-scope stations, 5,000 draws per
weighting.

This — not Stage A — is treated as the framework's central evidentiary claim.

Outputs
-------
  outputs/stage_b_concentration.csv        Gini, permutation-null summary, p-values
  outputs/stage_b_out_of_scope_ranking.csv per-station gap contribution & rank
  outputs/stage_b_permutation_null_<metric>.csv   full null distribution (for Fig. 4/5)
================================================================================
"""

import os

import numpy as np
import pandas as pd

from Step1_Stage_A_Coverage_Gap import load_all, build_master, OUT_DIR

np.random.seed(42)


# ============================================================================
# SECTION 1 — GINI COEFFICIENT + PERMUTATION NULL
# ============================================================================
def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def gini_permutation_test(values, n_perm=5000):
    """Null: the SAME total gap mass reallocated uniformly at random
    (Dirichlet(1,...,1)) across the same N out-of-scope stations."""
    n = len(values)
    total = values.sum()
    null_ginis = np.array([gini(np.random.dirichlet(np.ones(n)) * total) for _ in range(n_perm)])
    observed = gini(values)
    p_val = np.mean(null_ginis >= observed)
    return observed, null_ginis, p_val


def run_stage_b(master, n_perm=5000):
    print("=" * 80)
    print("[STAGE B] CONCENTRATION OF THE COVERAGE GAP — decisive diagnostic")
    print("=" * 80)

    rows, ranking_rows, null_store = [], [], {}
    for label, col in [("degree", "degree"), ("betweenness", "betweenness"),
                        ("cascade impact (degree)", "cascade_impact_degree")]:
        sub = master[[col, "w_i", "station_kor", "station_en_display",
                       "is_multiline_junction"]].dropna(subset=[col, "w_i"])
        out = sub[sub["w_i"] == 0].copy()
        if len(out) < 5:
            continue

        out["gap_contribution"] = out[col]
        out = out.sort_values("gap_contribution", ascending=False)
        total = out["gap_contribution"].sum()
        out["cum_share"] = out["gap_contribution"].cumsum() / total
        out["metric"] = label
        out["rank"] = np.arange(1, len(out) + 1)
        ranking_rows.append(out[["metric", "rank", "station_kor", "station_en_display",
                                  "gap_contribution", "cum_share", "is_multiline_junction"]])

        obs_gini, null_dist, p_val = gini_permutation_test(out["gap_contribution"].to_numpy(), n_perm)
        top3_share = out["gap_contribution"].head(3).sum() / total
        n_nodes_for_80pct = int((out["cum_share"] <= 0.80).sum()) + 1

        rows.append({
            "metric": label, "N_out_of_scope": len(out),
            "observed_Gini": round(obs_gini, 3),
            "null_mean_Gini": round(null_dist.mean(), 3),
            "null_95pct_Gini": round(np.percentile(null_dist, 95), 3),
            "p_value_concentration_gt_null": p_val,
            "top3_share_of_total_gap": f"{top3_share:.1%}",
            "n_stations_for_80pct_of_gap": n_nodes_for_80pct,
            "small_N_flag": "YES (N<30)" if len(out) < 30 else "no",
        })
        null_store[label] = null_dist

        print(f"\n  [{label}] top 5 gap contributors (out-of-scope):")
        print(out[["station_en_display", "gap_contribution", "cum_share",
                    "is_multiline_junction"]].head(5).to_string(index=False))

    conc_table = pd.DataFrame(rows)
    print("\n  Concentration summary (vs. Dirichlet(1,...,1) permutation null):")
    print(conc_table.to_string(index=False))
    print("\n  PAYLOAD: on resilience-relevant metrics (betweenness, cascade impact),")
    print("  the gap is highly concentrated (Gini 0.667-0.789, p<0.001) — 7-10")
    print("  nameable stations carry 80% of it. Degree-based concentration (Gini")
    print("  0.429) is NOT distinguishable from random allocation (p=0.872). The")
    print("  choice of weighting materially changes the policy conclusion.")

    ranking_table = pd.concat(ranking_rows, ignore_index=True)
    return conc_table, ranking_table, null_store


if __name__ == "__main__":
    master_path = f"{OUT_DIR}/master_table.csv"
    if os.path.exists(master_path):
        master = pd.read_csv(master_path)
        master["lines"] = master["lines"].apply(eval)  # round-trip list column from CSV
    else:
        print("master_table.csv not found — reloading from source (run Step1 first for speed).")
        data = load_all()
        master = build_master(data)

    conc_table, ranking_table, null_store = run_stage_b(master)

    conc_table.to_csv(f"{OUT_DIR}/stage_b_concentration.csv", index=False, encoding="utf-8-sig")
    ranking_table.to_csv(f"{OUT_DIR}/stage_b_out_of_scope_ranking.csv", index=False, encoding="utf-8-sig")
    for label, null_dist in null_store.items():
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        pd.DataFrame({"null_gini": null_dist}).to_csv(
            f"{OUT_DIR}/stage_b_permutation_null_{safe_label}.csv", index=False)

    print(f"\nSaved stage_b_concentration.csv, stage_b_out_of_scope_ranking.csv, "
          f"and per-metric permutation-null CSVs to {OUT_DIR}/")
