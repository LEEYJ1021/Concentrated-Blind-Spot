# -*- coding: utf-8 -*-
"""
================================================================================
STEP 6 — CROSS-STAGE CONVERGENCE TEST
--------------------------------------------------------------------------------
Concentrated Blind Spots: Diagnosing and Prioritizing Structural Coverage Gaps
in Corridor-Based Rail Decarbonization Networks

Compares the top-5 out-of-scope stations from Stage B (Step2, betweenness-
based concentration ranking) and Stage C (Step3, prescriptive greedy
selection). The observed overlap (out of 5) is benchmarked against a
permutation null in which two independent five-station sets are drawn without
replacement from the 34-station out-of-scope candidate pool, 20,000 draws; the
empirical p-value is the share of null draws whose overlap is at least as
large as observed.

This result is read jointly with Step5's feature-independence audit rather
than in isolation: a meaningful share of any observed overlap may trace to
shared topological origin between collinear Stage C inputs and the Stage B
metric, rather than to fully independent triangulation. See README.md,
"Headline findings," item 6.

Outputs
-------
  outputs/stage6_convergence_test.csv          observed overlap, p-value, station lists
  outputs/stage6_convergence_null_distribution.csv   full null distribution (for Fig. 8)
  outputs/stage6_rank_comparison.csv           side-by-side Stage B / Stage C ranks (for Fig. 9)
================================================================================
"""

import os

import numpy as np
import pandas as pd

from Step1_Stage_A_Coverage_Gap import load_all, build_master, OUT_DIR
from Step2_Stage_B_Concentration import run_stage_b
from Step3_Stage_C_Prescriptive_Selection import run_stage_c

N_PERM = 20000
TOP_K = 5


# ============================================================================
# SECTION 1 — PERMUTATION TEST FOR CHANCE OVERLAP
# ============================================================================
def overlap_permutation_pvalue(set_a, set_b, candidate_pool, k=TOP_K, n_perm=N_PERM):
    """Null: two independent k-station sets drawn without replacement from the
    same candidate pool. Returns (observed overlap, null distribution, p-value)."""
    observed = len(set_a & set_b)
    pool = list(candidate_pool)
    n = len(pool)
    rng = np.random.default_rng(42)
    null_overlaps = np.empty(n_perm, dtype=int)
    for i in range(n_perm):
        draw_a = set(rng.choice(pool, size=k, replace=False))
        draw_b = set(rng.choice(pool, size=k, replace=False))
        null_overlaps[i] = len(draw_a & draw_b)
    p_val = np.mean(null_overlaps >= observed)
    return observed, null_overlaps, p_val


def run_convergence_test(master, data):
    print("=" * 80)
    print("[CONVERGENCE TEST] Stage B (concentration) vs. Stage C (prescriptive)")
    print("=" * 80)

    conc_table, ranking_table, _ = run_stage_b(master)
    trace_df, cum_table, _, _, value_map = run_stage_c(master, data)

    name_map_en = master.set_index("station_eng")["station_en_display"]

    stage_b_top5 = set(
        ranking_table[ranking_table["metric"] == "betweenness"]
        .sort_values("gap_contribution", ascending=False)
        .head(TOP_K)["station_en_display"]
    )
    stage_c_top5 = set(trace_df["station"])

    candidate_pool = set(master.loc[master["w_i"] == 0, "station_en_display"])

    observed, null_dist, p_val = overlap_permutation_pvalue(stage_b_top5, stage_c_top5, candidate_pool)

    print(f"\n  Stage B top-5 (concentration ranking, betweenness): {sorted(stage_b_top5)}")
    print(f"  Stage C top-5 (prescriptive selection):                {sorted(stage_c_top5)}")
    print(f"\n  Observed overlap: {observed}/{TOP_K}")
    print(f"  Permutation null ({N_PERM:,} draws of two independent {TOP_K}-station "
          f"sets from a {len(candidate_pool)}-station pool): p = {p_val:.4f}")

    if p_val < 0.01:
        verdict = ("Overlap is far larger than a chance draw would produce — but see "
                   "Step5's feature-independence audit before reading this as fully "
                   "independent triangulation.")
    elif p_val < 0.05:
        verdict = "Overlap exceeds chance at conventional significance."
    else:
        verdict = "Overlap is not statistically distinguishable from chance at this sample size."
    print(f"  {verdict}")

    result_row = pd.DataFrame([{
        "stage_b_top5": ", ".join(sorted(stage_b_top5)),
        "stage_c_top5": ", ".join(sorted(stage_c_top5)),
        "observed_overlap": observed,
        "candidate_pool_size": len(candidate_pool),
        "n_permutations": N_PERM,
        "p_value": p_val,
    }])

    null_df = pd.DataFrame({"null_overlap": null_dist})

    # Side-by-side rank comparison table for the shared stations plus any Stage-C-only picks
    b_rank = (ranking_table[ranking_table["metric"] == "betweenness"]
              .sort_values("gap_contribution", ascending=False)
              .reset_index(drop=True))
    b_rank["stage_b_rank"] = b_rank.index + 1
    b_rank_map = dict(zip(b_rank["station_en_display"], b_rank["stage_b_rank"]))

    c_rank_map = dict(zip(trace_df["station"], trace_df["rank"]))

    all_stations = sorted(stage_b_top5 | stage_c_top5,
                          key=lambda s: c_rank_map.get(s, b_rank_map.get(s, 99)))
    rank_rows = [{
        "station": s,
        "stage_b_rank": b_rank_map.get(s, None),
        "stage_c_rank": c_rank_map.get(s, None),
    } for s in all_stations]
    rank_table = pd.DataFrame(rank_rows)

    print("\n  Side-by-side rank comparison:")
    print(rank_table.to_string(index=False))

    return result_row, null_df, rank_table


if __name__ == "__main__":
    master_path = f"{OUT_DIR}/master_table.csv"
    if os.path.exists(master_path):
        master = pd.read_csv(master_path)
        master["lines"] = master["lines"].apply(eval)
        data = load_all()
    else:
        print("master_table.csv not found — reloading from source (run Step1 first for speed).")
        data = load_all()
        master = build_master(data)

    result_row, null_df, rank_table = run_convergence_test(master, data)

    result_row.to_csv(f"{OUT_DIR}/stage6_convergence_test.csv", index=False, encoding="utf-8-sig")
    null_df.to_csv(f"{OUT_DIR}/stage6_convergence_null_distribution.csv", index=False, encoding="utf-8-sig")
    rank_table.to_csv(f"{OUT_DIR}/stage6_rank_comparison.csv", index=False, encoding="utf-8-sig")

    print(f"\nSaved stage6_convergence_test.csv, stage6_convergence_null_distribution.csv, "
          f"and stage6_rank_comparison.csv to {OUT_DIR}/")
