# -*- coding: utf-8 -*-
"""
================================================================================
STEP 5 — QUANTIFYING FEATURE INDEPENDENCE BETWEEN STAGE B AND STAGE C
--------------------------------------------------------------------------------
Concentrated Blind Spots: Diagnosing and Prioritizing Structural Coverage Gaps
in Corridor-Based Rail Decarbonization Networks

Stage C's value function (Step3) is built from features that are not
literally the Stage B metric (betweenness / cascade impact) — but "not the
same named feature" is a necessary, not sufficient, independence criterion.
This script computes the actual Pearson and Spearman correlations, across all
53 stations, between betweenness centrality (the Stage B metric) and each of
the four Stage C input features (closeness, eigenvector, demand, utilization
centrality), reporting r-squared as the share of variance in each Stage C
input that is linearly explained by betweenness alone.

This reframes the paper's central circularity safeguard from a binary claim
("the features are distinct") into a graded, falsifiable one: a Stage C input
with low shared variance contributes genuinely independent evidence to the
Step6 convergence test, while a Stage C input with high shared variance
functions, in practice, as a near-duplicate of the Stage B metric.

Outputs
-------
  outputs/stage5_feature_independence.csv    Pearson r, Spearman rho, r^2, interpretation
================================================================================
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from Step1_Stage_A_Coverage_Gap import load_all, build_master, OUT_DIR

STAGE_B_METRIC = "betweenness"
STAGE_C_INPUTS = ["closeness", "eigenvector", "demand_centrality", "utilization_centrality"]

# Interpretation thresholds for r^2 (shared variance), stated explicitly so the
# labeling rule is auditable rather than an implicit judgment call.
MODERATE_THRESHOLD = 0.50  # r^2 below this -> "moderately distinct"; at/above -> "strongly collinear"


def run_feature_independence_audit(master):
    print("=" * 80)
    print("[FEATURE INDEPENDENCE AUDIT] Stage B metric vs. each Stage C input feature")
    print("=" * 80)

    sub = master[[STAGE_B_METRIC] + STAGE_C_INPUTS].dropna()
    print(f"  N = {len(sub)} stations with complete centrality data.")

    rows = []
    for feature in STAGE_C_INPUTS:
        pearson_r, pearson_p = stats.pearsonr(sub[STAGE_B_METRIC], sub[feature])
        spearman_rho, spearman_p = stats.spearmanr(sub[STAGE_B_METRIC], sub[feature])
        r_squared = pearson_r ** 2
        interpretation = "Strongly collinear" if r_squared >= MODERATE_THRESHOLD else "Moderately distinct"
        rows.append({
            "stage_c_input_feature": feature,
            "pearson_r_with_betweenness": round(pearson_r, 3),
            "pearson_p": pearson_p,
            "spearman_rho": round(spearman_rho, 3),
            "spearman_p": spearman_p,
            "r_squared_shared_variance": round(r_squared, 3),
            "interpretation": interpretation,
        })

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))

    distinct = table[table["interpretation"] == "Moderately distinct"]["stage_c_input_feature"].tolist()
    collinear = table[table["interpretation"] == "Strongly collinear"]["stage_c_input_feature"].tolist()

    print(f"\n  Moderately distinct from betweenness (genuinely independent evidence): {distinct}")
    print(f"  Strongly collinear with betweenness (near-duplicate of the Stage B metric): {collinear}")
    print("\n  CONSEQUENCE FOR STAGE C: the de-circularization safeguard in Step3 — building")
    print("  the value function from features not literally equal to the Stage B metric —")
    print("  is a necessary but not sufficient condition. Only the features listed as")
    print("  'Moderately distinct' above contribute genuinely independent evidence to the")
    print("  Step6 convergence test; features listed as 'Strongly collinear' restate a")
    print("  large share of the Stage B signal under a different name.")

    return table


if __name__ == "__main__":
    master_path = f"{OUT_DIR}/master_table.csv"
    if os.path.exists(master_path):
        master = pd.read_csv(master_path)
    else:
        print("master_table.csv not found — reloading from source (run Step1 first for speed).")
        data = load_all()
        master = build_master(data)

    table = run_feature_independence_audit(master)
    table.to_csv(f"{OUT_DIR}/stage5_feature_independence.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved stage5_feature_independence.csv to {OUT_DIR}/")
