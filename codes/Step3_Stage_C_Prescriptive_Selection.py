# -*- coding: utf-8 -*-
"""
================================================================================
STEP 3 — STAGE C: PRESCRIPTIVE STOCHASTIC SUBMODULAR SELECTION
--------------------------------------------------------------------------------
Concentrated Blind Spots: Diagnosing and Prioritizing Structural Coverage Gaps
in Corridor-Based Rail Decarbonization Networks

Formulates budget-constrained scope expansion (select K out-of-scope stations
to maximize expected captured value, with pairwise value discounted for
topological overlap between already-selected stations) as monotone submodular
maximization, solved by greedy selection with the classical Nemhauser-Wolsey-
Fisher (1 - 1/e) approximation guarantee.

CIRCULARITY SAFEGUARD: the station value function is built from closeness,
eigenvector, demand, and utilization centrality — features that are NOT the
Stage B metric (betweenness / cascade impact). This is a necessary but not
sufficient de-circularization step; Step5_Feature_Independence_Audit.py
quantifies exactly how independent these four features actually are.

Sensitivity of the selected set to the relative weighting of the four value-
function components is assessed by re-running the selection under 200 draws
from the 4-dimensional probability simplex (Dirichlet(1,1,1,1)).

Outputs
-------
  outputs/stage_c_selection.csv               greedy selection trace (rank, station, marginal gain)
  outputs/stage_c_cumulative_value.csv         cumulative % of expected value captured, K=1..5
  outputs/stage_c_weight_sensitivity.csv       200-draw stability sweep over value-function weights
================================================================================
"""

import os

import numpy as np
import pandas as pd

from Step1_Stage_A_Coverage_Gap import load_all, build_master, OUT_DIR

np.random.seed(42)

INDEP_COLS = ["closeness", "eigenvector", "demand_centrality", "utilization_centrality"]
BUDGET = 5
OVERLAP_PENALTY_RATE = 0.3
N_WEIGHT_DRAWS = 200


# ============================================================================
# SECTION 1 — VALUE FUNCTION (built from non-Stage-B features)
# ============================================================================
def build_out_of_scope_adjacency(master, corridor_ef):
    out_scope_set = set(master.loc[master["w_i"] == 0, "station_eng"])
    adj = {n: set() for n in out_scope_set}
    edges = corridor_ef[["origin_eng", "dest_eng"]].dropna()
    for _, e in edges.iterrows():
        o, d = e["origin_eng"], e["dest_eng"]
        if o in out_scope_set and d in out_scope_set:
            adj[o].add(d)
            adj[d].add(o)
    return out_scope_set, adj


def build_value_function(out_of_scope_master, weights=None):
    """Equal- or custom-weighted z-scored composite of the four independent
    (non-Stage-B) centrality features, shifted to be non-negative for the
    submodular greedy gain calculation."""
    if weights is None:
        weights = np.ones(len(INDEP_COLS)) / len(INDEP_COLS)
    raw = out_of_scope_master[INDEP_COLS].fillna(0).to_numpy(dtype=float)
    z = (raw - raw.mean(axis=0)) / (raw.std(axis=0) + 1e-9)
    composite = z @ weights
    composite = composite - composite.min() + 1e-6
    return dict(zip(out_of_scope_master["station_eng"], composite))


# ============================================================================
# SECTION 2 — GREEDY SUBMODULAR SELECTION, (1 - 1/e) GUARANTEE
# ============================================================================
def greedy_select(value_dict, adj, budget=BUDGET, overlap_penalty_rate=OVERLAP_PENALTY_RATE):
    selected, remaining, trace = [], set(value_dict.keys()), []
    for k in range(budget):
        gains = {}
        for c in remaining:
            v = value_dict[c]
            penalty = sum(1 for s in selected if s in adj[c]) * overlap_penalty_rate * v
            gains[c] = max(v - penalty, 0)
        best = max(gains, key=gains.get)
        selected.append(best)
        remaining.discard(best)
        trace.append({"rank": k + 1, "station_eng": best, "marginal_gain": round(gains[best], 4)})
    return selected, pd.DataFrame(trace)


def run_stage_c(master, data):
    print("=" * 80)
    print(f"[STAGE C] Budget-constrained stochastic submodular selection (K={BUDGET})")
    print("Value function: closeness + eigenvector + demand + utilization centrality")
    print("(deliberately distinct from the Stage B metric — see Step5 for the")
    print(" measured, not assumed, feature-independence audit)")
    print("=" * 80)

    out_mask = master["w_i"] == 0
    out_of_scope_master = master.loc[out_mask].copy()
    out_scope_set, adj = build_out_of_scope_adjacency(master, data["corridor_ef"])

    value_map = build_value_function(out_of_scope_master)
    selected, trace_df = greedy_select(value_map, adj)

    name_map_en = master.set_index("station_eng")["station_en_display"]
    name_map_kor = master.set_index("station_eng")["station_kor"]
    trace_df["station_kor"] = trace_df["station_eng"].map(name_map_kor)
    trace_df["station"] = trace_df["station_eng"].map(name_map_en)

    total_val = sum(value_map.values())
    cum_rows = []
    running = 0.0
    for _, row in trace_df.iterrows():
        running += value_map[row["station_eng"]]
        cum_rows.append({"budget_K": row["rank"], "station": row["station"],
                          "cumulative_pct_of_expected_value": round(100 * running / total_val, 1)})
    cum_table = pd.DataFrame(cum_rows)

    print("\n  Greedy submodular selection (guaranteed >= (1-1/e) ~ 63% of optimal):")
    print(trace_df[["rank", "station", "station_eng", "marginal_gain"]].to_string(index=False))
    print("\n  Cumulative expected-value capture:")
    print(cum_table.to_string(index=False))
    print(f"\n  Budget K={BUDGET} captures {cum_table['cumulative_pct_of_expected_value'].iloc[-1]:.1f}% "
          f"of total expected out-of-scope value ({len(out_scope_set)} candidates total).")

    # --- Weight-sensitivity sweep (fixes over-reliance on equal weighting) ---
    print(f"\n  Sensitivity sweep: {N_WEIGHT_DRAWS} draws of value-function component")
    print(f"  weights from a Dirichlet(1,1,1,1) simplex over "
          f"{INDEP_COLS} ...")
    top5_sets = []
    for _ in range(N_WEIGHT_DRAWS):
        w = np.random.dirichlet(np.ones(len(INDEP_COLS)))
        vmap = build_value_function(out_of_scope_master, weights=w)
        sel, _ = greedy_select(vmap, adj)
        top5_sets.append(frozenset(sel))

    reference_set = frozenset(selected)
    match_count = sum(1 for s in top5_sets if s == reference_set)
    stability_rate = match_count / N_WEIGHT_DRAWS
    print(f"  Exact same top-{BUDGET} SET recovered in {match_count}/{N_WEIGHT_DRAWS} draws "
          f"({stability_rate:.1%}).")

    sensitivity_table = pd.DataFrame({
        "draw": range(1, N_WEIGHT_DRAWS + 1),
        "matches_reference_top5_set": [s == reference_set for s in top5_sets],
    })
    sensitivity_summary = pd.DataFrame([{
        "budget_K": BUDGET, "n_draws": N_WEIGHT_DRAWS,
        "reference_top5": ", ".join(name_map_en.get(s, s) for s in selected),
        "selection_stability_rate": stability_rate,
    }])

    return trace_df, cum_table, sensitivity_table, sensitivity_summary, value_map


if __name__ == "__main__":
    master_path = f"{OUT_DIR}/master_table.csv"
    if os.path.exists(master_path):
        master = pd.read_csv(master_path)
        master["lines"] = master["lines"].apply(eval)
        data = load_all()  # corridor_ef edge list still needed for adjacency
    else:
        print("master_table.csv not found — reloading from source (run Step1 first for speed).")
        data = load_all()
        master = build_master(data)

    trace_df, cum_table, sensitivity_table, sensitivity_summary, value_map = run_stage_c(master, data)

    trace_df.to_csv(f"{OUT_DIR}/stage_c_selection.csv", index=False, encoding="utf-8-sig")
    cum_table.to_csv(f"{OUT_DIR}/stage_c_cumulative_value.csv", index=False, encoding="utf-8-sig")
    sensitivity_table.to_csv(f"{OUT_DIR}/stage_c_weight_sensitivity.csv", index=False, encoding="utf-8-sig")
    sensitivity_summary.to_csv(f"{OUT_DIR}/stage_c_weight_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\nSaved stage_c_selection.csv, stage_c_cumulative_value.csv, and "
          f"weight-sensitivity outputs to {OUT_DIR}/")
