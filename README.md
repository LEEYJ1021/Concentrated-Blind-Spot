# Concentrated Blind Spots: Diagnosing and Prioritizing Structural Coverage Gaps in Corridor-Based Rail Decarbonization Networks

**A de-circularized diagnostic-to-prescriptive framework for the Korean national freight rail network**

> **One-line summary.** When a rail decarbonization program is planned around a handful of designated corridors, the majority of stations are left formally out of scope. Is the structural risk sitting outside that scope diffuse, or is it concentrated on a short, nameable, policy-actionable list of stations — and does an independently constructed, budget-constrained prescriptive procedure recover the same list? We answer both questions with real Korean national freight rail network data (53 stations, four designated corridors), a permutation-benchmarked concentration analysis, a synthetic-network robustness check, and a measured — not assumed — feature-independence audit between the diagnostic and prescriptive stages.

---

## Table of contents

- [Motivation](#motivation)
- [Headline findings](#headline-findings)
- [Figures](#figures)
- [Repository structure](#repository-structure)
- [Methods overview](#methods-overview)
- [Data & reproducibility](#data--reproducibility)
- [Limitations](#limitations)
- [Future research](#future-research)

---

## Motivation

National freight-rail decarbonization programs are rarely designed for an entire network at once. Budget, monitoring capacity, and political attention are directed toward a subset of high-throughput corridors, and optimization models are fitted, validated, and reported against that designated scope. This leaves a structural coverage gap between where optimization effort is spent and where structural risk actually resides — but detecting that gap is only half the problem. A rail operator or ministry planner does not need another descriptive statistic; they need a short, defensible, budget-constrained action list, together with evidence that the list is not simply restating the diagnosis by construction.

This repository develops and empirically validates a two-stage decision-support pipeline for the Korean national freight rail network:

- a **diagnostic stage** that tests whether out-of-scope structural risk is concentrated or diffuse, benchmarked against an explicit permutation null and, critically, stress-tested against an ensemble of synthetic network realizations rather than resting on the single empirical topology alone;
- a **prescriptive stage** that recovers a budget-constrained priority set from a value function built from centrality features distinct from the diagnostic metric, whose actual shared variance with that metric is measured rather than assumed.

The result is a single, coherent narrative — concentration, then prescription, then a convergence test read jointly with a quantitative independence audit — rather than a loosely stitched sequence of independent analyses.

---

## Headline findings

1. **The coverage gap is large in point estimate, but statistically inconclusive on its own at this sample size (N = 53).** Optimization Coverage Gap point estimates run 59.6–67.5% across four centrality weightings, but the 95% bootstrap CIs do not exclude 50% after Holm–Bonferroni correction (m = 4). This is treated as a motivating diagnostic, not a confirmed finding, and is superseded by the sharper test below. *(Fig. 3)*

2. **Out-of-scope structural risk is not diffuse — it is sharply concentrated.** Among the 34 out-of-scope stations, betweenness-based concentration (Gini = 0.789) and cascade-impact-based concentration (Gini = 0.667) both exceed a 5,000-draw uniform-allocation permutation null at p < 0.001. Degree-based concentration (Gini = 0.429) is statistically indistinguishable from random allocation (p = 0.872) — the choice of weighting materially changes the policy conclusion. *(Fig. 4, Fig. 5)*

3. **The concentration finding is not an artifact of the single empirical topology.** Across an eight-realization ensemble of degree-preserving synthetic rewirings of the network, the betweenness-weighted cascade-impact Gini coefficient remains statistically indistinguishable in magnitude (mean = 0.673, SD = 0.032) and remains significant against a realization-specific permutation null in every one of the eight realizations (8/8, p < 0.001). Station-level identity is also substantially preserved: the mean overlap between each synthetic realization's top-5 stations and the empirical topology's top-5 is 3.75 out of 5.

4. **A budget-constrained stochastic submodular optimizer recovers a stable, deployable five-station priority set.** Using a value function built from closeness, eigenvector, demand, and utilization centrality — features that are not the diagnostic metric itself — greedy selection identifies **Obong, Goedong, Busan New Port, Ssangryong, and Gwangyang** as the top-5 priority stations under a budget of K = 5, capturing 79.4% of total expected out-of-scope value and reproducing the identical top-5 set in 100% of 200 draws from the scenario-probability simplex. *(Fig. 6, Fig. 7)*

5. **Feature independence between the two stages is measured, not assumed.** Closeness and eigenvector centrality — two of the four prescriptive-stage inputs — share modest variance with betweenness (r² = 0.27 and 0.38, respectively), supporting a genuine independent-evidence interpretation. Demand-based and utilization-based centrality, the other two inputs, are strongly collinear with betweenness (r² = 0.77 and 0.84). This reframes the paper's central circularity safeguard from a binary claim ("the features are distinct") into a graded, falsifiable one.

6. **The diagnostic and prescriptive rankings converge at a level unlikely to arise by chance — but the convergence is honestly reported as partial, not fully independent.** The prescriptive top-5 shares four of its five stations with the diagnostic top-5 concentration ranking (Obong, Goedong, and Busan New Port recur across both; Ssangryong and Gwangyang are recovered by the prescriptive stage alone), an overlap benchmarked against a 20,000-draw permutation null at p = 0.001. Read jointly with finding 5, roughly half of this agreement traces to genuinely distinct evidence (closeness, eigenvector) and roughly half to shared topological origin with the diagnostic metric (demand, utilization). *(Fig. 8, Fig. 9)*

---

## Figures

| | | |
|---|---|---|
| ![Fig 1](figures/main/fig1_framework.png) | ![Fig 2](figures/main/fig2_data_pipeline.png) | ![Fig 3](figures/main/fig3_ocg_forest.png) |
| **Fig. 1.** Overall pipeline — network, scope definition, out-of-scope candidate pool, Stage A→B/C→convergence flow. | **Fig. 2.** Data pipeline — raw sources, preprocessing, and the three analysis-ready datasets feeding Stages A–C. | **Fig. 3.** Optimization Coverage Gap forest plot — bootstrap and Holm–Bonferroni-adjusted CIs against the 50% reference line. |
| ![Fig 4](figures/main/fig4_lorenz.png) | ![Fig 5](figures/main/fig5_permutation_null.png) | ![Fig 6](figures/main/fig6_combined_map.png) |
| **Fig. 4.** Lorenz curves of the out-of-scope structural gap by weighting, with Gini coefficients. | **Fig. 5.** Observed Gini coefficients against their permutation-null distributions (degree, betweenness, cascade impact). | **Fig. 6.** Combined map — concentration diagnosis (left) and stochastic greedy Top-K=5 selection (right) over real geography. |
| ![Fig 7](figures/main/fig7_cumulative_value.png) | ![Fig 8](figures/main/fig8_convergence_null.png) | ![Fig 9](figures/main/fig9_convergence_grid.png) |
| **Fig. 7.** Cumulative expected value captured by the greedy selection, K = 1…5, against the (1 − 1/e) worst-case guarantee. | **Fig. 8.** Diagnostic–prescriptive top-5 overlap against a 20,000-draw permutation null (observed overlap = 4/5, p = 0.001). | **Fig. 9.** Side-by-side rank comparison of the diagnostic (Stage B) and prescriptive (Stage C) top-5 station lists. |

---

## Repository structure

```
rail-coverage-gap/
├── README.md
├── LICENSE
├── codes/
│   ├── Step1_Stage_A_Coverage_Gap.py
│   ├── Step2_Stage_B_Concentration.py
│   ├── Step3_Stage_C_Prescriptive_Selection.py
│   ├── Step4_Synthetic_Rewiring_Robustness.py
│   ├── Step5_Feature_Independence_Audit.py
│   └── Step6_Convergence_Test.py
├── figures/
│   └── main/               # Fig. 1–9 (paper-ready, 300 dpi)
└── docs/
    ├── methods_overview.md
    └── limitations_and_methods_supplement.md
```

> **Data and analysis code.** The two source repositories (station-level network topology and cascade/centrality metrics) and the analysis notebook are maintained separately and referenced with pinned commit SHAs in the data-provenance manifest (see [Data & reproducibility](#data--reproducibility)) so results are exactly reproducible even if the source repositories change.

---

## Methods overview

See [`docs/methods_overview.md`](docs/methods_overview.md) for the full pipeline. In brief:

1. **Stage A — Optimization Coverage Gap (motivating diagnostic).** Bootstrap-CI'd share of network importance (degree, betweenness, cascade impact under degree- and betweenness-based removal) sitting outside the four designated corridors, with Holm–Bonferroni correction across the four weightings tested (m = 4).
2. **Stage B — Concentration of the coverage gap (decisive diagnostic).** Gini coefficients and Lorenz curves of the Stage A gap computed among out-of-scope stations only, benchmarked against a symmetric-Dirichlet permutation null (5,000 draws).
3. **Synthetic-network robustness check.** Betweenness-weighted cascade impact recomputed on an eight-realization ensemble of degree-preserving double-edge-swap rewirings of the empirical topology, each benchmarked against its own realization-specific permutation null (2,000 draws), to test whether the concentration finding reflects the degree distribution rather than the specific empirical edge list.
4. **Stage C — Prescriptive stochastic submodular selection.** Budget-constrained greedy selection (K = 5) of out-of-scope stations to maximize expected captured value, with pairwise value discounted for topological overlap, providing a (1 − 1/e) approximation guarantee. The value function is built from closeness, eigenvector, demand, and utilization centrality — deliberately distinct from the Stage B metric — and stability-tested across 200 draws from the scenario-probability simplex.
5. **Feature-independence audit.** Pearson and Spearman correlations, across all 53 stations, between betweenness centrality (the Stage B metric) and each of the four Stage C input features, reported as r² (shared variance) rather than asserted as independence.
6. **Cross-stage convergence test.** The top-5 out-of-scope stations from Stage B and Stage C compared, with the observed overlap benchmarked against a permutation null in which two independent five-station sets are drawn without replacement from the 34-station candidate pool (20,000 draws), read jointly with the feature-independence audit rather than in isolation.

---

## Data & Reproducibility

* **Data sources.** All analyses are based exclusively on publicly available data retrieved from the following GitHub repositories:
  * **Concentrated-Blind-Spot** (analysis code and synthetic-network ensemble)
    https://github.com/LEEYJ1021/Concentrated-Blind-Spot
  * **korea-freight-rail-resilience-analysis** (underlying network topology and centrality/cascade-impact data)
    https://github.com/LEEYJ1021/korea-freight-rail-resilience-analysis
* **Data provenance.** For every downloaded file, the analysis automatically records the repository URL, retrieval timestamp (UTC), exact Git commit SHA, and file size. These provenance records enable complete traceability and should be reported in the manuscript's Data Availability statement.
* **Version-controlled analysis.** The analytical pipeline is designed to operate on version-pinned datasets whenever possible. Consequently, future updates to the source repositories do not affect the reproducibility of the reported results.
* **Randomization and reproducibility.** Random seeds are fixed throughout the analysis (`numpy.random.seed(42)` for the primary pipeline). The synthetic-network robustness ensemble uses eight independently fixed seeds (42, 137, 256, 512, 1024, 2048, 4096, 8192), each deterministically reproducible from the same repository. All stochastic procedures — bootstrap confidence intervals, permutation tests, and scenario-probability sensitivity sweeps — report both the number of iterations (5,000 / 2,000 / 20,000 / 200, as appropriate) and the resulting empirical distributions rather than only point estimates.
* **Statistical safeguards.** Holm–Bonferroni correction for multiple comparisons (Stage A), permutation-based null construction directly from the observed data (Stage B), degree-preserving synthetic rewiring for topology-level robustness, a value function deliberately built from non-Stage-B features (Stage C), a measured feature-independence audit rather than a qualitative one (Stage C ↔ Stage B), and a permutation-benchmarked chance-overlap null for the convergence claim (Stage B ↔ Stage C).
* **Computational environment.** The analysis records the Python version and major package versions used, to facilitate independent reproduction.

---

## Limitations

We would rather a reader find these here than have a reviewer find them first:

- **Out-of-scope N = 34 is small.** This is the direct reason Stage A's confidence intervals are wide and the Stage B–C convergence test has limited statistical power to detect anything short of near-total overlap.
- **This is a single-case design in the analytic-generalization sense.** Its claims should be read as generalizations about the mechanism linking policy-defined scope boundaries to network-structural risk, not as statistical generalizations to a population of national rail networks. The synthetic-rewiring robustness check narrows, but does not eliminate, this limitation: a rewiring ensemble is still drawn from the same network's own degree distribution and cannot substitute for genuinely independent replication on a different country's rail network.
- **The Stage C "de-circularization" safeguard is necessary but not sufficient on its own.** Building the prescriptive value function from features that are not literally the Stage B metric is a necessary starting point, but the feature-independence audit shows that not all nominally distinct features are equally independent — two of the four Stage C inputs (demand, utilization) turn out to be strongly collinear with betweenness (r² = 0.77–0.85).
- **The reported 4/5 convergence overlap should not be read as "two fully independent methods agree."** Roughly half of that agreement is attributable to shared topological origin rather than independent triangulation; see finding 6 above.
- **Degree-based concentration is not statistically distinguishable from random allocation.** Whichever centrality weighting is used to justify scope-expansion decisions materially changes the conclusion, and this sensitivity should be stated explicitly in any operational use of this framework.

Full text: [`docs/limitations_and_methods_supplement.md`](docs/limitations_and_methods_supplement.md).

---

## Future research

This repository deliberately restricts its scope to the diagnostic-prescriptive pairing that the data support at conventional significance. Several related analyses were developed on the same underlying dataset but are intentionally kept out of the central claim above, and are noted here as directions for follow-on work rather than folded into the present pipeline:

- **Re-estimating the Stage C value function using only the genuinely independent inputs.** A natural extension is to rebuild the prescriptive value function using closeness and eigenvector centrality alone — the two features shown here to be only moderately correlated with betweenness — and test whether the resulting priority set still recovers a comparable share of expected out-of-scope value and a comparable overlap with the Stage B ranking. If the overlap holds, the convergence claim would rest on an unambiguously independent value function; if it drops substantially, that is itself informative about how much of the current convergence is genuinely non-circular.
- **Replication on a second national rail network.** The single-network limitation discussed above is addressed here through synthetic rewiring, not independent replication. Applying the same diagnostic-prescriptive pipeline to a structurally comparable freight network in another country would be the most direct way to strengthen external validity in the statistical-generalization sense.
- **A predictive transfer layer with exact Shapley attribution.** A companion analysis (not part of this repository's central claim) develops a graph-propagated transfer model that extrapolates in-scope resilience-value labels to out-of-scope stations, with exact closed-form SHAP attribution rather than sampled attribution, made possible by a deliberately linear model specification.
- **Dynamic spillover-burden reallocation under alternative carbon-policy scenarios.** A related line of analysis examines how out-of-scope structural burden shifts as decarbonization policy intensifies across multiple carbon-pricing scenarios, using the same underlying network and centrality data.
- **An audited large-language-model narration layer for non-technical stakeholders.** A separate exploratory component pairs an LLM-based narrative explainer with a deterministic, code-based verification layer, validated with an adversarial positive-control stress test, to translate the technical diagnostic and prescriptive outputs above into stakeholder-facing language without sacrificing numerical fidelity.

These four directions are reported separately because each answers a distinct methodological question from the one addressed in this repository's central diagnostic-to-prescriptive pipeline.
