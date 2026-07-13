# Concentrated Blind Spots: Structural Coverage Gaps in Corridor-Based Rail Decarbonization Networks

**Project report — diagnostic-to-prescriptive analysis of the Korean national freight rail network**

---

## Table of contents

- [1. Overview](#1-overview)
- [2. Institutional setting](#2-institutional-setting)
- [3. Data](#3-data)
- [4. Analytical pipeline](#4-analytical-pipeline)
  - [4.1 Stage A — Optimization Coverage Gap](#41-stage-a--optimization-coverage-gap-motivating-diagnostic)
  - [4.2 Stage B — Concentration of the coverage gap](#42-stage-b--concentration-of-the-coverage-gap-decisive-diagnostic)
  - [4.3 Robustness check — synthetic degree-preserving network ensemble](#43-robustness-check--synthetic-degree-preserving-network-ensemble)
  - [4.4 Stage C — Prescriptive stochastic submodular selection](#44-stage-c--prescriptive-stochastic-submodular-selection)
  - [4.5 Feature-independence audit](#45-feature-independence-audit)
  - [4.6 Cross-stage convergence test](#46-cross-stage-convergence-test)
- [5. Results summary](#5-results-summary)
- [6. Statistical safeguards](#6-statistical-safeguards)
- [7. Figures](#7-figures)
- [8. Repository structure](#8-repository-structure)
- [9. Running the pipeline](#9-running-the-pipeline)
- [10. Data & reproducibility](#10-data--reproducibility)
- [11. Limitations](#11-limitations)
- [12. Future research](#12-future-research)

---

## 1. Overview

Corridor-based freight-rail decarbonization programs typically concentrate optimization and monitoring effort on a designated subset of high-throughput corridors, leaving the majority of stations formally out of scope. This project asks three linked questions about the real topology and demand data of the Korean national freight rail network:

1. Is the structural risk residing outside the designated scope **diffuse or concentrated**?
2. Does an independently constructed, budget-constrained **prescriptive selection procedure recover the same handful of stations** that the diagnostic analysis identifies, and how independent are the two procedures' input features **in practice rather than by assumption**?
3. Does the concentration finding **survive when tested against network realizations other than the single empirical topology**?

The pipeline is organized as a coupled diagnostic-to-prescriptive sequence rather than a set of loosely related analyses: a motivating diagnostic (Stage A), a decisive diagnostic (Stage B), a robustness check on Stage B against synthetic network realizations, a prescriptive stage (Stage C), a quantitative feature-independence audit between Stage B and Stage C, and a cross-stage convergence test. Every stage is benchmarked against an explicit statistical null rather than reported as a bare point estimate, and every safeguard is reported at the level of evidence it actually supports — including where that evidence is partial rather than complete (see [Section 6](#6-statistical-safeguards) and [Section 11](#11-limitations)).

---

## 2. Institutional setting

The empirical setting is the national freight rail network of the Republic of Korea, comprising **53 freight-handling stations**. Four corridors — **Gyeongbu, Chungbuk, Yeongdong, and Jungang** — are treated as the designated decarbonization scope, covering **19 stations** either directly or as multi-line junctions partially assigned to a designated corridor. The remaining **34 stations are formally out of scope**.

Each station *i* is assigned a scope weight w<sub>i</sub> ∈ [0, 1], equal to the share of its non-branch line assignments that fall within the designated corridors; w<sub>i</sub> = 0 for out-of-scope stations. Station-to-line assignments are cross-validated against the national rail operator's published line topology, with a manually verified mapping (`VERIFIED_MAPPING` in `Step1_Stage_A_Coverage_Gap.py`) resolving multi-line junctions and branch-line ambiguities.

---

## 3. Data

Three data components feed the pipeline:

| Component | Description | Source file(s) |
|---|---|---|
| Station attribute table | Coordinates, line assignments, hub classification for all 53 stations | `freight_stations_parsed.csv` |
| Centrality dataset | Degree, betweenness, closeness, eigenvector, demand, utilization, and composite centrality per station, computed on the full national freight topology | `centrality_data.csv` |
| Cascade-impact datasets | Exact change in network-wide centrality mass following each station's removal, under degree-, betweenness-, closeness-, and eigenvector-based removal | `cascade_*_exact.csv` |
| Corridor efficiency / edge list | Origin-destination trip volumes used to reconstruct the network's edge structure | `corridor_efficiency_summary.csv` |

All datasets are drawn from two version-controlled public repositories (see [Section 10](#10-data--reproducibility)).

---

## 4. Analytical pipeline

### 4.1 Stage A — Optimization Coverage Gap (motivating diagnostic)

*Script: [`Step1_Stage_A_Coverage_Gap.py`](codes/Step1_Stage_A_Coverage_Gap.py)*

For a centrality or cascade-impact weighting m<sub>i</sub>, the Optimization Coverage Gap (OCG) is the share of total weighted centrality mass attributable to out-of-scope stations:

```
OCG = Σᵢ [ mᵢ × (1 − wᵢ) ] / Σᵢ mᵢ
```

computed for four weightings: raw degree, raw betweenness, cascade impact under degree-based removal, and cascade impact under betweenness-based removal. A 95% confidence interval is obtained by station-level bootstrap resampling (5,000 replicates). Because four related weightings are tested against an implicit null of OCG = 50%, a Holm–Bonferroni correction (m = 4) is applied.

**Result:** point estimates trend large across all four weightings — 62.6% (degree), 59.6% (betweenness), 67.5% (cascade impact, degree removal), 66.6% (cascade impact, betweenness removal) — but at this sample size (N = 53, or N = 33 for the betweenness-restricted cascade weighting) the 95% bootstrap confidence intervals are wide and none excludes the 50% reference value after correction. This stage is deliberately treated as a **motivating trend, not a confirmed finding**, and is superseded by Stage B.

### 4.2 Stage B — Concentration of the coverage gap (decisive diagnostic)

*Script: [`Step2_Stage_B_Concentration.py`](codes/Step2_Stage_B_Concentration.py)*

Among the 34 out-of-scope stations, the Gini coefficient of each weighting is computed to test whether the coverage gap is diffuse or concentrated:

```
G = ( N + 1 − 2·Σᵢ [(N − i + 1)·x₍ᵢ₎] / Σᵢ xᵢ ) / N,   x₍1₎ ≤ x₍2₎ ≤ … ≤ x₍ₙ₎
```

A permutation null is constructed by reallocating the same total weighted mass uniformly at random across the same N out-of-scope stations via a symmetric Dirichlet(1,…,1) draw, repeated 5,000 times per weighting. The empirical p-value is the share of null-distribution Gini coefficients at or above the observed value. **This stage — not Stage A — is the pipeline's central evidentiary claim.**

| Weighting | Observed Gini | Null mean | p (concentration > null) | Top-3 share | Stations for 80% |
|---|---|---|---|---|---|
| Degree | 0.429 | 0.485 | 0.872 | 31.9% | 18 |
| Betweenness | 0.789 | 0.486 | < 0.001 | 56.1% | 7 |
| Cascade impact (degree) | 0.667 | 0.486 | < 0.001 | 67.2% | 10 |

*Table B1. Gini concentration of the coverage gap among out-of-scope stations (N = 34), benchmarked against a Dirichlet(1,…,1) permutation null (5,000 draws).*

Degree-based concentration is **not** distinguishable from a uniform-random allocation of the same total mass. Betweenness- and cascade-impact-based concentration, by contrast, both exceed the permutation null sharply: 7 stations account for 80% of the out-of-scope betweenness mass, and 10 stations account for 80% of out-of-scope cascade impact. The choice of weighting materially changes the policy conclusion.

### 4.3 Robustness check — synthetic degree-preserving network ensemble

*Script: [`Step4_Synthetic_Rewiring_Robustness.py`](codes/Step4_Synthetic_Rewiring_Robustness.py)*

A single national network is, definitionally, a sample of one topology, so the Stage B concentration finding is vulnerable to the objection that the specific empirical wiring — rather than the network's more general structural properties — is responsible for the observed result. To test this directly, an ensemble of **eight synthetic network realizations** was generated by **degree-preserving double-edge-swap rewiring** of the empirical topology (each seeded independently: 42, 137, 256, 512, 1024, 2048, 4096, 8192). Each realization preserves the exact empirical degree sequence — the same 53 nodes and the same total edge count — while randomizing which specific pairs of stations are connected. Betweenness-weighted cascade impact is recomputed on each realization, and its Gini coefficient benchmarked against a realization-specific permutation null (2,000 draws), exactly as in Stage B.

| Realization (seed) | N | Gini | Permutation-null mean | p |
|---|---|---|---|---|
| Empirical topology | 33 | 0.686 | 0.484 | 0.0002 |
| Synthetic — seed 42 | 53 | 0.692 | 0.491 | < 0.0005 |
| Synthetic — seed 137 | 53 | 0.646 | 0.490 | < 0.0005 |
| Synthetic — seed 256 | 53 | 0.699 | 0.489 | < 0.0005 |
| Synthetic — seed 512 | 53 | 0.653 | 0.491 | < 0.0005 |
| Synthetic — seed 1024 | 53 | 0.614 | 0.491 | 0.0005 |
| Synthetic — seed 2048 | 53 | 0.700 | 0.490 | < 0.0005 |
| Synthetic — seed 4096 | 53 | 0.699 | 0.490 | < 0.0005 |
| Synthetic — seed 8192 | 53 | 0.680 | 0.491 | < 0.0005 |

*Table B2. Betweenness-weighted cascade-impact Gini concentration on the empirical topology and across an eight-realization ensemble of degree-preserving synthetic rewirings.*

Across the ensemble, the Gini coefficient ranges from 0.614 to 0.700 (mean = 0.673, SD = 0.032), and **every realization's observed Gini exceeds its own realization-specific permutation null at p < 0.001**. Station-level identity is also substantially — though not perfectly — preserved: the empirical topology's top-5 stations by cascade impact are Jecheon Yard, Obong, Goedong, Busan New Port, and Donghae. Across the eight synthetic realizations, Busan New Port and Jecheon Yard appear in the top-5 in all eight realizations, Goedong and Donghae in five, and Obong in four; **the mean overlap between each synthetic realization's top-5 and the empirical topology's top-5 is 3.75 out of 5**. Concentration this sharp is therefore not an idiosyncrasy of the one empirical wiring pattern; it recurs across every randomization that preserves only the network's degree sequence, indicating the skew is a structural consequence of the degree distribution itself, not of any particular set of station-to-station connections.

### 4.4 Stage C — Prescriptive stochastic submodular selection

*Script: [`Step3_Stage_C_Prescriptive_Selection.py`](codes/Step3_Stage_C_Prescriptive_Selection.py)*

A budget-constrained scope-expansion problem is formulated as selecting K out-of-scope stations to maximize expected captured value, with pairwise value discounted for topological overlap with already-selected stations. Because this objective is monotone and submodular, greedy selection provides a guaranteed approximation ratio to the optimum:

```
f(S_greedy) ≥ (1 − 1/e) · f(S*),   e ≈ 2.71828
```

Station value is defined from a composite of **closeness, eigenvector, demand, and utilization centrality** — these four features were selected specifically because they are not the Stage B metric itself. This is a **necessary but not sufficient** de-circularization step; [Section 4.5](#45-feature-independence-audit) quantifies exactly how independent these four features actually are, rather than treating their distinct names as evidence of independence.

| Station | Greedy selection rank (K = 5) | Selection frequency (200 draws) |
|---|---|---|
| Obong | 1 | 100% |
| Goedong | 2 | 100% |
| Busan New Port | 3 | 100% |
| Ssangryong | 4 | 100% |
| Gwangyang | 5 | 100% |

*Table C1. Stage C station-selection stability under a 200-draw sensitivity sweep over the relative weighting of the four value-function components.*

Greedy submodular selection identifies **Obong, Goedong, Busan New Port, Ssangryong, and Gwangyang** as the top-five priority stations under a budget of K = 5, jointly capturing **79.4%** of total expected out-of-scope value and reproducing exactly the same top-five set in **100% of 200 sensitivity draws**, indicating the selection is not an artifact of a specific weighting assumption. Because greedy selection under a monotone submodular objective is guaranteed to capture at least (1 − 1/e) ≈ 63% of the optimal achievable value, the 79.4% figure sits close to the theoretical ceiling for a five-station budget.

### 4.5 Feature-independence audit

*Script: [`Step5_Feature_Independence_Audit.py`](codes/Step5_Feature_Independence_Audit.py)*

Rather than asserting that the Stage C value function is independent of the Stage B metric because it is built from differently named centrality measures, this audit computes the **actual Pearson and Spearman correlations**, across all 53 stations, between betweenness centrality (the Stage B metric) and each of the four Stage C inputs. The squared Pearson correlation (r²) is reported as the share of variance in each Stage C input that is linearly explained by betweenness alone.

| Stage C input feature | Pearson r with betweenness | Spearman ρ | r² (shared variance) | Interpretation |
|---|---|---|---|---|
| Closeness centrality | 0.520*** | 0.447*** | 27.0% | Moderately distinct |
| Eigenvector centrality | 0.615*** | 0.495*** | 37.8% | Moderately distinct |
| Demand centrality | 0.875*** | 0.772*** | 76.6% | Strongly collinear |
| Utilization centrality | 0.919*** | 0.879*** | 84.5% | Strongly collinear |

*Table C2. Pearson and Spearman correlation between betweenness centrality (Stage B metric) and each Stage C input feature, N = 53 stations. \*\*\* p < 0.001.*

The results are asymmetric across the four features in a way a qualitative independence claim would not have surfaced. Closeness and eigenvector centrality — reach- and influence-based measures — share well under half their variance with betweenness, supporting a genuine independent-evidence interpretation. Demand and utilization centrality, by contrast, are **strongly collinear** with betweenness (r² = 0.77–0.85): a station that scores highly on betweenness will, with very high probability, also score highly on these two features, largely because all three quantities are driven by the same small set of high-throughput junctions. **The practical implication: Stage C's composite value function is only genuinely reinforced by two of its four inputs.**

### 4.6 Cross-stage convergence test

*Script: [`Step6_Convergence_Test.py`](codes/Step6_Convergence_Test.py)*

The top-5 out-of-scope stations from Stage B and Stage C are compared. The observed overlap (out of 5) is benchmarked against a permutation null in which two independent five-station sets are drawn without replacement from the 34-station candidate pool, 20,000 times; the empirical p-value is the share of null draws whose overlap is at least as large as observed.

**Result:** the Stage C selection shares **four of its five stations** with the Stage B concentration ranking (Obong, Goedong, and Busan New Port recur across both; Ssangryong and Gwangyang are recovered by Stage C alone) — an overlap far larger than the permutation null predicts (**p = 0.001**).

This result is read jointly with the feature-independence audit rather than in isolation. Roughly half of Stage C's composite value function (demand and utilization centrality) is strongly collinear with the Stage B metric, so a meaningful share of the observed overlap is close to what a mechanical relationship between the two stages would produce on its own. The other half (closeness and eigenvector centrality) is only moderately correlated with betweenness, and this portion of the agreement is closer to genuine convergent validity. **The honest summary is a two-tier one:** the observed overlap is real and statistically far from chance, but a substantial share of its magnitude is attributable to shared topological origin rather than to two evidentially independent procedures reaching the same conclusion.

---

## 5. Results summary

1. The coverage gap is large in point estimate (59.6–67.5%) but statistically inconclusive on its own at N = 53; Stage A is a motivating trend, not a confirmed finding.
2. Out-of-scope structural risk is **not diffuse — it is sharply concentrated** on betweenness- and cascade-impact-based weightings (Gini 0.667–0.789, p < 0.001); degree-based concentration is statistically indistinguishable from random allocation.
3. The concentration finding **is not an artifact of the single empirical topology**: it survives an eight-realization synthetic rewiring ensemble at p < 0.001 in every realization, with 3.75/5 mean station-identity overlap.
4. A budget-constrained stochastic submodular optimizer recovers a **stable, deployable five-station priority set** — Obong, Goedong, Busan New Port, Ssangryong, Gwangyang — capturing 79.4% of expected out-of-scope value with 100% selection stability.
5. Feature independence between the two stages is **measured, not assumed**: two of four Stage C inputs are moderately distinct from the Stage B metric (r² = 0.27–0.38), two are strongly collinear (r² = 0.77–0.85).
6. The diagnostic and prescriptive rankings converge at a level unlikely to arise by chance (4/5 overlap, p = 0.001) — but the convergence is **honestly reported as partial**, roughly half genuinely independent evidence and half shared topological origin.

---

## 6. Statistical safeguards

| Threat to validity | Safeguard applied | Where applied |
|---|---|---|
| Small out-of-scope sample size (N = 34) inflating false-positive concentration claims | Permutation null constructed directly from the observed data | Stage B |
| Multiple simultaneous centrality weightings inflating family-wise error rate | Holm–Bonferroni correction (m = 4) | Stage A |
| Sampling uncertainty around bootstrap point estimates | 5,000-replicate station-level bootstrap | Stage A |
| Circularity between diagnostic and prescriptive value functions | Stage C built from features not used as the Stage B metric | Stage C |
| Chance agreement between two independently constructed priority rankings | 20,000-draw permutation null on top-5 set overlap | Convergence test |
| Concentration finding specific to one empirical topology | 8-realization degree-preserving synthetic rewiring ensemble | Robustness check |
| Assumed rather than measured independence between Stage B and Stage C features | Pearson/Spearman correlation and r² shared-variance audit, all 4 Stage C inputs | Independence audit |

---

## 7. Figures

| | | |
|---|---|---|
| ![Fig 1](figures/main/fig1_framework.png) | ![Fig 2](figures/main/fig2_data_pipeline.png) | ![Fig 3](figures/main/fig3_ocg_forest.png) |
| **Fig. 1.** Overall pipeline — network, scope definition, out-of-scope candidate pool, Stage A→B/C→convergence flow. | **Fig. 2.** Data pipeline — raw sources, preprocessing, and the three analysis-ready datasets feeding Stages A–C. | **Fig. 3.** Optimization Coverage Gap forest plot — bootstrap and Holm–Bonferroni-adjusted CIs against the 50% reference line. |
| ![Fig 4](figures/main/fig4_lorenz.png) | ![Fig 5](figures/main/fig5_permutation_null.png) | ![Fig 6](figures/main/fig6_combined_map.png) |
| **Fig. 4.** Lorenz curves of the out-of-scope structural gap by weighting, with Gini coefficients. | **Fig. 5.** Observed Gini coefficients against their permutation-null distributions (degree, betweenness, cascade impact). | **Fig. 6.** Combined map — concentration diagnosis (left) and stochastic greedy Top-K=5 selection (right) over real geography. |
| ![Fig 7](figures/main/fig7_cumulative_value.png) | ![Fig 8](figures/main/fig8_convergence_null.png) | ![Fig 9](figures/main/fig9_convergence_grid.png) |
| **Fig. 7.** Cumulative expected value captured by the greedy selection, K = 1…5, against the (1 − 1/e) worst-case guarantee. | **Fig. 8.** Diagnostic–prescriptive top-5 overlap against a 20,000-draw permutation null (observed overlap = 4/5, p = 0.001). | **Fig. 9.** Side-by-side rank comparison of the diagnostic (Stage B) and prescriptive (Stage C) top-5 station lists. |

---

## 8. Repository structure

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
│   └── main/               # Fig. 1–9 (300 dpi)
└── docs/
    ├── methods_overview.md
    └── limitations_and_methods_supplement.md
```

Each script writes its outputs to a local `outputs/` directory as CSVs, and later scripts import loader functions and (where relevant) results directly from earlier scripts (e.g., `Step6` imports `run_stage_b` from `Step2` and `run_stage_c` from `Step3`) rather than duplicating logic.

---

## 9. Running the pipeline

Requires `numpy`, `pandas`, `scipy`, and (for Step 4 only) `networkx`:

```bash
pip install numpy pandas scipy networkx --break-system-packages
```

Run in order from the `codes/` directory (later steps depend on `outputs/master_table.csv` written by Step 1, and will transparently reload from source if it is missing):

```bash
python Step1_Stage_A_Coverage_Gap.py
python Step2_Stage_B_Concentration.py
python Step3_Stage_C_Prescriptive_Selection.py
python Step4_Synthetic_Rewiring_Robustness.py
python Step5_Feature_Independence_Audit.py
python Step6_Convergence_Test.py
```

Each script is independently runnable and prints a labeled console report of its stage's findings in addition to writing CSV outputs.

---

## 10. Data & reproducibility

* **Data sources.** All analyses are based exclusively on publicly available data retrieved from two GitHub repositories:
  * **Concentrated-Blind-Spot** (analysis code and synthetic-network ensemble) — https://github.com/LEEYJ1021/Concentrated-Blind-Spot
  * **korea-freight-rail-resilience-analysis** (underlying network topology and centrality/cascade-impact data) — https://github.com/LEEYJ1021/korea-freight-rail-resilience-analysis
* **Version-controlled analysis.** The pipeline is designed to operate on version-pinned datasets whenever possible, so future updates to the source repositories do not affect the reproducibility of the reported results.
* **Randomization and reproducibility.** Random seeds are fixed throughout (`numpy.random.seed(42)` for the primary pipeline). The synthetic-network robustness ensemble uses eight independently fixed seeds (42, 137, 256, 512, 1024, 2048, 4096, 8192), each deterministically reproducible from the same repository. All stochastic procedures — bootstrap confidence intervals, permutation tests, and sensitivity sweeps — report both the number of iterations (5,000 / 2,000 / 20,000 / 200, as appropriate) and the resulting empirical distributions rather than only point estimates.
* **Computational environment.** Each script records the packages it depends on in its own header; no hidden shared state is assumed across scripts beyond the intermediate CSVs each stage writes.

---

## 11. Limitations

We would rather these be documented here than discovered by a downstream reader:

- **Out-of-scope N = 34 is small.** This is the direct reason Stage A's confidence intervals are wide and the Stage B–C convergence test has limited statistical power to detect anything short of near-total overlap.
- **This is a single-case network design.** Its claims should be read as generalizations about the mechanism linking policy-defined scope boundaries to network-structural risk, not as statistical generalizations to a population of national rail networks. The synthetic-rewiring robustness check narrows, but does not eliminate, this limitation: a rewiring ensemble is still drawn from the same network's own degree distribution and cannot substitute for genuinely independent replication on a different country's rail network.
- **The Stage C "de-circularization" safeguard is necessary but not sufficient on its own.** Building the prescriptive value function from features that are not literally the Stage B metric is a necessary starting point, but the feature-independence audit shows that not all nominally distinct features are equally independent — two of the four Stage C inputs (demand, utilization) turn out to be strongly collinear with betweenness (r² = 0.77–0.85).
- **The reported 4/5 convergence overlap should not be read as "two fully independent methods agree."** Roughly half of that agreement is attributable to shared topological origin rather than independent triangulation.
- **Degree-based concentration is not statistically distinguishable from random allocation.** Whichever centrality weighting is used to justify scope-expansion decisions materially changes the conclusion, and this sensitivity should be stated explicitly in any operational use of this framework.

Full text: [`docs/limitations_and_methods_supplement.md`](docs/limitations_and_methods_supplement.md).

---

## 12. Future research

This project deliberately restricts its scope to the diagnostic-prescriptive pairing that the data support at conventional significance. Several related analyses were developed on the same underlying dataset but are intentionally kept out of the central pipeline above, and are noted here as directions for follow-on work:

- **Re-estimating the Stage C value function using only the genuinely independent inputs.** A natural extension is to rebuild the prescriptive value function using closeness and eigenvector centrality alone — the two features shown here to be only moderately correlated with betweenness — and test whether the resulting priority set still recovers a comparable share of expected out-of-scope value and a comparable overlap with the Stage B ranking. If the overlap holds, the convergence claim would rest on an unambiguously independent value function; if it drops substantially, that is itself informative about how much of the current convergence is genuinely non-circular.
- **Replication on a second national rail network.** The single-network limitation discussed above is addressed here through synthetic rewiring, not independent replication. Applying the same diagnostic-prescriptive pipeline to a structurally comparable freight network in another country would be the most direct way to strengthen external validity.
- **A predictive transfer layer with exact Shapley attribution.** A companion analysis develops a graph-propagated transfer model that extrapolates in-scope resilience-value labels to out-of-scope stations, with exact closed-form SHAP attribution rather than sampled attribution, made possible by a deliberately linear model specification.
- **Dynamic spillover-burden reallocation under alternative carbon-policy scenarios.** A related line of analysis examines how out-of-scope structural burden shifts as decarbonization policy intensifies across multiple carbon-pricing scenarios, using the same underlying network and centrality data.
- **An audited large-language-model narration layer for non-technical stakeholders.** A separate exploratory component pairs an LLM-based narrative explainer with a deterministic, code-based verification layer, validated with an adversarial positive-control stress test, to translate the technical diagnostic and prescriptive outputs above into stakeholder-facing language without sacrificing numerical fidelity.

These directions are reported separately because each answers a distinct methodological question from the one addressed in this project's central diagnostic-to-prescriptive pipeline.
