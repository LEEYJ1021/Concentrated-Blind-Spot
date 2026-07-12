# Concentrated Blind Spots: Auditing Structural Coverage Gaps in Corridor-Based Rail Decarbonization Optimization
**A statistically-audited diagnostic framework for the Korean freight rail network**
> **One-line summary.** When a rail decarbonization program is planned around a handful of "core" corridors, which stations get structurally left out of the optimization scope — and is that gap random noise or a short, nameable, policy-actionable list? We answer this with real KORAIL network data, four independent analytical lenses (diagnostic → explanatory → predictive → prescriptive), and—critically—a full battery of statistical stress tests (permutation nulls, Holm–Bonferroni correction, nested cross-validation, a de-circularized robustness check, and an adversarial positive control for the LLM-audit layer) so that every headline result in this repository has been rigorously challenged for robustness, validity, and reproducibility.

---

## Table of contents

- [Why this fits the Special Issue](#why-this-fits-the-special-issue)
- [Headline findings — with honest statistical framing](#headline-findings--with-honest-statistical-framing)
- [Figures](#figures)
- [Repository structure](#repository-structure)
- [Methods overview](#methods-overview)
- [Data & reproducibility](#data--reproducibility)
- [Limitations (stated up front, not buried)](#limitations-stated-up-front-not-buried)

---

## Research Contributions

This study integrates industrial engineering, network science, explainable artificial intelligence, and generative AI to develop a rigorous framework for evaluating rail-network decarbonization strategies. The framework combines four complementary analytical perspectives—diagnostic, explanatory, predictive, and prescriptive—while emphasizing robustness, interpretability, and reproducibility throughout the analytical pipeline.

| Research contribution                                        | How this study delivers it                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Network-based diagnosis of rail decarbonization coverage** | Graph-theoretic analysis of the real KORAIL network identifies structurally underserved stations using degree, betweenness, and cascade-impact centrality under simulated node removals (RQ1–RQ2).                                                                                                          |
| **Scenario-based resilience assessment**                     | Dynamic spillover-burden reallocation is evaluated across five K-ETS carbon-policy scenarios (RQ3), supported by Spearman correlation analysis and permutation-based convergence testing.                                                                                                                   |
| **Explainable and trustworthy AI**                           | An exact closed-form SHAP audit layer is combined with a deterministic narrative verifier implemented in Python, and validated through adversarial stress tests that inject known incorrect claims to evaluate detection sensitivity and specificity.                                                       |
| **Responsible use of generative AI**                         | A narrowly scoped Explainer–Auditor multi-agent framework distinguishes narrative generation from deterministic numerical verification. Narrative consistency is evaluated using Fleiss' κ across three architecturally distinct local language models, while numerical fidelity is verified independently. |
| **Optimization under uncertainty**                           | A stochastic, budget-constrained submodular greedy optimization framework provides a formal (1 − 1/e) approximation guarantee (RQ5) and is evaluated across 200 randomly generated scenario-probability configurations.                                                                                     |
| **Rail-network decarbonization planning**                    | The complete analytical framework is evaluated under five policy scenarios representing alternative K-ETS carbon-cost trajectories, from the 2023 baseline to progressively strengthened decarbonization pathways.                                                                                          |


---

## Headline findings — with honest statistical framing

The original analysis found that four independent methods — static concentration, dynamic reallocation, predictive attribution, and prescriptive optimization — pointed at an overlapping short list of stations (Obong, Goedong, Busan New Port, Ssangryong, Gwangyang). A first-pass version of this project treated that overlap as self-evident. **It isn't, and a referee will say so immediately, because RQ5's original value function was mechanically built from RQ2 and RQ3's own outputs.** The version described here fixes that and reports what survives:

1. **RQ2 (static concentration) is real and non-trivial.** The structural gap among out-of-scope stations is not diffuse: Gini coefficients of 0.43–0.79 across three centrality weightings are all significantly higher (p < 0.001, 5,000-draw permutation test) than a null in which the same total gap mass is allocated uniformly at random across the same 34 candidate stations. Ten stations carry roughly 80% of the resilience-relevant gap. *(Fig. 2b, S2)*

2. **RQ1 (how large is the gap) is reported as a trend, not a confirmed finding.** Point estimates of the Optimization Coverage Gap run 60–68%, but with N = 53 stations the bootstrap CIs do not exclude 50% under any centrality weighting — and this conclusion is unchanged before and after Holm–Bonferroni correction across the four weightings tested (m = 4). We report both the raw and corrected thresholds rather than picking whichever makes the story look better. *(Fig. 2a, S1)*

3. **RQ3 (dynamic reallocation) shows real reconcentration.** As K-ETS policy intensifies from the 2023 baseline (S1) toward NDC alignment (S5), spillover burden shifts disproportionately *toward* already-burdened stations (Spearman ρ = 0.524, p = 0.010, N = 23). *(Fig. 2c, 3-heatmap)*

4. **RQ4 (predictive attribution) is interpretable by construction.** Because the transfer model is linear (SGC-propagated ridge regression), its SHAP values are exact, not sampled. Predictions for out-of-scope stations are driven almost entirely by closeness centrality (mean |SHAP| = 0.159 vs. ≤ 0.03 for every other feature) — an auditable, reportable fact about what the extrapolation step actually uses. The ridge penalty is selected by nested leave-one-corridor-out cross-validation (α = 0.1), not asserted, and label augmentation (N = 19 → 95) is explicitly reframed as a repeated-measures design (19 independent stations × 5 scenarios), not treated as a five-fold increase in independent sample size. *(Fig. 2d–e, S3)*

5. **RQ5 (prescriptive optimization), once de-circularized, still converges with RQ2 — but not with RQ3 or RQ4.** Rebuilding the greedy submodular value function on features never used in RQ2 or RQ3 (closeness, eigenvector, demand/utilization centrality), the top-5 selection still overlaps RQ2's top-5 at a rate far beyond chance (4/5 shared stations, permutation p = 0.001). Overlap with RQ3 and RQ4's top-5 lists, however, is not statistically distinguishable from random K = 5 draws (p = 0.15–0.58). **We report this honestly: the "four-way convergence" headline is downgraded to a genuine two-way (RQ2 ↔ RQ5) convergence plus directionally consistent, but not independently significant, agreement from RQ3 and RQ4.** *(Fig. 2f, 2h, S4, S5)*

6. **The LLM-audit layer's perfect scores are backed by a positive control.** The deterministic verifier passes 100% of 34 real narratives (mean Faithfulness Score 0.856, 95% bootstrap CI [0.849, 0.863]) — a result that would otherwise look suspiciously clean. An adversarial stress test injecting known-wrong claims at controlled corruption rates confirms the verifier detects genuine errors at 79–86% sensitivity while maintaining ~100% specificity, so the real-data result reads as *verified correctness*, not a rubber stamp. *(Fig. 2g, S6, S7)*

---

## Figures

### Main results

| | | |
|---|---|---|
| ![Fig 1](figures/main/fig1_korea_map_triptych.png) | ![Fig 2a](figures/main/fig2a_rq1_coverage_gap_forest.png) | ![Fig 2b](figures/main/fig2b_rq2_lorenz_curves.png) |
| **Fig. 1.** Korea map triptych — coverage gap, reallocation, and greedy selection, all on real KOSTAT geography. | **Fig. 2a.** RQ1 forest plot — CIs do not exclude 50% at N = 53. | **Fig. 2b.** RQ2 Lorenz curves — betweenness and cascade-impact gaps are sharply concentrated. |
| ![Fig 2c](figures/main/fig2c_rq3_reallocation_bars.png) | ![Fig 2d](figures/main/fig2d_rq4_shap_importance.png) | ![Fig 2e](figures/main/fig2e_rq4_augmentation_mae.png) |
| **Fig. 2c.** RQ3 top/bottom-5 spillover-share movers, S1→S5. | **Fig. 2d.** RQ4 global SHAP importance — closeness dominates. | **Fig. 2e.** RQ4 augmentation — repeated-measures MAE improvement. |
| ![Fig 2f](figures/main/fig2f_rq5_cumulative_value.png) | ![Fig 2g](figures/main/fig2g_llm_audit_faithfulness.png) | ![Fig 2h](figures/main/fig2h_convergence_matrix.png) |
| **Fig. 2f.** RQ5 cumulative value capture, greedy K = 1…5. | **Fig. 2g.** LLM-audit Faithfulness Score distribution (n = 34). | **Fig. 2h.** Four-method rank-convergence matrix. |
| ![Fig 3a](figures/main/fig3_shap_waterfall_topstation.png) | ![Fig 3b](figures/main/fig3_reallocation_heatmap.png) | |
| **Fig. 3a.** SHAP waterfall for the highest-predicted out-of-scope station. | **Fig. 3b.** Spillover-share evolution, all 15 most-shifted stations × 5 scenarios. | |

### Robustness supplement

| | | |
|---|---|---|
| ![Fig S1](figures/supplement/figS1_rq1_multiplicity.png) | ![Fig S2](figures/supplement/figS2_rq2_gini_permutation.png) | ![Fig S3](figures/supplement/figS3_rq4_alpha_selection.png) |
| **Fig. S1.** RQ1 with Holm–Bonferroni correction (m = 4) — no weighting significant. | **Fig. S2.** RQ2 Gini vs. 5,000-draw permutation null — p < 0.001 all metrics. | **Fig. S3.** RQ4 ridge regularization path — α selected by nested CV, not hardcoded. |
| ![Fig S4](figures/supplement/figS4_rq5_stability.png) | ![Fig S5](figures/supplement/figS5_convergence_significance.png) | ![Fig S6](figures/supplement/figS6_verifier_stress_test.png) |
| **Fig. S4.** RQ5 station-selection stability across 200 random scenario-probability draws (100% recovery). | **Fig. S5.** Convergence significance — de-circularized RQ5 vs. RQ2/RQ3/RQ4. | **Fig. S6.** LLM-verifier positive-control stress test — sensitivity/specificity vs. injected corruption rate. |
| ![Fig S7](figures/supplement/figS7_faithfulness_ci.png) | | |
| **Fig. S7.** Faithfulness score with 95% bootstrap CI (n = 34), replacing the bare point estimate. | | |

---

## Repository structure

```
rail-coverage-gap/
├── README.md
├── LICENSE
├── CITATION.cff
├── figures/
│   ├── main/              # Fig. 1–3 (paper-ready, 300 dpi)
│   └── supplement/        # Fig. S1–S7 (robustness supplement)
└── docs/
    ├── methods_overview.md
    ├── limitations_and_methods_supplement.md
    └── cfp_alignment_notes.md
```

> **Data and analysis code.** The two source repositories (station-level network topology and cascade/centrality metrics) and the analysis notebook are maintained separately and referenced with pinned commit SHAs and SHA-256 checksums in the data-provenance manifest (see [Data & reproducibility](#data--reproducibility)) so results are exactly reproducible even if the source repositories change.

---

## Methods overview

See [`docs/methods_overview.md`](docs/methods_overview.md) for the full pipeline. In brief:

1. **RQ1 — Optimization Coverage Gap.** Bootstrap-CI'd share of network importance (by degree / betweenness / cascade-impact-under-removal) sitting outside the four in-scope corridors.
2. **RQ2 — Concentration.** Gini coefficients and Lorenz curves of the RQ1 gap across out-of-scope stations, benchmarked against a Dirichlet-permutation null.
3. **RQ3 — Reallocation.** Spillover-burden share of out-of-scope stations under five real K-ETS scenarios (S1 2023 → S5 NDC), tested for reconcentration via Spearman correlation between baseline share and share-change.
4. **RQ4 — Predictive transfer + exact SHAP audit.** A 2-hop SGC-propagated ridge regression transfers in-scope VSS (value-of-stochastic-solution) labels to out-of-scope stations; because the model is linear, SHAP attributions are computed in closed form rather than sampled. Ridge α selected by nested leave-one-corridor-out CV. Labels augmented via a genuine repeated-measures design (5 K-ETS scenarios × 19 independent stations).
5. **RQ5 — Prescriptive optimization.** Budget-constrained greedy submodular selection (K = 5) over out-of-scope stations, with a (1 − 1/e) approximation guarantee, rebuilt on a value function independent of RQ2/RQ3's own metrics to remove circularity, and stability-tested across 200 random scenario-probability draws.
6. **LLM audit layer.** A schema-validated Explainer agent (qwen2.5:14b) produces narrative explanations of SHAP attributions; a deterministic Python verifier checks direction/rank/magnitude claims against ground truth; a three-model Auditor panel (llama3.1:8b, mistral:7b-instruct, qwen2.5:7b-instruct) judges narrative overclaiming and readability, with inter-rater reliability reported via Fleiss' κ. The verifier's sensitivity is validated with a synthetic adversarial stress test.

---

## Data & reproducibility

- All source data are retrieved from two public repositories; exact commit SHAs and SHA-256 checksums of every retrieved file are recorded at analysis time and should be cited in the manuscript's Data Availability statement.
- All cross-validation that touches the repeated-measures RQ4 dataset is clustered at the corridor level (leave-one-corridor-out), never at the observation level, to respect the non-independence of the 5 scenario-repeats per station.
- Random seeds are fixed (`numpy.random.seed(42)` for the main pipeline; separate seeds documented per robustness check) for point-estimate reproducibility; bootstrap and permutation distributions are reported with sample sizes (5,000 / 10,000 / 20,000 draws as appropriate) rather than as single numbers.

---

## Limitations (stated up front, not buried)

We would rather a reader find these in the README than have a reviewer find them first:

- **Out-of-scope N = 34 is small.** Several RQ2/RQ5 comparisons are flagged explicitly wherever N < 30.
- **RQ4's "N = 95" is 19 independent stations, not 95 independent observations.** Every reported metric states the effective independent N alongside the raw row count.
- **RQ1 does not, on its own, statistically establish that the coverage gap exceeds 50%** at current sample size. We report it as a trend motivating RQ2–RQ5, not as a standalone confirmed result.
- **RQ5's apparent "four-way convergence" is, honestly, a two-way statistically-significant convergence (RQ2 ↔ RQ5) plus directionally consistent but non-significant agreement with RQ3 and RQ4.** See [Headline findings](#headline-findings--with-honest-statistical-framing) above and Fig. S5.
- **The transfer model (SGC-ridge) is intentionally simple (linear).** This is what makes exact SHAP possible, but it also means the "AI extrapolation" being audited is not a deep model — a reviewer working in that CFP topic may reasonably ask whether the trustworthiness contribution generalizes to nonlinear architectures. We treat this as a scoping decision to be stated explicitly, not something to obscure.
- **Scenario probabilities used in RQ5's expected-value objective are a modeling assumption**, sensitivity-tested across 200 draws from the probability simplex (Fig. S4) rather than asserted as ground truth.

Full text: [`docs/limitations_and_methods_supplement.md`](docs/limitations_and_methods_supplement.md).
