# Methods Overview

This document expands the summary in the main `README.md` into a fuller methods narrative, organized in the diagnostic → explanatory → predictive → prescriptive sequence used throughout the project. It is written to be lifted, with light editing, into a manuscript's Methods section.

## 0. Scope definition

The Korean freight rail network is partitioned into an **in-scope** set — four corridors (Gyeongbu, Chungbuk, Yeongdong, Jungang) that carry the bulk of current decarbonization-optimization attention — and an **out-of-scope** set of remaining stations. Each station's scope weight `w_i` is the fraction of its (non-branch) line assignments that fall inside the four in-scope corridors; multi-line junction stations are flagged separately so they are not silently double-counted as "fully out of scope."

Station-to-line resolution for stations not covered by the original curated dataset uses a manually verified mapping cross-checked against MOLIT (Ministry of Land, Infrastructure and Transport) records. This mapping is exposed in full in the analysis code so it can be audited line-by-line.

## 1. RQ1 — Optimization Coverage Gap (OCG)

For four centrality/impact weightings (degree, betweenness, cascade-impact-under-degree-removal, cascade-impact-under-betweenness-removal), the OCG is the share of total network importance carried by out-of-scope stations:

```
OCG = Σ_i [ weight_i * (1 − w_i) ] / Σ_i weight_i
```

A nonparametric bootstrap (5,000 resamples) produces a 95% CI around each point estimate. Because four weightings are tested against the same implicit null (OCG = 50%), a Holm–Bonferroni correction (m = 4) is applied to the two-sided bootstrap p-values, and both corrected and uncorrected conclusions are reported side by side.

## 2. RQ2 — Concentration of the gap

Restricting to out-of-scope stations only, the RQ1 gap contribution per station is ranked and a Gini coefficient and Lorenz curve are computed per weighting. To give the Gini coefficients a real comparator (an isolated "Gini = 0.667" number is not self-interpreting), each is benchmarked against a permutation null: the same total gap mass is reallocated uniformly at random (Dirichlet(1,...,1)) across the same N out-of-scope stations, 5,000 times, and an empirical p-value is computed as the fraction of null draws with Gini ≥ observed.

## 3. RQ3 — Spillover-burden reallocation

Using real corridor-level Value of the Stochastic Solution (VSS) figures under five K-ETS carbon-policy scenarios (S1 2023 baseline through S5 NDC-aligned intensification) and observed origin–destination trip volumes, each out-of-scope station's spillover exposure is estimated as a trip-weighted average of the VSS of adjacent in-scope corridors. Spillover shares are normalized within each scenario, and the S1→S5 share change is regressed (Spearman) against the S1 baseline share to test whether burden **reconcentrates** on already-high-share stations or diffuses toward previously low-share stations.

## 4. RQ4 — Predictive transfer with an exact SHAP audit layer

**Feature propagation.** Seven station-level centrality/demand features are z-scored and propagated two hops through a symmetrically-normalized adjacency matrix (Simplified Graph Convolution, SGC) built from observed corridor edges.

**Transfer model.** A ridge regression is fit on in-scope stations, where the label is the VSS/E[RP] percentage of the station's home corridor. Because this model is linear in the propagated features, SHAP attribution is available in closed form — `shap_j = (x_j − x̄_j) * β_j` — rather than approximated by sampling, which both saves computation and removes an entire class of approximation-error caveats a reviewer might otherwise raise.

**Hyperparameter selection.** The ridge penalty α is chosen by nested leave-one-corridor-out cross-validation over a log-spaced grid (0.1–100), not fixed a priori.

**Label augmentation.** The original label set (N = 19, one VSS label per in-scope station under the S1 baseline only) is expanded to N = 95 by reusing each station's VSS labels under all five real, previously-collected K-ETS scenarios, with a scenario-intensity feature appended. This is explicitly a **repeated-measures design** — 19 independent stations × 5 scenario-repeats — and every reported metric states the effective independent N alongside the raw observation count. Cross-validation remains clustered at the corridor level throughout.

## 5. RQ5 — Prescriptive optimization (budget-constrained scope expansion)

**De-circularization.** An earlier version of this analysis defined the RQ5 "value" of adding an out-of-scope station to the optimization scope as a direct function of RQ2's cascade-impact metric and RQ3's spillover metric — which mechanically guarantees agreement with RQ2/RQ3 and makes any "convergence" claim circular. The value function used here instead is an equal-weighted composite of four features that enter **neither** RQ2 nor RQ3's target variables: closeness centrality, eigenvector centrality, demand centrality, and utilization centrality.

**Selection.** A greedy algorithm selects K = 5 stations to maximize the diminishing-returns-adjusted (submodular) sum of value, with an overlap penalty for adjacency to already-selected stations, carrying the standard (1 − 1/e) approximation guarantee for monotone submodular maximization under a cardinality constraint.

**Convergence testing.** Overlap between this independently-defined RQ5 top-5 and each of RQ2/RQ3/RQ4's top-5 lists is tested against a permutation null of two random 5-station draws from the 34-station out-of-scope pool (20,000 draws), reporting an exact p-value per pairwise comparison rather than an eyeballed "same stations" claim.

**Sensitivity.** Because the scenario probabilities used in the (original, spillover-based) RQ5 objective are themselves a modeling assumption, a 200-draw sweep over the probability simplex reports how often the same top-5 station set is recovered.

## 6. LLM audit layer (Explainer / deterministic Verifier / Auditor panel)

The design principle is: **do not ask an LLM to check another LLM's arithmetic.**

1. **Explainer agent** (qwen2.5:14b) receives the SHAP attributions for the smallest feature subset covering 80% of a station's total |SHAP| mass (a defensible, non-arbitrary claim-selection rule, replacing an earlier fixed-threshold cutoff) and produces a schema-validated JSON narrative in Korean plus structured claims (feature, direction, rank, magnitude label).
2. **Deterministic Verifier** (plain Python) checks every claim against the ground-truth SHAP values, computing continuous direction accuracy and rank accuracy (not just binary pass/fail) and combining them with a semantic-similarity term into a single Faithfulness Score (weights 0.4 / 0.3 / 0.3, stated explicitly and reported with the underlying components).
3. **Auditor panel** (three architecturally distinct local models: llama3.1:8b, mistral:7b-instruct, qwen2.5:7b-instruct) independently judges only the narrative's *language quality* — overclaiming and readability — never re-deriving the numbers. Inter-rater reliability is reported via Fleiss' κ rather than raw percent agreement.
4. **Positive control.** Because the deterministic verifier passes 100% of real narratives, a synthetic stress test injects known-wrong claims (flipped direction, shifted rank) at controlled corruption rates (0–100%) and measures the verifier's detection sensitivity and specificity, confirming the perfect real-data score reflects genuine correctness rather than a permissive verifier.
5. **Prompt-injection defense.** Both agent system prompts explicitly instruct the model to treat all input values as inert data, and station names are sanitized (control characters and instruction-like substrings stripped) before templating — stated honestly as defense-in-depth, not a formal injection-proofing guarantee.
