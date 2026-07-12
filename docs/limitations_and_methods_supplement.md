# Limitations and Methods Supplement

*Drop-in text for a manuscript's Limitations / Methods-caveats section. Each item maps to a specific, checkable artifact in this repository rather than a general disclaimer.*

1. **Non-circularity of RQ5.** To avoid RQ5's prescriptive value function being mechanically derived from RQ2's concentration metric (cascade-impact under degree removal) or RQ3's spillover-share metric, RQ5 is defined here on an independent composite of closeness centrality, eigenvector centrality, demand centrality, and utilization centrality — none of which enter RQ2 or RQ3's target variables. Convergence between this independently-defined RQ5 selection and the RQ2/RQ3/RQ4 top-5 lists is tested against a permutation null of random out-of-scope picks (Fig. S5). **Result:** significant overlap with RQ2 (p = 0.001); overlap with RQ3 and RQ4 is directionally consistent but not statistically distinguishable from chance (p = 0.15–0.58) at K = 5.

2. **Effective sample size in RQ4.** The augmented dataset contains 95 observations drawn from only 19 independent stations, each measured under 5 K-ETS scenarios — a repeated-measures design, not 95 independent samples. All cross-validation is clustered at the corridor level (leave-one-corridor-out) to respect this dependence structure; the effective independent N is reported alongside every metric.

3. **Hyperparameter selection.** The ridge penalty (α) for the SGC-ridge transfer model is selected by nested leave-one-corridor-out cross-validation over a log-spaced grid (0.1, 0.5, 1, 2, 5, 10, 20, 50, 100), not fixed a priori. The selected value is α = 0.1 (Fig. S3).

4. **Multiple comparisons.** RQ1 reports bootstrap CIs for the Optimization Coverage Gap under 4 distinct centrality weightings. A Holm–Bonferroni correction (m = 4) is applied to the implied two-sided test against a 50% null; the qualitative conclusion — CIs do not exclude 50% at N = 53 — is unchanged before and after correction (Fig. S1).

5. **Statistical comparator for concentration.** RQ2's Gini coefficients are benchmarked against a permutation null in which the same total structural-gap mass is allocated uniformly at random (Dirichlet(1,...,1)) across the same N out-of-scope stations. Observed concentration exceeds this null at p < 0.001 for all three centrality weightings tested (Fig. S2).

6. **Scenario-probability sensitivity.** The K-ETS scenario probabilities used in RQ5's expected-value objective are a modeling assumption. A sensitivity sweep over 200 draws from the probability simplex shows the same top-5 station set is recovered in 100% of draws (Fig. S4) — i.e., the selection is robust to this particular assumption even though the assumption itself is not empirically derived.

7. **LLM-audit verifier validation.** The deterministic verifier underlying the Explainer/Auditor faithfulness pipeline achieves a perfect pass rate on 34 real out-of-scope narratives. To confirm this reflects genuine claim correctness rather than a permissive verifier, a synthetic stress test injects known-incorrect claims (flipped direction / shifted rank) at controlled corruption rates and measures detection sensitivity and specificity (Fig. S6). Faithfulness scores on the real narratives are additionally reported with 95% bootstrap confidence intervals rather than as a bare point estimate (Fig. S7).

8. **Data provenance.** All source CSVs are retrieved from two public GitHub repositories. Exact commit SHAs and SHA-256 checksums of the retrieved bytes are recorded in a data-provenance manifest for reproducibility; these are cited in the manuscript's Data Availability statement.

9. **Small-N disclosure.** The out-of-scope station set has N = 34; several RQ2/RQ5 comparisons therefore carry a small-sample flag. This is stated explicitly wherever N < 30, rather than left implicit.

10. **Model simplicity as a scoping choice.** The RQ4 transfer model is intentionally linear (SGC-propagated ridge regression). This is what makes exact, non-sampled SHAP attribution possible and is central to the "AI trustworthiness" contribution claimed for the Special Issue — but it also means the audited "AI extrapolation" step is simpler than a deep model. We treat this as a stated scoping decision (auditability traded against model capacity) rather than an unstated one, and note it as a natural direction for follow-up work with nonlinear transfer models and sampled or kernel SHAP.
