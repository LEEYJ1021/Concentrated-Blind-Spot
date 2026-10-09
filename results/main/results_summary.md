# Re-analysis summary (auto-generated; seed 20261009)

## Setting
- 53 stations, 86 service edges, 10 lines, 217 trains. **Timetable reference year: UNKNOWN (author must state).**
- Design space: 1022 line subsets. Reference scope (top-4 lines by train-km): ['Gyeongbu', 'Jungang', 'Chungbuk', 'Jeolla']; original four lines: ['Gyeongbu', 'Chungbuk', 'Yeongdong', 'Jungang']; identical: False.

## Key numbers
- OCG reference/original: 0.220 / 0.326; baselines (1 - mean w): 0.224 / 0.339; effects -0.004 / -0.012.
- Pool size (reference): 8. Primary indicator (eff_loss_dist) Gini: 0.727 (in-scope 0.798); random-pool p=0.5577; Dirichlet p=0.0014.
- DIBI top-5: ['JecheonYard', 'Donghae', 'Dodam', 'BusanNewPort', 'Dongsan']; invariant blind spots: []; Spearman(DIBI, naive)=0.959.
- Stage C (K=5, lambda=0.3): greedy ['Hwangdeung', 'Suncheon', 'Gwangyang', 'Seokpo', 'Gunsan']; exact ['Gwangyang', 'Suncheon', 'Gunsan', 'Seokpo', 'Hwangdeung']; greedy/exact = 1.0; monotone violations 0, submodular violations 0.
- H3 overlaps / information-matched p: [('Full', 4, 0.94488), ('Service-augmented', 4, 0.69927), ('Distinct only', 3, 0.87866)].
- Moran I range (k=3..8): [-0.21227209382506174, 0.0] vs E[I]=-0.1429; p range [0.634, 1.0]; power at rho=0.5 (k=5) = 0.10.
- Rewiring (degree-preserving) Gini: observed 0.764 vs ensemble nan +/- nan; p=0.6983 (n=1000).
- Specification curve: 3% of 88 specifications have p<.05 vs random-pool null.

## Recalibrated hypotheses (data-driven verdicts)
- H1' (out-of-scope value is more concentrated than a random same-size station set; Holm-adjusted): **NOT SUPPORTED**
- H2' (observed concentration is unusual vs degree-preserving rewired ensembles): **NOT SUPPORTED**
- H3' (Stage B/C agreement exceeds an information-matched null): **NOT SUPPORTED**
- H4' (DIBI adds information beyond naive ranking): **NOT SUPPORTED**
- H5' (global spatial clustering detected): **NOT DETECTED (interpret with power above; never as proof of dispersion)**

## Wording rules for the manuscript
- Say "train-service connection network", never "physical topology".
- Say "no detected global spatial autocorrelation", never "dispersion".
- Say "agreement between partially information-sharing procedures", never "independent validation".
- Say "coverage of the composite objective", never "% of structural risk".
- K=5 is a cardinality limit, not a budget (unless the cost-weighted version is used).
- Legacy 'cascade impact' = single-node removal disconnection + simplified overload redistribution (see F01B).
