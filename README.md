# Auditing designated corridor scopes: null-model calibration and a screening shortlist for Korea's freight-rail network

Code and result tables for the revised manuscript. The repository was restructured after review: the earlier
version (Stage A-C, Moran's I, "coverage gap" claims) is kept in `legacy_v1/` and is **superseded**; several of
its headline numbers (e.g. 39.3% captured value, 4/5 convergence, "dispersion") are not carried into the revision.

## 1. What the study asks

- **RQ1.** Is an out-of-scope audit statistic valid under its reference distribution, and what can it detect at
  the scale of a national freight network?
- **RQ2.** What does the calibrated audit say about Korea's four-line scope (53 stations)?

The network is a **train-service connection network** built from the freight timetable (217 trains, 10 lines,
53 stations, 86 undirected edges), not track topology. The four-line scope is an analytic boundary defined by
line membership, not traced to an official designation document. Indicators are structural and service
measures; **no emissions are evaluated.**

## 2. Main findings (see manuscript for tables)

| Hypothesis | Result |
|---|---|
| H1 Size | Dirichlet(1) reference rejects 0-100% of random scopes depending on indicator and topology; scope-matched random pool holds nominal size (4.3-5.5%). |
| H2 Power | No statistic reaches power 0.8 with up to 5 planted blind spots (N = 53, 34 outside); OCG reaches 0.43 at m = 5; top-5 leakage cannot reach p <= 0.05 (min p = 0.097). |
| H3 Detectability | Power rises with N and falls with out-of-scope share; Korean configuration 0.20-0.40; about 0.90 at N = 100 with 10% planted (upper bound). |
| H4 Korea | OCG effects -0.135 / -0.099 / +0.010 (distance-weighted efficiency loss / betweenness / service train-km); out-minus-in Gini differences positive but Holm p >= 0.33; sign flips under service-share scopes. Result is **undetermined**, not absent. |
| H5 Typicality | All six statistics lie inside 95% prediction intervals of degree-preserving rewired ensembles (1,000 each, plain and length-constrained). |
| H6 Spatial | No global clustering detected (Moran's I < E[I] for all weights, two-sided p 0.08-0.70); minimum detectable rho lies between 0.7 and 0.9. |
| H7 Core | Goedong (1.00), Obong (0.92), Busan New Port (0.83) in the top five across 24 specifications; external consistency not testable (Holm p = 1.0). |

These are screening results and a reporting standard, not an investment ranking.

## 3. Repository layout

```
codes/        00 bundle builder -> 01 phase-0 audit -> 02 main -> 03 S -> 04 P -> 05 C -> 06 D -> 07 E
results/      CSV tables and *_numbers.json written by each stage (main, supp_S, supp_C, supp_D, supp_E)
figures/      manuscript_figures/ (final figures and source data), reanalysis_v1/ (pipeline output)
docs/         notes
legacy_v1/    earlier pipeline (Step1-7) and figures, kept for traceability only
run_all.sh    runs 00-07 in order
```

## 4. Seeds and repetitions (manuscript Table A5)

| Stage | Script | Seed |
|---|---|---|
| Main analysis | `02_main_reanalysis.py` | 20261009 |
| Supplement S (scope, selection, rewiring, spatial) | `03_...` | 20261010 |
| Calibration C (size, power, synthetic) | `05_...` | 20261011 |
| Statistics D | `06_...` | 20261012 |
| Detectability E | `07_...` | 20261013 |

Pre-specified settings (alpha = 0.05, equivalence margin 0.05, core frequency 0.8, planted m = 0,1,2,3,5) are
fixed at the top of each script. Analysis was frozen after step E. The pre-specified selection rule chose the
Gini difference G(out) - G(in); OCG is reported alongside as a **post-hoc inclusion** (its single size-screen
exceedance, 1 of 30 cells, is consistent with chance).

## 5. Running

```bash
pip install -r requirements.txt
export RAIL_ROOT=/absolute/path/to/AI_Rail_OM   # folder containing Data/ (raw Korail CSVs, not redistributed)
QUICK=1 bash run_all.sh                          # smoke test, never cite
bash run_all.sh                                  # full run (minutes to tens of minutes)
```

`RAIL_ROOT` must be set to an absolute path. The scripts run from `codes/`, so the default (`.`) will not find `Data/`.
Order matters: `03` and `05` read tables from `02`; `04` reads `03`; `07` reads `06`.

## 6. Data

- Raw Korail files (reservations, special terms, station detail, route distance, kilopost, etc.) are **not**
  redistributed; place them in `$RAIL_ROOT/Data/`. Two source files are truncated in the current export
  (`역간최단거리`, `표준적하시간`); results do not depend on them in the main analysis.
- Network inputs: `rail-freight-decarbonization` and `korea-freight-rail-resilience-analysis` (GitHub), fetched by
  `00_build_data_bundle_v3.py` and cached.
- The timetable reference year is stated in the manuscript (Section 3.1).

## 7. Known limitations

- Planted-blind-spot power is an upper bound (most favourable configuration).
- Single network representation and timetable year; scope not traced to an official instrument.
- Structural indicators are not validated against operational or environmental outcomes.
- Moran's I has limited power at n = 34; "not detected" is not evidence of dispersion.
- The legacy "cascade impact" is mostly topological disconnection plus a simplified overload increment
  (alpha = 0.5, loads not recomputed); it is reported as such, not as a load-redistributing cascade model.
- Candidate list (facility-location step) shares information with the diagnostic ranking; agreement is not
  independent corroboration.

## 8. Citation

See the manuscript. Release tag `v2.0-revision` corresponds to the submitted analysis; the commit hash is
recorded in the manuscript's reproducibility statement.
