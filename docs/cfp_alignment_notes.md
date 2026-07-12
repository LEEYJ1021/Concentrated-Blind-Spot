# CFP Alignment Notes — Transportation Research Part E, VSI: CIE52

*Working notes for the cover letter. Guest editors: M. Fathi, T. Arbaoui, D. Battini, A. Dolgui, K. Hadj-Hamou. Submission deadline: 31 December 2026. Article type on submission: "VSI: CIE52".*

## Direct topic mapping

The Special Issue call lists sixteen topic areas "not limited to." This project touches six directly and can be framed in the cover letter using the CFP's own language rather than paraphrasing it into something less recognizable to the guest editors:

- *"AI, machine learning, and big data analytics for optimizing freight and passenger flows"* — RQ1–RQ2 use real KORAIL network topology and simulated-removal cascade metrics.
- *"Data-driven disruption management, predictive risk modeling, and strategies for resilient and adaptive supply chains"* — RQ3's five-scenario spillover reallocation is a direct disruption/predictive-risk framing.
- *"AI security, robustness, and trustworthiness in transportation systems"* — the exact-SHAP audit layer and the adversarially stress-tested deterministic verifier are the clearest fit in the whole submission; lead with this in the cover letter, not bury it.
- *"Generative AI for logistics innovation and decision support"* — the Explainer/Auditor pipeline, scoped honestly (narrative judgment only, never numeric verification).
- *"Industrial engineering methods ... optimization"* — RQ5's submodular greedy selection with a formal approximation guarantee.
- *"Green transportation strategies and decarbonization of logistics networks"* — the whole framework is conditioned on Korea's K-ETS carbon-cost trajectory.

## What the cover letter should NOT claim

- Do **not** describe the RQ2–RQ5 overlap as an unconditional "four-way convergence." State it as: RQ2 ↔ RQ5 convergence is statistically significant (permutation p = 0.001); RQ3 and RQ4 point the same direction but are not independently significant at K = 5. This is a more defensible and, if anything, more interesting claim, because it shows the authors tested rather than assumed convergence.
- Do **not** lean on the "N = 95" augmented dataset size without immediately clarifying the effective independent N (19 stations × 5 scenario repeats).
- Do **not** present the RQ1 coverage-gap point estimates (60–68%) as a confirmed finding; they are explicitly a trend not excluding 50% at current N.

## Suggested emphasis order for the cover letter

1. Lead with the AI-trustworthiness contribution (exact SHAP + adversarially-validated deterministic verifier) since it is the CFP topic with the thinnest existing literature and the clearest methodological novelty here.
2. Follow with the honest-statistics framing across RQ1–RQ5 as a methodological-rigor signal — reviewers reward papers that pre-empt their own objections.
3. Close with the real-world policy relevance: a short, nameable station list (not a systemic redesign) that a policymaker could act on at low cost, grounded in real Korean network geography (Fig. 1).

## Realistic risk factors to flag internally (not for the cover letter)

- Data source is two personal GitHub repositories; a provenance manifest (commit SHA + SHA-256) mitigates but does not eliminate a reviewer's concern about independent verifiability of the underlying topology and scope-mapping decisions.
- N = 34 out-of-scope stations and N = 19 independent RQ4 labels are small by Q1-transportation-journal standards; this is disclosed rather than hidden, but disclosure does not remove the substantive limitation.
- The RQ4 transfer model is linear by design (to keep SHAP exact); a referee working in the "AI trustworthiness" topic area specifically may ask whether the audit approach generalizes to nonlinear/deep models. Have an answer ready (see limitations item 10) rather than being surprised by the question.
