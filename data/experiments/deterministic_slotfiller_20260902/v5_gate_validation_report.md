# v5 gated synonym substitution — validation result

**Acceptance test (per plan):** the `category_tier == "prototype"` gate
(≥2 independently-matched `exact_keywords`) must block v4's two known-bad
hits — a car-repair sentence ("Diagnosed and corrected vehicle alignment
issues...") getting cited as evidence for unrelated software/product
requirements, purely because "issues"→"problems" is a safe word-level
synonym.

**Result: FAILED.** The gate blocked 0 of 3 attempted substitutions,
including both known-bad hits from v4, plus surfaced a third hit under the
same gate:

| Job | Requirement | exact_keywords | Gate tier |
|---|---|---|---|
| Forward Deployed Software Engineer, New Grad | "...use large scale data to solve valuable business problems" | `['problems', 'using']` | prototype |
| Principal Product Manager, Enterprise Applications | "...define pro[duct strategy]..." | `['alignment', 'problems']` | prototype |
| Product Support Specialist | "...minimize risk of unexpected issues..." | `['customer', 'issues', 'problems']` | prototype |

**Root cause:** the gate assumed "≥2 matched keywords" means "independently,
meaningfully corroborated." In practice, one of the two (or three) matched
keywords is often itself a near-contentless connector word — "using" (matched
because both the job posting and the evidence sentence happen to contain the
word "using," which carries no topical information) or "alignment" (matched
literally by coincidence, not because either job is actually about vehicle
alignment). Requiring 2 keyword hits instead of 1 doesn't filter out weak
matches when either hit can itself be low-information. This is the same
underlying problem behind the whole night's `match_score` findings (mean
1.5 keywords available, generic scaffolding words like "using"/"information"
dominating the exact_keywords pool) applied one level up: the corroboration
gate is built from the same low-information signal it's supposed to guard
against.

**Decision:** synonym substitution (v4 ungated, v5 gated) is **not**
promoted further. Both designs, independently, converge on the same failure
mode against real data. Not treating this as "needs a smarter gate" and
iterating further tonight — a keyword-count threshold is structurally the
wrong tool here regardless of the threshold value, since it can't
distinguish a specific/informative keyword from a generic one without
something like inverse-document-frequency weighting, which is materially
more scope than this feature justified. Inflection-tolerant matching
(Phase 1, shipped to production) remains the validated, safe win from this
line of investigation.
