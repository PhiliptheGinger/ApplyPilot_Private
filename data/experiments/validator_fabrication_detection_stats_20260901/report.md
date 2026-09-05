# Statistical Test: Fixed-Validator Fabrication-Detection Recall

Follow-up to the two new validator gaps found while verifying the earlier fabrication fixes:
1. **Bare fictional-role headers** with no parseable "Title at Company" (e.g. IterateCV's fabricated "ALIGNMENT TECHNIQUES" entry) weren't checked against the known-employer list at all.
2. **In-bullet job-posting-phrase injection** (grafting job vocabulary onto an otherwise correctly-attributed bullet, e.g. "utilizing data to inform comedic timing" on a Stand-up Comedian entry) wasn't detected by anything.

Both are now implemented in `validator.py`. This experiment measures how well the **complete, fixed validator** (all 6 fabrication checks together) actually performs, using a real, representative sample of the job data ApplyPilot processes — not 5 hand-picked examples.

## Method

Rather than running an LLM 200 times (slow, non-reproducible, and confounds "did the model fabricate" with "did the check catch it"), this injects **controlled, randomized-but-realistic** fabrications directly into resume data built from real profile evidence. The fabrication varies per job — drawing on that job's own real description text — so each of the 200 trials is a genuinely different instance, not the same example repeated 200 times. (A deterministic check run on identical input 200 times has zero sampling variance and would tell us nothing; varying the fabrication per real job is what makes this a meaningful set of independent trials.)

- **Population**: 27,338 real jobs in the live DB with a full description over 500 characters.
- **Sample**: 200 jobs, simple random sample, fixed seed (`20260901`) for reproducibility.
- **Clean baseline**: built *programmatically* from `profile.json`'s own authoritative `experience_inventory` (not any historical generated-resume file) — guaranteed free of the pre-existing placeholder-date/title-mismatch artifacts found in real historical resumes earlier in this investigation, which would otherwise confound the false-positive measurement with issues unrelated to this test.
- **Five conditions per job**: `clean` (no fabrication — measures false-positive rate), `skill` (a fabricated skill drawn from that job's own vocabulary, excluding generic category words), `date` (one entry's date shifted 3-12 years outside its real range), `bare_header` (a fabricated prior job using the target job's own title/company — the exact real-world pattern found twice in the earlier bake-off), `keyword_injection` (2-3 distinctive job-vocabulary words grafted onto an unrelated real bullet).
- **Detection measured precisely**: whether *that specific mechanism's own signal* fired (by message content), not just whether the overall result failed — `bare_header` and `keyword_injection` are deliberately WARNING-tier by design, so a raw pass/fail check would have missed them entirely (this happened during development — see Corrections below).
- **Confidence intervals**: Wilson score interval (more accurate than the normal approximation for proportions near 0% or 100%, which is exactly where a working validator should sit), computed at 95% and 99%.

## Results (n=200 per condition)

| Condition | Rate | 95% CI | 99% CI |
|---|---|---|---|
| Clean (false-positive rate) | 0/200 = **0.0%** | [0.0%, 1.9%] | [0.0%, 3.2%] |
| Fabricated skill | 200/200 = **100.0%** | [98.1%, 100.0%] | [96.8%, 100.0%] |
| Fabricated date | 200/200 = **100.0%** | [98.1%, 100.0%] | [96.8%, 100.0%] |
| Bare fictional-role/employer header | 200/200 = **100.0%** | [98.1%, 100.0%] | [96.8%, 100.0%] |
| Keyword injection | 199/200 = **99.5%** | [97.2%, 99.9%] | [95.9%, 99.9%] |

Zero false positives across 200 genuinely clean resumes, and every fabrication type caught at or above 99.5%, with the lower bound of every 99% CI staying above 95.9%.

## Corrections made during this test (documented, not glossed over)

1. **A real production bug, found only by this large-N test, not by any hand-picked example**: `validate_json_fields` joined the SKILLS dict's category values with a bare space (`" ".join(...)`). When a fabricated skill landed in a *different category* than the real ones and neither value ended in a delimiter, the two categories could merge into one claim (e.g. `{"Other": "Outreach", "Fabricated": "Salesforce"}` → `"outreach salesforce"`), which then falsely substring-matched the real "outreach" skill and silently exempted the fabrication riding along with it. First surfaced as a 0% catch rate on `skill` across all 200 trials. **Fixed**: join with `", "` instead. Every hand-picked test case up to this point happened to keep multi-item claims within a single, comma-separated category value, which is exactly why this had never surfaced before — the value of running a large, varied sample rather than trusting a handful of examples.
2. **A measurement bug in this test, not the validator**: the first pass measured "detected" as `not result["passed"]` (any error). `bare_header` and `keyword_injection` are deliberately WARNING-tier (advisory, not blocking, per the design precedent set by `check_claim_inflation`/`check_agency_inflation`), so they never fail validation on their own — the raw pass/fail metric showed 0% "catch rate" for both, which was wrong: manual inspection confirmed the specific warnings *were* firing correctly every time. Fixed by checking for each mechanism's own specific message text in errors *or* warnings, not just overall pass/fail.
3. **A test-harness quality issue**: the fabricated-skill picker initially chose generic category/qualifier words from job descriptions ("Language", "Technologies", "Proficient") that the validator's own stopword filtering correctly ignores as non-skill-like — a real fabrication is a plausible tool/tech name ("Databricks"), not a bare word like "Language". This produced a misleadingly low 96.5% skill catch rate that was really "the validator correctly ignored 7 words that don't look like skill claims," not "7 fabrications slipped through." Fixed by excluding the validator's own `_SKILL_CLAIM_STOPWORDS` from the picker; corrected rate is 100%.

## The one remaining miss (1/200, keyword_injection)

Job: "Application Support Engineer." The injected bullet ("Tires, utilizing troubleshooting and configure to support ancestry.") happened to include "troubleshooting" — a word that is *genuinely* grounded for the matched entry (Alignment Technician's own `relevance_categories` includes "hands-on troubleshooting"), so it doesn't count as injected. That left only 2 candidate words ("configure", "ancestry") against the check's `>= 2` threshold — right at the boundary, and one fell short after evidence-grounding was subtracted. This is expected, low-severity fuzziness in a deliberately heuristic, WARNING-tier natural-language check, not a structural gap — consistent with why it was designed as advisory rather than blocking in the first place.

## Conclusion

The fixed validator (all 6 fabrication checks, including the two implemented in this session) catches essentially all of the fabrication patterns found across this entire investigation, on a real, representative, randomly varied sample of the job data ApplyPilot actually processes — with zero false positives on genuinely clean resumes. The one miss and the three corrections made along the way are documented above rather than smoothed over; the corrections (one real production bug, two test-methodology bugs) are themselves evidence that testing against a large varied sample surfaces real issues a handful of hand-picked examples cannot.
