# v6 IDF-weighted gate — validation result

**Setup:** computed real IDF weights from 30,184 job postings in the local
DB (`compute_idf_weights.py`). Spot-check confirmed the weights behave as
expected: `the`=1.01, `and`=1.00 (true stopwords), `using`=1.81,
`customer`=1.80 (generic), `issues`=1.96, `problems`=2.11 (moderate),
`troubleshooting`=2.48, `python`=2.68, `alignment`=3.20 (specific). Gated
substitution on the sum of matched keywords' IDF weights instead of a raw
count, calibrated across 5 thresholds against the same 100-job sample and
v5's 3 known hits.

| threshold | applied | blocked | known-bad still through |
|---|---|---|---|
| 3.0 | 3 | 0 | 3/3 |
| 4.0 | 2 | 1 | 2/3 |
| 5.0 | 2 | 1 | 2/3 |
| 6.0 | 0 | 3 | 0/3 |
| 7.0 | 0 | 3 | 0/3 |

**Result: no threshold both blocks every known-bad hit AND preserves any
yield.** At 6.0+, all 3 known-bad hits are blocked — but so is everything
else; applied drops to 0. At 4.0–5.0, IDF correctly blocks the one hit that
was purely a generic-connector-word problem (`problems`+`using`,
weighted=3.91) — proving the mechanism works exactly as designed for that
case — but the other two hits (`alignment`+`problems`, weighted=5.30;
`customer`+`issues`+`problems`, weighted=5.87) still get through at a
*higher* score than the case IDF successfully blocks.

**Root cause, and why it's not a threshold-tuning problem:** `alignment`
has HIGH global IDF (3.20) not because it's a rare, specific signal, but
because it's a **polysemous word used in two unrelated narrow domains** —
"strategic/stakeholder alignment" (common corporate jargon in the Product
Manager posting) and "wheel alignment" (the candidate's actual automotive
evidence). IDF measures corpus-wide rarity; it has no mechanism to detect
that a rare word is rare *because it means two different things in two
different contexts*, which is exactly the failure case here. Raising the
threshold enough to also catch this collision blocks every other match too
— because in this dataset, there is no case where a synonym-substitution
opportunity coincides with two keywords that are BOTH genuinely specific
AND used in the same sense as the candidate's evidence.

**Decision: close out word-level synonym substitution as a line of
investigation.** This is the third independently-tried gate design
(ungated v4, raw-count v5, IDF-weighted v6) and the third failure against
real data — two blocked nothing useful, this one only partially worked and
revealed a deeper problem (word-sense ambiguity) that no simple corpus
statistic fixes. Not attempting a fourth gate design without new
information; the honest conclusion is that this specific feature doesn't
have a real, positive-yield use case in this dataset, gate design aside.
Inflection-tolerant matching (shipped, Phase 1) remains the one validated
win from the whole "widen keyword matching" investigation.

**Separate, real byproduct worth keeping:** `idf_weights.json` and
`compute_idf_weights.py` are still a reusable artifact — same IDF weights
could reasonably improve `schemas.classify_category_tier`'s prototype/
near_prototype split generally (a bigger, unvalidated change, out of scope
here) or other places raw keyword counting currently stands in for
"how corroborated is this match." Not pursuing that now given three
consecutive negative results on the narrower feature this was built for —
flagging as a separate, later idea, not a natural next step from this one.
