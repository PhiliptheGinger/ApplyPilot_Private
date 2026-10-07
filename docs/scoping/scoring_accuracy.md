# Scoring accuracy: requirement framing (FW28, FW15, FW14) and stand-up (FW21)

Scoped 2026-10-07.

## FW28 / FW14 / FW15 — "is this phrasing a hard requirement?"

### Today

`deterministic_fallback.extract_years_required` decides whether a years
mention is a requirement by looking for a small set of qualifier words near
it: `_REQUIRED_CONTEXT_RE = required | must have | minimum of`, plus a
tightly scoped "minimum N years" check, plus section-header handling
(decisions #82, #86–92). Every new phrasing ("Seeking 10+ years",
"commercial experience", "Founding Engineer") has been a one-off regex
addition with its own false-positive scare (#87, #90, #92).

### Proposal: classify framing, don't enumerate it

Reuse the two pieces the codebase already trusts:

1. **Frame-style classification** (the shape of `schemas.select_frame`):
   classify the *sentence* that contains a years/degree mention into
   `hard_requirement | preferred | descriptive | unrelated` from several
   weak signals rather than one keyword:
   - modal strength: required/must/need/minimum/seeking/looking for (hard) vs.
     preferred/nice/bonus/plus/ideally (soft);
   - section context: nearest header is Requirements/Qualifications vs.
     Preferred/Bonus vs. About us/Benefits;
   - subject: "you have / candidates with / we're seeking someone with"
     (about the applicant) vs. "we have / our team has / founded in"
     (about the company — the #92 "About [Company]" false positive);
   - list membership: a bullet inside a requirements list.
2. **Claim-tier lattice turned around** (FW29's top idea): score the
   posting's own verbs (architect, own, lead, set direction) with the
   existing `CLAIM_TIERS` to produce an *implicit seniority* signal for
   postings with no explicit years — the "Founding Engineer"/research-role
   gap in FW15(a)/(b).

Rule: the classifier only ever *adds* a cap or a requirement when the
signals agree; disagreement leaves today's behavior unchanged. That keeps
it from regressing the ~1000 Claude-direct audited scores.

### Validation plan (needs the live DB)

- Re-run the classifier over the 1,000+ `claude_direct` jobs (decisions
  #94–128) and require ≥ 98% agreement with the audited years/degree
  outcomes before shipping.
- Sample 50 sentences each for "seeking", "commercial experience", and
  "founding" from the corpus and hand-label them.

### FW15(c) — defense / law-enforcement contractors

A small, high-precision keyword set in the *company description*
("defense", "DoD", "national security", "warfighter", "law enforcement
agencies") gated on appearing in an "About us" paragraph, not a client
list. Also needs a corpus false-positive check (defense keywords appear in
cybersecurity postings that serve banks).

**Where:** classifier code and unit tests are cloud-ok; both validation
passes need the real DB.

## FW21 — is stand-up's title regex still needed?

The test can't be run fairly yet: the Standup `qualifications` entry isn't
wired into evidence matching at all (decision #136), and until 2026-09-15 it
had no prose of its own. Prerequisite: move qualifications into
`rank_profile_evidence` like the other inventories. Then compare inclusion
on the same 15+15 real jobs decision #144 used. Size M, needs the live
profile and DB.

## Decision needed

- OK to start FW28 as an *additive* classifier gated on signal agreement,
  validated against the Claude-direct audit set before it changes any
  score?
