"""Cognitive-linguistic schema library for resume tailoring and cover letters.

Computes ONE structured representation per job -- job requirements, ranked
candidate evidence (reusing local_tailor.py's deterministic grounding), and
a compositional cognitive-linguistic analysis of each SUPPORTED requirement
-- then renders it as a compact guidance block that both tailor.py and
cover_letter.py prepend to their existing prompts.

Design constraints (2026-08-23 architecture note, updated 2026-08-24):
  - Zero LLM calls anywhere in this module. Everything here is deterministic
    Python over already-computed data (local_tailor.py's requirement
    extraction / evidence ranking / auto-resolution). This keeps the
    routing decision intact: scoring stays cloud-only, and this module adds
    no new local-model OR cloud-model traffic -- it only gives the cloud
    model a more structured prompt for calls it was already going to make.
  - A schema is only ever assigned to a requirement that is already
    SUPPORTED by grounded evidence (see build_job_schema_representation).
    Ambiguous/unresolved requirements get no schema and are omitted from
    the rendered guidance -- this module never upgrades an uncertain match
    into a confident-looking rhetorical structure.
  - The per-job representation is cached in-process (keyed by job URL +
    description hash) so tailor and cover-letter generation for the same
    job in the same run share one computation instead of two.
  - This module does NOT replace local_tailor.py's grounding logic (the
    _pair_candidate_evidence / _auto_resolve_requirements thresholds are
    untouched) -- it only reads the same deterministic primitives and adds
    a rhetorical-structure layer on top, read-only.

2026-08-24 architecture expansion
----------------------------------
Investigated ~25 cognitive-linguistic systems (frame semantics, event
structure, causal structure, force dynamics, image schemas, binding/role,
prototype theory, domain-transfer typing, viewpoint, figure-ground,
profiling, granularity, salience, information structure, construction
grammar, conceptual metaphor, metonymy, blending, claim-strength/
entailment, discourse coherence, construal operations). Several of these
turned out to be redundant relabelings of the SAME extractable signal (a
migration's "SOURCE-PATH-GOAL" image schema *is* frame=migration + an
event transition; force dynamics *is* frame + causal structure under a
different vocabulary) -- implementing them as separate parallel fields
would be indirection, not new information, so they were deliberately
deferred (documented at the bottom of this docstring) rather than
implemented. What's implemented below are the systems found to add
genuinely orthogonal, deterministically-extractable, testable information:

  FRAME            -- compact situational ontology (~9 frames), generalizes
                       the old single-purpose troubleshooting regex.
  EVENT TYPE        -- accomplishment (has a stated outcome) vs. activity
                       (ongoing responsibility, no stated outcome).
  CATEGORY TIER     -- graded prototype scale (prototype / near_prototype /
                       peripheral / unsupported), replacing the old binary
                       literal/synonym match_kind with an honest 4-way
                       scale grounded in actual matched-term counts.
  CLAIM STRENGTH    -- THE centerpiece safety layer: a 5-tier verb lattice
                       (participation < execution < implementation < design
                       < authority). Each supported requirement gets a
                       ceiling derived from the EVIDENCE'S OWN description
                       text (a human already vetted that wording), and
                       check_claim_strength() is a deterministic POST-
                       generation check callers apply to realized text --
                       see local_tailor.request_local_realization for where
                       this is actually enforced (drops any bullet whose
                       realized verb exceeds its evidence's ceiling).
  VIEWPOINT         -- one coarse, job-title-derived professional lens per
                       job (engineering / data / operations / support /
                       leadership / general) -- the same underlying
                       evidence can be legitimately presented from a
                       different angle depending on the target role, with
                       zero change to the underlying facts.
  SALIENCE ORDER    -- merges figure-ground / profiling / information-
                       structure into one ranking of which conceptual
                       element (outcome / mechanism / technology / context)
                       should open the realized sentence.
  CONSTRUCTION      -- BULLET_SCHEMAS entries now carry a concrete example
                       `pattern`, not just abstract slot names.
  DISCOURSE         -- flag_repetition() marks requirements whose (frame,
                       cognitive_schema) pair repeats 3+ times, so the
                       realizer is nudged toward lexical variety rather
                       than every bullet opening identically.
  PROVENANCE        -- each requirement's representation now carries the
                       literal evidence source text it derives from,
                       persisted for audit rather than just the evidence
                       item's name.

Deliberately DEFERRED (with rationale, not merely omitted):
  - Force dynamics: re-describes frame+event+causal structure under a
    different vocabulary; no additional signal is deterministically
    extractable from our evidence data beyond what FRAME+EVENT TYPE +
    CLAIM STRENGTH already capture.
  - Image schemas (SOURCE-PATH-GOAL, BLOCKAGE-REMOVAL-FLOW, etc.): same
    redundancy -- a migration's image schema IS frame=migration plus an
    event-structure transition. A parallel tag would be relabeling.
  - Conceptual metaphor: explicitly internal-only per its own spec, and
    for the same reason as image schemas, collapses into FRAME + EVENT.
  - Metonymy / reference-point as a separate structure: this is exactly
    the JUSTIFICATION for the claim-strength ceiling (bounding "Python"
    to what the evidence's own verb supports), not new information beyond
    it.
  - Blending / conceptual integration as a separate structure: "structure
    transfers, facts don't" is exactly what CATEGORY TIER + the
    domain_transfer cognitive schema already enforce.
  - Granularity/specificity as a separate system: materially covered
    already by exact_keywords vs. synonym_concepts (use the job's own
    term when the match is literal; state the transferable principle
    otherwise) -- a distinct "zoom level" field would mostly restate that.
  - Construal operations (FOREGROUND/ZOOM_IN/SHIFT_VIEWPOINT/...) as an
    explicit named-operation layer: their effects are exactly what
    VIEWPOINT/SALIENCE/FRAME compute directly. Achieving the same
    functional outcome via direct computation is simpler than adding an
    operation-application indirection layer on top.

Public API
----------
  build_job_schema_representation(job, profile) -> dict
      The full computation: requirements (each with frame/event_type/
      category_tier/claim_ceiling/salience_order/provenance/schema/
      lexical anchors), a job-level viewpoint, and a selected summary
      schema. No caching.

  get_or_build_job_schema(job, profile) -> dict
      Cached wrapper -- the one callers (tailor.py, cover_letter.py) should
      actually use.

  format_schema_guidance(representation, max_requirements=8) -> str
      Render a representation as a compact prompt section. Returns "" when
      there is nothing supported to show, so callers can `if guidance:`.

  check_claim_strength(realized_text, ceiling) -> dict
      Deterministic post-generation safety check -- see local_tailor.py's
      request_local_realization for its enforcement call site.

  clear_schema_cache() -> None
      Test/debug helper -- drops the in-process cache.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from collections import Counter

from applypilot.scoring.local_tailor import (
    _auto_resolve_requirements,
    _split_requirement_lines,
    _synonym_hit,
    _term_in_text,
    rank_profile_evidence,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Cognitive/event schemas -- reusable conceptual structures, independent
#    of output format (a resume bullet and a cover-letter sentence can both
#    realize the same cognitive schema).
# ---------------------------------------------------------------------------

COGNITIVE_SCHEMAS: dict[str, dict] = {
    "action_object_outcome": {
        "description": "Direct, literally-matched skill: did X, on Y, producing Z.",
        "slots": ["action", "object", "outcome"],
    },
    "problem_diagnosis_repair_verification": {
        "description": (
            "Troubleshooting/support work: something was wrong, it was investigated, fixed, and confirmed working."
        ),
        "slots": ["problem", "diagnosis", "repair", "verification"],
    },
    "evidence_claim": {
        "description": "A factual capability grounded directly in one verified evidence item.",
        "slots": ["evidence", "claim"],
    },
    "domain_transfer": {
        "description": (
            "Experience from one domain expressed as a transferable principle "
            "applicable to the target domain -- never claims the other "
            "domain's job title or context as if it were the same job."
        ),
        "slots": ["source_domain_experience", "transferable_principle", "target_relevance"],
    },
    "prevention": {
        "description": (
            "A risk or failure mode was identified and averted BEFORE it "
            "occurred (force-dynamics: an opposing force was prevented from "
            "taking effect) -- distinct from problem_diagnosis_repair_"
            "verification, which assumes the failure already happened."
        ),
        "slots": ["risk", "intervention", "prevented_outcome"],
    },
}

# ---------------------------------------------------------------------------
# 2. Resume/discourse schemas -- reusable document/bullet structures.
#    Each now carries a concrete example `pattern` (construction grammar,
#    #18) alongside its abstract slot list, so the realizer sees a real
#    sentence shape, not just slot names.
# ---------------------------------------------------------------------------

BULLET_SCHEMAS: dict[str, dict] = {
    "action_object_context_outcome": {
        "description": "Default bullet shape for a direct, literally-matched requirement.",
        "slots": ["action", "object", "context", "outcome"],
        "pattern": "[Action] [object] [context], [outcome].",
        "discourse_function": "foreground candidate agency and concrete technical capability",
    },
    "problem_action_result": {
        "description": "Bullet shape for troubleshooting/support/incident-resolution evidence.",
        "slots": ["problem", "action", "result"],
        "pattern": "[Diagnosed/identified problem], [action taken], [verified result].",
        "discourse_function": "foreground diagnostic competence",
    },
    "domain_transfer_bullet": {
        "description": (
            "Bullet shape for evidence matched via a synonym/paraphrase, not a "
            "literal term -- states the transferable principle rather than "
            "implying identical context or domain."
        ),
        "slots": ["source_experience", "transferable_principle", "target_relevance"],
        "pattern": "[Source experience], applying [transferable principle] relevant to [target context].",
        "discourse_function": "foreground legitimate transferable capability without overclaiming domain fit",
    },
    "prevention_intervention_result": {
        "description": (
            "Bullet shape for evidence describing a risk/failure that was "
            "averted BEFORE it happened, not one that was fixed after the "
            "fact -- see force_relation. Distinct from problem_action_result, "
            "which assumes something already broke."
        ),
        "slots": ["risk", "intervention", "prevented_outcome"],
        "pattern": "Identified [risk] and [intervention], preventing [outcome that would otherwise have occurred].",
        "discourse_function": "foreground reliability and proactive risk management",
    },
}

SUMMARY_SCHEMAS: dict[str, dict] = {
    "identity_domain_strength_evidence_value": {
        "description": (
            "Used when 2+ requirements are directly (literally) supported -- "
            "lead with concrete, already-demonstrated strength."
        ),
        "slots": ["identity", "domain", "strength", "evidence", "target_value"],
    },
    "capability_background_transfer_value": {
        "description": (
            "Used when support is thinner or mostly transferable/synonym "
            "matches -- lead with transferable capability rather than "
            "overclaiming direct domain fit."
        ),
        "slots": ["current_capability", "relevant_background", "transferable_strength", "employer_value"],
    },
}

# ---------------------------------------------------------------------------
# 3. Frame semantics (#A) -- compact situational ontology. Generalizes the
#    old single-purpose _TROUBLESHOOTING_HINTS regex (kept as the
#    diagnostic_support frame's hints, unchanged wording) into a small,
#    extensible set. Checked in a fixed priority order below so a more
#    specific frame wins over a coincidental generic-capability match.
# ---------------------------------------------------------------------------

FRAMES: dict[str, dict] = {
    "diagnostic_support": {
        "description": "Something was wrong; it was investigated, fixed, and confirmed working.",
        "slots": ["problem", "investigation", "cause", "intervention", "verification"],
        "hints": (
            "troubleshoot",
            "diagnos",
            "incident",
            "repair",
            "root cause",
            "root-cause",
            "resolve",
            "resolution",
            "technical support",
            "help desk",
            "support ticket",
            "outage",
        ),
    },
    "deployment": {
        "description": "A software/infrastructure change was moved into a target environment.",
        "slots": ["artifact", "environment", "method", "result"],
        "hints": ("deploy", "release", "rollout", "provision", "ci/cd", "continuous integration"),
    },
    "automation": {
        "description": "A manual/repetitive process was replaced or reduced by a programmatic one.",
        "slots": ["manual_process", "mechanism", "result"],
        "hints": ("automat", "script", "manual process", "streamlin", "eliminate manual"),
    },
    "migration": {
        "description": "A system or process moved from one state/platform to another.",
        "slots": ["source_state", "target_state", "method"],
        "hints": ("migrat", "upgrad", "moderniz", "transition", "port to", "cutover"),
    },
    "development": {
        "description": "New functionality or software was built.",
        "slots": ["artifact", "method", "result"],
        "hints": ("develop", "build", "implement", "programming", "software engineer"),
    },
    "analysis": {
        "description": "Data or information was examined to produce a finding or decision.",
        "slots": ["input", "method", "finding"],
        "hints": ("analy", "report", "metric", "dashboard", "insight", "data-driven"),
    },
    "leadership_project": {
        "description": "Coordinated people or an initiative toward a defined outcome.",
        "slots": ["initiative", "scope", "result"],
        "hints": ("led ", "lead ", "manage", "coordinat", "mentor", "supervis", "own the"),
    },
    "customer_support": {
        "description": "Direct interaction with a user or customer to address a stated need.",
        "slots": ["user", "need", "response"],
        "hints": ("customer", "client", "user support", "inquiries", "service desk"),
    },
    "security": {
        "description": "Protecting a system, data, or process against unauthorized access or risk.",
        "slots": ["asset", "threat", "control", "outcome"],
        "hints": ("security", "vulnerabilit", "compliance", "access control", "encrypt", "audit"),
    },
    "optimization": {
        "description": "Existing performance/efficiency was measured and improved without changing what the system does.",
        "slots": ["target", "bottleneck", "change", "improvement"],
        "hints": ("optimiz", "performance", "latency", "throughput", "efficien", "cost reduction"),
    },
    "monitoring": {
        "description": "Ongoing observation of a system's health or behavior to detect issues early.",
        "slots": ["system", "signal", "threshold", "response"],
        "hints": ("monitor", "alert", "observability", "dashboard", "on-call", "sla"),
    },
    "integration": {
        "description": "Two or more systems/services were connected so they work together.",
        "slots": ["system_a", "system_b", "interface", "result"],
        "hints": ("integrat", "api", "webhook", "connector", "interoperab", "third-party"),
    },
    "training_mentorship": {
        "description": "Transferred knowledge or skill to others, distinct from managing their work output.",
        "slots": ["learner", "content", "method"],
        "hints": ("train", "mentor", "onboard", "teach", "coach", "knowledge transfer", "documentation for"),
    },
    "general_capability": {
        "description": "Fallback frame -- no more specific frame's hints matched this requirement.",
        "slots": ["action", "object", "result"],
        "hints": (),
    },
}

# Priority order for frame selection -- more specific frames checked first
# so e.g. "automate the deployment pipeline" resolves to automation (the
# requirement's actual point) rather than deployment (an incidental word).
_FRAME_PRIORITY = (
    "diagnostic_support",
    "security",
    "monitoring",
    "automation",
    "integration",
    "migration",
    "deployment",
    "optimization",
    "analysis",
    "leadership_project",
    "training_mentorship",
    "customer_support",
    "development",
)


def select_frame(requirement_text: str) -> str:
    """Deterministically pick a frame for one requirement's text. Always
    returns a valid FRAMES key -- "general_capability" when nothing more
    specific matches."""
    text_lower = requirement_text.lower()
    for frame_name in _FRAME_PRIORITY:
        hints = FRAMES[frame_name]["hints"]
        if any(h in text_lower for h in hints):
            return frame_name
    return "general_capability"


# ---------------------------------------------------------------------------
# 4. Event structure (#5) -- accomplishment vs. activity (simplified
#    Vendler-style distinction). An "accomplishment" evidence description
#    states a concrete outcome/completion; an "activity" describes ongoing
#    responsibility with no stated result. Feeds both salience ordering
#    (accomplishments profile their outcome first) and, implicitly, claim
#    safety (an "activity" shouldn't be realized as if it concluded with a
#    result the evidence never states).
# ---------------------------------------------------------------------------

# Explicit verb-form list (base/past/-ing/3rd-person), same precision
# rationale as _CLAIM_VERB_PATTERNS below -- a loose `\w*` stem would
# false-positive on unrelated words.
_ACCOMPLISHMENT_VERB_RE = re.compile(
    r"\b(reduced|reducing|reduces|"
    r"increased|increasing|increases|"
    r"improved|improving|improves|"
    r"eliminated|eliminating|eliminates|"
    r"resolved|resolving|resolves|"
    r"delivered|delivering|delivers|"
    r"completed|completing|completes|"
    r"launched|launching|launches|"
    r"cut|cuts|cutting|"
    r"saved|saving|saves|"
    r"automated|automating|automates|"
    r"migrated|migrating|migrates|"
    r"built|building|builds|"
    r"shipped|shipping|ships|"
    r"deployed|deploying|deploys|"
    r"achieved|achieving|achieves|"
    r"grew|growing|grows|"
    r"decreased|decreasing|decreases|"
    r"restored|restoring|restores|"
    r"fixed|fixing|fixes|"
    r"repaired|repairing|repairs|"
    r"solved|solving|solves|"
    r"corrected|correcting|corrects|"
    r"verified|verifying|verifies)\b",
    re.IGNORECASE,
)


def classify_event_type(evidence_text: str) -> str:
    """ "accomplishment" if the evidence's own text contains a completion/
    outcome verb, "activity" otherwise (ongoing responsibility, nothing
    stated as concluded)."""
    if evidence_text and _ACCOMPLISHMENT_VERB_RE.search(evidence_text):
        return "accomplishment"
    return "activity"


# Force dynamics (#C), narrowly scoped: everything else evidence describes
# assumes a problem already happened and was then fixed (that's what
# diagnostic_support + the accomplishment verbs above already capture).
# "Prevention" is the one genuinely distinct force-dynamic relation with no
# existing construction -- an opposing force (a risk/failure mode) was
# averted BEFORE it took effect. Deliberately NOT a full force-dynamics
# system (resistance/control/external-pressure/etc. were evaluated and
# rejected -- see schemas.py's module docstring -- they all reduce to
# frame + event_type under a different vocabulary; only prevention doesn't).
_PREVENTION_RE = re.compile(
    r"\b(prevented|preventing|prevents|"
    r"avoided|avoiding|avoids|"
    r"averted|averting|averts|"
    r"mitigated|mitigating|mitigates)\b",
    re.IGNORECASE,
)


def detect_force_relation(evidence_text: str) -> str | None:
    """Returns "prevention" if the evidence's own text describes averting a
    risk/failure before it occurred, else None. None is the overwhelmingly
    common case -- this is a narrow detector, not a general classifier."""
    if evidence_text and _PREVENTION_RE.search(evidence_text):
        return "prevention"
    return None


# ---------------------------------------------------------------------------
# 5. Category tier (#10) -- graded prototype scale, replacing the old
#    binary literal/synonym match_kind with an honest 4-tier scale grounded
#    in data we already compute (exact-keyword count). "transferable_
#    analogue" (a deeper cross-domain STRUCTURAL similarity, e.g. warehouse
#    workflow optimization <-> deployment pipeline optimization) is a real
#    theoretical category but NOT implemented as a distinct automatic tier
#    here -- we have no reliable deterministic signal to distinguish it
#    from an ordinary peripheral/synonym match without real semantic
#    parsing, and faking that precision would be worse than not claiming
#    it. Synonym matches are conservatively tiered "peripheral".
# ---------------------------------------------------------------------------

CATEGORY_TIERS = ("prototype", "near_prototype", "peripheral", "unsupported")


def classify_category_tier(match_kind: str, exact_keyword_count: int) -> str:
    """Deterministic category-membership tier for one requirement/evidence
    pairing. match_kind is local_tailor's own literal/synonym/semantic/
    unknown classification (see _match_kind below)."""
    if match_kind == "literal":
        return "prototype" if exact_keyword_count >= 2 else "near_prototype"
    if match_kind == "synonym":
        return "peripheral"
    if match_kind == "semantic":
        # 2026-08-31: raw embedding-similarity provenance (local_tailor's
        # semantic_match.py) is deliberately never eligible for prototype/
        # near_prototype (literal) OR peripheral (curated synonym) --
        # real-data validation found both raw cosine similarity and the
        # existing Qwen3 arbitration prompt unreliable at distinguishing
        # genuine semantic matches from false positives (see
        # local_tailor._auto_resolve_requirements' docstring for the
        # specific evidence). Stays "unsupported" -- the same conservative
        # outcome an unrecognized match_kind already got -- until some
        # future, explicitly-verified confirmation step exists to promote
        # it no higher than "peripheral", never "prototype"/"near_prototype".
        return "unsupported"
    return "unsupported"


# ---------------------------------------------------------------------------
# 6. Claim-strength lattice (#22) -- THE central safety layer.
#
# A 5-tier ordinal scale of what an action verb actually claims. The
# CEILING for a given piece of evidence is derived from the verb ALREADY
# PRESENT in that evidence's own description/factual_concepts text in
# profile.json -- a human already vetted that wording, so it's ground
# truth, not something re-derived from job-relevance or schema choice.
# check_claim_strength() is the deterministic POST-generation check: does
# the REALIZED text use a verb stronger than the evidence supports? Never
# used to grant permission to claim MORE -- only ever to catch and reject
# claiming more. See local_tailor.request_local_realization for the actual
# enforcement (drops any bullet that fails this check).
# ---------------------------------------------------------------------------

CLAIM_TIERS = ("participation", "execution", "implementation", "design", "authority")
_CLAIM_TIER_RANK = {tier: i for i, tier in enumerate(CLAIM_TIERS)}

# Explicit verb-form lists (base/past/-ing/3rd-person), not loose `\w*`
# stems -- a stem like `us\w*` would false-positive on "user"/"usable"/
# "usually". Verbose but precise: only the exact forms are matched.
_CLAIM_VERB_PATTERNS: dict[str, re.Pattern] = {
    "participation": re.compile(
        r"\b(used|using|uses|utilized|utilizing|utilizes|utilised|utilising|"
        r"worked with|working with|assisted|assisting|assists|"
        r"supported|supporting|supports|"
        r"participated in|participating in|participates in|"
        r"helped|helping|helps|"
        r"contributed to|contributing to|contributes to|"
        r"exposure to|familiar with|learned|learning|knowledge of|"
        # 2026-09-03: added while wiring real experience_inventory
        # `responsibilities` text into ceiling detection (see
        # _evidence_own_text) -- these communication/collaboration verbs
        # are common in non-technical resume evidence (customer service,
        # sales, hands-on trade roles) and match the existing
        # participation-tier character (worked with/assisted/supported).
        r"explained|explaining|explains|"
        r"coordinated|coordinating|coordinates|"
        r"communicated|communicating|communicates)\b",
        re.IGNORECASE,
    ),
    "execution": re.compile(
        r"\b(configured|configuring|configures|"
        r"operated|operating|operates|"
        r"ran|running|runs|"
        r"maintained|maintaining|maintains|"
        r"deployed|deploying|deploys|"
        r"applied|applying|applies|"
        r"administered|administering|administers|"
        r"monitored|monitoring|monitors|"
        r"executed|executing|executes|"
        # 2026-09-03: same real-data pass as above -- procedural,
        # hands-on-trade verbs (diagnosing/fixing a specific issue via an
        # established procedure) match this tier's existing character
        # (configured/operated/maintained/applied), one step below
        # implementation's "built something new".
        r"diagnosed|diagnosing|diagnoses|"
        r"corrected|correcting|corrects|"
        r"performed|performing|performs|"
        r"installed|installing|installs|"
        r"prepared|preparing|prepares)\b",
        re.IGNORECASE,
    ),
    "implementation": re.compile(
        r"\b(implemented|implementing|implements|"
        r"built|building|builds|"
        r"developed|developing|develops|"
        r"created|creating|creates|"
        r"automated|automating|automates|"
        r"integrated|integrating|integrates|"
        r"wrote|writing|writes|"
        r"coded|coding|codes|"
        r"delivered|delivering|delivers)\b",
        re.IGNORECASE,
    ),
    "design": re.compile(
        # "engineers"/"architects" as bare -s forms deliberately excluded --
        # both are far more often the PLURAL-NOUN job title ("worked with
        # senior engineers") than the rare 3rd-person-singular verb, and a
        # false positive here wrongly inflates the ceiling this ONE tier
        # below authority gates; "designed/designing" and "engineered"
        # already cover the realistic past-tense/gerund verb phrasing
        # resumes actually use.
        #
        # "engineering" the bare gerund is ALSO excluded (2026-09-02) --
        # found via the deterministic slot-filler prototype scanning real
        # profile.json factual_concepts: "Feature Engineering" (a data-
        # science technique NAME, from CAP Predictor's factual_concepts)
        # matched this pattern and wrongly elevated that evidence's claim
        # ceiling to "design", the same tier "architected" gates just
        # below. Same noun/verb-collision class as the -s-form exclusion
        # above -- "X engineering" (feature/data/software/prompt/requirements
        # engineering) is overwhelmingly a field/discipline name in resume
        # and job-posting text, not a first-person action verb. "engineered"
        # (unambiguous past tense) has no comparable collision and stays.
        r"\b(designed|designing|"
        r"engineered|"
        r"planned|planning|plans)\b",
        re.IGNORECASE,
    ),
    # 2026-08-24: this tier used to also include led/managed/owned/directed/
    # spearheaded/drove -- conflating individual TECHNICAL authority
    # (architecting a system solo) with PEOPLE authority (leading a team),
    # which are genuinely different claims (see the AGENCY axis below).
    # Evidence supporting "architected X" should not silently license "led
    # the team that architected X" -- that's a people-authority claim this
    # axis was never designed to validate. The people-verbs moved to
    # _AGENCY_VERB_PATTERNS.
    # "architects" (bare -s form) deliberately excluded -- same
    # noun/verb-collision risk as "engineers" above, and this is the
    # HIGHEST tier, so a false positive here is the most dangerous case:
    # evidence text merely mentioning "senior architects" as colleagues
    # would wrongly earn this evidence an authority-tier ceiling.
    "authority": re.compile(r"\b(architected|architecting)\b", re.IGNORECASE),
}


def detect_claim_tier(text: str) -> str | None:
    """Return the HIGHEST claim-tier verb found anywhere in `text`, or None
    if no recognized claim verb appears at all (e.g. a noun-phrase
    fragment with no verb, or vocabulary this lattice doesn't cover)."""
    if not text:
        return None
    found = [tier for tier, pattern in _CLAIM_VERB_PATTERNS.items() if pattern.search(text)]
    if not found:
        return None
    return max(found, key=lambda t: _CLAIM_TIER_RANK[t])


def _evidence_own_text(evidence_item: dict) -> str:
    """The full text an evidence item's own record actually contains,
    joining description + factual_concepts + responsibilities.

    2026-09-03: `responsibilities` was missing here even though tailor.py's
    canonical-inventory builder and validator.py's vocabulary-injection
    check both already fall back to it (a real, already-authored field --
    several experience_inventory entries, e.g. "National Tire and Battery /
    Mavis", have description=None and factual_concepts=None but a real,
    verb-bearing `responsibilities` list, e.g. "Diagnosed and corrected
    vehicle alignment issues..."). Without this, claim/agency-tier
    detection and event-type/force-relation classification below were
    silently blind to that text and defaulted every such entry to the
    lowest tier -- not because the evidence was thin, but because this one
    function never looked at the field where it actually lives. Single
    shared helper so claim/agency ceiling and the build_job_schema_
    representation evidence_text used for provenance/event_type/
    force_relation all read the same fields, rather than three
    independently-maintained field lists drifting apart again.
    """
    parts = [evidence_item.get("description")]
    concepts = evidence_item.get("factual_concepts")
    if concepts:
        parts.append(" ".join(str(c) for c in concepts))
    responsibilities = evidence_item.get("responsibilities")
    if responsibilities:
        parts.append(" ".join(str(r) for r in responsibilities))
    return " ".join(p for p in parts if p)


def claim_ceiling_for_evidence(evidence_item: dict) -> str:
    """The strongest claim tier this evidence item's OWN text supports.
    Defaults to the most conservative tier ("participation") when no
    recognized verb is found -- silence in the evidence is never license
    to claim more, only ever a reason to claim less.

    Secondary signal (#L, metonymy/domain compression): a bare skill
    mention ("Python") typically has no description at all -- correctly
    defaults to "participation" (the resume shouldn't imply more than
    "mentioned/used" for an unelaborated skill). `evidence_level`/
    `proficiency` free-text (skills_inventory's own fields) is checked as
    a fallback ONLY when the description/factual_concepts/responsibilities
    gave no verb -- an own-text verb always wins when present, since it's
    the more specific, more directly-authored signal.
    """
    text = _evidence_own_text(evidence_item)
    tier = detect_claim_tier(text)
    if tier:
        return tier
    level_text = " ".join(str(evidence_item.get(k) or "") for k in ("evidence_level", "proficiency"))
    return detect_claim_tier(level_text) or "participation"


def check_claim_strength(realized_text: str, ceiling: str) -> dict:
    """Deterministic post-generation check: does `realized_text` use a
    claim verb stronger than `ceiling` permits?

    Returns {"passed": bool, "detected_tier": str | None, "ceiling": str,
    "violation": str | None}. `detected_tier` is None (and passed=True)
    when the realized text uses no recognized claim verb at all -- this
    lattice only rejects OVERCLAIMING, it never requires a claim verb to
    be present.
    """
    detected = detect_claim_tier(realized_text)
    if detected is None:
        return {"passed": True, "detected_tier": None, "ceiling": ceiling, "violation": None}
    if _CLAIM_TIER_RANK[detected] > _CLAIM_TIER_RANK.get(ceiling, 0):
        return {
            "passed": False,
            "detected_tier": detected,
            "ceiling": ceiling,
            "violation": f"'{detected}'-tier claim exceeds evidence ceiling '{ceiling}'",
        }
    return {"passed": True, "detected_tier": detected, "ceiling": ceiling, "violation": None}


# ---------------------------------------------------------------------------
# 6b. Agency axis (#O) -- ORTHOGONAL to the technical-depth lattice above,
# not a replacement for it. The gap this closes: evidence supporting
# "architected X" (solo technical authority, tier=authority on the
# technical-depth axis) does NOT by itself license "led the team that
# architected X" -- that's a claim about PEOPLE authority/organizational
# scope, a genuinely different dimension. Two evidence items can be
# technical-depth-equal but agency-different (a solo architect vs. a team
# lead who assigned the work to others), and the realization must respect
# BOTH ceilings independently, not just the max of one merged scale.
#
# Deliberately only 2 axes, not the 4 (agency/technical-depth/scope/
# authority) that could theoretically be drawn: "scope" (task vs. system
# vs. org-wide) requires a RELATIVE SIZE judgment from one-line evidence
# text that no reliable regex heuristic exists for -- faking that
# precision would be worse than not claiming it (deferred, documented).
# "Authority" was folded INTO agency rather than kept as a 4th axis --
# once agency covers "was this an individual contribution or did it
# involve directing others," a separate authority/decision-maker axis
# would need a different deterministic signal than agency already uses,
# and none was found; they'd end up detecting the same vocabulary twice.
# ---------------------------------------------------------------------------

AGENCY_TIERS = ("individual_contributor", "owner", "team_lead", "director")
_AGENCY_TIER_RANK = {tier: i for i, tier in enumerate(AGENCY_TIERS)}

_AGENCY_VERB_PATTERNS: dict[str, re.Pattern] = {
    "owner": re.compile(
        r"\b(owned|owning|owns|independently|solely|individually)\b",
        re.IGNORECASE,
    ),
    "team_lead": re.compile(
        # "leads" as a bare -s form deliberately excluded (2026-09-03,
        # same noun/verb-collision class as "engineers"/"architects" above
        # and "engineering" in the claim-tier lattice) -- found via real
        # profile.json data: AMP Smart's responsibilities include "Identify
        # qualified leads," where "leads" is the NOUN (sales prospects),
        # not the verb "leads [a team]" -- and wrongly earned that
        # evidence a team_lead-tier agency ceiling. "led"/"leading" (past
        # tense/gerund) have no comparable collision and stay.
        r"\b(led|leading|managed|managing|manages|"
        r"supervised|supervising|supervises)\b",
        re.IGNORECASE,
    ),
    "director": re.compile(
        r"\b(directed|directing|directs|"
        r"spearheaded|spearheading|spearheads|"
        r"drove|driving|drives)\b",
        re.IGNORECASE,
    ),
}


def detect_agency_tier(text: str) -> str | None:
    """Return the HIGHEST agency-tier phrase found in `text`, or None if
    nothing beyond individual-contributor-level phrasing is present.
    "individual_contributor" (the lowest tier) has no dedicated vocabulary
    -- it's the default absence of any higher-agency signal, the same
    "silence is conservative, never a stronger default" philosophy as the
    technical-depth axis."""
    if not text:
        return None
    found = [tier for tier, pattern in _AGENCY_VERB_PATTERNS.items() if pattern.search(text)]
    if not found:
        return None
    return max(found, key=lambda t: _AGENCY_TIER_RANK[t])


def agency_ceiling_for_evidence(evidence_item: dict) -> str:
    """The strongest agency tier this evidence item's OWN text supports.
    Defaults to "individual_contributor" (the most conservative tier) when
    no team/ownership language is found."""
    return detect_agency_tier(_evidence_own_text(evidence_item)) or "individual_contributor"


def check_agency_strength(realized_text: str, ceiling: str) -> dict:
    """Deterministic post-generation check, same shape/contract as
    check_claim_strength but for the agency axis: does `realized_text`
    claim more organizational/people authority than the evidence
    supports? ("Led the team..." when the evidence only shows individual
    contribution.)"""
    detected = detect_agency_tier(realized_text)
    if detected is None:
        return {"passed": True, "detected_tier": None, "ceiling": ceiling, "violation": None}
    if _AGENCY_TIER_RANK[detected] > _AGENCY_TIER_RANK.get(ceiling, 0):
        return {
            "passed": False,
            "detected_tier": detected,
            "ceiling": ceiling,
            "violation": f"'{detected}'-tier agency claim exceeds evidence ceiling '{ceiling}'",
        }
    return {"passed": True, "detected_tier": detected, "ceiling": ceiling, "violation": None}


def check_causal_claim(realized_text: str, evidence_text: str) -> dict:
    """Deterministic check for unsupported CAUSAL/OUTCOME claims (#6),
    distinct from the agency-verb lattice above. "Used Python" needing no
    causal support is fine; "Used Python to automate X, reducing manual
    effort" asserts a causal outcome ("reducing...") that must itself be
    grounded.

    If `realized_text` uses one of the strong outcome verbs (reduced,
    eliminated, increased, improved, saved, etc. -- the same vocabulary
    classify_event_type looks for), the underlying `evidence_text` must
    ALSO use one of those verbs somewhere -- i.e. the human who wrote the
    evidence already asserted that outcome, so the realization is
    restating it, not inventing it. Weaker constructions (used, supported,
    contributed to, helped, participated in, worked with) never trigger
    this check regardless of evidence, since they don't claim a causal
    result at all.

    Returns {"passed": bool, "violation": str | None}.
    """
    if not _ACCOMPLISHMENT_VERB_RE.search(realized_text or ""):
        return {"passed": True, "violation": None}
    if _ACCOMPLISHMENT_VERB_RE.search(evidence_text or ""):
        return {"passed": True, "violation": None}
    return {
        "passed": False,
        "violation": (
            "realized text asserts a causal/outcome claim "
            "(e.g. reduced/eliminated/increased/improved) that the "
            "underlying evidence text does not itself establish"
        ),
    }


# Figure-ground (#F/#G): a resume-specific heuristic for a known weakness
# -- passive voice backgrounds the candidate as agent ("X was diagnosed")
# instead of foregrounding them ("Diagnosed X"). Deliberately a SOFT
# (logged, not rejecting) check, unlike claim-strength/agency/causal/
# metric above: "was/were + past participle" has real false positives
# (describing a SYSTEM's prior state as context, e.g. "the legacy system
# was outdated", is legitimate and not a claim-safety problem) where the
# others don't -- the primary enforcement is the AGENT instruction shown
# to the realizer in format_schema_guidance; this is a diagnostic signal,
# not a gate, matching decision #19's warnings-not-errors precedent.
_PASSIVE_VOICE_RE = re.compile(r"\b(was|were|is|are|been|being)\s+\w+ed\b", re.IGNORECASE)


def check_passive_voice(realized_text: str) -> dict:
    """Detects (does not reject) passive-voice constructions that could
    background the candidate as agent. Returns {"passed": bool,
    "violation": str | None} for symmetry with the other check_* functions,
    but callers should log rather than drop on failure -- see the module
    docstring's rationale above."""
    if _PASSIVE_VOICE_RE.search(realized_text or ""):
        return {
            "passed": False,
            "violation": "realized text may use passive voice, backgrounding the candidate as agent",
        }
    return {"passed": True, "violation": None}


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")


def check_metric_fabrication(
    realized_text: str,
    evidence_text: str,
    known_metrics: list[str] | None = None,
) -> dict:
    """Deterministic check (#P, sharpening the causal check): a
    legitimately-grounded VERB ("reduced", supported by the evidence) can
    still be paired with a FABRICATED MAGNITUDE ("by 30%") that nothing
    supports -- check_causal_claim alone wouldn't catch this, since it
    only checks that the OUTCOME VERB is grounded, not that a specific
    NUMBER riding along with it is. Every number-bearing token in
    `realized_text` must appear verbatim in `evidence_text` or in
    `known_metrics` (the profile's own resume_facts.real_metrics list --
    the same ground-truth set the CLOUD tailoring prompt's existing HARD
    RULE, "Do NOT change real numbers", already protects; this closes the
    equivalent gap for degraded mode, which previously had no numeric
    check at all).

    Text with no numbers always passes -- this only rejects a NEW,
    ungrounded number, never requires one to be present.
    """
    numbers = _NUMBER_RE.findall(realized_text or "")
    if not numbers:
        return {"passed": True, "violation": None}
    haystack = evidence_text or ""
    known = known_metrics or []
    for n in numbers:
        if n in haystack:
            continue
        if any(n in str(m) for m in known):
            continue
        return {
            "passed": False,
            "violation": f"number '{n}' in realized text is not present in the evidence text or known real_metrics",
        }
    return {"passed": True, "violation": None}


# ---------------------------------------------------------------------------
# 7. Viewpoint (#12) -- one coarse, job-title-derived professional lens per
#    job. The SAME underlying evidence can be legitimately presented from a
#    different angle depending on the target role (an automation project
#    reads as "implementation" from an engineering viewpoint, or as
#    "reduced manual work" from an operations viewpoint) -- no fact
#    changes, only which aspect is foregrounded. Deliberately a small fixed
#    vocabulary, not an elaborate ontology.
# ---------------------------------------------------------------------------

VIEWPOINTS = ("engineering", "data", "operations", "support", "leadership", "general")

_VIEWPOINT_HINTS: dict[str, tuple[str, ...]] = {
    "leadership": ("engineering manager", "team lead", "director", "head of", " vp ", "principal", "staff engineer"),
    "data": ("data scientist", "data engineer", "analytics", "machine learning", "data analyst"),
    "operations": (
        "devops",
        "site reliability",
        " sre",
        "infrastructure engineer",
        "platform engineer",
        "systems administrator",
        "it operations",
    ),
    "support": ("support engineer", "technical support", "help desk", "customer support", "service desk"),
    "engineering": ("software engineer", "developer", "backend", "frontend", "full stack", "full-stack", "programmer"),
}
_VIEWPOINT_PRIORITY = ("leadership", "data", "operations", "support", "engineering")


def select_viewpoint(job: dict) -> str:
    """Coarse professional lens derived from the job title (checked first,
    more specific viewpoints before the generic "engineering" bucket) and,
    if the title alone doesn't resolve it, the first 500 chars of the
    description. Always returns a VIEWPOINTS member; "general" when
    nothing matches."""
    title = (job.get("title") or "").lower()
    for viewpoint in _VIEWPOINT_PRIORITY:
        if any(h in title for h in _VIEWPOINT_HINTS[viewpoint]):
            return viewpoint
    text = f"{title} {(job.get('full_description') or '')[:500]}".lower()
    for viewpoint in _VIEWPOINT_PRIORITY:
        if any(h in text for h in _VIEWPOINT_HINTS[viewpoint]):
            return viewpoint
    return "general"


# 2026-08-24 (#Q): viewpoint was a label only -- it didn't change anything
# about what got emphasized. This is the smallest way to make it actually
# influence realization without any new extraction: a short, fixed phrase
# per viewpoint, shown to the realizer alongside the existing salience
# ordering. Still zero facts change -- only which already-true aspect of
# the same evidence gets foregrounded.
VIEWPOINT_EMPHASIS: dict[str, str] = {
    "engineering": "technical implementation detail and concrete capability",
    "data": "data flow, transformation, and analytical rigor",
    "operations": "reliability, efficiency, and reduction of manual effort",
    "support": "responsiveness and resolution of the user's actual problem",
    "leadership": "scope of ownership and outcome, not just the mechanism",
    "general": "the outcome and its direct relevance to this role",
}


# ---------------------------------------------------------------------------
# 8. Salience ordering (#13/#14/#16/#17 merged) -- one deterministic
#    ranking of which conceptual element should open the realized
#    sentence. Four theoretical lenses (figure-ground, profiling,
#    attention/salience, information structure) collapse into a single
#    representation because they all answer the same operational question:
#    what comes first.
# ---------------------------------------------------------------------------


def rank_salience(event_type: str) -> list[str]:
    """Which element profiles first. Accomplishments (a stated outcome
    exists) lead with the outcome -- "Reduced deployment failures by..."
    reads stronger than opening with the mechanism. Activities (no stated
    outcome) lead with the action itself, since there's no result to open
    on without inventing one."""
    if event_type == "accomplishment":
        return ["outcome", "mechanism", "technology", "context"]
    return ["action", "technology", "context", "outcome"]


# ---------------------------------------------------------------------------
# 9. Discourse coherence (#23) -- flags (does not silently rewrite) bullet
#    slots whose (frame, cognitive_schema) pair repeats 3+ times among the
#    supported requirements, so the realizer is nudged toward lexical
#    variety instead of every bullet opening identically. Never changes
#    WHICH schema was assigned -- diversity is a phrasing nudge, not a
#    reinterpretation of evidence.
# ---------------------------------------------------------------------------


def flag_repetition(requirements: list[dict]) -> None:
    """In-place: sets requirement["vary_phrasing"] = True on every
    supported requirement whose (frame, cognitive_schema) pair occurs 3+
    times among the supported requirements."""
    supported = [r for r in requirements if r.get("supported") and r.get("schema")]
    counts = Counter((r["frame"], r["schema"]["cognitive_schema"]) for r in supported)
    for r in supported:
        key = (r["frame"], r["schema"]["cognitive_schema"])
        r["vary_phrasing"] = counts[key] >= 3


# ---------------------------------------------------------------------------
# 10. Deterministic classification and selection -- no LLM call below.
# ---------------------------------------------------------------------------


def _match_kind(requirement_text: str, evidence_item: dict) -> str:
    """Classify HOW an already-resolved requirement/evidence pair matched:
    "literal" (an exact term from the evidence appears in the requirement
    text), "synonym" (matched only via local_tailor.py's curated paraphrase
    table), "semantic" (evidence_item carries local_tailor.py's
    semantic_match.py provenance marker and has no literal/synonym term
    overlap), or "unknown" (matched some other way -- treated
    conservatively, same as synonym, by callers).

    Read-only classification for rendering/schema-selection purposes only --
    never used to decide whether a match is valid; that decision was already
    made by local_tailor.py's _auto_resolve_requirements before this
    function is ever called. Note: literal/synonym are checked FIRST and
    take priority even for a semantically-provenanced item -- an item
    _auto_resolve_requirements added via semantic recall that also happens
    to share a literal/curated term with this specific requirement line is
    classified by that stronger signal, not weakened to "semantic".
    """
    text_lower = requirement_text.lower()
    terms = evidence_item.get("matched_terms") or []
    if any(_term_in_text(t, text_lower) for t in terms):
        return "literal"
    if any(_synonym_hit(t, text_lower) for t in terms):
        return "synonym"
    if evidence_item.get("provenance") == "semantic":
        return "semantic"
    return "unknown"


def select_schema_for_requirement(
    requirement_text: str,
    evidence_item: dict,
    match_kind: str,
    frame: str | None = None,
    force_relation: str | None = None,
) -> dict:
    """Deterministically pick a (cognitive_schema, bullet_schema) pair for
    one supported requirement. Never called for an unsupported requirement
    -- see build_job_schema_representation.

    `frame`, if given, takes priority over the old inline troubleshooting-
    hint regex (the diagnostic_support frame now owns that classification)
    -- kept optional so existing direct callers (and old tests) that don't
    pass a frame still get correct behavior via the text-based fallback.

    `force_relation` (see detect_force_relation), when "prevention", takes
    priority over the frame-based dispatch below -- averting a failure
    before it happens is a distinct construction from diagnosing/fixing
    one that already happened, regardless of which frame the requirement
    text otherwise resembles.
    """
    if match_kind != "literal":
        return {"cognitive_schema": "domain_transfer", "bullet_schema": "domain_transfer_bullet"}
    if force_relation == "prevention":
        return {"cognitive_schema": "prevention", "bullet_schema": "prevention_intervention_result"}
    if frame is None:
        frame = select_frame(requirement_text)
    if frame == "diagnostic_support":
        return {
            "cognitive_schema": "problem_diagnosis_repair_verification",
            "bullet_schema": "problem_action_result",
        }
    if evidence_item.get("type") in ("skill", "certification"):
        return {"cognitive_schema": "evidence_claim", "bullet_schema": "action_object_context_outcome"}
    return {"cognitive_schema": "action_object_outcome", "bullet_schema": "action_object_context_outcome"}


def select_summary_schema(requirements: list[dict]) -> str:
    """Pick a summary schema from how many requirements are directly
    (literally) supported vs. only transferable/synonym matches."""
    literal_supported = sum(
        1
        for r in requirements
        if r.get("supported") and r["schema"] and r["schema"]["cognitive_schema"] != "domain_transfer"
    )
    if literal_supported >= 2:
        return "identity_domain_strength_evidence_value"
    return "capability_background_transfer_value"


# ---------------------------------------------------------------------------
# 11. Composition: one representation per job
# ---------------------------------------------------------------------------


def build_job_schema_representation(job: dict, profile: dict) -> dict:
    """Compute the full per-job structured representation. No LLM call.

    Reuses local_tailor.py's deterministic requirement-extraction and
    evidence-ranking verbatim (same functions the local-first planner
    uses), then adds a compositional cognitive-linguistic layer on top:

      requirements: list of
        {requirement, importance, supported, resume_evidence,
         exact_keywords, synonym_concepts, schema, ambiguous,
         frame, event_type, category_tier, claim_ceiling,
         salience_order, provenance, vary_phrasing}
        -- schema/frame/event_type/category_tier/claim_ceiling/
        salience_order/provenance are all None/[]/"" unless supported is
        True. "ambiguous" (bool) marks a requirement with 2+ tied
        deterministic candidates that this function does not attempt to
        resolve (that's local_tailor.py's LLM-assisted job, not this
        one's) -- it gets no schema/frame/etc. either.
      viewpoint: one job-level VIEWPOINTS member (see select_viewpoint).
      summary_schema: one of SUMMARY_SCHEMAS' keys.
      evidence_considered: how many ranked evidence items were available.
    """
    top_n = int(os.environ.get("APPLYPILOT_LOCAL_EVIDENCE_TOPN", "6"))
    ranked_evidence = rank_profile_evidence(job, profile, top_n=top_n)
    requirement_lines, _dropped = _split_requirement_lines(job.get("full_description") or "")
    resolved, candidates = _auto_resolve_requirements(requirement_lines, ranked_evidence)

    requirements: list[dict] = []
    for i, line in enumerate(requirement_lines, start=1):
        entry: dict = {
            "requirement": line["text"],
            "importance": line["importance"],
            "supported": False,
            "resume_evidence": [],
            "exact_keywords": [],
            "synonym_concepts": [],
            "schema": None,
            "ambiguous": False,
            "frame": None,
            "event_type": None,
            "force_relation": None,
            "category_tier": None,
            "claim_ceiling": None,
            "agency_ceiling": None,
            "salience_order": [],
            "provenance": [],
        }
        evidence_ids = resolved.get(i) or []
        if evidence_ids:
            item = ranked_evidence[evidence_ids[0] - 1]
            kind = _match_kind(line["text"], item)
            frame = select_frame(line["text"])
            entry["supported"] = True
            entry["resume_evidence"] = [item["name"]]
            entry["frame"] = frame
            if kind == "literal":
                entry["exact_keywords"] = sorted(
                    t for t in (item.get("matched_terms") or []) if _term_in_text(t, line["text"].lower())
                )
            else:
                entry["synonym_concepts"] = [item["name"]]
            entry["category_tier"] = classify_category_tier(kind, len(entry["exact_keywords"]))

            # rank_profile_evidence wraps the raw profile.json item under
            # item["item"] (item itself is {"type", "name", "score",
            # "matched_terms", "item"}) -- the descriptive text this
            # section needs (description/factual_concepts/responsibilities,
            # see _evidence_own_text) lives on the RAW item, not the
            # wrapper. Using the wrapper directly here silently always
            # returned "" (a real bug caught by manual verification: an
            # evidence description literally containing "Automated..."
            # still produced ceiling="participation" because
            # .get("description") on the wrapper is always None).
            raw_item = item.get("item") or {}
            evidence_text = _evidence_own_text(raw_item)
            entry["event_type"] = classify_event_type(evidence_text)
            force_relation = detect_force_relation(evidence_text)
            entry["force_relation"] = force_relation
            entry["schema"] = select_schema_for_requirement(
                line["text"],
                item,
                kind,
                frame=frame,
                force_relation=force_relation,
            )
            entry["claim_ceiling"] = claim_ceiling_for_evidence(raw_item)
            entry["agency_ceiling"] = agency_ceiling_for_evidence(raw_item)
            entry["salience_order"] = rank_salience(entry["event_type"])
            if evidence_text:
                entry["provenance"] = [{"source_type": item.get("type") or "unknown", "text": evidence_text[:300]}]
        elif candidates.get(i):
            entry["ambiguous"] = True
        requirements.append(entry)

    flag_repetition(requirements)

    return {
        "job_url": job.get("url"),
        "requirements": requirements,
        "viewpoint": select_viewpoint(job),
        "summary_schema": select_summary_schema(requirements),
        "evidence_considered": len(ranked_evidence),
    }


# ---------------------------------------------------------------------------
# 12. Per-process cache -- computed once per job, reused across tailor/cover
# ---------------------------------------------------------------------------

_CACHE: dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()


def _cache_key(job: dict) -> str:
    desc_hash = hashlib.md5((job.get("full_description") or "").encode("utf-8")).hexdigest()
    return f"{job.get('url') or ''}:{desc_hash}"


def get_or_build_job_schema(job: dict, profile: dict) -> dict:
    """Cached wrapper around build_job_schema_representation.

    Keyed by (job URL, description hash) so a later re-enrichment of the
    same URL with different description text doesn't serve a stale
    representation. Thread-safe for the --workers>1 case; the build itself
    may run more than once under a race, which is harmless (pure function,
    last write wins) -- the lock only protects the dict access.
    """
    key = _cache_key(job)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached
    rep = build_job_schema_representation(job, profile)
    with _CACHE_LOCK:
        _CACHE[key] = rep
    return rep


def clear_schema_cache() -> None:
    """Test/debug helper: drop all cached representations."""
    with _CACHE_LOCK:
        _CACHE.clear()


# ---------------------------------------------------------------------------
# 13. Rendering -- shared by tailor.py and cover_letter.py
# ---------------------------------------------------------------------------


def format_schema_guidance(representation: dict | None, max_requirements: int = 8) -> str:
    """Render a job schema representation as a compact prompt section.

    Only supported requirements (schema is not None) are shown -- ambiguous
    and unsupported requirements are deliberately omitted here, not just
    de-prioritized, so the guidance can never read as confident about
    something this module didn't actually ground. Returns "" when nothing
    supported exists, so callers can `if guidance:` before including it.
    """
    if not representation or not representation.get("requirements"):
        return ""

    shown_lines: list[str] = []
    for r in representation["requirements"]:
        if not r.get("supported") or not r.get("schema"):
            continue
        if len(shown_lines) >= max_requirements:
            break
        schema = r["schema"]
        cog = COGNITIVE_SCHEMAS.get(schema["cognitive_schema"], {})
        bullet = BULLET_SCHEMAS.get(schema["bullet_schema"], {})
        frame = FRAMES.get(r.get("frame") or "", {})
        anchor_bits = []
        if r["exact_keywords"]:
            anchor_bits.append("use these exact terms: " + ", ".join(r["exact_keywords"]))
        if r["synonym_concepts"]:
            anchor_bits.append(
                "transferable evidence, do not overclaim identical domain: " + ", ".join(r["synonym_concepts"])
            )
        if r.get("vary_phrasing"):
            anchor_bits.append("this frame repeats elsewhere -- vary the phrasing/opening")
        anchor_text = ("\n    " + "; ".join(anchor_bits)) if anchor_bits else ""
        salience = r.get("salience_order") or []
        lead_with = f" -- lead with {salience[0]}" if salience else ""
        force_note = ""
        if r.get("force_relation") == "prevention":
            force_note = (
                "\n    force relation: prevention -- this averted a failure BEFORE it happened, not one already fixed"
            )
        shown_lines.append(
            f"- [{r['importance']}] {r['requirement']}\n"
            f"    evidence: {', '.join(r['resume_evidence'])}\n"
            f"    frame: {r.get('frame')} ({frame.get('description', '')})\n"
            f"    cognitive frame: {schema['cognitive_schema']} -- {cog.get('description', '')}\n"
            f"    bullet shape ({schema['bullet_schema']}): {bullet.get('pattern', '')} "
            f"[{bullet.get('discourse_function', '')}]"
            f"{lead_with}{force_note}\n"
            f"    technical-depth ceiling: {r.get('claim_ceiling')} -- do not use a stronger "
            f"technical verb (e.g. 'designed'->'architected') than this evidence supports\n"
            f"    agency ceiling: {r.get('agency_ceiling')} -- do not claim more people/"
            f"organizational authority (e.g. 'led the team') than this evidence supports -- "
            f"these are SEPARATE limits, a strong technical-depth ceiling does not raise this one"
            f"{anchor_text}"
        )

    if not shown_lines:
        return ""

    emphasis = VIEWPOINT_EMPHASIS.get(representation.get("viewpoint") or "general", "")
    lines = [
        (
            "JOB SCHEMA GUIDANCE (deterministic, evidence-grounded -- express these "
            "points using the given cognitive frame/bullet shape as a structural "
            "guide, not verbatim text; still write fresh wording):"
        ),
        (
            f"VIEWPOINT: {representation.get('viewpoint', 'general')} -- present this "
            f"candidate's experience from this professional angle where it fits "
            f"naturally, emphasizing {emphasis}; do not change any fact to fit the angle."
        ),
        (
            "AGENT: the candidate must be the grammatical subject/agent of every "
            'bullet -- "Diagnosed X", never the passive "X was diagnosed".'
        ),
        *shown_lines,
    ]

    summary_key = representation.get("summary_schema")
    if summary_key and summary_key in SUMMARY_SCHEMAS:
        s = SUMMARY_SCHEMAS[summary_key]
        lines.append(f"SUMMARY SCHEMA: {summary_key} -- {s['description']} (slots: {' -> '.join(s['slots'])})")

    return "\n".join(lines)
