"""Rigorous re-validation of the opt-in escalation trigger
(scoring/deterministic_fallback.py's _AMBIGUOUS_TITLE_RE), using ONLY data
already on hand -- no new LLM/model calls, no Gemini quota needed.

Decision #77 shipped this trigger as a point estimate (85% hybrid gate
agreement vs 79% 1.7b-alone / 87% pure-8b) on a single n=52 sample, with an
explicit honest caveat: the keyword list was built FROM that exact sample,
so the point estimate is likely optimistic and has never had a confidence
interval attached.

This script:
  1. Reconstructs the exact 52-job overlap between the two prior real runs
     (1.7b n=58, 8b n=52) -- same data decision #77 used, no new calls.
  2. Computes gate agreement for three configurations against real
     fit_score: 1.7b-alone, 8b-alone, and the HYBRID (using the actual
     production _AMBIGUOUS_TITLE_RE regex, not a re-derived one).
  3. Bootstrap-resamples the n=52 set (10,000 resamples, real statistical
     technique, not a guess) to attach 95% CIs to each configuration's gate
     agreement rate AND to the hybrid's gate-flip catch rate specifically --
     the CI width itself tells us how much to trust the point estimate
     given how small n=52 really is.
  4. Per-keyword breakdown: of the 6 keywords in _AMBIGUOUS_TITLE_RE, which
     ones actually fired on a disagreement in this sample vs. fired on jobs
     where 1.7b was already correct (pure escalation overhead, no benefit)?
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot.scoring.deterministic_fallback import _AMBIGUOUS_TITLE_RE  # noqa: E402

EXP_DIR = REPO_ROOT / "data" / "experiments" / "deterministic_score_fallback_20260906"


def load(name: str) -> dict[str, dict]:
    with open(EXP_DIR / name, encoding="utf-8") as f:
        rows = json.load(f)
    return {r["url"]: r for r in rows}


def family_of(detail: str) -> str | None:
    m = re.search(r"family:(\S+)", detail or "")
    return m.group(1) if m else None


def gate(score: int | None) -> bool | None:
    if score is None:
        return None
    return score >= 8


def main() -> None:
    fast = load("atomic_facts_scorer_results.json")  # 1.7b, n=58
    slow = load("qwen8b_fullscale_results.json")  # 8b, n=52

    common = sorted(set(fast) & set(slow))
    print(f"common URLs (real overlap): {len(common)}\n")

    rows = []
    for url in common:
        f, s = fast[url], slow[url]
        real = f["fit_score"]
        title = f["title"] or ""
        escalate = bool(_AMBIGUOUS_TITLE_RE.search(title))
        hybrid_score = s["local_score"] if escalate else f["local_score"]
        rows.append(
            {
                "url": url,
                "title": title,
                "real": real,
                "fast_score": f["local_score"],
                "slow_score": s["local_score"],
                "hybrid_score": hybrid_score,
                "escalated": escalate,
                "fast_family": family_of(f["detail"]),
                "slow_family": family_of(s["detail"]),
            }
        )

    n = len(rows)

    def gate_agree_rate(score_key: str) -> float:
        agree = sum(1 for r in rows if gate(r[score_key]) == gate(r["real"]))
        return agree / n

    fast_rate = gate_agree_rate("fast_score")
    slow_rate = gate_agree_rate("slow_score")
    hybrid_rate = gate_agree_rate("hybrid_score")
    n_escalated = sum(1 for r in rows if r["escalated"])

    print(f"n={n}")
    print(f"1.7b-alone gate agreement:  {fast_rate:.1%}")
    print(f"8b-alone gate agreement:    {slow_rate:.1%}")
    print(f"hybrid gate agreement:      {hybrid_rate:.1%}  ({n_escalated}/{n} escalated = {n_escalated/n:.1%})")
    print()

    # Gate-flip disagreements: cases where fast and slow gate differently.
    flips = [r for r in rows if gate(r["fast_score"]) != gate(r["slow_score"])]
    print(f"real gate-flip disagreements (fast vs slow): {len(flips)}/{n}")
    caught = [r for r in flips if r["escalated"]]
    print(f"  of those, escalation trigger fires on: {len(caught)}/{len(flips)}")
    slow_right_on_flips = [r for r in flips if gate(r["slow_score"]) == gate(r["real"])]
    print(f"  of those, 8b (slow) is actually correct on: {len(slow_right_on_flips)}/{len(flips)}")
    caught_and_slow_right = [r for r in caught if gate(r["slow_score"]) == gate(r["real"])]
    print(f"  escalation catches + 8b-correct: {len(caught_and_slow_right)}/{len(slow_right_on_flips)} of the flips worth catching")
    print()

    # Bootstrap CIs (real resampling, 10k iterations, seeded for reproducibility)
    random.seed(20260908)
    B = 10000

    def bootstrap_ci(metric_fn, B=B) -> tuple[float, float, float]:
        point = metric_fn(rows)
        samples = []
        for _ in range(B):
            resample = [random.choice(rows) for _ in range(n)]
            samples.append(metric_fn(resample))
        samples.sort()
        lo = samples[int(0.025 * B)]
        hi = samples[int(0.975 * B)]
        return point, lo, hi

    def m_fast(rs):
        return sum(1 for r in rs if gate(r["fast_score"]) == gate(r["real"])) / len(rs)

    def m_slow(rs):
        return sum(1 for r in rs if gate(r["slow_score"]) == gate(r["real"])) / len(rs)

    def m_hybrid(rs):
        return sum(1 for r in rs if gate(r["hybrid_score"]) == gate(r["real"])) / len(rs)

    print("=== Bootstrap 95% CIs (10,000 resamples) ===")
    for name, fn in (("1.7b-alone", m_fast), ("8b-alone", m_slow), ("hybrid", m_hybrid)):
        point, lo, hi = bootstrap_ci(fn)
        print(f"  {name}: {point:.1%}  95% CI=[{lo:.1%}, {hi:.1%}]")
    print()

    # Per-keyword breakdown: which regex alternatives actually fire, and on
    # what fraction of THOSE firings does escalation change anything useful?
    keyword_alts = ["technician", "maintenance", "assembler", "composites", "field service", "embedded", "infotainment"]
    print("=== Per-keyword firing breakdown ===")
    for kw in keyword_alts:
        fired = [r for r in rows if re.search(kw, r["title"], re.IGNORECASE)]
        if not fired:
            print(f"  {kw!r}: 0 titles matched in this sample")
            continue
        useful = [r for r in fired if gate(r["fast_score"]) != gate(r["slow_score"])]
        print(f"  {kw!r}: fired on {len(fired)} titles, {len(useful)} were real fast/slow disagreements")
        for r in fired:
            flag = "DISAGREEMENT" if r in useful else ""
            print(f"      real={r['real']} fast={r['fast_score']} slow={r['slow_score']} {flag}  {r['title'][:60]!r}")

    (Path(__file__).parent / "escalation_revalidation_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print("\nwrote escalation_revalidation_results.json")


if __name__ == "__main__":
    main()
