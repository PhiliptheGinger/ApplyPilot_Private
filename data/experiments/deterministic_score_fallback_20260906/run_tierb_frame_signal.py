"""Plan step 1 (bail-early gate): does FRAME-alignment alone predict real
LLM fit_score better than the failed literal-keyword attempt (8% exact,
59% gate agreement, see run_spotcheck.py)?

Pulls the full real labeled dataset (jobs with a genuine LLM-judged
fit_score -- excludes _check_ineligible's own deterministic rejections),
does a stratified train/held-out split, computes a frame-coverage-ratio
signal on train, fits score-band thresholds by grid search on train only,
and reports the SAME honest metrics on held-out only.

No production code touched.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from applypilot import config  # noqa: E402

config.load_env()

from applypilot import database  # noqa: E402
from applypilot.config import load_profile  # noqa: E402
from applypilot.scoring.local_tailor import _split_requirement_lines  # noqa: E402
from applypilot.scoring.schemas import _evidence_own_text, select_frame  # noqa: E402

HERE = Path(__file__).parent
SEED = 20260906


def _profile_evidence_items(profile: dict) -> list[dict]:
    items: list[dict] = []
    for key in ("experience_inventory", "project_inventory", "skills_inventory", "certifications"):
        for item in profile.get(key) or []:
            if isinstance(item, dict) and item.get("resume_allowed") is not False and item.get("name"):
                items.append(item)
    return items


def frame_signal(job: dict, ev_frames: list[str]) -> dict | None:
    """Returns coverage_ratio (per-requirement: does any evidence item share
    its frame? excludes the 'general_capability' fallback from counting as
    a real match -- matching on the default/no-signal frame would be
    meaningless) and a job-level distribution-overlap (Jaccard over the SET
    of frames present, job side vs. evidence side) as a second candidate
    feature."""
    lines, _ = _split_requirement_lines(job.get("full_description") or "")
    if not lines:
        return None
    req_frames = [select_frame(line["text"]) for line in lines]
    ev_frame_set = {f for f in ev_frames if f != "general_capability"}

    matched = sum(1 for f in req_frames if f != "general_capability" and f in ev_frame_set)
    total = len(req_frames)
    coverage_ratio = matched / total if total else None

    req_frame_set = {f for f in req_frames if f != "general_capability"}
    union = req_frame_set | ev_frame_set
    jaccard = (len(req_frame_set & ev_frame_set) / len(union)) if union else 0.0

    return {"coverage_ratio": coverage_ratio, "jaccard": jaccard, "matched": matched, "total": total}


def stratified_split(rows: list[dict], seed: int, train_frac: float = 0.7):
    by_band: dict[str, list[dict]] = {}
    for r in rows:
        s = r["fit_score"]
        band = "1-4" if s <= 4 else ("5-7" if s <= 7 else "8-10")
        by_band.setdefault(band, []).append(r)
    rng = random.Random(seed)
    train, held = [], []
    for band, items in by_band.items():
        rng.shuffle(items)
        cut = int(len(items) * train_frac)
        train.extend(items[:cut])
        held.extend(items[cut:])
    rng.shuffle(train)
    rng.shuffle(held)
    return train, held


def fit_thresholds(train_scored: list[dict]) -> list[tuple[float, int]]:
    """Grid search: for a small set of ratio cutpoints, find the boundaries
    that maximize agreement with real fit_score, evaluated ONLY on train.
    Returns a sorted list of (min_ratio, score) bands, highest ratio first.
    """
    # Candidate cutpoints from the actual train-set ratio distribution.
    ratios = sorted({r["ratio"] for r in train_scored if r["ratio"] is not None})
    if not ratios:
        return [(0.0, 2)]

    best = None
    # Try a handful of 2-cutpoint schemes (low/mid/high) since the real
    # distribution is bimodal (poor 1-4 vs good 8-10, thin middle) -- a
    # single linear ramp is exactly what failed in the naive attempt.
    candidates = [0.0] + ratios + [1.0]
    for hi_cut in candidates:
        for lo_cut in candidates:
            if lo_cut > hi_cut:
                continue

            def predict(ratio):
                if ratio is None:
                    return 2
                if ratio >= hi_cut:
                    return 9
                if ratio >= lo_cut:
                    return 5
                return 2

            agree = sum(
                1
                for r in train_scored
                if (predict(r["ratio"]) >= 8) == (r["fit_score"] >= 8)
            )
            if best is None or agree > best[0]:
                best = (agree, lo_cut, hi_cut)

    _, lo_cut, hi_cut = best
    return [(hi_cut, 9), (lo_cut, 5), (0.0, 2)]


def apply_thresholds(ratio: float | None, bands: list[tuple[float, int]]) -> int:
    if ratio is None:
        return 2
    for cutoff, score in bands:
        if ratio >= cutoff:
            return score
    return 2


def report(name: str, rows: list[dict]) -> None:
    have = [r for r in rows if r["det_score"] is not None]
    n = len(have)
    if n == 0:
        print(f"{name}: no scorable rows")
        return
    exact = sum(1 for r in have if r["det_score"] == r["fit_score"])
    within2 = sum(1 for r in have if abs(r["det_score"] - r["fit_score"]) <= 2)
    gate = sum(1 for r in have if (r["det_score"] >= 8) == (r["fit_score"] >= 8))
    # Confusion on the 1-4 / 8-10 poles specifically (ignore the thin 5-7 band).
    poles = [r for r in have if r["fit_score"] <= 4 or r["fit_score"] >= 8]
    pole_agree = sum(1 for r in poles if (r["det_score"] >= 8) == (r["fit_score"] >= 8)) if poles else 0
    print(
        f"{name}: n={n} exact={100*exact/n:.0f}% within2={100*within2/n:.0f}% "
        f"gate_agree={100*gate/n:.0f}% pole_agree={100*pole_agree/len(poles):.0f}% (n_poles={len(poles)})"
    )


def main():
    conn = database.get_connection()
    profile = load_profile()
    ev_items = _profile_evidence_items(profile)
    ev_frames = [select_frame(_evidence_own_text(item)) for item in ev_items]
    print("profile evidence items:", len(ev_items))
    print("evidence frames:", Counter(ev_frames))

    rows = conn.execute(
        """
        SELECT url, title, fit_score, full_description FROM jobs
        WHERE fit_score IS NOT NULL
          AND (score_reasoning IS NULL OR score_reasoning NOT LIKE '%Ineligible:%')
          AND full_description IS NOT NULL
        """
    ).fetchall()
    rows = [dict(r) for r in rows]
    print("total labeled (non-prefilter) jobs:", len(rows))

    train_jobs, held_jobs = stratified_split(rows, seed=SEED)
    print("train:", len(train_jobs), "held-out:", len(held_jobs))

    def compute(jobs):
        out = []
        for job in jobs:
            sig = frame_signal(job, ev_frames)
            ratio = sig["coverage_ratio"] if sig else None
            out.append({**job, "ratio": ratio, "jaccard": sig["jaccard"] if sig else None})
        return out

    train_scored = compute(train_jobs)
    held_scored = compute(held_jobs)

    bands = fit_thresholds(train_scored)
    print("fitted bands (ratio_cutoff -> score), highest first:", bands)

    for r in train_scored:
        r["det_score"] = apply_thresholds(r["ratio"], bands)
    for r in held_scored:
        r["det_score"] = apply_thresholds(r["ratio"], bands)

    print()
    report("TRAIN (fit on this)", train_scored)
    report("HELD-OUT (real validation)", held_scored)

    print()
    print("Baseline from run_spotcheck.py (naive literal-keyword ratio, n=40): exact=8% gate_agree=59%")

    with open(HERE / "tierb_results.json", "w") as f:
        json.dump({"bands": bands, "train": train_scored, "held_out": held_scored}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
