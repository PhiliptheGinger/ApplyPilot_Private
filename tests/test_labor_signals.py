"""Regression tests for the positive labor/ownership-structure signal
detector (CLAUDE.md decision #206 addendum, built 2026-09-25).

Real corpus check found one false positive worth pinning explicitly:
"union membership" inside ordinary EEO/non-discrimination boilerplate
must NOT match as a unionization signal (the exact false-positive shape
already documented for bare "military" in the ethical-exclusion list).
"""

from __future__ import annotations

from applypilot.scoring.labor_signals import (
    LABOR_SIGNAL_BONUS_CAP,
    detect_labor_signals,
    labor_signal_score_adjustment,
)


def _job(desc: str, title: str = "Software Engineer") -> dict:
    return {"title": title, "full_description": desc}


def test_no_signals_on_ordinary_posting():
    job = _job("We are looking for a Software Engineer with 3+ years of experience.")
    assert detect_labor_signals(job) == []


def test_esop_line_detected():
    job = _job("Benefits include: 401(k) Savings Plan, Employee Stock Ownership Plan (ESOP), dental and vision.")
    assert "employee_ownership" in detect_labor_signals(job)


def test_collective_bargaining_agreement_detected():
    job = _job("This is an hourly position governed by the IAM Collective Bargaining agreement.")
    assert "unionized" in detect_labor_signals(job)


def test_eeo_boilerplate_union_membership_is_not_a_false_positive():
    """The real false positive found during verification: bare EEO
    protected-characteristic boilerplate must not be mistaken for a real
    unionization signal."""
    job = _job(
        "We do not discriminate based on race, color, religion, sex, national origin, "
        "disability, status as a crime victim, protected veteran status, political "
        "affiliation, union membership, or any other characteristic protected by law."
    )
    assert "unionized" not in detect_labor_signals(job)


def test_public_benefit_corporation_detected():
    job = _job("Anthropic is a public benefit corporation headquartered in San Francisco.")
    assert "benefit_corporation" in detect_labor_signals(job)


def test_b_corp_certified_detected():
    job = _job("BCorp Certified, WWF Partnership, volunteer days, and so much more.")
    assert "b_corp_certified" in detect_labor_signals(job)


def test_profit_sharing_detected():
    job = _job("We offer a competitive profit-sharing plan alongside base salary.")
    assert "profit_sharing" in detect_labor_signals(job)


def test_worker_cooperative_detected():
    job = _job("Our company operates as a worker-owned cooperative.")
    assert "worker_cooperative" in detect_labor_signals(job)


def test_adjustment_zero_for_no_signals():
    delta, note = labor_signal_score_adjustment([])
    assert delta == 0
    assert note == ""


def test_adjustment_scales_with_signal_count_up_to_cap():
    delta_one, _ = labor_signal_score_adjustment(["employee_ownership"])
    assert delta_one == 1

    delta_many, note = labor_signal_score_adjustment(
        ["employee_ownership", "unionized", "b_corp_certified", "profit_sharing"]
    )
    assert delta_many == LABOR_SIGNAL_BONUS_CAP
    assert "employee ownership" in note
