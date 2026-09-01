import json

import pytest

from applypilot.database import current_state, transition_state
from applypilot.scoring.plan_experiment import build_structured_plan, validate_structured_plan


PROFILE = {
    "experience_inventory": [
        {
            "name": "Help Desk Technician",
            "resume_allowed": True,
            "relevance_categories": ["troubleshoot", "support", "root cause", "python"],
            "description": "Diagnosed user issues and resolved root cause with scripts.",
        }
    ],
    "project_inventory": [],
    "skills_inventory": [],
    "certifications": [],
}


def _job(rowid: int = 1) -> dict:
    return {
        "rowid": rowid,
        "url": "https://example.com/job/exp-1",
        "title": "Support Technician",
        "company": "Acme",
        "site": "linkedin",
        "full_description": "- Troubleshoot hardware issues and identify root cause\n- Python scripting experience\n",
    }


def test_validate_structured_plan_rejects_unknown_evidence_reference():
    plan = build_structured_plan(_job(), PROFILE)
    if plan["requirements"]:
        plan["requirements"][0]["status"] = "supported"
        plan["requirements"][0]["evidence_ids"] = [99999]
    result = validate_structured_plan(plan)
    assert not result["passed"]
    assert any("unknown evidence id" in e for e in result["errors"])


def test_validate_structured_plan_rejects_unsupported_with_evidence_ids():
    plan = build_structured_plan(_job(), PROFILE)
    if plan["requirements"]:
        plan["requirements"][0]["status"] = "unsupported"
        plan["requirements"][0]["evidence_ids"] = [1]
    result = validate_structured_plan(plan)
    assert not result["passed"]
    assert any("unsupported requirements must not reference evidence_ids" in e for e in result["errors"])


def test_experiment_tailor_plan_cli_is_read_only_and_writes_report(tmp_db, seed_job, monkeypatch, tmp_path):
    import applypilot.cli as cli_mod

    conn = tmp_db()
    row = seed_job(
        conn,
        url_suffix="exp-cli",
        title="Support Technician",
        full_description="- Troubleshoot hardware issues and identify root cause\n- Python scripting\n",
        tailored_resume_path=None,
        cover_letter_path=None,
        fit_score=9,
        company="Acme",
    )
    rowid = conn.execute("SELECT rowid FROM jobs WHERE url = ?", (row["url"],)).fetchone()[0]
    transition_state(conn, row["url"], "scored", reason="test setup", force=True)
    conn.execute("UPDATE jobs SET tailor_attempts = 2 WHERE url = ?", (row["url"],))
    conn.commit()

    before_state = current_state(conn, row["url"])
    before_attempts = conn.execute("SELECT COALESCE(tailor_attempts, 0) FROM jobs WHERE url = ?", (row["url"],)).fetchone()[0]
    before_path = conn.execute("SELECT tailored_resume_path FROM jobs WHERE url = ?", (row["url"],)).fetchone()[0]

    monkeypatch.setattr(
        "applypilot.config.load_profile",
        lambda: PROFILE,
    )

    out_path = tmp_path / "q2_report.json"
    cli_mod.experiment_tailor_plan_cmd(
        job_id=[rowid],
        url=None,
        output=str(out_path),
        with_local_planner=False,
        simulate_realization_failure=True,
        max_jobs=5,
    )

    after_state = current_state(conn, row["url"])
    after_attempts = conn.execute("SELECT COALESCE(tailor_attempts, 0) FROM jobs WHERE url = ?", (row["url"],)).fetchone()[0]
    after_path = conn.execute("SELECT tailored_resume_path FROM jobs WHERE url = ?", (row["url"],)).fetchone()[0]

    assert after_state == before_state
    assert after_attempts == before_attempts
    assert after_path == before_path

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["mode"] == "read_only"
    assert report["summary"]["job_count"] == 1
    assert report["jobs"][0]["plan_validation"]["passed"] is True
    assert report["jobs"][0]["realization_boundary"]["validated_plan_reusable_after_failure"] is True


def test_experiment_tailor_plan_cli_requires_explicit_targets():
    import applypilot.cli as cli_mod

    with pytest.raises(cli_mod.typer.Exit) as excinfo:
        cli_mod.experiment_tailor_plan_cmd(
            job_id=None,
            url=None,
            output=None,
            with_local_planner=False,
            simulate_realization_failure=True,
            max_jobs=5,
        )
    assert excinfo.value.exit_code == 1


def test_experiment_local_invoked_uses_actual_planner_metadata(tmp_db, seed_job, monkeypatch, tmp_path):
    import applypilot.cli as cli_mod

    conn = tmp_db()
    row = seed_job(
        conn,
        url_suffix="exp-cli-invoked",
        title="Support Technician",
        full_description="- Troubleshoot hardware issues and identify root cause\n",
        tailored_resume_path=None,
        cover_letter_path=None,
        fit_score=9,
        company="Acme",
    )
    rowid = conn.execute("SELECT rowid FROM jobs WHERE url = ?", (row["url"],)).fetchone()[0]

    monkeypatch.setattr("applypilot.config.load_profile", lambda: PROFILE)
    monkeypatch.setattr("applypilot.llm.is_local_configured", lambda: True)

    monkeypatch.setattr(
        "applypilot.scoring.local_tailor.rank_profile_evidence",
        lambda _job, _profile, top_n=6: [
            {
                "type": "skill",
                "name": "Python",
                "score": 1,
                "matched_terms": ["python"],
                "item": {"name": "Python", "resume_allowed": True},
            }
        ],
    )
    monkeypatch.setattr(
        "applypilot.scoring.local_tailor._split_requirement_lines",
        lambda _text: ([{"text": "Req one", "importance": "required"}], []),
    )
    monkeypatch.setattr(
        "applypilot.scoring.local_tailor._auto_resolve_requirements",
        lambda _reqs, _ranked: ({1: [1]}, {1: [1]}),
    )
    monkeypatch.setattr(
        "applypilot.scoring.local_tailor.get_local_tailoring_plan",
        lambda _resume, _job, _profile, return_meta=False: (
            {
                "requirements": [
                    {
                        "requirement": "Req one",
                        "importance": "required",
                        "resume_evidence": ["Python"],
                        "supported": True,
                    }
                ]
            },
            {"llm_called": True},
        )
        if return_meta
        else {
            "requirements": [
                {
                    "requirement": "Req one",
                    "importance": "required",
                    "resume_evidence": ["Python"],
                    "supported": True,
                }
            ]
        },
    )

    out_path = tmp_path / "q2_report_invoked.json"
    cli_mod.experiment_tailor_plan_cmd(
        job_id=[rowid],
        url=None,
        output=str(out_path),
        with_local_planner=True,
        simulate_realization_failure=True,
        max_jobs=5,
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    local = report["jobs"][0]["local_planner"]
    assert local["invoked"] is True
    assert local["invoked_predicted"] is False
    assert local["invocation_prediction_mismatch"] is True


def test_experiment_adapter_rejects_unmapped_supported_evidence(tmp_db, seed_job, monkeypatch, tmp_path):
    import applypilot.cli as cli_mod

    conn = tmp_db()
    row = seed_job(
        conn,
        url_suffix="exp-cli-unmapped",
        title="Support Technician",
        full_description="- Troubleshoot hardware issues and identify root cause\n",
        tailored_resume_path=None,
        cover_letter_path=None,
        fit_score=9,
        company="Acme",
    )
    rowid = conn.execute("SELECT rowid FROM jobs WHERE url = ?", (row["url"],)).fetchone()[0]

    monkeypatch.setattr("applypilot.config.load_profile", lambda: PROFILE)
    monkeypatch.setattr("applypilot.llm.is_local_configured", lambda: True)
    monkeypatch.setattr(
        "applypilot.scoring.local_tailor.get_local_tailoring_plan",
        lambda _resume, _job, _profile, return_meta=False: (
            {
                "requirements": [
                    {
                        "requirement": "Unmapped requirement text",
                        "importance": "required",
                        "resume_evidence": ["Definitely Not In Catalog"],
                        "supported": True,
                    }
                ]
            },
            {"llm_called": True},
        )
        if return_meta
        else {
            "requirements": [
                {
                    "requirement": "Unmapped requirement text",
                    "importance": "required",
                    "resume_evidence": ["Definitely Not In Catalog"],
                    "supported": True,
                }
            ]
        },
    )

    out_path = tmp_path / "q2_report_unmapped.json"
    cli_mod.experiment_tailor_plan_cmd(
        job_id=[rowid],
        url=None,
        output=str(out_path),
        with_local_planner=True,
        simulate_realization_failure=True,
        max_jobs=5,
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    local = report["jobs"][0]["local_planner"]
    assert local["plan_validation"]["passed"] is True
    assert "adapter_diagnostics" in local
    unmapped = local["adapter_diagnostics"]["unmapped_evidence_by_requirement"]
    assert unmapped
    assert unmapped[0]["unmapped_resume_evidence"] == ["Definitely Not In Catalog"]
    assert not any("supported requirements must reference evidence_ids" in e for e in local["plan_validation"]["errors"])
