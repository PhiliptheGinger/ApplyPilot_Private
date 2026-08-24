from applypilot.scoring import fact_approval


def _profile() -> dict:
    return {
        "skills_boundary": {
            "languages": ["python", "sql", "go"],
            "tools": ["kubernetes", "terraform"],
        }
    }


def test_extract_facts_from_resume_json_filters_to_allowed_skills():
    data = {
        "skills": {
            "Languages": "Python, Rust, SQL",
            "Tools": "Kubernetes; Terraform",
        },
        "experience": [
            {
                "header": "Software Engineer at ExampleCo",
                "subtitle": "Python | 2021-2024",
                "bullets": ["Reduced latency by 35%", "Scaled to 100 services"],
            }
        ],
        "projects": [],
        "education": [{"institution": "State University", "degree": "BS CS", "dates": "2018-2022"}],
    }

    facts = fact_approval.extract_facts_from_resume_json(data, _profile())

    assert "skill:python" in facts
    assert "skill:sql" in facts
    assert "skill:kubernetes" in facts
    assert "skill:rust" not in facts
    assert "metric:35%" in facts
    assert "metric:100" in facts


def test_auto_approval_requires_subset_of_approved_union(tmp_path, monkeypatch):
    cache = tmp_path / "approved_resume_facts.json"
    monkeypatch.setattr(fact_approval, "_FACT_CACHE_PATH", cache)

    a = {"skill:python", "experience:header:engineer at a", "metric:35%"}
    c = {"skill:sql", "projects:header:etl pipeline", "metric:100"}

    fact_approval.record_approved_facts(a, source="A")
    fact_approval.record_approved_facts(c, source="C")

    # Subset of A should auto-approve.
    b = {"skill:python", "metric:35%"}
    assert fact_approval.is_auto_approvable(b)

    # Subset of union(A, C) should auto-approve.
    d = {"skill:python", "skill:sql", "metric:100"}
    assert fact_approval.is_auto_approvable(d)

    # Any new fact outside approved union should fail auto-approval.
    e = {"skill:python", "metric:100", "experience:header:unknown company"}
    assert not fact_approval.is_auto_approvable(e)


def test_empty_candidate_never_auto_approves(tmp_path, monkeypatch):
    cache = tmp_path / "approved_resume_facts.json"
    monkeypatch.setattr(fact_approval, "_FACT_CACHE_PATH", cache)

    fact_approval.record_approved_facts({"skill:python"}, source="seed")
    assert not fact_approval.is_auto_approvable(set())
