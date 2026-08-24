"""Regression tests for the 2026-08-23 KeyError: 'summary' incident.

A malformed final-attempt LLM response missing "summary" (or "skills") is
already flagged as a validation error by validate_json_fields, but
tailor_resume's "last attempt, assemble whatever we got" path
(tailor.py:789) still called assemble_resume_text(data, profile)
unconditionally -- and that function did data["summary"] / data["skills"]
as direct dict subscripts, raising KeyError and killing the whole job
instead of returning a status="failed_validation" result.

Fix: assemble_resume_text uses .get() with explicit defaults for both
fields (and logs a warning when either is missing) so a malformed response
degrades to a visibly incomplete section instead of crashing.
"""

import unittest
from unittest.mock import MagicMock, patch

from applypilot.scoring.tailor import assemble_resume_text

PROFILE = {
    "personal": {"full_name": "Jordan Lee", "email": "jordan@example.com"},
    "resume_facts": {},
    "skills_boundary": {"languages": ["Python"]},
}


class TestAssembleResumeTextMissingSummary(unittest.TestCase):
    def test_missing_summary_does_not_raise(self):
        data = {
            "title": "Software Engineer",
            "skills": {"Languages": "Python"},
            "experience": [],
            "projects": [],
            "education": [],
        }
        text = assemble_resume_text(data, PROFILE)  # must not raise KeyError
        self.assertIn("SUMMARY", text)

    def test_missing_summary_logs_a_warning(self):
        data = {
            "title": "Software Engineer",
            "skills": {"Languages": "Python"},
            "experience": [],
            "projects": [],
            "education": [],
        }
        with self.assertLogs("applypilot.scoring.tailor", level="WARNING") as cm:
            assemble_resume_text(data, PROFILE)
        self.assertTrue(any("summary" in m.lower() for m in cm.output))


class TestAssembleResumeTextMissingSkills(unittest.TestCase):
    def test_missing_skills_does_not_raise(self):
        data = {
            "title": "Software Engineer",
            "summary": "An engineer who builds things.",
            "experience": [],
            "projects": [],
            "education": [],
        }
        text = assemble_resume_text(data, PROFILE)  # must not raise KeyError
        self.assertIn("TECHNICAL SKILLS", text)

    def test_missing_both_summary_and_skills_does_not_raise(self):
        data = {"title": "Software Engineer", "experience": [], "projects": [], "education": []}
        text = assemble_resume_text(data, PROFILE)
        self.assertIn("SUMMARY", text)
        self.assertIn("TECHNICAL SKILLS", text)


class TestAssembleResumeTextStillWorksForWellFormedData(unittest.TestCase):
    def test_well_formed_data_unaffected(self):
        data = {
            "title": "Software Engineer",
            "summary": "Builds reliable backend systems.",
            "skills": {"Languages": "Python, Go"},
            "experience": [{"header": "Engineer at X", "bullets": ["Built stuff"]}],
            "projects": [],
            "education": [],
        }
        text = assemble_resume_text(data, PROFILE)
        self.assertIn("Builds reliable backend systems.", text)
        self.assertIn("Python, Go", text)


class TestTailorResumeFinalAttemptMissingSummary(unittest.TestCase):
    """Reproduces the exact reported crash end to end: the LLM's response on
    the LAST retry attempt omits "summary" (and therefore fails
    validate_json_fields), and tailor_resume must return a
    status="failed_validation" result instead of raising."""

    def test_final_attempt_missing_summary_returns_failed_validation_not_raise(self):
        from applypilot.scoring import tailor as tailor_mod

        job = {
            "title": "Support Technician",
            "url": "https://example.com/j1",
            "full_description": "- Troubleshoot issues\n",
        }
        malformed_response = (
            '{"title":"Support Technician","skills":{"Languages":"Python"},'
            '"experience":[],"projects":[],"education":[]}'  # "summary" omitted
        )
        mock_client = MagicMock()
        mock_client.chat.return_value = malformed_response
        mock_client.has_cloud_available = lambda: True

        with (
            patch.object(tailor_mod, "get_stage_client", return_value=mock_client),
            patch.object(tailor_mod, "is_local_configured", return_value=False),
        ):
            # max_retries=0 -- the FIRST attempt is also the LAST attempt,
            # exercising tailor.py:789's "last attempt, assemble whatever we
            # got" path on the very first try.
            tailored, report = tailor_mod.tailor_resume(
                "Original resume.",
                job,
                PROFILE,
                max_retries=0,
            )

        self.assertEqual(report["status"], "failed_validation")
        self.assertIsInstance(tailored, str)  # assembled something, didn't crash
        self.assertIn("SUMMARY", tailored)


if __name__ == "__main__":
    unittest.main()
