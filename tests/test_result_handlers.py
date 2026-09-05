"""Tests for apply/result_handlers.py's free-text result inference.

2026-09-04: this function had no test coverage at all before this file.
Added while fixing a real false-positive bug found by checking it against
realistic agent output rather than trusting it as-is (see the negation-cue
guard in _infer_result_from_output's module comment).
"""

import unittest

from applypilot.apply.result_handlers import _infer_result_from_output


class TestInferResultFromOutput(unittest.TestCase):
    def test_genuine_strong_success_phrase_still_detected(self):
        self.assertEqual(
            _infer_result_from_output("The application submitted successfully and I received a confirmation email."),
            "applied",
        )

    def test_genuine_weak_success_needs_two_signals(self):
        text = "Great news -- your application was sent to the hiring team and application received confirmation shown on screen."
        self.assertEqual(_infer_result_from_output(text), "applied")

    def test_single_weak_signal_alone_is_not_enough(self):
        self.assertIsNone(_infer_result_from_output("The page showed an 'application sent' banner briefly."))

    def test_negated_failure_containing_success_phrase_not_misread(self):
        """2026-09-04 regression: the exact bug found -- a FAILURE
        description that happens to contain a literal success phrase as a
        substring must not be misread as 'applied'."""
        cases = [
            "I was unable to get the application submitted successfully due to a CAPTCHA that could not be solved.",
            "I could not get this application submitted successfully; giving up after 3 attempts.",
            "The form failed to confirm the application was sent to the employer.",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertIsNone(_infer_result_from_output(text))

    def test_negation_far_from_the_phrase_does_not_suppress_a_real_positive(self):
        """A negation cue elsewhere in the output, unrelated to the actual
        success sentence, must not suppress a genuine positive -- the
        negation window is scoped to text immediately before the match."""
        text = (
            "The site was unable to load the cover letter preview earlier, but that was resolved. "
            "The application submitted successfully and I received a confirmation email. Thank you for applying!"
        )
        self.assertEqual(_infer_result_from_output(text), "applied")

    def test_captcha_pattern_still_detected(self):
        self.assertEqual(
            _infer_result_from_output("This listing is blocked by captcha and cannot proceed further."),
            "captcha",
        )

    def test_already_applied_pattern_still_detected(self):
        self.assertEqual(
            _infer_result_from_output("The site says: you have already applied to this position."),
            "already_applied",
        )

    def test_no_signal_returns_none(self):
        self.assertIsNone(_infer_result_from_output("I navigated to the page and looked around."))


if __name__ == "__main__":
    unittest.main()
