"""Retry cover-letter generation for the one real cover_failed job that still
qualifies under current scoring standards (Ramp 'Partner Development
Representative | Accounting' -- deterministic_combine confirms family=
customer_facing_or_sales, years=None, cs_degree=False -> 9, still well above
the funnel's min_score=8). The other real cover_failed job (PNC/Tempus
'Software Engineer Associate') was corrected down to fit_score=5 in a sibling
script and is excluded here.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from applypilot import config  # noqa: E402

config.load_env()

from applypilot.scoring.cover_letter import run_cover_letters  # noqa: E402

URL = "https://jobs.ashbyhq.com/ramp/b55447c0-4adc-42eb-9ca2-f88fd44e0e5b"


def main():
    result = run_cover_letters(job_ids=[URL], doc_format="pdf", max_age_days=0)
    print(result)


if __name__ == "__main__":
    main()
