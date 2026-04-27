"""Rewrite embedded-ATS URLs to their canonical form.

Many companies (Databricks, Stripe, Pinterest, Airbnb, …) embed a
Greenhouse application form as an iframe on their own careers page. The
parent URL carries a ``?gh_jid=N`` query param that identifies the
underlying Greenhouse job. The agent can fill the form much faster on
the iframe's source URL directly (no parent-page noise, no captcha
iframe, smaller a11y snapshots) — so we rewrite at discovery / enrich
time before the apply pipeline ever sees the parent URL.

Canonical Greenhouse form:
    https://job-boards.greenhouse.io/{slug}/jobs/{gh_jid}

The slug map below was bootstrapped from the top hosts in our queue.
Add an entry whenever a new employer shows up — verify with::

    curl -sIo /dev/null -w '%{http_code}\\n' \\
        https://job-boards.greenhouse.io/{slug}/jobs/{any_gh_jid}

200/301/302 means the slug is valid; 404 means it isn't.
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse


# Bare host (no leading "www.") → Greenhouse tenant slug.
# Verified 2026-04-26 by probing job-boards.greenhouse.io/{slug}/jobs/{id}.
GREENHOUSE_HOST_SLUGS: dict[str, str] = {
    "stripe.com":              "stripe",
    "databricks.com":          "databricks",
    "pinterestcareers.com":    "pinterest",
    "careers.airbnb.com":      "airbnb",
    "jobs.dropbox.com":        "dropbox",
    "cast.ai":                 "castai",
    "sproutsocial.com":        "sproutsocial",
    "samsara.com":             "samsara",
    "instacart.careers":       "instacart",
    "hubspot.com":             "hubspot",
    "kentik.com":              "kentik",
    "consensys.io":            "consensys",
    "abnormal.ai":             "abnormalsecurity",
    "careers.toasttab.com":    "toast",
    "netskope.com":            "netskope",
    "upsun.com":               "upsun",
    "prizepicks.com":          "prizepicks",
    "fortisgames.com":         "fortisgames",
    "kaseya.com":              "kaseya",
    "nebius.com":              "nebius",
}


def canonicalize_application_url(url: str) -> str:
    """Return the canonical apply URL for ``url`` if a known rewrite applies.

    Currently rewrites:

      * ``https://{host}/...?gh_jid=N`` → ``https://job-boards.greenhouse.io/{slug}/jobs/N``
        when ``{host}`` is in :data:`GREENHOUSE_HOST_SLUGS`.

    All other URLs (including URLs already on a Greenhouse host) are
    returned unchanged.
    """
    if not url or not url.startswith(("http://", "https://")):
        return url

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]

    # Already on Greenhouse — nothing to do.
    if "greenhouse.io" in host:
        return url

    # gh_jid embed → canonical Greenhouse URL
    qs = parse_qs(parsed.query)
    gh_jid_vals = qs.get("gh_jid")
    if gh_jid_vals:
        gh_jid = gh_jid_vals[0]
        if gh_jid.isdigit():
            slug = GREENHOUSE_HOST_SLUGS.get(host)
            if slug:
                return f"https://job-boards.greenhouse.io/{slug}/jobs/{gh_jid}"

    return url
