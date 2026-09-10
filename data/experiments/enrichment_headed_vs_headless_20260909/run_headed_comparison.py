"""Real headed-vs-headless comparison for enrichment's detail-page scrape,
requested directly: "let's see if enrichment does better headed, at least
with a decent enough sample." CLAUDE.md decision #77 documented Intel/
Workday as the flagship case (an explicit 20s wait for Workday's own
description selector never resolving -- looks like bot-detection serving a
decoy shell to the enrichment browser, which launches headless with none
of the apply-stage's extra stealth measures).

Pulls a fresh batch of real Intel job URLs directly from Workday's public
search JSON API (no bot-detection concern there -- it's a plain API, not a
rendered page) via discovery.workday.workday_search, then runs the real,
unmodified detail.scrape_detail_page against each URL twice: once with the
current headless=True config, once with headless=False. Same code path
both times (detail.py's own launch_opts, toggled via a module-level
override for this experiment only -- see the accompanying monkeypatch).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from patchright.sync_api import sync_playwright  # noqa: E402

from applypilot.discovery.workday import load_employers, workday_search  # noqa: E402
from applypilot.enrichment.detail import UA, _STEALTH_INIT_SCRIPT, scrape_detail_page  # noqa: E402


def get_sample_urls(n: int = 10) -> list[str]:
    employers = load_employers()
    intel = employers["intel"]
    result = workday_search(intel, "technician", limit=n * 2)
    jobs = result.get("jobPostings", [])
    base = intel["base_url"]
    site_id = intel["site_id"]
    urls = [f"{base}/{site_id}{j['externalPath']}" for j in jobs]
    return urls[:n]


def run_batch(urls: list[str], headless: bool) -> list[dict]:
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, chromium_sandbox=True)
        context = browser.new_context(user_agent=UA)
        context.add_init_script(_STEALTH_INIT_SCRIPT)
        page = context.new_page()
        for url in urls:
            t0 = time.time()
            r = scrape_detail_page(page, url)
            r["url"] = url
            r["headless"] = headless
            r["wall_time"] = time.time() - t0
            r["desc_len"] = len(r.get("full_description") or "")
            results.append(r)
            print(
                f"  [{'headless' if headless else 'headed  '}] "
                f"status={r['status']:8s} tier={r.get('tier_used')} "
                f"desc_len={r['desc_len']:5d} time={r['wall_time']:.1f}s  {url[-60:]}"
            )
        browser.close()
    return results


def main() -> None:
    urls = get_sample_urls(n=10)
    print(f"{len(urls)} fresh Intel job URLs pulled live from Workday's search API:")
    for u in urls:
        print(" ", u)

    print("\n=== HEADLESS (current default) ===")
    headless_results = run_batch(urls, headless=True)

    print("\n=== HEADED (headless=False) ===")
    headed_results = run_batch(urls, headless=False)

    out = REPO_ROOT / "data/experiments/enrichment_headed_vs_headless_20260909/results.json"
    out.write_text(json.dumps({"headless": headless_results, "headed": headed_results}, indent=2))

    def summarize(label, results):
        ok = sum(1 for r in results if r["status"] == "ok")
        partial = sum(1 for r in results if r["status"] == "partial")
        err = sum(1 for r in results if r["status"] == "error")
        avg_len = sum(r["desc_len"] for r in results) / len(results) if results else 0
        avg_time = sum(r["wall_time"] for r in results) / len(results) if results else 0
        print(f"{label}: ok={ok} partial={partial} error={err} avg_desc_len={avg_len:.0f} avg_time={avg_time:.1f}s")

    print("\n=== SUMMARY ===")
    summarize("headless", headless_results)
    summarize("headed  ", headed_results)


if __name__ == "__main__":
    main()
