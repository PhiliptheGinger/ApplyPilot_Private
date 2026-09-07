"""One-off diagnostic for the WeWorkRemotely/Intel-Workday boilerplate
enrichment bug (CLAUDE.md Future Work item 5, decision #75).

Reuses detail.py's own browser setup (patchright, headless chromium, same
UA) so results reflect what production enrichment actually sees.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from applypilot.enrichment.detail import UA  # noqa: E402
from patchright.sync_api import sync_playwright  # noqa: E402

WWR_URL = "https://weworkremotely.com/remote-jobs/samsara-staff-software-engineer"
INTEL_URL = "https://intel.wd1.myworkdayjobs.com/External/job/Module-Process-Engineer_JR0284332"


def diagnose_wwr(page):
    print("=== WeWorkRemotely ===")
    page.goto(WWR_URL, timeout=45000, wait_until="domcontentloaded")
    page.wait_for_load_state("domcontentloaded", timeout=15000)
    time.sleep(1)

    # Find every element containing "PROMOTED" to see the ad widget's own classes.
    promoted = page.evaluate(
        """() => {
            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            let node, hits = [];
            while ((node = walker.nextNode())) {
                if (node.textContent.includes('PROMOTED')) {
                    let el = node.parentElement;
                    for (let i = 0; i < 4 && el; i++) {
                        hits.push({depth: i, tag: el.tagName, id: el.id, cls: el.className});
                        el = el.parentElement;
                    }
                    break;
                }
            }
            return hits;
        }"""
    )
    print("PROMOTED widget ancestor chain:", promoted)

    # Look for likely real-description containers by known WWR conventions.
    candidates = page.evaluate(
        """() => {
            const sels = ['.listing-container', '#job-listing-show-container', '.job-listing',
                          '.listing-body', '#job-show-container', '.company-card', 'section',
                          '[class*="description"]', '[id*="job"]'];
            const out = [];
            for (const s of sels) {
                document.querySelectorAll(s).forEach(el => {
                    const t = el.innerText.trim();
                    if (t.length > 80) {
                        out.push({sel: s, id: el.id, cls: (el.className || '').toString().slice(0,80), len: t.length, snippet: t.slice(0,150)});
                    }
                });
            }
            return out;
        }"""
    )
    print(f"\ncandidate real-content containers ({len(candidates)}):")
    for c in candidates[:15]:
        print(" ", c)


def diagnose_intel(page):
    print("\n=== Intel / Workday ===")
    page.goto(INTEL_URL, timeout=45000, wait_until="domcontentloaded")
    page.wait_for_load_state("domcontentloaded", timeout=15000)
    body_len_early = page.evaluate("() => document.body.innerText.length")
    print(f"body text length right after domcontentloaded: {body_len_early}")

    try:
        page.wait_for_selector('[data-automation-id="jobPostingDescription"]', timeout=20000)
        found = True
    except Exception as e:  # noqa: BLE001
        found = False
        print(f"wait_for_selector('[data-automation-id=jobPostingDescription]') failed: {e}")

    body_len_late = page.evaluate("() => document.body.innerText.length")
    print(f"body text length after waiting for the automation-id selector: {body_len_late}")
    print(f"selector found: {found}")

    if found:
        desc = page.evaluate(
            "() => document.querySelector('[data-automation-id=\"jobPostingDescription\"]').innerText"
        )
        print(f"description length: {len(desc)}")
        print("snippet:", desc[:300])


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, chromium_sandbox=True)
        context = browser.new_context(user_agent=UA)
        page = context.new_page()
        try:
            diagnose_wwr(page)
        except Exception as e:  # noqa: BLE001
            print("WWR diagnosis failed:", e)
        try:
            diagnose_intel(page)
        except Exception as e:  # noqa: BLE001
            print("Intel diagnosis failed:", e)
        browser.close()


if __name__ == "__main__":
    main()
