# Reaching LinkedIn jobs (FW64 → FW38) and the locality data pack (FW71)

Scoped 2026-10-07.

## FW38 — 6,108 LinkedIn jobs with no application_url

Known (decisions #184, FW38): jobspy never fills `job_url_direct`; upgrading
jobspy doesn't help (tested); and automating linkedin.com directly is ruled
out (decision #168). The human-first flow (#187) works but costs a human
click per job.

## FW64 — find the same job on the employer's own ATS

Every LinkedIn row has a title, a company name and a location. Many of
those employers post the same requisition on a public board this project
already crawls (Greenhouse, Lever, Ashby, Workday, SmartRecruiters).

### Matching design

1. **Company resolution.** Map the LinkedIn company string to a board:
   normalize (lowercase, strip "Inc", "LLC", punctuation), then look it up
   in the existing employer registries (`config/*_employers.yaml`) and in
   `jobs.company` values extracted from ATS URLs (FW55 now makes those
   reliable). Unknown companies: try the public board endpoints by slug
   guess (`boards-api.greenhouse.io/v1/boards/{slug}`, `api.lever.co/v0/
   postings/{slug}`, `api.ashbyhq.com/posting-api/job-board/{slug}`) and
   cache hits and misses.
2. **Posting match** within that board: title similarity (token-set ratio
   after removing seniority noise like "II", "Sr."), location compatibility
   (reuse `location_filter`), and posted-date proximity. Accept only above a
   high threshold; anything ambiguous stays `manual_only`.
3. **Write** the ATS posting URL to `application_url`, record
   `matched_from='linkedin'` + confidence, and let the normal pipeline take
   it. Never touches linkedin.com.

### Duplicate detection (the other half of FW64)

The same matcher, run at insert time across sources, marks a new row as a
duplicate of an existing one (same company + matching title + compatible
location within 30 days) so a job can't be applied to twice. The browser
extension can then warn when you open a posting the DB already has as
`applied`.

### Stages

1. Offline matcher + measured precision on a hand-labeled sample of 100
   LinkedIn rows (needs the live DB to sample from; matching code is
   cloud-ok). **Ship only at ≥ 95% precision.**
2. Batch job `applypilot match-linkedin --limit N` that fills
   `application_url` for confident matches.
3. Insert-time duplicate check and the extension warning.

## FW71 — locality data pack

Two separate datasets:

1. **Places** (auto-fill the location filter). Public-domain US Census
   place/county files, or GeoNames (CC-BY), reduced to one small file per
   state: place name, county, lat/long. The wizard asks for a commute
   radius and fills `accept_patterns` with every place inside it. A
   per-state file is roughly 100–400 KB; download on demand from a pinned
   URL and cache in `~/.applypilot/geo/`. Cloud-buildable end to end.
2. **Employers** near the candidate and which ATS each uses. The NC and
   Seattle lists in `docs/` are hand-built examples. Sourcing at scale
   without scraping LinkedIn: ATS public boards already list a company's
   office locations per posting, so crawling known boards and indexing by
   location builds the dataset as a by-product of discovery. Seed it from
   the existing YAML lists.

### Decided 2026-10-07 (#234)

- US first: Census place files (public domain). GeoNames for worldwide later.
- Hosted on GitHub (a data branch or release asset), never shipped in the package. The wizard downloads only the user's state (or country, later) and caches it in `~/.applypilot/geo/`.
- Employer index: built from discovery runs and seed lists, published on GitHub the same way; nothing large stored locally.

### Original decisions needed

- Places source: Census (public domain, US-only) or GeoNames (CC-BY,
  worldwide, requires attribution)?
- Should the employer index ship in the repo, or only be built locally
  from each user's own discovery runs?
