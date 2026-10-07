# Login and 2FA automation (FW43 research → FW44 → FW66)

Scoped 2026-10-07. FW43 is an explicit research gate the user set before any
login/2FA automation is built: "this should be scouted out before we really
try it."

## What exists

- The apply agent already reads **email** verification codes from Gmail
  (prompt.py instructs up to three searches, spam included).
- Three **SMS relay** readers exist but are wired into nothing:
  `tracking/sms_client.py` (Twilio), `tracking/google_voice_client.py`
  (Gmail-forwarded Google Voice texts), `tracking/adb_sms_client.py`
  (USB-connected Android). A phone-2FA wall today ends in
  `needs_human:sms_verification`.
- Worker Chrome profiles persist across runs (decision #32), so a one-time
  manual login to an employer's ATS may already carry over. Not confirmed.

## Research findings

How login pages detect automation (from vendor write-ups on credential
stuffing and account-takeover defense):

1. **Cross-account device fingerprint correlation** is the strongest
   signal: one device logging into many *different* accounts. ApplyPilot is
   the opposite (one person, one account per employer), which is the
   lowest-risk profile there is.
2. **Velocity**: bursts of logins or failures. A failed-then-retried login
   loop is the real danger; one attempt per employer is not.
3. **Behavioral timing**: fields filled and submitted in milliseconds.
   The Claude agent's natural pacing already looks human; the deterministic
   engine now pauses between fields (FW65).
4. **Headless/stealth fingerprints**: the workers use headed Chrome for
   Testing plus the extension's stealth scripts (decision #42).
5. **IP reputation**: residential home IP — fine.

Sources: [cside: credential stuffing detection](https://cside.com/blog/how-to-detect-credential-stuffing),
[cside: account takeover prevention](https://cside.com/blog/account-takeover-prevention-browser-layer-detection),
[Castle: 6-day credential stuffing attack](https://blog.castle.io/anatomy-of-a-6-day-credential-stuffing-attack-from-2-2m-residential-ips/),
[Workable: managing automated applications](https://help.workable.com/hc/en-us/articles/35293126257815-Managing-AI-generated-and-automated-job-applications),
[Why Workday asks for a new account per company](https://jobwizard.ai/blog/why-does-workday-keep-asking-me-to-make-a-new-account-for-every-company).

**Assessment:** for one real person logging into their *own* accounts, at
human pace, from their home IP, in a headed browser, the detection risk of
*reading a code and typing it in* is low — lower than the form-filling the
pipeline already does. The real risks are behavioral, and they are fixable
with rules:

- never retry a failed login or code more than once per employer per day;
- never request a new code in a loop (rate-limited per employer);
- never create a second account at an employer that already has one
  (`accounts` table, decision #174);
- stop and notify (FW53) on any "unusual activity" / lockout page.

Separate, ATS-side risk: some employers flag *AI-generated applications*
(Workable's article). That is about application content, not login, and is
out of scope here.

## Proposed FW44 design (not built)

1. Prompt step: when the agent sees an SMS/phone-code field, it outputs
   `NEEDS_CODE:sms:<url>` instead of escalating immediately.
2. `run_job` polls the configured relay (`get_latest_verification_code`,
   window = 5 min) every 10 s for up to 3 min; on a code, sends it to the
   agent via the existing resume-prompt mechanic. On no code: today's
   `needs_human:sms_verification`.
3. Guards above enforced in code (a per-employer counter in the DB), not
   only in the prompt.
4. Never memoize a login path in `successful_paths.py` (FW43's explicit
   caution) until a month of logs shows no lockouts.

**Done when:** a supervised run on one employer with phone 2FA completes
with the relay, and the counters provably stop a second attempt.

## Decisions needed

- Is this assessment enough to lift the FW43 gate for **reading and
  entering codes on the candidate's own accounts**? (Account creation stays
  manual either way.)
- Which relay to make the default for FW44: Google Voice (free, needs the
  Gmail forward set up) or ADB (free, needs the phone plugged in)?
