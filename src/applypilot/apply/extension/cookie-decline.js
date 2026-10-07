// ApplyPilot: decline cookie-consent banners (FW51, 2026-10-07).
//
// Saves the apply agent a tool call (and the confusion of a modal over the
// form) on most pages, and declines optional tracking instead of accepting
// it. Conservative by design -- application forms contain buttons like
// "Decline to self-identify" that must never be clicked:
//
//   1. Known consent-manager buttons (OneTrust, Cookiebot, Didomi, ...) are
//      clicked by their own reject-button selectors.
//   2. Otherwise, only inside an element whose id/class contains "cookie"
//      AND whose text mentions cookies, click a button whose WHOLE label is
//      a reject phrase ("Reject all", "Decline", "Only necessary", ...).
//      A label that mentions "accept" is never clicked.
//
// Turn off by setting chrome.storage.local "autoDeclineCookies" to false.

(function () {
  'use strict';
  if (window.__apCookieDecline) return;
  window.__apCookieDecline = true;

  var KNOWN_REJECT_SELECTORS = [
    '#onetrust-reject-all-handler',
    '#CybotCookiebotDialogBodyButtonDecline',
    '#CybotCookiebotDialogBodyLevelButtonLevelOptinDeclineAll',
    '#didomi-notice-disagree-button',
    '.osano-cm-denyAll',
    '.cky-btn-reject',
    '.cmplz-deny',
    '.iubenda-cs-reject-btn',
    '.cm-btn-decline',
    '[data-tid="banner-decline"]',
    '#truste-consent-required',
    '.cc-deny',
  ];

  var REJECT_LABEL = /^(reject( all)?( cookies)?|reject (optional|non-essential|additional) cookies|decline( all)?( cookies)?|deny( all)?|refuse( all)?|(only|use only|use) (necessary|essential|required)( cookies)?( only)?|(necessary|essential|strictly necessary) (cookies )?only|continue without accepting|no,? thanks)$/i;

  function visible(el) {
    if (!el || !el.getBoundingClientRect) return false;
    var r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return false;
    var s = window.getComputedStyle(el);
    return s.visibility !== 'hidden' && s.display !== 'none';
  }

  function labelOf(el) {
    return (el.innerText || el.textContent || el.value || el.getAttribute('aria-label') || '')
      .replace(/\s+/g, ' ').trim();
  }

  function tryKnown() {
    for (var i = 0; i < KNOWN_REJECT_SELECTORS.length; i++) {
      var el = document.querySelector(KNOWN_REJECT_SELECTORS[i]);
      if (el && visible(el) && !/accept/i.test(labelOf(el))) {
        el.click();
        return KNOWN_REJECT_SELECTORS[i];
      }
    }
    return null;
  }

  function tryGeneric() {
    var containers = document.querySelectorAll('[id*="cookie" i], [class*="cookie" i]');
    for (var i = 0; i < containers.length; i++) {
      var box = containers[i];
      if (!visible(box)) continue;
      if (!/cookie/i.test(box.innerText || '')) continue;
      var buttons = box.querySelectorAll('button, [role="button"], a, input[type="button"], input[type="submit"]');
      for (var j = 0; j < buttons.length; j++) {
        var b = buttons[j];
        var label = labelOf(b);
        if (!label || label.length > 60 || /accept|agree|allow/i.test(label)) continue;
        if (REJECT_LABEL.test(label) && visible(b)) {
          b.click();
          return 'generic: ' + label;
        }
      }
    }
    return null;
  }

  function attempt() {
    try {
      return tryKnown() || tryGeneric();
    } catch (e) {
      return null;
    }
  }

  function run() {
    var done = attempt();
    if (done) {
      window.__apCookieDeclined = done;
      return;
    }
    // Banners often appear a few seconds after load: watch for up to 15s.
    var deadline = Date.now() + 15000;
    var pending = false;
    var obs = new MutationObserver(function () {
      if (pending) return;
      pending = true;
      setTimeout(function () {
        pending = false;
        var hit = attempt();
        if (hit || Date.now() > deadline) {
          if (hit) window.__apCookieDeclined = hit;
          obs.disconnect();
        }
      }, 250);
    });
    obs.observe(document.documentElement, { childList: true, subtree: true });
    setTimeout(function () { obs.disconnect(); }, 15500);
  }

  try {
    if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
      chrome.storage.local.get({ autoDeclineCookies: true }, function (prefs) {
        if (prefs && prefs.autoDeclineCookies === false) return;
        run();
      });
      return;
    }
  } catch (e) { /* fall through */ }
  run();
})();
