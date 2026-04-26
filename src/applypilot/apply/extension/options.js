// ApplyPilot Settings page.
//
// Hash-routed sidebar nav; each section is a stub that subsequent steps
// (Integrations, Q&A, Preferences, Credentials, Sessions) will fill in.
//
// Backend access mirrors popup.js: probe http://localhost:(BASE_PORT+wid)
// for the worker that has the always-on HTTP server up. /api/* endpoints
// already include CORS headers, so fetch() works from chrome-extension://
// origins.

const BASE_PORT   = 7380;
const MAX_WORKERS = 5;

// ── Sidebar nav: hash-routed section toggle ───────────────────────────────────

const SECTIONS = [
  'integrations',
  'preferences',
  'qa',
  'credentials',
  'sessions',
  'about',
];

function showSection(name) {
  if (!SECTIONS.includes(name)) name = 'integrations';
  document.querySelectorAll('nav a').forEach(a => {
    a.classList.toggle('active', a.dataset.section === name);
  });
  document.querySelectorAll('main section').forEach(s => {
    s.classList.toggle('active', s.id === `sec-${name}`);
  });
  if (location.hash.slice(1) !== name) {
    history.replaceState(null, '', `#${name}`);
  }
}

document.querySelectorAll('nav a').forEach(a => {
  a.addEventListener('click', () => showSection(a.dataset.section));
});

window.addEventListener('hashchange', () => {
  showSection(location.hash.slice(1) || 'integrations');
});

// ── Backend helper ────────────────────────────────────────────────────────────

/**
 * Probe each worker port for an endpoint until one responds OK.
 * Returns the parsed JSON body, or throws if all probes fail.
 */
async function probeWorkers(path, opts = {}) {
  const errors = [];
  for (let wid = 0; wid < MAX_WORKERS; wid++) {
    try {
      const r = await fetch(`http://localhost:${BASE_PORT + wid}${path}`, {
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(2000),
        ...opts,
      });
      if (r.ok) return r.json();
      errors.push(`worker-${wid}: HTTP ${r.status}`);
    } catch (e) {
      errors.push(`worker-${wid}: ${e.message}`);
    }
  }
  throw new Error(
    `No worker responded. Run \`applypilot apply\`. ` +
    `Probed: ${errors.join('; ')}`
  );
}

// Exposed so future section modules can reuse the same probe logic.
window.AP = { probeWorkers, BASE_PORT, MAX_WORKERS };

// ── About: populate version + extension ID ────────────────────────────────────

(async () => {
  try {
    const m = await fetch(chrome.runtime.getURL('manifest.json'))
      .then(r => r.json());
    document.getElementById('ver').textContent = `v${m.version}`;
    document.getElementById('about-version').textContent = m.version;
  } catch { /* ignore */ }
  document.getElementById('about-id').textContent = chrome.runtime.id;
})();

// ── Initial route ─────────────────────────────────────────────────────────────

showSection(location.hash.slice(1) || 'integrations');
