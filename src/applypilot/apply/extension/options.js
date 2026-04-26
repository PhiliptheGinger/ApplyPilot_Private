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

// Listeners notified when the active section changes; populated by each
// section's own module below. Each listener gets (newName, oldName).
const _sectionListeners = [];
let _currentSection = null;

function onSectionChange(fn) { _sectionListeners.push(fn); }

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
  const old = _currentSection;
  _currentSection = name;
  for (const fn of _sectionListeners) {
    try { fn(name, old); } catch (e) { console.error(e); }
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

// ── Integrations section ──────────────────────────────────────────────────────

const INTEGRATIONS_POLL_MS = 5000;
let _integrationsTimer = null;

function pillForGmail(status) {
  switch (status) {
    case 'valid':         return { cls: 'pill-ok',   text: '✓ Valid' };
    case 'expiring_soon': return { cls: 'pill-warn', text: '⚠ Expiring' };
    case 'expired':       return { cls: 'pill-err',  text: '✗ Expired' };
    case 'missing':       return { cls: 'pill-mute', text: '— Not configured' };
    default:              return { cls: 'pill-err',  text: '✗ Error' };
  }
}

function fmtRelative(seconds) {
  const abs = Math.abs(seconds);
  let unit;
  if (abs >= 86400)     unit = `${Math.floor(abs / 86400)}d`;
  else if (abs >= 3600) unit = `${Math.floor(abs / 3600)}h`;
  else if (abs >= 60)   unit = `${Math.floor(abs / 60)}m`;
  else                  unit = `${abs}s`;
  return seconds < 0 ? `${unit} ago` : `in ${unit}`;
}

function esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderIntegrations(data) {
  const out = [];

  // ── Gmail ──
  const g = data.gmail || {};
  const gp = pillForGmail(g.status);
  out.push(`<div class="card">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
      <strong style="font-size:15px">Gmail</strong>
      <span class="pill ${gp.cls}">${gp.text}</span>
    </div>`);
  if (g.configured) {
    out.push(`<div style="font-size:12px;color:#94a3b8;line-height:1.6">`);
    out.push(`Access token ${fmtRelative(g.expires_in_seconds || 0)}<br>`);
    out.push(`Scopes: <code style="color:#cbd5e1">${esc(g.scopes || 'unknown')}</code>`);
    out.push(`</div>`);
  } else {
    out.push(`<div style="font-size:12px;color:#94a3b8;line-height:1.6">
      No credentials at <code>~/.gmail-mcp/credentials.json</code>.
      Re-authentication will create them.
    </div>`);
  }
  out.push(`<div style="margin-top:12px;display:flex;gap:8px">
    <button class="btn btn-warn" data-int-action="gmail-reauth">Re-authenticate Gmail</button>
    <button class="btn btn-secondary" data-int-action="refresh">Refresh status</button>
  </div>
  <div id="gmail-feedback" style="margin-top:8px;font-size:12px;color:#94a3b8"></div>
  </div>`);

  // ── ATS sessions ──
  out.push(`<h2>ATS Sessions</h2>`);
  const sessions = data.ats_sessions || [];
  if (!sessions.length) {
    out.push(`<div class="placeholder">
      No saved sessions. The pipeline saves ATS cookies automatically after each
      successful apply.
    </div>`);
  } else {
    out.push(`<div class="card" style="padding:0">
      <table style="width:100%;border-collapse:collapse;font-size:13px">
        <thead><tr style="border-bottom:1px solid #2d2d4e">
          <th style="text-align:left;padding:10px 16px;color:#94a3b8;font-weight:500">ATS</th>
          <th style="text-align:left;padding:10px 16px;color:#94a3b8;font-weight:500">Cookies</th>
          <th style="text-align:left;padding:10px 16px;color:#94a3b8;font-weight:500">Age</th>
        </tr></thead>
        <tbody>`);
    for (const s of sessions) {
      const ageStr = s.age_hours == null ? '—'
        : s.age_hours < 1 ? `${Math.round(s.age_hours * 60)}m`
        : s.age_hours < 24 ? `${s.age_hours.toFixed(1)}h`
        : `${Math.floor(s.age_hours / 24)}d`;
      out.push(`<tr style="border-bottom:1px solid #1e293b">
        <td style="padding:10px 16px"><strong>${esc(s.slug)}</strong></td>
        <td style="padding:10px 16px">${s.has_cookies ? '✓' : '—'}</td>
        <td style="padding:10px 16px;color:#94a3b8">${ageStr}</td>
      </tr>`);
    }
    out.push(`</tbody></table></div>`);
  }

  return out.join('');
}

async function loadIntegrations() {
  const target = document.getElementById('integrations-content');
  if (!target) return;
  try {
    const data = await window.AP.probeWorkers('/api/integrations');
    target.innerHTML = renderIntegrations(data);
    bindIntegrationActions();
  } catch (err) {
    target.innerHTML = `<div class="placeholder">
      Could not reach the apply backend.<br>
      Run <code>applypilot apply</code> in a terminal.<br>
      <span style="color:#475569;font-size:11px;display:block;margin-top:8px">${esc(err.message)}</span>
    </div>`;
  }
}

function bindIntegrationActions() {
  document.querySelectorAll('[data-int-action]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const action = btn.dataset.intAction;
      const fb = document.getElementById('gmail-feedback');
      btn.disabled = true;
      try {
        if (action === 'gmail-reauth') {
          if (fb) fb.textContent = 'Starting Gmail OAuth flow — your browser will open the consent screen…';
          const r = await window.AP.probeWorkers('/api/integrations/gmail/reauth', {
            method: 'POST', body: '{}',
          });
          if (r.status === 'started') {
            if (fb) fb.textContent = 'OAuth flow running in a browser tab. After you approve, click "Refresh status".';
          } else {
            if (fb) fb.textContent = `Failed: ${r.error || 'unknown error'}`;
          }
        } else if (action === 'refresh') {
          await loadIntegrations();
        }
      } catch (e) {
        if (fb) fb.textContent = `Error: ${e.message}`;
      } finally {
        btn.disabled = false;
      }
    });
  });
}

// Auto-poll while the Integrations section is the active one. Stop polling
// when the user navigates away so we don't keep hitting the backend forever.
function startIntegrationsPoll() {
  if (_integrationsTimer) return;
  loadIntegrations();
  _integrationsTimer = setInterval(loadIntegrations, INTEGRATIONS_POLL_MS);
}
function stopIntegrationsPoll() {
  if (_integrationsTimer) {
    clearInterval(_integrationsTimer);
    _integrationsTimer = null;
  }
}

onSectionChange(name => {
  if (name === 'integrations') startIntegrationsPoll();
  else                          stopIntegrationsPoll();
});

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
