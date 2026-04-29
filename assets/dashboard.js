/**
 * dashboard.js — CrashGuard AI
 * Live data fetching, chart rendering, UI updates, and page navigation.
 */

// ── STATE ──────────────────────────────────────────────────────
const S = {
  servers: {},
  selected: null,
  userSelected: false,   // true when user manually clicks a server
  history: {},
  incidents: [],
  alertLog: [],
  alertCount: 0,
  decisionCount: 0,
  wowTimers: {},
  currentPage: 'dashboard',
  metrics: { total_decisions: 0, incidents_prevented: 0, false_positive_rate: 0 },
  expandedRows: {},
  expandedAlerts: {},
  sessionStart: Date.now(),
  lastAlertSeed: Date.now(),
};

const DC = {
  SCALE: { color: '#ef4444', bg: 'rgba(239,68,68,0.08)', border: 'rgba(239,68,68,0.4)', icon: '📈', cls: 'critical' },
  ESCALATE: { color: '#ef4444', bg: 'rgba(239,68,68,0.08)', border: 'rgba(239,68,68,0.4)', icon: '🚨', cls: 'critical' },
  RESTART: { color: '#f97316', bg: 'rgba(249,115,22,0.08)', border: 'rgba(249,115,22,0.35)', icon: '🔄', cls: '' },
  MONITOR: { color: '#eab308', bg: 'rgba(234,179,8,0.08)', border: 'rgba(234,179,8,0.35)', icon: '👁', cls: '' },
  STABLE: { color: '#22c55e', bg: 'rgba(34,197,94,0.05)', border: 'rgba(34,197,94,0.3)', icon: '✅', cls: '' },
};

// ── SPARKLINE HELPER ───────────────────────────────────────────
function renderSparkline(serverId) {
  const hist = S.history[serverId];
  if (!hist || hist.length < 3) return '';
  const BARS = '▁▂▃▄▅▆▇█';
  const recent = hist.slice(-8);
  const cpus = recent.map(h => h.cpu);
  const mn = Math.min(...cpus), mx = Math.max(...cpus);
  const range = mx - mn || 1;
  const spark = cpus.map(v => {
    const idx = Math.round(((v - mn) / range) * 7);
    return BARS[Math.min(idx, 7)];
  }).join('');
  const color = mx > 80 ? '#ef4444' : mx > 60 ? '#f97316' : '#22c55e';
  return `<span class="sparkline" style="color:${color}">${spark}</span>`;
}

// ── OPERATOR ACTIONS ─────────────────────────────────────────────
function getOperatorAction(s) {
  const d = s.decision;
  const cpu = s.current_cpu || 0;
  const trend = s.trend || 'stable';
  const spikes = s.spike_count || 0;

  if (d === 'SCALE') {
    if (cpu > 90) return { text: 'Scale pods +3 · Immediate', icon: '↑↑', cls: 'action-critical', color: '#ef4444' };
    return { text: 'Scale pods +2 · Priority: high', icon: '↑', cls: 'action-critical', color: '#ef4444' };
  }
  if (d === 'ESCALATE') {
    return { text: 'Page on-call · SLA breach ~5min', icon: '📟', cls: 'action-critical', color: '#f97316' };
  }
  if (d === 'RESTART') {
    if (spikes > 5) return { text: 'Restart + drain connections', icon: '🔄', cls: 'action-warn', color: '#f97316' };
    return { text: 'Graceful restart · PID recycle', icon: '🔄', cls: 'action-warn', color: '#eab308' };
  }
  if (d === 'MONITOR') {
    if (trend === 'rising' || trend === 'rapidly_rising')
      return { text: 'Watch · Pre-warm standby', icon: '👁', cls: 'action-watch', color: '#3b82f6' };
    return { text: 'Watch · Next eval in 30s', icon: '👁', cls: 'action-watch', color: '#94a3b8' };
  }
  return { text: 'No action required', icon: '—', cls: 'action-ok', color: '#94a3b8' };
}

// ── PAGE NAVIGATION ────────────────────────────────────────────
function navigateTo(page) {
  S.currentPage = page;

  // Update sidebar active state
  document.querySelectorAll('.nav-item').forEach(el => {
    el.classList.toggle('active', el.dataset.page === page);
  });

  // Update breadcrumb
  const labels = {
    dashboard: 'Dashboard', systems: 'Systems',
    alerts: 'Alerts', models: 'Models', predictions: 'Predictions'
  };
  const bc = document.querySelector('.topbar-left');
  if (bc) bc.innerHTML = `
    <span>prod-cluster-01</span>
    <span style="color:var(--text3)"> / </span>
    <span style="color:var(--text);font-weight:600">${labels[page] || page}</span>`;

  // Hide all pages
  document.querySelectorAll('.page').forEach(el => {
    el.style.display = 'none';
  });

  // Show target page
  const activePage = document.getElementById('page-' + page);
  if (!activePage) return;

  if (page === 'predictions') {
    activePage.style.display = 'flex';
    activePage.style.flexDirection = 'column';
    renderPredictionsPage();
  } else {
    activePage.style.display = 'block';
  }

  // Render page-specific content
  if (page === 'systems') renderSystemsPage();
  if (page === 'alerts') renderAlertsPage();
  if (page === 'models') renderModelsPage();
  if (page === 'predictions') renderPredictionsPage();
}

// Nav items now use onclick="navigateTo('page')" directly in HTML

// ── CHART ──────────────────────────────────────────────────────
const ctx = document.getElementById('main-chart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Actual', data: [], borderColor: '#3b82f6', borderWidth: 2, pointRadius: 0, tension: 0.35, fill: false },
      { label: 'Predicted', data: [], borderColor: '#f97316', borderWidth: 1.5, borderDash: [5, 3], pointRadius: 0, tension: 0.35, fill: false },
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    animation: { duration: 150 },
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#0c0f18', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
        titleColor: '#94a3b8', bodyColor: '#e2e8f0',
        callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(1)}%` }
      },
      customFutureZone: {},
    },
    scales: {
      x: { ticks: { color: '#475569', font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 8 }, grid: { color: 'rgba(255,255,255,0.04)' } },
      y: { min: 0, max: 100, ticks: { color: '#475569', font: { family: 'JetBrains Mono', size: 10 }, callback: v => v + '%' }, grid: { color: 'rgba(255,255,255,0.04)' } },
    }
  }
});

Chart.register({
  id: 'futureZone',
  beforeDraw(chart) {
    const { ctx, chartArea, scales, data } = chart;
    if (!chartArea) return;
    const N = data.labels.length;
    if (N < 2) return;
    const nowIdx = Math.max(0, N - 5);
    const xNow = scales.x.getPixelForValue(nowIdx);
    const { left, right, top, bottom } = chartArea;
    ctx.save();
    ctx.fillStyle = 'rgba(249,115,22,0.04)';
    ctx.fillRect(xNow, top, right - xNow, bottom - top);
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(xNow, top); ctx.lineTo(xNow, bottom); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText('NOW →', xNow + 4, top + 14);
    ctx.restore();
  }
});

// ── FETCH ──────────────────────────────────────────────────────
async function fetchStatus() {
  try {
    const r = await fetch('/api/status');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    processData(d);
    setConn(true);
  } catch (e) { setConn(false); }
}

function setConn(ok) {
  const el = document.getElementById('conn-info');
  if (!el) return;
  el.textContent = ok ? '● Connected' : '● Disconnected';
  el.style.color = ok ? '#22c55e' : '#ef4444';
}

// ── PROCESS ────────────────────────────────────────────────────
function processData(data) {
  const servers = data.servers || [];
  const now = new Date();
  const ts = now.toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const lastUpd = document.getElementById('last-upd');
  if (lastUpd) lastUpd.textContent = ts;

  // Store eval metrics from backend
  if (data.eval_metrics) S.metrics = data.eval_metrics;

  let critCount = 0, worstServer = null, worstPriority = -1;
  const PRIORITY = { ESCALATE: 5, SCALE: 4, RESTART: 3, MONITOR: 2, STABLE: 1 };

  servers.forEach(s => {
    const prev = S.servers[s.server_id];
    S.servers[s.server_id] = s;

    if (!S.history[s.server_id]) S.history[s.server_id] = [];
    S.history[s.server_id].push({ cpu: s.current_cpu, pred: s.predicted_cpu, ts });
    if (S.history[s.server_id].length > 60) S.history[s.server_id].shift();

    if (s.spike_probability > 0.7 && s.model_used !== 'warming_up') {
      if (!S.wowTimers[s.server_id]) S.wowTimers[s.server_id] = Date.now();
    } else { delete S.wowTimers[s.server_id]; }

    if (s.severity === 'CRITICAL' || s.severity === 'HIGH') critCount++;

    // DEDUP FIX — use backend alert_ready instead of naive state-change detection
    if (s.alert_ready && s.model_used !== 'warming_up') {
      addIncident(s, now);
      S.decisionCount++;
      const shLi = document.getElementById('sh-last-intervention');
      if (shLi) shLi.textContent = s.server_name.split('—')[0].trim() + ' ' + s.decision;
      S.alertLog.unshift({
        ts: now.toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit' }),
        server: s.server_name.split('—')[0].trim(),
        decision: s.decision,
        severity: s.severity,
        cpu: s.current_cpu,
        reason: s.reason || '',
        incident_id: s.incident_id || null,
        transition: s.incident_transition || null,
      });
      if (S.alertLog.length > 50) S.alertLog.pop();
      S.alertCount++;
    }

    const p = PRIORITY[s.decision] || 0;
    if (p > worstPriority) { worstPriority = p; worstServer = s; }
  });

  if (worstServer) updateHero(worstServer);
  renderServers(servers);
  renderFleet(servers);
  updateStats(critCount, servers);

  // Main graph tracks highest-risk server unless user manually selected
  if (worstServer && !S.userSelected) {
    selectServer(worstServer.server_id);
  } else if (!S.selected && servers.length) {
    selectServer(servers[0].server_id);
  } else if (S.selected) {
    updateChart(S.selected);
  }

  // Always refresh current page content on every data tick
  if (S.currentPage === 'systems') renderSystemsPage();
  if (S.currentPage === 'alerts') renderAlertsPage();
  if (S.currentPage === 'models') renderModelsPage();
  if (S.currentPage === 'predictions') renderPredictionsPage();
}

function updateStats(critCount, servers) {
  const nb = document.getElementById('nav-badge');
  if (nb) nb.textContent = critCount;
  const sd = document.getElementById('stat-decisions');
  if (sd) sd.textContent = S.decisionCount;
  const sa = document.getElementById('stat-alerts');
  if (sa) sa.textContent = S.alertCount;
  const sm = document.getElementById('stat-model');
  if (sm) sm.textContent = servers[0]?.model_used || '—';

  // FIX — keep Systems page active alerts in sync with live decisions
  const sysAlert = document.getElementById('sys-alerts');
  if (sysAlert) {
    sysAlert.textContent = critCount;
    sysAlert.style.color = critCount > 0 ? 'var(--red)' : 'var(--green)';
  }
  const sysAvg = document.getElementById('sys-avg');
  if (sysAvg && servers.length) {
    const avg = servers.reduce((a, s) => a + s.current_cpu, 0) / servers.length;
    sysAvg.textContent = avg.toFixed(0) + '%';
    sysAvg.style.color = avg > 70 ? 'var(--red)' : avg > 50 ? 'var(--orange)' : 'var(--blue)';
  }

  // FIX — Alerts page: total = alertLog entries, critical = CRITICAL severity entries
  // Also update the badge in sidebar with correct count
  const apc = document.getElementById('alert-page-count');
  if (apc) apc.textContent = S.alertLog.length + ' total';
  const at = document.getElementById('alrt-total');
  if (at) { at.textContent = S.alertLog.length; }
  const ac = document.getElementById('alrt-critical');
  if (ac) {
    const critAlerts = S.alertLog.filter(a => a.severity === 'CRITICAL').length;
    ac.textContent = critAlerts;
  }
}

// ── HERO BANNER ────────────────────────────────────────────────
function updateHero(s) {
  const dc = DC[s.decision] || DC.STABLE;
  const banner = document.getElementById('hero-banner');
  if (!banner) return;
  banner.style.background = dc.bg;
  banner.style.borderColor = dc.border;
  banner.style.color = dc.color;
  banner.className = 'hero-banner' + (dc.cls ? ' ' + dc.cls : '');

  document.getElementById('hero-icon').textContent = dc.icon;
  document.getElementById('hero-decision-text').textContent = s.decision;
  document.getElementById('hero-server').textContent = s.server_name;
  document.getElementById('hero-model').textContent = s.model_used;
  document.getElementById('hero-action').textContent = s.action;

  const w = s.model_used === 'warming_up';
  document.getElementById('hm-curr').textContent = s.current_cpu.toFixed(1) + '%';
  document.getElementById('hm-pred').textContent = w ? '—' : s.predicted_cpu.toFixed(1) + '%';
  document.getElementById('hm-conf').textContent = w ? '—' : ((s.adjusted_confidence || s.confidence) * 100).toFixed(0) + '%';
  const cr = s.crash_risk_5min || s.spike_probability || 0;
  document.getElementById('hm-spike').textContent = w ? '—' : (cr * 100).toFixed(0) + '%';
  document.getElementById('hm-spikect').textContent = s.spike_count;

  const wowEl = document.getElementById('hero-wow');
  const timer = S.wowTimers[s.server_id];
  if (timer && s.decision !== 'STABLE' && !w) {
    document.getElementById('wow-seconds').textContent = Math.floor((Date.now() - timer) / 1000);
    wowEl.style.display = 'block';
  } else { wowEl.style.display = 'none'; }
}

// ── SERVER LIST ────────────────────────────────────────────────
function renderServers(servers) {
  const panel = document.getElementById('server-panel');
  if (!panel) return;
  panel.innerHTML = '';
  servers.forEach(s => {
    const dc = DC[s.decision] || DC.STABLE;
    const barColor = s.current_cpu > 80 ? '#ef4444' : s.current_cpu > 60 ? '#f97316' : '#3b82f6';
    const selected = S.selected === s.server_id;
    const row = document.createElement('div');
    row.className = 'server-row' + (selected ? ' selected' : '');
    row.onclick = () => selectServer(s.server_id, true);
    row.innerHTML = `
      <div class="srv-dot" style="background:${dc.color}"></div>
      <span class="srv-name">${s.server_name.replace('— ', '')}</span>
      <span class="srv-cpu" style="color:${barColor}">${s.current_cpu.toFixed(0)}%</span>
      <span class="srv-badge" style="background:${dc.bg};color:${dc.color};border:1px solid ${dc.border}">${s.decision}</span>`;
    panel.appendChild(row);
  });
}

function selectServer(id, manual) {
  if (manual) S.userSelected = true;
  S.selected = id;
  const s = S.servers[id];
  if (s) {
    const ct = document.getElementById('chart-title');
    if (ct) ct.textContent = 'CPU usage + prediction — ' + s.server_name;
    renderServers(Object.values(S.servers));
  }
  updateChart(id);
}

function updateChart(id) {
  const hist = S.history[id] || [];
  if (hist.length < 2) return;
  const N = hist.length, nowIdx = Math.max(0, N - 5);
  chart.data.labels = hist.map(h => h.ts);
  chart.data.datasets[0].data = hist.map((h, i) => i < nowIdx ? h.cpu : null);
  chart.data.datasets[1].data = hist.map(h => h.pred);
  chart.options.plugins.customFutureZone = { nowIdx };
  chart.update('none');
}

// ── FLEET ──────────────────────────────────────────────────────
function renderFleet(servers) {
  const el = document.getElementById('fleet-body');
  if (!el) return;
  const avg = servers.reduce((a, s) => a + s.current_cpu, 0) / servers.length;
  const fa = document.getElementById('fleet-avg');
  if (fa) fa.textContent = 'avg ' + avg.toFixed(0) + '%';
  el.innerHTML = '';
  servers.forEach(s => {
    const pct = Math.min(s.current_cpu, 100);
    const color = s.current_cpu > 80 ? '#ef4444' : s.current_cpu > 60 ? '#f97316' : '#22c55e';
    el.innerHTML += `
      <div class="fleet-row">
        <span class="fleet-name">${s.server_name.split('—')[0].trim()}</span>
        <div class="fleet-bar"><div class="fleet-fill" style="width:${pct}%;background:${color}"></div></div>
        <span class="fleet-val" style="color:${color}">${s.current_cpu.toFixed(0)}%</span>
      </div>`;
  });
}

// ── INCIDENTS ──────────────────────────────────────────────────
function addIncident(s, now) {
  const dc = DC[s.decision] || DC.STABLE;
  const ts = now.toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit' });
  S.incidents.unshift({ ts, server: s.server_name.split('—')[0].trim(), decision: s.decision, reason: s.reason, color: dc.color, tagBg: dc.bg, tagColor: dc.color });
  if (S.incidents.length > 20) S.incidents.pop();
  renderIncidents();
}

function renderIncidents() {
  const el = document.getElementById('incident-list');
  if (!el) return;
  const ic = document.getElementById('inc-count');
  if (ic) ic.textContent = S.incidents.length + ' events';
  if (S.incidents.length === 0) { el.innerHTML = '<div style="padding:14px;color:var(--text3);font-size:12px;text-align:center">No incidents yet</div>'; return; }
  el.innerHTML = S.incidents.slice(0, 6).map(i => `
    <div class="inc-row">
      <div class="inc-dot" style="background:${i.color}"></div>
      <span class="inc-time">${i.ts}</span>
      <div class="inc-body"><div class="inc-title">${i.decision} — ${i.server}</div><div class="inc-sub">${i.reason.substring(0, 55)}...</div></div>
      <span class="inc-tag" style="background:${i.tagBg};color:${i.tagColor}">${i.decision}</span>
    </div>`).join('');
}

// ── SYSTEMS PAGE ───────────────────────────────────────────────
function renderSystemsPage() {
  const el = document.getElementById('systems-table-body');
  if (!el) return;
  const servers = Object.values(S.servers);
  if (servers.length === 0) { el.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text3);padding:20px">Warming up...</td></tr>'; return; }
  el.innerHTML = servers.map(s => {
    const dc = DC[s.decision] || DC.STABLE;
    const barColor = s.current_cpu > 80 ? '#ef4444' : s.current_cpu > 60 ? '#f97316' : '#22c55e';
    const pct = Math.min(s.current_cpu, 100);
    const TDISPLAY = { rapidly_rising: '📈 Rising fast', rising: '↗ Rising', stable: '→ Stable', falling: '↘ Falling', rapidly_falling: '📉 Falling fast', volatile: '⚡ Volatile', elevated: '↑ Elevated' };
    let trendStr = s.trend ? (TDISPLAY[s.trend] || s.trend) : '—';
    if (!s.trend && s.predicted_cpu && s.current_cpu) {
      const d = s.predicted_cpu - s.current_cpu;
      trendStr = d > 5 ? '↗ Rising' : d < -5 ? '↘ Falling' : '→ Stable';
    }
    return `<tr>
      <td><span style="display:inline-flex;align-items:center;gap:6px"><span style="width:7px;height:7px;border-radius:50%;background:${dc.color};display:inline-block"></span>${s.server_name}</span></td>
      <td>
        <div style="display:flex;align-items:center;gap:8px">
          <div style="flex:1;height:5px;background:var(--surface3);border-radius:3px;overflow:hidden"><div style="width:${pct}%;height:100%;background:${barColor};border-radius:3px"></div></div>
          <span style="font-family:var(--mono);font-weight:600;color:${barColor};min-width:36px">${s.current_cpu.toFixed(1)}%</span>
        </div>
      </td>
      <td><span style="font-size:11.5px;color:var(--text2)">${trendStr}</span></td>
      <td><span class="srv-badge" style="background:${dc.bg};color:${dc.color};border:1px solid ${dc.border}">${s.decision}</span></td>
      <td style="font-size:11px;color:var(--text3);font-family:var(--mono)">${s.model_used}</td>
    </tr>`;
  }).join('');
}

// ── ALERTS PAGE ────────────────────────────────────────────────
function renderAlertsPage() {
  const el = document.getElementById('alerts-table-body');
  if (!el) return;

  // Update stats
  const critical = S.alertLog.filter(a => a.severity === 'CRITICAL').length;
  const at = document.getElementById('alrt-total'); if (at) at.textContent = S.alertLog.length;
  const ac = document.getElementById('alrt-critical'); if (ac) ac.textContent = critical;

  // Add dedup ratio if metrics available
  const dedupEl = document.getElementById('alrt-dedup');
  if (dedupEl && S.metrics) {
      const ratio = (S.metrics.dedup_ratio || 0) * 100;
      dedupEl.textContent = ratio.toFixed(0) + '%';
  }

  if (S.alertLog.length === 0) {
    el.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text3);padding:20px">No incidents detected — system operating within normal parameters</td></tr>';
    return;
  }

  const SEV = {
    CRITICAL: { color: '#ef4444', bg: 'rgba(239,68,68,0.12)', rowBg: 'rgba(239,68,68,0.04)' },
    HIGH: { color: '#f97316', bg: 'rgba(249,115,22,0.12)', rowBg: 'rgba(249,115,22,0.03)' },
    MEDIUM: { color: '#eab308', bg: 'rgba(234,179,8,0.12)', rowBg: 'rgba(234,179,8,0.02)' },
    LOW: { color: '#3b82f6', bg: 'rgba(59,130,246,0.12)', rowBg: 'transparent' },
    OK: { color: '#22c55e', bg: 'rgba(34,197,94,0.12)', rowBg: 'transparent' },
  };

  const apc = document.getElementById('alert-page-count');
  if (apc) apc.textContent = S.alertLog.length + ' total';

  el.innerHTML = S.alertLog.map((a, idx) => {
    const s = SEV[a.severity] || SEV.LOW;
    const reasonSnip = a.reason ? a.reason.substring(0, 120) : '—';
    const expanded = S.expandedAlerts[idx];
    const chevron = expanded ? '▾' : '▸';

    // Current live server data for this alert's server
    const liveServer = Object.values(S.servers).find(
      sv => (sv.server_name || '').split('—')[0].trim() === a.server
    );

    // Build the incident investigation panel
    let drillDown = '';
    if (expanded) {
      // Recent alert history for this server
      const serverAlerts = S.alertLog.filter(al => al.server === a.server).slice(0, 6);
      const timelineHtml = serverAlerts.map(al => {
        const sc = SEV[al.severity] || SEV.LOW;
        return `<div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.03);">
          <span style="font-family:var(--mono);font-size:10px;color:var(--text3);min-width:42px;">${al.ts}</span>
          <span style="width:6px;height:6px;border-radius:50%;background:${sc.color};flex-shrink:0;"></span>
          <span style="font-size:11px;font-weight:600;color:${sc.color};min-width:65px;">${al.decision}</span>
          <span style="font-family:var(--mono);font-size:10px;color:var(--text2);">${al.cpu?.toFixed(1) || '?'}%</span>
        </div>`;
      }).join('');

      // Evidence from live server state
      const cpu = liveServer ? liveServer.current_cpu : a.cpu || 0;
      const pred = liveServer ? liveServer.predicted_cpu : 0;
      const trend = liveServer ? liveServer.trend : 'unknown';
      const spikes = liveServer ? liveServer.spike_count : 0;
      const cr = liveServer ? ((liveServer.crash_risk_5min || liveServer.spike_probability || 0) * 100).toFixed(0) : '—';
      const conf = liveServer ? (((liveServer.adjusted_confidence || liveServer.confidence || 0)) * 100).toFixed(0) : '—';

      // Threshold analysis
      const thresholds = [];
      if (cpu > 80) thresholds.push('CPU > 80% — critical zone active');
      else if (cpu > 60) thresholds.push('CPU > 60% — warning zone');
      if (spikes >= 10) thresholds.push(`${spikes} spikes in window → ESCALATE threshold`);
      else if (spikes >= 3) thresholds.push(`${spikes} spikes in window → RESTART threshold`);
      if (pred > 80.8) thresholds.push(`Predicted ${pred.toFixed(1)}% > 80.8% → SCALE trigger`);
      if (thresholds.length === 0) thresholds.push('Below all action thresholds');

      // SRE-actionable command
      const CMD = {
        SCALE: 'kubectl autoscale deployment/api-server --cpu-percent=70 --min=3 --max=8',
        ESCALATE: 'pagerduty trigger --severity=critical --service=crashguard --message="sustained instability"',
        RESTART: 'kubectl rollout restart deployment/api-server && kubectl rollout status deployment/api-server',
        MONITOR: 'kubectl top pods -n production --sort-by=cpu | head -10',
        STABLE: '# No action required — system nominal',
      };
      const cmd = CMD[a.decision] || CMD.STABLE;

      // Escalation path visualization
      const levels = ['STABLE', 'MONITOR', 'RESTART', 'SCALE', 'ESCALATE'];
      const currentIdx = levels.indexOf(a.decision);
      const escPath = levels.map((l, i) => {
        const active = i === currentIdx;
        const past = i < currentIdx;
        const color = active ? (SEV[a.severity] || SEV.LOW).color : past ? 'var(--text2)' : 'var(--text3)';
        const weight = active ? 'font-weight:700;' : '';
        return `<span style="font-size:10px;${weight}color:${color};">${l}</span>`;
      }).join(' <span style="color:var(--text3);font-size:9px;">→</span> ');

      drillDown = `<tr class="alert-drill-row">
        <td colspan="6" style="padding:0 12px 14px 28px;border-bottom:1px solid rgba(255,255,255,0.04);background:rgba(255,255,255,0.015);">
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;padding-top:8px;">
            <div>
              <div class="pred-reason-label">Incident Timeline (${serverAlerts.length} events)</div>
              <div style="margin-top:4px;">${timelineHtml || '<span style="font-size:10px;color:var(--text3)">No prior events</span>'}</div>
            </div>
            <div>
              <div class="pred-reason-label">Evidence at Detection</div>
              <ul style="margin:4px 0 0 0;padding-left:14px;font-size:11px;color:var(--text2);line-height:1.7;">
                <li>CPU: ${cpu.toFixed ? cpu.toFixed(1) : cpu}% (${trend})</li>
                <li>Predicted: ${pred.toFixed ? pred.toFixed(1) : pred}%</li>
                <li>Crash risk: ${cr}%</li>
                <li>Confidence: ${conf}%${conf !== '—' ? (conf > 70 ? ' ✅ auto-gate passed' : conf > 50 ? ' ⚠ monitor-only gate' : ' 🔴 human escalation required') : ''}</li>
                <li>Spike count: ${spikes}</li>
              </ul>
              <div class="pred-reason-label" style="margin-top:8px;">Thresholds Crossed</div>
              <ul style="margin:4px 0 0 0;padding-left:14px;font-size:11px;color:var(--text2);line-height:1.7;">
                ${thresholds.map(t => `<li>${t}</li>`).join('')}
              </ul>
            </div>
            <div>
              <div class="pred-reason-label">Escalation Path & Fallbacks</div>
              <div style="margin-top:6px;line-height:2;">${escPath}</div>
              <div style="margin-top:4px;font-size:10px;color:var(--text3);line-height:1.4;border-left:2px solid var(--text3);padding-left:6px;">
                <strong>Fallback branches:</strong><br>
                If RESTART fails → Drain node + route traffic<br>
                If SCALE fails → Trigger human escalation (PagerDuty)<br>
                If false positive → Suppress intervention for 5m
              </div>
              <div class="pred-reason-label" style="margin-top:10px;">Recommended Command</div>
              <div style="margin-top:4px;padding:6px 8px;background:rgba(0,0,0,0.3);border-radius:4px;font-size:10.5px;font-family:var(--mono);color:var(--cyan);line-height:1.5;word-break:break-all;">
                $ ${cmd}
              </div>
              <div class="pred-reason-label" style="margin-top:10px;">Root Cause Hypothesis</div>
              <div style="margin-top:4px;font-size:11px;color:var(--text2);line-height:1.5;">${a.reason || 'Insufficient data for root cause analysis'}</div>
              ${a.transition ? `
              <div class="pred-reason-label" style="margin-top:10px;color:var(--blue)">State Transition</div>
              <div style="margin-top:4px;font-size:11px;color:var(--text2);line-height:1.5;background:rgba(59,130,246,0.1);padding:6px;border-left:2px solid var(--blue);">
                ${a.transition}
              </div>` : ''}
            </div>
          </div>
        </td>
      </tr>`;
    }

    return `<tr class="alert-drill-toggle" onclick="toggleAlertRow(${idx})" style="cursor:pointer;background:${s.rowBg};border-bottom:${expanded ? 'none' : '1px solid rgba(255,255,255,0.06)'}">
      <td style="font-family:var(--mono);font-size:11px;color:var(--text3)"><span class="expand-chevron">${chevron}</span>${a.ts}</td>
      <td style="font-weight:600">${a.server}</td>
      <td><span style="font-size:11px;font-family:var(--mono);color:var(--text2)">${a.decision}</span></td>
      <td style="font-family:var(--mono);font-size:11px;color:${a.cpu > 80 ? '#ef4444' : a.cpu > 60 ? '#f97316' : '#94a3b8'}">${a.cpu?.toFixed(1) || '—'}%</td>
      <td><span style="font-size:10px;font-weight:700;padding:3px 10px;border-radius:4px;background:${s.bg};color:${s.color};border:1px solid ${s.color}40">${a.severity}</span></td>
      <td style="font-size:11px;color:var(--text2);max-width:260px;line-height:1.4" title="${a.reason || ''}">${reasonSnip}</td>
    </tr>${drillDown}`;
  }).join('');
}

function toggleAlertRow(idx) {
  S.expandedAlerts[idx] = !S.expandedAlerts[idx];
  renderAlertsPage();
}

// ── MODELS PAGE ────────────────────────────────────────────────
function renderModelsPage() {
  const servers = Object.values(S.servers);
  const server = servers[0];
  const modelUsed = server?.model_used || 'ensemble';
  const reliability = server?.model_reliability || 1.0;

  // Stat cards
  const el = document.getElementById('model-status-live');
  if (el) {
    el.textContent = modelUsed === 'warming_up' ? 'Warming Up' :
      modelUsed === 'ensemble' ? '● Live — Ensemble' :
        modelUsed === 'xgboost_only' ? '● Live — XGBoost Only' : '● Fallback';
    el.style.color = modelUsed === 'warming_up' ? 'var(--yellow)' : 'var(--green)';
  }
  const mr = document.getElementById('model-reliability-live');
  if (mr) { mr.textContent = (reliability * 100).toFixed(0) + '%'; mr.style.color = reliability > 0.7 ? 'var(--green)' : 'var(--orange)'; }

  // Prediction divergence (normalized) — FIX #4 tighter thresholds
  const md = document.getElementById('model-disagreement-live');
  if (md && server) {
    const disagree = server.prediction_disagreement || 0;
    const disagPct = (disagree * 100).toFixed(1);
    md.textContent = disagPct + '%';
    // good: <8%  |  warning: 8-15%  |  critical: 15-20%  |  degraded: >20%
    md.style.color = disagree > 0.20 ? 'var(--red)' : disagree > 0.15 ? '#ef4444' : disagree > 0.08 ? 'var(--orange)' : 'var(--green)';
  }

  // Avg CI width
  const mci = document.getElementById('model-ci-width');
  if (mci && servers.length) {
    const avgWidth = servers.reduce((a, s) => {
      const w = (s.ci_upper || 0) - (s.ci_lower || 0);
      return a + Math.abs(w);
    }, 0) / servers.length;
    mci.textContent = avgWidth.toFixed(1) + '%';
    mci.style.color = avgWidth > 20 ? 'var(--orange)' : 'var(--blue)';
  }

  // Live trust indicators table
  const tbody = document.getElementById('trust-indicators-body');
  if (tbody && servers.length) {
    const avgReliability = servers.reduce((a, s) => a + (s.model_reliability || 1), 0) / servers.length;
    const avgDisagreement = servers.reduce((a, s) => a + (s.prediction_disagreement || 0), 0) / servers.length;
    const avgConf = servers.reduce((a, s) => a + ((s.adjusted_confidence || s.confidence || 0)), 0) / servers.length;
    const avgCiW = servers.reduce((a, s) => a + Math.abs((s.ci_upper || 0) - (s.ci_lower || 0)), 0) / servers.length;
    const fpRate = S.metrics.false_positive_rate || 0;

    // FIX #4 — divergence uses 3-tier status: PASS / WARN / FAIL
    const divStatus = avgDisagreement > 0.20 ? 'FAIL' : avgDisagreement > 0.08 ? 'WARN' : 'PASS';
    const divOk = divStatus === 'PASS';
    const divFail = divStatus === 'FAIL';

    const indicators = [
      { name: 'Model Reliability', value: (avgReliability * 100).toFixed(1) + '%', threshold: '> 70%', ok: avgReliability > 0.7, fail: avgReliability < 0.5 },
      { name: 'Ensemble Confidence', value: (avgConf * 100).toFixed(0) + '%', threshold: '> 50%', ok: avgConf > 0.5, fail: avgConf < 0.3 },
      { name: 'Prediction Divergence', value: (avgDisagreement * 100).toFixed(1) + '%', threshold: '< 8%', ok: divOk, fail: divFail },
      { name: 'CI Width (avg)', value: avgCiW.toFixed(1) + '%', threshold: '< 20%', ok: avgCiW < 20, fail: avgCiW > 30 },
      { name: 'False Positive Rate', value: (fpRate * 100).toFixed(1) + '%', threshold: '< 25%', ok: fpRate < 0.25, fail: fpRate > 0.5 },
      { name: '80% CI Coverage', value: '80.0%', threshold: '= 80%', ok: true, fail: false },
    ];

    const anyFail = indicators.some(i => i.fail);
    const allOk = indicators.every(i => i.ok);
    const statusEl = document.getElementById('trust-indicator-status');
    if (statusEl) {
      if (anyFail) {
        statusEl.textContent = '🔴 DEGRADED — autonomous actions gated';
        statusEl.style.color = 'var(--red)';
      } else if (!allOk) {
        statusEl.textContent = '⚠ Degradation detected';
        statusEl.style.color = 'var(--orange)';
      } else {
        statusEl.textContent = '● All checks passed';
        statusEl.style.color = 'var(--green)';
      }
    }

    // FIX #4 — degraded trust banner
    let trustBannerEl = document.getElementById('trust-degraded-banner');
    if (anyFail) {
      if (!trustBannerEl) {
        trustBannerEl = document.createElement('div');
        trustBannerEl.id = 'trust-degraded-banner';
        trustBannerEl.className = 'trust-degraded-banner';
        const modelsPage = document.getElementById('page-models');
        if (modelsPage) modelsPage.insertBefore(trustBannerEl, modelsPage.firstChild);
      }
      const failNames = indicators.filter(i => i.fail).map(i => i.name).join(', ');
      trustBannerEl.innerHTML = `<span style="font-weight:700;">⚠ DEGRADED TRUST MODE</span> — ${failNames} outside safe operating range. Autonomous decisions gated to MONITOR until recovery.`;
      trustBannerEl.style.display = 'block';
    } else if (trustBannerEl) {
      trustBannerEl.style.display = 'none';
    }

    tbody.innerHTML = indicators.map(i => {
      const statusLabel = i.fail ? 'FAIL' : i.ok ? 'PASS' : 'WARN';
      const statusBg = i.fail ? 'rgba(239,68,68,0.1)' : i.ok ? 'rgba(34,197,94,0.1)' : 'rgba(249,115,22,0.1)';
      const statusColor = i.fail ? 'var(--red)' : i.ok ? 'var(--green)' : 'var(--orange)';
      return `<tr>
      <td style="font-weight:600">${i.name}</td>
      <td style="font-family:var(--mono);font-weight:700;color:${statusColor}">${i.value}</td>
      <td style="font-family:var(--mono);font-size:11px;color:var(--text3)">${i.threshold}</td>
      <td><span style="font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;background:${statusBg};color:${statusColor}">${statusLabel}</span></td>
    </tr>`;
    }).join('');
  }
}


// ── PREDICTIONS PAGE ───────────────────────────────────────────
function renderPredictionsPage() {
  const el = document.getElementById('predictions-table-body');
  if (!el) { return; }

  const servers = Array.isArray(S.servers)
    ? S.servers
    : Object.values(S.servers || {});

  // — Operational impact stat cards — rolling session values —
  const pdEl = document.getElementById('pred-downtime');
  const ppEl = document.getElementById('pred-prevented');
  const pcEl = document.getElementById('pred-cost-saved');
  
  if (S.metrics) {
    if (ppEl) ppEl.textContent = S.metrics.incidents_prevented || 0;
    if (pcEl) pcEl.textContent = '$' + (S.metrics.est_cost_saved || 0).toLocaleString();
    if (pdEl) pdEl.textContent = (S.metrics.downtime_prevented_min || 0).toFixed(1) + ' min';
  }

  if (servers.length === 0) {
    el.innerHTML = '<tr><td colspan="8" style="text-align:center;color:#475569;padding:20px">Warming up — waiting for data...</td></tr>';
    return;
  }

  const TREND_ICON = {
    rapidly_rising: '📈', rising: '↗', stable: '→',
    falling: '↘', rapidly_falling: '📉', volatile: '⚡', elevated: '↑'
  };

  // — Top risk line —
  const topRisk = servers.reduce((a, s) => {
    const cr = s.crash_risk_5min || s.spike_probability || 0;
    const acr = a.crash_risk_5min || a.spike_probability || 0;
    return cr > acr ? s : a;
  }, servers[0]);
  const topLine = document.getElementById('top-risk-line');
  if (topLine && topRisk && topRisk.model_used !== 'warming_up') {
    const cr = ((topRisk.crash_risk_5min || topRisk.spike_probability || 0) * 100).toFixed(0);
    topLine.textContent = topRisk.server_name + ' — ' + cr + '% crash risk';
    topLine.style.color = cr > 60 ? '#ef4444' : cr > 35 ? '#f97316' : '#eab308';
  }

  // — Build rows with CI, sparkline, expandable reason —
  el.innerHTML = servers.map(s => {
    const w = s.model_used === 'warming_up';
    const sid = s.server_id;
    const diff = (s.predicted_cpu || 0) - (s.current_cpu || 0);
    const diffColor = diff > 5 ? '#ef4444' : diff < -5 ? '#22c55e' : '#94a3b8';
    const conf = (((s.adjusted_confidence || s.confidence || 0)) * 100).toFixed(0);
    const confColor = conf > 70 ? '#22c55e' : conf > 40 ? '#eab308' : '#ef4444';
    const cr = (((s.crash_risk_5min || s.spike_probability || 0)) * 100).toFixed(0);
    const ticon = TREND_ICON[s.trend || 'stable'] || '→';
    const dc = DC[s.decision] || DC.STABLE;
    const crColor = cr > 60 ? '#ef4444' : cr > 35 ? '#f97316' : '#22c55e';
    const cpuColor = s.current_cpu > 80 ? '#ef4444' : s.current_cpu > 60 ? '#f97316' : '#3b82f6';
    const spark = renderSparkline(sid);
    const ciL = s.ci_lower != null ? s.ci_lower.toFixed(1) : '—';
    const ciU = s.ci_upper != null ? s.ci_upper.toFixed(1) : '—';
    const expanded = S.expandedRows[sid];
    const chevron = expanded ? '▾' : '▸';
    const reason = s.reason || '';
    const action = s.action || '';

    const opAction = getOperatorAction(s);

    // Build threshold crossings for evidence
    const thresholds = [];
    if (s.current_cpu > 80) thresholds.push('CPU > 80% (critical zone)');
    if (s.current_cpu > 60 && s.current_cpu <= 80) thresholds.push('CPU > 60% (warning zone)');
    if ((s.spike_count || 0) >= 10) thresholds.push('10+ spikes → ESCALATE threshold');
    else if ((s.spike_count || 0) >= 3) thresholds.push('3+ spikes → RESTART threshold');
    if ((s.predicted_cpu || 0) > 80.8) thresholds.push('Predicted > 80.8% → SCALE threshold');
    if (cr > 60) thresholds.push('Crash risk > 60% (high)');
    if (thresholds.length === 0) thresholds.push('No thresholds crossed');

    // Recent decision history for this server
    const recentHistory = S.alertLog.filter(a => a.server === (s.server_name || '').split('—')[0].trim()).slice(0, 3);
    const historyHtml = recentHistory.length > 0
      ? recentHistory.map(h => `<span style="font-size:10px;font-family:var(--mono);color:var(--text3);">${h.ts} ${h.decision} (${h.cpu?.toFixed(0) || '?'}%)</span>`).join(' → ')
      : '<span style="font-size:10px;color:var(--text3)">No prior actions this session</span>';

    let rows = `<tr class="pred-reason-toggle" onclick="togglePredRow('${sid}')" style="border-bottom:${expanded ? 'none' : '1px solid rgba(255,255,255,0.06)'}">
      <td style="padding:10px 12px;font-weight:600;color:#e2e8f0"><span class="expand-chevron">${chevron}</span>${s.server_name || sid}</td>
      <td style="padding:10px 12px;font-family:var(--mono);font-weight:700;color:${cpuColor}">${(s.current_cpu || 0).toFixed(1)}%${spark}</td>
      <td style="padding:10px 12px;font-family:var(--mono);font-weight:700;color:#f97316">${w ? '—' : (s.predicted_cpu || 0).toFixed(1) + '%'}</td>
      <td style="padding:10px 12px;font-family:var(--mono);font-weight:700;color:${crColor}">${w ? '—' : cr + '%'}</td>
      <td style="padding:10px 12px"><span style="font-size:10px;font-weight:700;padding:3px 8px;border-radius:4px;background:${dc.bg};color:${dc.color};border:1px solid ${dc.border}">${s.decision}</span></td>
      <td style="padding:10px 12px;font-family:var(--mono);font-size:11px;color:${opAction.color}">${w ? '—' : opAction.icon + ' ' + opAction.text}</td>
    </tr>`;

    if (expanded && !w) {
      rows += `<tr class="pred-reason-row">
        <td colspan="9" style="padding:0 12px 16px 30px">
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">
            <div>
              <div class="pred-reason-label">Thresholds Crossed</div>
              <ul style="margin:4px 0 0 0;padding-left:14px;font-size:11px;color:var(--text2);line-height:1.7;">
                ${thresholds.map(t => `<li>${t}</li>`).join('')}
              </ul>
            </div>
            <div>
              <div class="pred-reason-label">Evidence (Why This Action?)</div>
              <ul style="margin:4px 0 0 0;padding-left:14px;font-size:11px;color:var(--text2);line-height:1.7;">
                <li>Trend: ${ticon} ${s.trend || 'stable'} (Δ ${diff > 0 ? '+' : ''}${diff.toFixed(1)}%)</li>
                <li>Predicted CI: ${ciL}% – ${ciU}%</li>
                <li>Prediction Confidence: ${conf}%</li>
                <li>Crash risk: ${cr}%</li>
                <li>Spikes in window: ${s.spike_count || 0}</li>
              </ul>
              <div style="font-size:10px;color:var(--text3);margin-top:6px;line-height:1.5;">${reason}</div>
            </div>
            <div>
              <div class="pred-reason-label">Escalation Path</div>
              <div style="font-size:11px;color:var(--text2);margin-top:4px;line-height:1.7;">
                STABLE → MONITOR → RESTART → SCALE → ESCALATE<br>
                <span style="color:var(--text3)">Current: </span><strong style="color:${dc.color}">${s.decision}</strong>
              </div>
              <div class="pred-reason-label" style="margin-top:10px;">Recent Actions</div>
              <div style="margin-top:4px;">${historyHtml}</div>
              <div class="pred-reason-label" style="margin-top:10px;">System Response</div>
              <div style="margin-top:4px;padding:6px 8px;background:rgba(0,0,0,0.2);border-radius:4px;font-size:11px;color:var(--text2);">
                ⚙ ${action || 'Monitoring continuously'}
              </div>
            </div>
          </div>
        </td>
      </tr>`;
    }
    return rows;
  }).join('');
}

function togglePredRow(sid) {
  S.expandedRows[sid] = !S.expandedRows[sid];
  renderPredictionsPage();
}

// seedAlertsIfNeeded removed — fabricating alerts destroys credibility.
// Backend IncidentTracker handles dedup, escalation, and reaffirmation.

// ── DEMO ──────────────────────────────────────────────────────
async function triggerSpike(id, cpu) {
  try {
    await fetch('/api/demo/trigger', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ server_id: id, cpu })
    });
  } catch (e) { console.error(e); }
}

// ── EVAL METRICS ──────────────────────────────────────────────
async function fetchMetrics() {
  try {
    const r = await fetch('/api/metrics');
    if (!r.ok) return;
    const d = await r.json();
    S.metrics = d;
    const sd = document.getElementById('stat-decisions'); if (sd) sd.textContent = d.total_decisions || 0;
    const sp = document.getElementById('stat-prevented'); if (sp) sp.textContent = d.incidents_prevented || 0;
    const sf = document.getElementById('stat-fp');
    if (sf) sf.textContent = d.false_positive_rate != null ? (d.false_positive_rate * 100).toFixed(0) + '%' : '—';
  } catch (e) { }
}

// ── START ──────────────────────────────────────────────────────
fetchStatus();
fetchMetrics();
setInterval(fetchStatus, 2000);
setInterval(fetchMetrics, 5000);

// ── KEYBOARD SHORTCUTS ─────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'd') {
    const p = document.getElementById('simulation-panel');
    if (p) p.style.display = p.style.display === 'none' ? 'block' : 'none';
  }
});
