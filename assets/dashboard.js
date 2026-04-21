/**
 * dashboard.js — CrashGuard AI
 * Live data fetching, chart rendering, UI updates, and page navigation.
 */

// ── STATE ──────────────────────────────────────────────────────
const S = {
    servers: {},
    selected: null,
    history: {},
    incidents: [],
    alertLog: [],
    alertCount: 0,
    decisionCount: 0,
    wowTimers: {},
    currentPage: 'dashboard',
};

const DC = {
    SCALE: { color: '#ef4444', bg: 'rgba(239,68,68,0.08)', border: 'rgba(239,68,68,0.4)', icon: '📈', cls: 'critical' },
    ESCALATE: { color: '#ef4444', bg: 'rgba(239,68,68,0.08)', border: 'rgba(239,68,68,0.4)', icon: '🚨', cls: 'critical' },
    RESTART: { color: '#f97316', bg: 'rgba(249,115,22,0.08)', border: 'rgba(249,115,22,0.35)', icon: '🔄', cls: '' },
    MONITOR: { color: '#eab308', bg: 'rgba(234,179,8,0.08)', border: 'rgba(234,179,8,0.35)', icon: '👁', cls: '' },
    STABLE: { color: '#22c55e', bg: 'rgba(34,197,94,0.05)', border: 'rgba(34,197,94,0.3)', icon: '✅', cls: '' },
};

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

    // Show/hide pages
    document.querySelectorAll('.page').forEach(el => {
        el.style.display = el.id === 'page-' + page ? 'flex' : 'none';
    });

    // Render page-specific content
    if (page === 'systems') renderSystemsPage();
    if (page === 'alerts') renderAlertsPage();
    if (page === 'models') renderModelsPage();
    if (page === 'predictions') renderPredictionsPage();
}

// Wire up nav items on load
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.nav-item[data-page]').forEach(el => {
        el.addEventListener('click', () => navigateTo(el.dataset.page));
    });
});

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

        if (prev && prev.decision !== s.decision && s.decision !== 'STABLE' && s.model_used !== 'warming_up') {
            addIncident(s, now);
            S.decisionCount++;
            // Track alerts
            S.alertLog.unshift({
                ts: now.toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit' }),
                server: s.server_name.split('—')[0].trim(),
                decision: s.decision,
                severity: s.severity,
                cpu: s.current_cpu,
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

    if (!S.selected && servers.length) selectServer(servers[0].server_id);
    else if (S.selected) updateChart(S.selected);

    // Refresh current non-dashboard page if visible
    if (S.currentPage === 'systems') renderSystemsPage();
    if (S.currentPage === 'alerts') renderAlertsPage();
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
        row.onclick = () => selectServer(s.server_id);
        row.innerHTML = `
      <div class="srv-dot" style="background:${dc.color}"></div>
      <span class="srv-name">${s.server_name.replace('— ', '')}</span>
      <span class="srv-cpu" style="color:${barColor}">${s.current_cpu.toFixed(0)}%</span>
      <span class="srv-badge" style="background:${dc.bg};color:${dc.color};border:1px solid ${dc.border}">${s.decision}</span>`;
        panel.appendChild(row);
    });
}

function selectServer(id) {
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

    if (S.alertLog.length === 0) {
        el.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text3);padding:20px">No alerts fired yet — trigger a spike from Demo Controls</td></tr>';
        return;
    }

    // FIX 3 — Strong severity colors with row background tinting
    const SEV = {
        CRITICAL: { color: '#ef4444', bg: 'rgba(239,68,68,0.12)', rowBg: 'rgba(239,68,68,0.04)' },
        HIGH: { color: '#f97316', bg: 'rgba(249,115,22,0.12)', rowBg: 'rgba(249,115,22,0.03)' },
        MEDIUM: { color: '#eab308', bg: 'rgba(234,179,8,0.12)', rowBg: 'rgba(234,179,8,0.02)' },
        LOW: { color: '#3b82f6', bg: 'rgba(59,130,246,0.12)', rowBg: 'transparent' },
        OK: { color: '#22c55e', bg: 'rgba(34,197,94,0.12)', rowBg: 'transparent' },
    };

    const apc = document.getElementById('alert-page-count');
    if (apc) apc.textContent = S.alertLog.length + ' total';

    el.innerHTML = S.alertLog.map(a => {
        const s = SEV[a.severity] || SEV.LOW;
        return `<tr style="background:${s.rowBg}">
      <td style="font-family:var(--mono);font-size:11px;color:var(--text3)">${a.ts}</td>
      <td style="font-weight:600">${a.server}</td>
      <td><span style="font-size:11px;font-family:var(--mono);color:var(--text2)">${a.decision}</span></td>
      <td style="font-family:var(--mono);font-size:11px;color:${a.cpu > 80 ? '#ef4444' : a.cpu > 60 ? '#f97316' : '#94a3b8'}">${a.cpu?.toFixed(1) || '—'}%</td>
      <td><span style="font-size:10px;font-weight:700;padding:3px 10px;border-radius:4px;background:${s.bg};color:${s.color};border:1px solid ${s.color}40">${a.severity}</span></td>
    </tr>`;
    }).join('');
}

// ── MODELS PAGE ────────────────────────────────────────────────
function renderModelsPage() {
    // Static but realistic — model info doesn't change at runtime
    const server = Object.values(S.servers)[0];
    const modelUsed = server?.model_used || 'ensemble';
    const reliability = server?.model_reliability || 1.0;
    const el = document.getElementById('model-status-live');
    if (el) {
        el.textContent = modelUsed === 'warming_up' ? 'Warming Up' :
            modelUsed === 'ensemble' ? '● Live — Ensemble' :
                modelUsed === 'xgboost_only' ? '● Live — XGBoost Only' : '● Fallback';
        el.style.color = modelUsed === 'warming_up' ? 'var(--yellow)' : 'var(--green)';
    }
    const mr = document.getElementById('model-reliability-live');
    if (mr) { mr.textContent = (reliability * 100).toFixed(0) + '%'; mr.style.color = reliability > 0.7 ? 'var(--green)' : 'var(--orange)'; }
}

// ── PREDICTIONS PAGE ───────────────────────────────────────────
function renderPredictionsPage() {
    const el = document.getElementById('predictions-table-body');
    if (!el) return;
    const servers = Object.values(S.servers);
    if (servers.length === 0) { el.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text3);padding:20px">Warming up...</td></tr>'; return; }

    const TREND_ICON = { rapidly_rising: '📈', rising: '↗', stable: '→', falling: '↘', rapidly_falling: '📉', volatile: '⚡', elevated: '↑' };

    // FIX 2 — Find top risk server
    let topRisk = null;
    servers.forEach(s => {
        const cr = s.crash_risk_5min || s.spike_probability || 0;
        if (!topRisk || cr > (topRisk.crash_risk_5min || topRisk.spike_probability || 0)) topRisk = s;
    });
    const banner = document.getElementById('top-risk-banner');
    if (banner && topRisk && topRisk.model_used !== 'warming_up') {
        const cr = (topRisk.crash_risk_5min || topRisk.spike_probability || 0);
        if (cr > 0.15) {
            banner.style.display = 'flex';
            document.getElementById('top-risk-server').textContent = topRisk.server_name;
            document.getElementById('top-risk-value').textContent = (cr * 100).toFixed(0) + '%';
            // Color by risk level
            const color = cr > 0.6 ? '#ef4444' : cr > 0.35 ? '#f97316' : '#eab308';
            const bg = cr > 0.6 ? 'rgba(239,68,68,0.08)' : cr > 0.35 ? 'rgba(249,115,22,0.08)' : 'rgba(234,179,8,0.08)';
            const border = cr > 0.6 ? 'rgba(239,68,68,0.35)' : cr > 0.35 ? 'rgba(249,115,22,0.35)' : 'rgba(234,179,8,0.35)';
            banner.style.background = bg;
            banner.style.borderColor = border;
            banner.querySelector('#top-risk-server').style.color = color;
            banner.querySelector('#top-risk-value').style.color = color;
            banner.querySelector('div > div').style.color = color;
        } else {
            banner.style.display = 'none';
        }
    }

    el.innerHTML = servers.map(s => {
        const diff = s.predicted_cpu - s.current_cpu;
        const diffColor = diff > 5 ? '#ef4444' : diff < -5 ? '#22c55e' : '#94a3b8';
        const conf = ((s.adjusted_confidence || s.confidence) * 100).toFixed(0);
        const cr = ((s.crash_risk_5min || s.spike_probability || 0) * 100).toFixed(0);
        const ticon = TREND_ICON[s.trend || 'stable'] || '→';
        const dc = DC[s.decision] || DC.STABLE;
        const crColor = cr > 60 ? '#ef4444' : cr > 35 ? '#f97316' : '#22c55e';
        return `<tr>
      <td style="font-weight:600">${s.server_name}</td>
      <td style="font-family:var(--mono);font-weight:700;color:${s.current_cpu > 80 ? '#ef4444' : s.current_cpu > 60 ? '#f97316' : '#3b82f6'}">${s.current_cpu.toFixed(1)}%</td>
      <td style="font-family:var(--mono);font-weight:700;color:var(--orange)">${s.model_used === 'warming_up' ? '—' : s.predicted_cpu.toFixed(1) + '%'}</td>
      <td style="font-family:var(--mono);color:${diffColor}">${s.model_used === 'warming_up' ? '—' : (diff > 0 ? '+' : '') + diff.toFixed(1) + '%'}</td>
      <td style="font-family:var(--mono);color:var(--green)">${s.model_used === 'warming_up' ? '—' : conf + '%'}</td>
      <td style="font-size:14px">${ticon} <span style="font-size:11px;color:var(--text3)">${s.trend || '—'}</span></td>
      <td style="font-family:var(--mono);font-weight:700;color:${crColor}">${s.model_used === 'warming_up' ? '—' : cr + '%'}</td>
      <td><span class="srv-badge" style="background:${dc.bg};color:${dc.color};border:1px solid ${dc.border}">${s.decision}</span></td>
    </tr>`;
    }).join('');
}

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
