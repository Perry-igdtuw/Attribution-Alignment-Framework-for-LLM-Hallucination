/* ============================================================
   FHI Dashboard — Interactive Charts & Data Binding
   ============================================================ */

// Embedded results data (from the pipeline run)
const RESULTS_DATA = [
  {
    "sample_id": "halueval_0",
    "fhi": 0.3151,
    "aas": 0.0714,
    "cis": 0.4644,
    "ess": 0.9815,
    "hcg": 0.4343,
    "predicted_hallucination": true,
    "true_hallucination": true
  },
  {
    "sample_id": "halueval_1",
    "fhi": 0.2864,
    "aas": 0.0233,
    "cis": 0.3539,
    "ess": 0.7932,
    "hcg": 0.0202,
    "predicted_hallucination": true,
    "true_hallucination": true
  },
  {
    "sample_id": "halueval_2",
    "fhi": 0.2745,
    "aas": 0.0952,
    "cis": 0.5713,
    "ess": 0.3985,
    "hcg": 0.2249,
    "predicted_hallucination": true,
    "true_hallucination": false
  },
  {
    "sample_id": "halueval_3",
    "fhi": 0.4200,
    "aas": 0.2381,
    "cis": 0.6795,
    "ess": 0.6832,
    "hcg": 0.1727,
    "predicted_hallucination": true,
    "true_hallucination": true
  },
  {
    "sample_id": "halueval_4",
    "fhi": 0.2961,
    "aas": 0.0500,
    "cis": 0.3692,
    "ess": 0.8420,
    "hcg": 0.1104,
    "predicted_hallucination": true,
    "true_hallucination": false
  }
];

// FHI Weight Constants (from metric_weights.yaml)
const WEIGHTS = { w1_aas: 0.25, w2_cis: 0.35, w3_ess: 0.25, w4_hcg: 0.15 };

// Chart.js global styling
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(99, 102, 241, 0.08)';

// ──── Utility Functions ────────────────────────────────────────────
function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function std(arr) {
    const m = mean(arr);
    return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
}

// ──── KPI Cards ────────────────────────────────────────────────────
function renderKPIs(data) {
    const metrics = ['fhi', 'aas', 'cis', 'ess', 'hcg'];
    metrics.forEach(m => {
        const vals = data.map(d => d[m]);
        const el = document.getElementById(`kpi-${m}-value`);
        if (el) {
            const v = mean(vals);
            el.textContent = v.toFixed(4);
            // Color coding for FHI
            if (m === 'fhi') {
                el.style.background = v > 0.5
                    ? 'linear-gradient(135deg, #34d399, #22d3ee)'
                    : 'linear-gradient(135deg, #fb7185, #f472b6)';
                el.style.webkitBackgroundClip = 'text';
                el.style.webkitTextFillColor = 'transparent';
            }
        }
    });

    // Sparklines
    metrics.forEach(m => {
        const canvas = document.getElementById(`spark-${m}`);
        if (!canvas) return;
        const vals = data.map(d => d[m]);
        new Chart(canvas, {
            type: 'line',
            data: {
                labels: data.map(d => d.sample_id),
                datasets: [{
                    data: vals,
                    borderColor: m === 'hcg' ? '#fb7185' : '#818cf8',
                    borderWidth: 2,
                    fill: true,
                    backgroundColor: m === 'hcg'
                        ? 'rgba(251, 113, 133, 0.08)'
                        : 'rgba(129, 140, 248, 0.08)',
                    pointRadius: 0,
                    tension: 0.4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
                scales: { x: { display: false }, y: { display: false } },
                animation: { duration: 1200, easing: 'easeOutQuart' }
            }
        });
    });
}

// ──── Radar Chart ──────────────────────────────────────────────────
function renderRadar(data) {
    const ctx = document.getElementById('radar-chart');
    const vals = [
        mean(data.map(d => d.aas)),
        mean(data.map(d => d.cis)),
        mean(data.map(d => d.ess)),
        mean(data.map(d => d.hcg)),
    ];

    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['AAS', 'CIS', 'ESS', 'HCG'],
            datasets: [{
                label: 'Gemma-2B-it',
                data: vals,
                borderColor: '#818cf8',
                backgroundColor: 'rgba(129, 140, 248, 0.15)',
                borderWidth: 2.5,
                pointRadius: 5,
                pointBackgroundColor: '#818cf8',
                pointBorderColor: '#1e1b4b',
                pointBorderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    ticks: { stepSize: 0.2, color: '#64748b', backdropColor: 'transparent', font: { size: 10 } },
                    grid: { color: 'rgba(99, 102, 241, 0.1)' },
                    angleLines: { color: 'rgba(99, 102, 241, 0.1)' },
                    pointLabels: { color: '#e2e8f0', font: { size: 13, weight: '600' } },
                }
            },
            plugins: {
                legend: { display: true, position: 'top', labels: { color: '#94a3b8', usePointStyle: true, pointStyle: 'circle' } },
            },
            animation: { duration: 1500, easing: 'easeOutQuint' }
        }
    });
}

// ──── FHI Distribution ─────────────────────────────────────────────
function renderDistribution(data) {
    const ctx = document.getElementById('fhi-distribution-chart');
    const fhiVals = data.map(d => d.fhi);

    // Create histogram bins
    const bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    const binLabels = bins.slice(0, -1).map((b, i) => `${b.toFixed(1)}–${bins[i+1].toFixed(1)}`);
    const counts = new Array(bins.length - 1).fill(0);
    fhiVals.forEach(v => {
        const idx = Math.min(Math.floor(v * 10), 9);
        counts[idx]++;
    });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: binLabels,
            datasets: [{
                label: 'Sample Count',
                data: counts,
                backgroundColor: counts.map((_, i) => {
                    const ratio = i / 9;
                    if (ratio < 0.5) return `rgba(251, 113, 133, ${0.3 + ratio})`;
                    return `rgba(52, 211, 153, ${ratio * 0.8})`;
                }),
                borderColor: counts.map((_, i) => {
                    if (i < 5) return '#fb7185';
                    return '#34d399';
                }),
                borderWidth: 1.5,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    borderColor: 'rgba(129, 140, 248, 0.3)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    titleFont: { weight: '600' },
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 10 } },
                    title: { display: true, text: 'FHI Score Range', color: '#94a3b8' }
                },
                y: {
                    grid: { color: 'rgba(99, 102, 241, 0.06)' },
                    title: { display: true, text: 'Count', color: '#94a3b8' },
                    ticks: { stepSize: 1 }
                }
            },
            animation: { duration: 1200, easing: 'easeOutQuart' }
        }
    });
}

// ──── Per-Sample Stacked Bar ───────────────────────────────────────
function renderSampleBreakdown(data) {
    const ctx = document.getElementById('sample-breakdown-chart');

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.sample_id),
            datasets: [
                {
                    label: 'AAS (w=0.25)',
                    data: data.map(d => d.aas * WEIGHTS.w1_aas),
                    backgroundColor: 'rgba(129, 140, 248, 0.8)',
                    borderRadius: 3,
                },
                {
                    label: 'CIS (w=0.35)',
                    data: data.map(d => d.cis * WEIGHTS.w2_cis),
                    backgroundColor: 'rgba(192, 132, 252, 0.8)',
                    borderRadius: 3,
                },
                {
                    label: 'ESS (w=0.25)',
                    data: data.map(d => d.ess * WEIGHTS.w3_ess),
                    backgroundColor: 'rgba(56, 189, 248, 0.8)',
                    borderRadius: 3,
                },
                {
                    label: '−HCG (w=0.15)',
                    data: data.map(d => -(d.hcg * WEIGHTS.w4_hcg)),
                    backgroundColor: 'rgba(251, 113, 133, 0.7)',
                    borderRadius: 3,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#94a3b8', usePointStyle: true, pointStyle: 'rectRounded', font: { size: 11 } }
                },
                tooltip: {
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    borderColor: 'rgba(129, 140, 248, 0.3)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`
                    }
                }
            },
            scales: {
                x: {
                    stacked: true,
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { family: "'JetBrains Mono', monospace", size: 11 } }
                },
                y: {
                    stacked: true,
                    grid: { color: 'rgba(99, 102, 241, 0.06)' },
                    title: { display: true, text: 'Weighted Contribution to FHI', color: '#94a3b8' }
                }
            },
            animation: { duration: 1400, easing: 'easeOutQuint' }
        }
    });
}

// ──── AAS vs CIS Scatter ───────────────────────────────────────────
function renderScatter(data) {
    const ctx = document.getElementById('scatter-chart');

    const scatterData = data.map(d => ({
        x: d.aas,
        y: d.cis,
        r: d.ess * 12 + 4,
        label: d.sample_id,
    }));

    new Chart(ctx, {
        type: 'bubble',
        data: {
            datasets: [{
                label: 'Samples',
                data: scatterData,
                backgroundColor: data.map(d =>
                    d.predicted_hallucination
                        ? 'rgba(251, 113, 133, 0.6)'
                        : 'rgba(52, 211, 153, 0.6)'
                ),
                borderColor: data.map(d =>
                    d.predicted_hallucination ? '#fb7185' : '#34d399'
                ),
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    borderColor: 'rgba(129, 140, 248, 0.3)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        label: (ctx) => {
                            const d = data[ctx.dataIndex];
                            return [
                                `${d.sample_id}`,
                                `AAS: ${d.aas.toFixed(4)}`,
                                `CIS: ${d.cis.toFixed(4)}`,
                                `FHI: ${d.fhi.toFixed(4)}`,
                            ];
                        }
                    }
                }
            },
            scales: {
                x: {
                    min: 0, max: 1,
                    title: { display: true, text: 'Attribution Alignment (AAS)', color: '#94a3b8' },
                    grid: { color: 'rgba(99, 102, 241, 0.06)' }
                },
                y: {
                    min: 0, max: 1,
                    title: { display: true, text: 'Causal Impact (CIS)', color: '#94a3b8' },
                    grid: { color: 'rgba(99, 102, 241, 0.06)' }
                }
            },
            animation: { duration: 1200, easing: 'easeOutQuart' }
        }
    });
}

// ──── FHI Weights Doughnut ─────────────────────────────────────────
function renderWeights() {
    const ctx = document.getElementById('weight-chart');

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['AAS (w₁)', 'CIS (w₂)', 'ESS (w₃)', 'HCG (w₄)'],
            datasets: [{
                data: [WEIGHTS.w1_aas, WEIGHTS.w2_cis, WEIGHTS.w3_ess, WEIGHTS.w4_hcg],
                backgroundColor: [
                    'rgba(129, 140, 248, 0.85)',
                    'rgba(192, 132, 252, 0.85)',
                    'rgba(56, 189, 248, 0.85)',
                    'rgba(251, 113, 133, 0.85)',
                ],
                borderColor: '#111827',
                borderWidth: 3,
                hoverOffset: 12,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#94a3b8',
                        usePointStyle: true,
                        pointStyle: 'circle',
                        font: { size: 12 },
                        padding: 16,
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    borderColor: 'rgba(129, 140, 248, 0.3)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        label: (ctx) => `${ctx.label}: ${(ctx.parsed * 100).toFixed(0)}%`
                    }
                }
            },
            animation: { animateRotate: true, duration: 1600, easing: 'easeOutQuint' }
        }
    });
}

// ──── Results Table ────────────────────────────────────────────────
function renderTable(data) {
    const tbody = document.getElementById('results-tbody');
    tbody.innerHTML = '';

    data.forEach(d => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="color: var(--text-primary); font-weight: 500;">${d.sample_id}</td>
            <td style="color: ${d.fhi > 0.5 ? 'var(--accent-emerald)' : 'var(--accent-rose)'}; font-weight: 600;">
                ${d.fhi.toFixed(4)}
            </td>
            <td>${d.aas.toFixed(4)}</td>
            <td>${d.cis.toFixed(4)}</td>
            <td>${d.ess.toFixed(4)}</td>
            <td>${d.hcg.toFixed(4)}</td>
            <td>
                <span class="badge-hallu ${d.predicted_hallucination ? 'true' : 'false'}">
                    ${d.predicted_hallucination ? '⚠ Hallucination' : '✓ Faithful'}
                </span>
            </td>
            <td>
                <span class="badge-hallu ${d.true_hallucination === null ? 'null' : (d.true_hallucination ? 'true' : 'false')}">
                    ${d.true_hallucination === null ? '—' : (d.true_hallucination ? 'Hallucination' : 'Factual')}
                </span>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

// ──── Status Badge ─────────────────────────────────────────────────
function updateStatus(data) {
    const statusText = document.getElementById('status-text');
    const badge = document.getElementById('pipeline-status');
    
    if (data.length > 0) {
        statusText.textContent = `${data.length} samples loaded`;
        badge.style.borderColor = 'rgba(52, 211, 153, 0.25)';
        badge.style.background = 'rgba(52, 211, 153, 0.1)';
    } else {
        statusText.textContent = 'No data';
        badge.style.borderColor = 'rgba(251, 191, 36, 0.25)';
        badge.style.background = 'rgba(251, 191, 36, 0.1)';
        badge.querySelector('.status-dot').style.background = '#fbbf24';
        statusText.style.color = '#fbbf24';
    }
}

// ──── Initialize ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const data = RESULTS_DATA;
    
    updateStatus(data);
    renderKPIs(data);
    renderRadar(data);
    renderDistribution(data);
    renderSampleBreakdown(data);
    renderScatter(data);
    renderWeights();
    renderTable(data);

    // Refresh button
    document.getElementById('refresh-btn').addEventListener('click', () => {
        location.reload();
    });
});
