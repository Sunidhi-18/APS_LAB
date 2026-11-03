/**
 * CYBERSHIELD XAI - FRONTEND ENGINE
 * Drives real-time AJAX scanning, SHAP waterfall rendering, model benchmarking,
 * Spark Big Data analytics visualization, and incident report exports.
 */

let shapWaterfallChart = null;
let modelMetricsBarChart = null;
let rocCurvesChart = null;
let sparkCorrChart = null;
let sparkClassCompChart = null;
let globalShapChart = null;

let currentScanResult = null;

// Initialize Dashboard upon DOM load
document.addEventListener("DOMContentLoaded", () => {
  // Execute default scan on startup
  executeScan();

  // Pre-load background tabs data
  loadModelBenchmarks();
  loadSparkStats();
  loadGlobalShap();
});

// =========================================================================
// TAB SWITCHING
// =========================================================================
function switchTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));

  const targetBtn = document.getElementById(`btn-tab-${tabId}`);
  const targetContent = document.getElementById(`tab-${tabId}`);

  if (targetBtn) targetBtn.classList.add("active");
  if (targetContent) targetContent.classList.add("active");
}

function applyPreset(url) {
  document.getElementById("target-url-input").value = url;
  executeScan();
}

// =========================================================================
// TAB 1: REAL-TIME URL SCANNER & SHAP WATERFALL
// =========================================================================
async function executeScan() {
  const urlInput = document.getElementById("target-url-input").value.trim();
  const modelSelect = document.getElementById("model-select").value;

  if (!urlInput) {
    alert("Please enter a valid URL to analyze.");
    return;
  }

  const loadingElem = document.getElementById("scan-loading");
  const resultsContainer = document.getElementById("scan-results-container");

  loadingElem.style.display = "block";
  resultsContainer.style.opacity = "0.3";

  try {
    const res = await fetch("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: urlInput, model: modelSelect })
    });

    if (!res.ok) throw new Error("Failed to scan URL");
    const data = await res.json();
    currentScanResult = data;

    renderScanResults(data);
  } catch (err) {
    console.error(err);
    alert("Error executing scan: " + err.message);
  } finally {
    loadingElem.style.display = "none";
    resultsContainer.style.opacity = "1";
  }
}

function renderScanResults(data) {
  // 1. Verdict & Risk Gauge
  const isPhish = data.verdict === "PHISHING";
  const badge = document.getElementById("verdict-badge");
  badge.textContent = isPhish ? "🚨 PHISHING DETECTED" : "✅ LEGITIMATE VERIFIED";
  badge.style.background = isPhish ? "rgba(239, 68, 68, 0.15)" : "rgba(16, 185, 129, 0.15)";
  badge.style.color = isPhish ? "var(--danger-red)" : "var(--success-green)";
  badge.style.borderColor = isPhish ? "var(--danger-red)" : "var(--success-green)";

  document.getElementById("risk-score-val").textContent = `${data.risk_score}%`;
  document.getElementById("risk-score-val").style.color = data.severity_color;
  
  const circleBar = document.getElementById("gauge-circle-bar");
  circleBar.style.stroke = data.severity_color;
  circleBar.setAttribute("stroke-dasharray", `${data.risk_score}, 100`);

  document.getElementById("res-model-used").textContent = data.selected_model;
  document.getElementById("res-latency").textContent = `${data.latency_ms} ms`;
  document.getElementById("res-severity").textContent = data.severity;
  document.getElementById("res-severity").style.color = data.severity_color;

  // 2. XAI Narrative
  const narrativeBox = document.getElementById("xai-narrative-box");
  narrativeBox.innerHTML = formatMarkdown(data.shap_explanation.narrative);

  // 3. Render SHAP Waterfall Chart
  renderShapWaterfall(data.shap_explanation.top_contributors);

  // 4. Render Attack Vectors
  const vectorsList = document.getElementById("attack-vectors-list");
  vectorsList.innerHTML = "";
  if (data.attack_vectors && data.attack_vectors.length > 0) {
    data.attack_vectors.forEach(v => {
      const item = document.createElement("div");
      item.className = "vector-item";
      item.style.borderLeft = `4px solid ${data.severity_color}`;
      item.innerHTML = `
        <div class="vector-icon">⚠️</div>
        <div>
          <div class="vector-title">${escapeHtml(v.type)}</div>
          <div class="vector-detail">${escapeHtml(v.detail)}</div>
        </div>
      `;
      vectorsList.appendChild(item);
    });
  } else {
    vectorsList.innerHTML = `<div style="color:var(--text-muted); font-size:13px;">No malicious attack vectors identified. Authentic web behavior.</div>`;
  }

  // 5. Render IOC Table
  const iocTable = document.getElementById("ioc-table-body");
  iocTable.innerHTML = "";
  data.iocs.forEach(ioc => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${escapeHtml(ioc.indicator)}</strong></td>
      <td style="font-family:var(--font-mono); font-size:12px;">${escapeHtml(ioc.value)}</td>
      <td>
        <span style="font-weight:700; color:${ioc.flagged ? 'var(--danger-red)' : 'var(--success-green)'};">
          ${ioc.flagged ? 'FLAGGED' : 'NORMAL'}
        </span>
      </td>
    `;
    iocTable.appendChild(tr);
  });

  // 6. Remediation Checklist
  const userRecs = document.getElementById("user-recs-list");
  userRecs.innerHTML = "";
  data.user_recommendations.forEach(r => {
    const li = document.createElement("li");
    li.textContent = r;
    userRecs.appendChild(li);
  });

  const socRecs = document.getElementById("soc-recs-list");
  socRecs.innerHTML = "";
  data.soc_recommendations.forEach(r => {
    const li = document.createElement("li");
    li.textContent = r;
    socRecs.appendChild(li);
  });
}

function renderShapWaterfall(contributors) {
  const ctx = document.getElementById("shap-waterfall-chart").getContext("2d");
  if (shapWaterfallChart) shapWaterfallChart.destroy();

  const labels = contributors.map(c => c.feature);
  const values = contributors.map(c => c.shap_value);
  const bgColors = values.map(v => v >= 0 ? "rgba(239, 68, 68, 0.85)" : "rgba(16, 185, 129, 0.85)");
  const borderColors = values.map(v => v >= 0 ? "#ef4444" : "#10b981");

  shapWaterfallChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "SHAP Impact Score (Positive = Malicious, Negative = Benign)",
        data: values,
        backgroundColor: bgColors,
        borderColor: borderColors,
        borderWidth: 1.5,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      scales: {
        x: {
          grid: { color: "rgba(51, 65, 85, 0.3)" },
          ticks: { color: "#94a3b8", font: { family: "Inter" } }
        },
        y: {
          grid: { display: false },
          ticks: { color: "#f8fafc", font: { family: "JetBrains Mono", size: 12, weight: "bold" } }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const item = contributors[ctx.dataIndex];
              return `Value: ${item.value} | SHAP: ${item.shap_value > 0 ? '+' : ''}${item.shap_value} [${item.impact.toUpperCase()}]`;
            },
            afterLabel: (ctx) => {
              return contributors[ctx.dataIndex].description;
            }
          }
        }
      }
    }
  });
}

async function exportThreatReport() {
  if (!currentScanResult) return;
  try {
    const res = await fetch("/api/export-report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: currentScanResult.target_url })
    });
    if (!res.ok) throw new Error("Export failed");
    const blob = await res.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = downloadUrl;
    a.download = `Threat_Report_${currentScanResult.analysis_id}.html`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } catch (err) {
    alert("Export error: " + err.message);
  }
}

// =========================================================================
// TAB 2: MODEL BENCHMARKS & ROC CURVES
// =========================================================================
async function loadModelBenchmarks() {
  try {
    const res = await fetch("/api/models");
    if (!res.ok) return;
    const data = await res.json();
    const metrics = data.metrics;
    const curves = data.curves;

    // Populate Comparison Table
    const tbody = document.getElementById("models-table-body");
    tbody.innerHTML = "";

    const modelNames = Object.keys(metrics);
    modelNames.forEach(mName => {
      const m = metrics[mName];
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong style="color:var(--primary-cyan);">${escapeHtml(mName)}</strong></td>
        <td><strong style="color:var(--success-green);">${m.accuracy}%</strong></td>
        <td>${m.precision}%</td>
        <td>${m.recall}%</td>
        <td><strong style="color:#fff;">${m.f1_score}%</strong></td>
        <td>${m.roc_auc}%</td>
        <td><span style="color:${m.fpr > 1 ? 'var(--warning-amber)' : 'var(--text-muted)'}">${m.fpr}%</span></td>
        <td>${m.train_time_sec}s</td>
        <td><span style="font-family:var(--font-mono);">${m.latency_ms_per_1000} ms</span></td>
      `;
      tbody.appendChild(tr);
    });

    // Render Metrics Comparison Bar Chart
    const ctxBar = document.getElementById("model-metrics-bar-chart").getContext("2d");
    if (modelMetricsBarChart) modelMetricsBarChart.destroy();

    modelMetricsBarChart = new Chart(ctxBar, {
      type: "bar",
      data: {
        labels: modelNames,
        datasets: [
          { label: "Accuracy (%)", data: modelNames.map(k => metrics[k].accuracy), backgroundColor: "rgba(56, 189, 248, 0.85)" },
          { label: "Precision (%)", data: modelNames.map(k => metrics[k].precision), backgroundColor: "rgba(59, 130, 246, 0.85)" },
          { label: "Recall (%)", data: modelNames.map(k => metrics[k].recall), backgroundColor: "rgba(16, 185, 129, 0.85)" },
          { label: "F1-Score (%)", data: modelNames.map(k => metrics[k].f1_score), backgroundColor: "rgba(168, 85, 247, 0.85)" }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { display: false } },
          y: { min: 98, max: 100.2, ticks: { color: "#94a3b8" }, grid: { color: "rgba(51, 65, 85, 0.3)" } }
        },
        plugins: {
          legend: { labels: { color: "#cbd5e1" } }
        }
      }
    });

    // Render ROC Curves Chart
    const ctxRoc = document.getElementById("roc-curves-chart").getContext("2d");
    if (rocCurvesChart) rocCurvesChart.destroy();

    const colors = ["#38bdf8", "#10b981", "#f59e0b", "#a855f7"];
    const rocDatasets = modelNames.map((name, i) => {
      const cData = curves[name]?.roc || { fpr: [0, 1], tpr: [0, 1] };
      const points = cData.fpr.map((fpr, idx) => ({ x: fpr, y: cData.tpr[idx] }));
      return {
        label: `${name} (AUC: ${metrics[name].roc_auc}%)`,
        data: points,
        borderColor: colors[i % colors.length],
        backgroundColor: "transparent",
        borderWidth: 2,
        tension: 0.1,
        pointRadius: 1
      };
    });

    // Add baseline random classifier line
    rocDatasets.push({
      label: "Random Chance (AUC: 50.0%)",
      data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      borderColor: "#64748b",
      borderDash: [5, 5],
      borderWidth: 1.5,
      pointRadius: 0
    });

    rocCurvesChart = new Chart(ctxRoc, {
      type: "line",
      data: { datasets: rocDatasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False Positive Rate (FPR)", color: "#94a3b8" }, ticks: { color: "#94a3b8" }, grid: { color: "rgba(51, 65, 85, 0.3)" } },
          y: { min: 0, max: 1.02, title: { display: true, text: "True Positive Rate (TPR / Recall)", color: "#94a3b8" }, ticks: { color: "#94a3b8" }, grid: { color: "rgba(51, 65, 85, 0.3)" } }
        },
        plugins: {
          legend: { labels: { color: "#cbd5e1" } }
        }
      }
    });

  } catch (err) {
    console.error("Error loading model benchmarks:", err);
  }
}

// =========================================================================
// TAB 3: SPARK BIG DATA STATS
// =========================================================================
async function loadSparkStats() {
  try {
    const res = await fetch("/api/spark-stats");
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById("spark-total-records").textContent = Number(data.total_records).toLocaleString();
    
    // Pearson Correlation Chart
    const ctxCorr = document.getElementById("spark-correlation-chart").getContext("2d");
    if (sparkCorrChart) sparkCorrChart.destroy();

    const topCorr = data.top_correlations.slice(0, 8);
    const corrLabels = topCorr.map(x => x[0]);
    const corrValues = topCorr.map(x => x[1]);

    sparkCorrChart = new Chart(ctxCorr, {
      type: "bar",
      data: {
        labels: corrLabels,
        datasets: [{
          label: "Pearson Correlation Coefficient with Phishing Target",
          data: corrValues,
          backgroundColor: corrValues.map(v => v > 0 ? "rgba(56, 189, 248, 0.85)" : "rgba(245, 158, 11, 0.85)"),
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { display: false } },
          y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(51, 65, 85, 0.3)" } }
        },
        plugins: {
          legend: { labels: { color: "#cbd5e1" } }
        }
      }
    });

    // Feature Comparison by Class Chart
    const ctxClass = document.getElementById("spark-class-comparison-chart").getContext("2d");
    if (sparkClassCompChart) sparkClassCompChart.destroy();

    const compareFeatures = ["URLLength", "DomainLength", "URLSimilarityIndex", "CharContinuationRate"];
    const phishStats = compareFeatures.map(f => data.feature_statistics.phishing[f]?.avg || 0);
    const legitStats = compareFeatures.map(f => data.feature_statistics.legitimate[f]?.avg || 0);

    sparkClassCompChart = new Chart(ctxClass, {
      type: "bar",
      data: {
        labels: compareFeatures,
        datasets: [
          { label: "Phishing URLs (Avg)", data: phishStats, backgroundColor: "rgba(239, 68, 68, 0.85)" },
          { label: "Legitimate URLs (Avg)", data: legitStats, backgroundColor: "rgba(16, 185, 129, 0.85)" }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { display: false } },
          y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(51, 65, 85, 0.3)" } }
        },
        plugins: {
          legend: { labels: { color: "#cbd5e1" } }
        }
      }
    });

  } catch (err) {
    console.error("Error loading Spark stats:", err);
  }
}

// =========================================================================
// TAB 4: GLOBAL SHAP XAI EXPLORER
// =========================================================================
async function loadGlobalShap() {
  try {
    const res = await fetch("/api/global-shap");
    if (!res.ok) return;
    const data = await res.json();

    const topFeatures = data.slice(0, 12);
    const labels = topFeatures.map(f => f.feature);
    const values = topFeatures.map(f => f.importance);

    const ctxGlobal = document.getElementById("global-shap-chart").getContext("2d");
    if (globalShapChart) globalShapChart.destroy();

    globalShapChart = new Chart(ctxGlobal, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "Global Mean Absolute SHAP Value (Impact on Prediction)",
          data: values,
          backgroundColor: "rgba(168, 85, 247, 0.85)",
          borderColor: "#a855f7",
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: "y",
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(51, 65, 85, 0.3)" } },
          y: { ticks: { color: "#f8fafc", font: { family: "JetBrains Mono", size: 12 } }, grid: { display: false } }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });

    // Populate Encyclopedia Table
    const tbody = document.getElementById("global-shap-table-body");
    tbody.innerHTML = "";
    data.forEach(item => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong style="color:var(--primary-cyan); font-family:var(--font-mono);">${escapeHtml(item.feature)}</strong></td>
        <td><strong style="color:#a855f7;">${item.importance}</strong></td>
        <td style="color:#cbd5e1; font-size:13px;">${escapeHtml(item.description)}</td>
      `;
      tbody.appendChild(tr);
    });

  } catch (err) {
    console.error("Error loading Global SHAP data:", err);
  }
}

// =========================================================================
// TAB 5: BATCH URL SCANNER
// =========================================================================
async function executeBatchScan() {
  const text = document.getElementById("batch-urls-input").value.trim();
  if (!text) {
    alert("Please provide at least one URL.");
    return;
  }

  const btn = event.target;
  const originalHtml = btn.innerHTML;
  btn.innerHTML = `<span>⏳</span> Scanning Batch...`;
  btn.disabled = true;

  try {
    const res = await fetch("/api/batch-scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls: text })
    });
    if (!res.ok) throw new Error("Batch scan failed");
    const data = await res.json();

    const panel = document.getElementById("batch-results-panel");
    panel.style.display = "block";
    document.getElementById("batch-summary-caption").textContent = 
      `Successfully processed ${data.total_scanned} URLs in ${data.total_time_sec}s (${data.avg_ms_per_url} ms/URL average).`;

    const tbody = document.getElementById("batch-table-body");
    tbody.innerHTML = "";
    data.results.forEach(r => {
      const isPhish = r.verdict === "PHISHING";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-family:var(--font-mono); font-size:12px; max-width:260px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${escapeHtml(r.url)}">${escapeHtml(r.url)}</td>
        <td>
          <span style="font-weight:700; color:${isPhish ? 'var(--danger-red)' : 'var(--success-green)'};">
            ${isPhish ? '🚨 PHISHING' : '✅ LEGITIMATE'}
          </span>
        </td>
        <td><strong>${r.risk_score}%</strong></td>
        <td>
          <span style="display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:700; background:${r.severity_color}22; color:${r.severity_color}; border:1px solid ${r.severity_color};">
            ${r.severity}
          </span>
        </td>
        <td style="font-size:12px; color:var(--text-muted);">${escapeHtml(r.primary_vector)}</td>
        <td>${r.is_ip ? '⚠️ Yes' : 'No'}</td>
        <td>${r.https ? '🔒 HTTPS' : 'HTTP'}</td>
      `;
      tbody.appendChild(tr);
    });

  } catch (err) {
    alert("Batch scanning error: " + err.message);
  } finally {
    btn.innerHTML = originalHtml;
    btn.disabled = false;
  }
}

// =========================================================================
// UTILITIES
// =========================================================================
function escapeHtml(str) {
  if (!str) return "";
  return String(str).replace(/[&<>"']/g, m => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[m]);
}

function formatMarkdown(text) {
  if (!text) return "";
  let formatted = escapeHtml(text);
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong style="color:#fff;">$1</strong>');
  formatted = formatted.replace(/\*(.*?)\*/g, '<em style="color:#94a3b8;">$1</em>');
  return formatted;
}
