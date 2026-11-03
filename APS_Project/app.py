"""
Flask Web Server & Cyber Threat Intelligence API
Serves the Interactive Dashboard and REST APIs for Phishing URL Detection,
Explainable AI (SHAP), Model Comparisons, and Apache Spark Big Data Analytics.
"""

import os
import sys
import json
import time
from flask import Flask, render_template, request, jsonify, Response, send_file
import io

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure local imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from feature_extractor import extract_features_from_url, extract_features_vector, FEATURE_NAMES
from xai_engine import XAIEngine
from threat_analyzer import ThreatAnalyzer

app = Flask(__name__)

# Initialize singletons for fast inference
print("[*] Initializing XAI Engine and Threat Analyzers for all models...")
analyzers = {
    "xgboost": ThreatAnalyzer(XAIEngine(model_name="xgboost")),
    "random_forest": ThreatAnalyzer(XAIEngine(model_name="random_forest")),
    "logistic_regression": ThreatAnalyzer(XAIEngine(model_name="logistic_regression"))
}
primary_analyzer = analyzers["xgboost"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

@app.route("/")
def index():
    """Renders the main Cybersecurity Dashboard."""
    return render_template("index.html")

@app.route("/api/scan", methods=["POST"])
def scan_url():
    """Scans an individual URL and returns prediction, SHAP explanation, and threat intelligence."""
    data = request.get_json(force=True) or {}
    url = data.get("url", "").strip()
    model_choice = data.get("model", "xgboost").lower()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    analyzer = analyzers.get(model_choice, primary_analyzer)
    t0 = time.time()
    report = analyzer.analyze_url(url)
    report["latency_ms"] = round((time.time() - t0) * 1000, 2)
    report["selected_model"] = model_choice.upper()

    return jsonify(report)

@app.route("/api/batch-scan", methods=["POST"])
def batch_scan():
    """Scans multiple URLs in batch and returns tabulated threat metrics."""
    data = request.get_json(force=True) or {}
    urls = data.get("urls", [])
    if isinstance(urls, str):
        urls = [u.strip() for u in urls.splitlines() if u.strip()]

    if not urls:
        return jsonify({"error": "No URLs provided"}), 400

    results = []
    t0 = time.time()
    for u in urls[:25]: # Cap at 25 for real-time responsiveness
        rep = primary_analyzer.analyze_url(u)
        results.append({
            "url": u,
            "verdict": rep["verdict"],
            "risk_score": rep["risk_score"],
            "severity": rep["severity"],
            "severity_color": rep["severity_color"],
            "primary_vector": rep["attack_vectors"][0]["type"] if rep["attack_vectors"] else "None",
            "is_ip": rep["features"]["IsDomainIP"] == 1,
            "https": rep["features"]["IsHTTPS"] == 1
        })
    total_time = round(time.time() - t0, 3)

    return jsonify({
        "results": results,
        "total_scanned": len(results),
        "total_time_sec": total_time,
        "avg_ms_per_url": round((total_time / max(len(results), 1)) * 1000, 2)
    })

@app.route("/api/models", methods=["GET"])
def get_model_metrics():
    """Returns comparative model performance benchmark data and ROC/PR curve points."""
    metrics_path = os.path.join(MODELS_DIR, "model_comparison_metrics.json")
    curves_path = os.path.join(MODELS_DIR, "roc_pr_curves.json")

    if not os.path.exists(metrics_path):
        return jsonify({"error": "Model metrics not found. Run training script first."}), 404

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    curves = {}
    if os.path.exists(curves_path):
        with open(curves_path, "r") as f:
            curves = json.load(f)

    return jsonify({
        "metrics": metrics,
        "curves": curves
    })

@app.route("/api/spark-stats", methods=["GET"])
def get_spark_stats():
    """Returns Apache Spark distributed dataset summary and Big Data benchmarks."""
    spark_path = os.path.join(DATA_DIR, "spark_analytics_summary.json")
    if not os.path.exists(spark_path):
        return jsonify({"error": "Spark summary not found."}), 404

    with open(spark_path, "r") as f:
        stats = json.load(f)

    return jsonify(stats)

@app.route("/api/global-shap", methods=["GET"])
def get_global_shap():
    """Returns precomputed global SHAP feature importance ranking."""
    shap_path = os.path.join(MODELS_DIR, "global_shap_importance.json")
    if not os.path.exists(shap_path):
        return jsonify({"error": "Global SHAP importance not found."}), 404

    with open(shap_path, "r") as f:
        data = json.load(f)

    return jsonify(data)

@app.route("/api/export-report", methods=["POST"])
def export_report():
    """Generates and serves a downloadable standalone HTML Threat Intelligence Report."""
    data = request.get_json(force=True) or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    report = primary_analyzer.analyze_url(url)
    html_content = primary_analyzer.generate_html_report(report)

    filename = f"Threat_Report_{report['analysis_id']}.html"
    return Response(
        html_content,
        mimetype="text/html",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[+] Phishing URL Detection & XAI Platform running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
