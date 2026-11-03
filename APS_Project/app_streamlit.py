"""
Streamlit Interactive Dashboard for Phishing URL Detection & Cyber Threat Analysis
Clean, simplified, and intuitive interface focusing on:
  1. Real-Time URL Threat Scanner with SHAP Explainability
  2. Model Benchmark Studio: Baseline Logistic Regression vs. Improved Random Forest & XGBoost
  3. Global Explainable AI (SHAP) Insights
  4. Multi-URL Batch Security Scanner
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="CyberShield XAI - Phishing URL Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure local imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from feature_extractor import extract_features_from_url, extract_features_vector, FEATURE_NAMES
from xai_engine import XAIEngine
from threat_analyzer import ThreatAnalyzer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

@st.cache_resource
def load_analyzers():
    return {
        "XGBoost (Improved)": ThreatAnalyzer(XAIEngine(model_name="xgboost")),
        "Random Forest (Improved)": ThreatAnalyzer(XAIEngine(model_name="random_forest")),
        "Logistic Regression (Baseline)": ThreatAnalyzer(XAIEngine(model_name="logistic_regression"))
    }

analyzers = load_analyzers()

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        font-weight: 600;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🛡️ CYBERSHIELD XAI")
st.sidebar.caption("Explainable AI Phishing URL Threat Intelligence")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 System Overview")
st.sidebar.markdown("- **Dataset**: PhiUSIIL (191,459 Records)")
st.sidebar.markdown("- **Feature Space**: 50 Engineered Signals")
st.sidebar.markdown("- **Baseline Model**: Logistic Regression")
st.sidebar.markdown("- **Improved Models**: Random Forest & XGBoost")
st.sidebar.markdown("- **Explainability**: SHAP TreeExplainer")
st.sidebar.markdown("---")
st.sidebar.info("💡 Select a tab to perform live URL threat triage, inspect model benchmarks, or view global XAI rankings.")

# Main Navigation Tabs (Simplified and Focused)
tabs = st.tabs([
    "🛡️ Real-Time URL Scanner",
    "📊 Model Benchmark Studio",
    "🧠 Explainable AI (SHAP)",
    "🚀 Multi-URL Batch Scanner"
])

# =========================================================================
# TAB 1: REAL-TIME URL SCANNER
# =========================================================================
with tabs[0]:
    st.header("🔍 Real-Time URL Threat Scanner")
    st.markdown("Decompose any URL into 50 structural indicators, detect malicious patterns, and generate instance-level SHAP explanations.")

    col_input, col_model = st.columns([3, 1])
    with col_input:
        target_url = st.text_input(
            "Enter target URL to analyze:",
            value="http://192.168.1.100/paypal/login.php?cmd=verify&token=4892842"
        )
    with col_model:
        selected_model_name = st.selectbox(
            "Classifier Model:",
            ["XGBoost (Improved)", "Random Forest (Improved)", "Logistic Regression (Baseline)"]
        )

    # Quick scenario buttons
    st.write("**Quick Test Scenarios:**")
    p1, p2, p3, p4, p5 = st.columns(5)
    if p1.button("🚨 IP Harvester"):
        target_url = "http://192.168.1.100/paypal/login.php?cmd=verify&token=4892842"
    if p2.button("⚠️ Typosquatting"):
        target_url = "https://chase-security-verify.update-accounts.xyz/signin"
    if p3.button("🪙 Crypto Scam"):
        target_url = "http://binance-airdrop-claim.eth-wallet.cc/connect"
    if p4.button("✅ Google Search"):
        target_url = "https://www.google.com/search?q=cybersecurity"
    if p5.button("✅ Wikipedia"):
        target_url = "https://en.wikipedia.org/wiki/Phishing"

    if st.button("⚡ Execute Threat Scan", type="primary"):
        analyzer = analyzers[selected_model_name]
        t0 = time.time()
        report = analyzer.analyze_url(target_url)
        elapsed_ms = round((time.time() - t0) * 1000, 2)

        st.markdown("---")
        
        # High-level Metrics
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        is_phish = report["verdict"] == "PHISHING"
        
        res_col1.metric("Verdict", report["verdict"], delta="MALICIOUS" if is_phish else "SAFE", delta_color="inverse")
        res_col2.metric("Risk Score", f"{report['risk_score']}%")
        res_col3.metric("Severity", report["severity"])
        res_col4.metric("Inference Latency", f"{elapsed_ms} ms")

        # XAI Explanation Narrative
        st.subheader("🧠 Threat Intelligence Narrative (XAI)")
        if is_phish:
            st.error(report["shap_explanation"]["narrative"])
        else:
            st.success(report["shap_explanation"]["narrative"])

        # Necessary Graph: SHAP Waterfall / Feature Contributions
        st.subheader("📉 Local Feature Impact (SHAP Feature Contributions)")
        shap_factors = report["shap_explanation"]["top_contributors"]
        df_shap = pd.DataFrame(shap_factors)
        
        fig_shap = px.bar(
            df_shap,
            x="shap_value",
            y="feature",
            orientation="h",
            color="impact",
            color_discrete_map={"malicious": "#ef4444", "benign": "#10b981"},
            labels={"shap_value": "SHAP Impact Score (+ Malicious / - Benign)", "feature": "Feature"},
            title="Key Feature Drivers for Current Prediction"
        )
        fig_shap.update_layout(yaxis=dict(autorange="reversed"), template="plotly_dark", height=320)
        st.plotly_chart(fig_shap, use_container_width=True)

        # Attack Vectors & IOCs
        col_vec, col_ioc = st.columns(2)
        with col_vec:
            st.subheader("🎯 Detected Attack Vectors")
            if report["attack_vectors"]:
                for v in report["attack_vectors"]:
                    st.error(f"**{v['type']}**: {v['detail']}")
            else:
                st.success("✓ No malicious attack vectors detected.")

        with col_ioc:
            st.subheader("🔍 Indicators of Compromise (IOC)")
            df_iocs = pd.DataFrame(report["iocs"])
            st.dataframe(df_iocs, use_container_width=True)

        # Download Report
        html_report = analyzer.generate_html_report(report)
        st.download_button(
            label="📥 Download Standalone HTML Threat Incident Report",
            data=html_report,
            file_name=f"Threat_Report_{report['analysis_id']}.html",
            mime="text/html"
        )

# =========================================================================
# TAB 2: MODEL BENCHMARK STUDIO
# =========================================================================
with tabs[1]:
    st.header("📊 Machine Learning Model Comparative Evaluation")
    st.markdown("Comparing **Baseline (Logistic Regression)** against **Improved Models (Random Forest and XGBoost)** trained on the full dataset of 191,459 URLs.")

    metrics_path = os.path.join(MODELS_DIR, "model_comparison_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

        rows = []
        for m_name, m in metrics_data.items():
            rows.append({
                "Model": m_name,
                "Type": m.get("type", "Classifier"),
                "Accuracy (%)": m["accuracy"],
                "Precision (%)": m["precision"],
                "Recall (%)": m["recall"],
                "F1-Score (%)": m["f1_score"],
                "ROC-AUC (%)": m["roc_auc"],
                "FPR (%)": m["fpr"],
                "Train Time (s)": m["train_time_sec"],
                "Latency / 1k (ms)": m["latency_ms_per_1000"]
            })

        df_metrics = pd.DataFrame(rows)
        st.dataframe(df_metrics, use_container_width=True)

        # Necessary Graph 1: Comparative Metric Bar Chart
        st.subheader("📈 Algorithm Metric Comparison")
        df_melt = pd.melt(
            df_metrics,
            id_vars=["Model"],
            value_vars=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)", "ROC-AUC (%)"],
            var_name="Metric",
            value_name="Score"
        )
        fig_bar = px.bar(
            df_melt,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            title="Baseline vs. Improved Ensemble Performance (38,292 Test Samples)",
            template="plotly_dark",
            range_y=[99.5, 100.1],
            color_discrete_map={
                "Logistic Regression": "#3B82F6",
                "Random Forest": "#10B981",
                "XGBoost": "#8B5CF6"
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Necessary Graph 2: ROC Curves
        curves_path = os.path.join(MODELS_DIR, "roc_pr_curves.json")
        if os.path.exists(curves_path):
            with open(curves_path, "r") as f:
                curves_data = json.load(f)

            col_roc, col_cm = st.columns([1, 1])
            with col_roc:
                st.subheader("🎯 Receiver Operating Characteristic (ROC)")
                fig_roc = go.Figure()
                color_map = {"Logistic Regression": "#3B82F6", "Random Forest": "#10B981", "XGBoost": "#8B5CF6"}
                for m_name, c_data in curves_data.items():
                    roc_pts = c_data["roc"]
                    fig_roc.add_trace(go.Scatter(
                        x=roc_pts["fpr"],
                        y=roc_pts["tpr"],
                        mode="lines",
                        name=f"{m_name} (AUC: {metrics_data[m_name]['roc_auc']}%)",
                        line=dict(color=color_map.get(m_name, "#ffffff"), width=2.5)
                    ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    line=dict(dash="dash", color="gray"),
                    name="Random Chance"
                ))
                fig_roc.update_layout(
                    title="ROC Discrimination Curves",
                    xaxis_title="False Positive Rate (FPR)",
                    yaxis_title="True Positive Rate (Recall)",
                    template="plotly_dark",
                    height=400
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            with col_cm:
                st.subheader("🔍 Confusion Matrix Comparison")
                cm_img_path = os.path.join(MODELS_DIR, "confusion_matrices.png")
                if os.path.exists(cm_img_path):
                    st.image(cm_img_path, caption="Confusion Matrices across Baseline and Improved Models on Held-Out Test Data", use_container_width=True)

# =========================================================================
# TAB 3: EXPLAINABLE AI (SHAP) EXPLORER
# =========================================================================
with tabs[2]:
    st.header("🧠 Explainable AI (SHAP) Global Insights")
    st.markdown("Understand which features drive phishing classification across the entire dataset using cooperative game theory.")

    shap_path = os.path.join(MODELS_DIR, "global_shap_importance.json")
    if os.path.exists(shap_path):
        with open(shap_path, "r") as f:
            global_shap = json.load(f)

        df_global = pd.DataFrame(global_shap[:15])
        
        # Necessary Graph: Global Feature Importance
        fig_global = px.bar(
            df_global,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Purples",
            labels={"importance": "Mean |SHAP| Value", "feature": "Feature"},
            title="Top 15 Most Influential Phishing Indicators (Full Dataset Global SHAP)",
            template="plotly_dark",
            height=450
        )
        fig_global.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_global, use_container_width=True)

        st.subheader("📖 Feature Rationale Dictionary")
        st.dataframe(df_global[["feature", "importance", "description"]], use_container_width=True)

# =========================================================================
# TAB 4: BATCH SCANNER
# =========================================================================
with tabs[3]:
    st.header("🚀 Multi-URL Batch Security Scanner")
    st.markdown("Paste multiple URLs for concurrent high-speed triage.")

    default_batch_text = """http://192.168.1.100/paypal/login.php?cmd=verify&token=4892842
https://chase-security-verify.update-accounts.xyz/signin
http://binance-airdrop-claim.eth-wallet.cc/connect
https://www.google.com/search?q=cybersecurity
https://en.wikipedia.org/wiki/Phishing"""

    batch_input = st.text_area("Target URLs (one per line):", value=default_batch_text, height=140)

    if st.button("⚡ Run Batch Threat Triage", type="primary"):
        urls = [u.strip() for u in batch_input.splitlines() if u.strip()]
        if urls:
            analyzer = analyzers["XGBoost (Improved)"]
            results = []
            progress_bar = st.progress(0)
            t_start = time.time()
            for idx, u in enumerate(urls[:25]):
                rep = analyzer.analyze_url(u)
                results.append({
                    "URL": u,
                    "Verdict": rep["verdict"],
                    "Risk Score": f"{rep['risk_score']}%",
                    "Severity": rep["severity"],
                    "Primary Vector": rep["attack_vectors"][0]["type"] if rep["attack_vectors"] else "None",
                    "Direct IP": "Yes" if rep["features"]["IsDomainIP"] == 1 else "No",
                    "HTTPS": "Yes" if rep["features"]["IsHTTPS"] == 1 else "No"
                })
                progress_bar.progress((idx + 1) / len(urls[:25]))

            total_elapsed = round(time.time() - t_start, 3)
            st.success(f"✓ Processed {len(results)} URLs in {total_elapsed}s ({round(total_elapsed/len(results)*1000, 2)} ms/URL average).")

            df_batch = pd.DataFrame(results)
            st.dataframe(df_batch, use_container_width=True)

            csv_data = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Batch Results CSV",
                data=csv_data,
                file_name="Batch_Phishing_Scan_Results.csv",
                mime="text/csv"
            )
