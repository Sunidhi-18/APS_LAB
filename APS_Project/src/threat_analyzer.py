"""
Cyber Threat Analysis & Incident Reporting Engine
Synthesizes machine learning classifications, SHAP explainability factors, and cybersecurity heuristic indicators
into actionable threat intelligence reports, risk scores, attack vector classifications, and exportable artifacts.
"""

import os
import sys
import time
import json
from urllib.parse import urlparse

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_features_from_url, extract_features_vector, FEATURE_NAMES
from xai_engine import XAIEngine

class ThreatAnalyzer:
    def __init__(self, xai_engine: XAIEngine = None):
        if xai_engine is None:
            self.xai = XAIEngine(model_name="xgboost")
        else:
            self.xai = xai_engine

    def analyze_url(self, url: str) -> dict:
        """
        Performs full end-to-end threat analysis on a given URL:
          1. Feature extraction
          2. ML Inference + SHAP Explanation
          3. Threat Vector Classification
          4. Indicators of Compromise (IOC) Analysis
          5. Risk Scoring & Severity Assignment
          6. SOC / User Defensive Recommendations
        """
        parsed_url = url.strip()
        if not parsed_url.startswith("http://") and not parsed_url.startswith("https://"):
            parsed_url = "http://" + parsed_url

        parsed = urlparse(parsed_url)
        domain = parsed.netloc.lower()
        protocol = parsed.scheme.upper()

        # 1. Feature Extraction
        features_dict = extract_features_from_url(parsed_url)
        feature_vector = [features_dict[k] for k in FEATURE_NAMES]

        # 2. XAI + Prediction
        xai_result = self.xai.explain_instance(feature_vector, top_k=8)
        phishing_prob = xai_result["prediction_proba"]
        verdict = xai_result["verdict"]

        # 3. Risk Score (0 - 100 scale)
        risk_score = round(phishing_prob * 100, 1)

        # 4. Severity Mapping
        if risk_score >= 85:
            severity = "CRITICAL"
            color = "#ef4444" # red
        elif risk_score >= 65:
            severity = "HIGH"
            color = "#f97316" # orange
        elif risk_score >= 40:
            severity = "MEDIUM"
            color = "#eab308" # yellow
        elif risk_score >= 20:
            severity = "LOW"
            color = "#3b82f6" # blue
        else:
            severity = "SAFE"
            color = "#22c55e" # green

        # 5. Attack Vector Classification
        attack_vectors = []
        if features_dict["IsDomainIP"] == 1:
            attack_vectors.append({
                "type": "Direct IP-Hosted Threat",
                "detail": "Adversary is hosting the phishing page directly on an unindexed IPv4 address to evade domain reputation filters."
            })
        if features_dict["URLSimilarityIndex"] < 50:
            attack_vectors.append({
                "type": "Brand Impersonation / Typosquatting",
                "detail": "Domain synthetically mimics a recognized high-traffic brand name to manipulate user trust."
            })
        if features_dict["Bank"] == 1 or features_dict["Pay"] == 1:
            attack_vectors.append({
                "type": "Financial Credential Harvester",
                "detail": "Targeted harvesting of financial institution login credentials, credit card details, or payment tokens."
            })
        if features_dict["Crypto"] == 1:
            attack_vectors.append({
                "type": "Cryptocurrency Drainer Scam",
                "detail": "Lures users to connect Web3 wallets or provide seed phrases/private keys."
            })
        if features_dict["HasObfuscation"] == 1:
            attack_vectors.append({
                "type": "URL Obfuscation & Evasion",
                "detail": "Employs character encoding (%xx, @) and path stuffing to mask malicious endpoints from standard web filters."
            })
        if features_dict["NoOfSubDomain"] >= 2:
            attack_vectors.append({
                "type": "Subdomain Stacking Evasion",
                "detail": "Embeds legitimate brand tokens inside nested subdomains of an attacker-controlled base domain."
            })

        if not attack_vectors and verdict == "PHISHING":
            attack_vectors.append({
                "type": "Anomalous Structural Pattern",
                "detail": "URL structure exhibits statistically correlated malicious heuristics across multiple lexical features."
            })

        # 6. Indicators of Compromise (IOCs)
        iocs = [
            {"indicator": "Domain", "value": domain, "flagged": features_dict["IsDomainIP"] == 1 or features_dict["URLSimilarityIndex"] < 60},
            {"indicator": "Protocol", "value": protocol, "flagged": protocol == "HTTP"},
            {"indicator": "Direct IP Host", "value": "Yes" if features_dict["IsDomainIP"] == 1 else "No", "flagged": features_dict["IsDomainIP"] == 1},
            {"indicator": "TLD Legitimate Probability", "value": f"{features_dict['TLDLegitimateProb']*100:.0f}%", "flagged": features_dict["TLDLegitimateProb"] < 0.40},
            {"indicator": "Subdomain Depth", "value": str(int(features_dict["NoOfSubDomain"])), "flagged": features_dict["NoOfSubDomain"] >= 2},
            {"indicator": "URL Length", "value": f"{int(features_dict['URLLength'])} chars", "flagged": features_dict["URLLength"] > 75},
            {"indicator": "Special Characters Ratio", "value": f"{features_dict['SpacialCharRatioInURL']*100:.1f}%", "flagged": features_dict["SpacialCharRatioInURL"] > 0.08},
            {"indicator": "Obfuscation Detected", "value": "Yes" if features_dict["HasObfuscation"] == 1 else "No", "flagged": features_dict["HasObfuscation"] == 1}
        ]

        # 7. Actionable Recommendations
        if verdict == "PHISHING":
            user_recommendations = [
                "DO NOT enter passwords, phone numbers, or credit card information on this site.",
                "Close the browser tab immediately and avoid downloading any suggested attachments.",
                "If credentials were submitted previously, reset passwords on the authentic service immediately and enable Multi-Factor Authentication (MFA)."
            ]
            soc_recommendations = [
                f"Block domain '{domain}' on enterprise DNS firewalls (Cisco Umbrella, Pi-hole, Cloudflare Gateway).",
                f"Sinkhole or blacklist the associated IP address across perimeter firewalls.",
                "Search SIEM proxy and firewall logs for any internal endpoints that established connections to this URL.",
                "Submit malicious URL report to Google Safe Browsing and PhishTank feeds."
            ]
        else:
            user_recommendations = [
                "URL structure demonstrates authentic domain architecture and security standards.",
                "Verify standard SSL padlock in the address bar before entering private information.",
                "Always ensure the root domain matches the exact organization you intend to visit."
            ]
            soc_recommendations = [
                "No immediate threat detected. Standard outbound perimeter inspection applies.",
                "Continue automated monitoring of domain changes and certificate renewals."
            ]

        report = {
            "target_url": parsed_url,
            "domain": domain,
            "protocol": protocol,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "analysis_id": f"IOC-{int(time.time())}-{hash(parsed_url) % 10000:04d}",
            "verdict": verdict,
            "risk_score": risk_score,
            "severity": severity,
            "severity_color": color,
            "attack_vectors": attack_vectors,
            "iocs": iocs,
            "shap_explanation": xai_result,
            "user_recommendations": user_recommendations,
            "soc_recommendations": soc_recommendations,
            "features": features_dict
        }

        return report

    def generate_html_report(self, report: dict) -> str:
        """Generates a standalone, beautifully formatted cybersecurity HTML threat report."""
        verdict = report["verdict"]
        severity = report["severity"]
        color = report["severity_color"]
        url = report["target_url"]
        
        iocs_html = "".join([
            f"<tr style='background: {'#3b1111' if ioc['flagged'] else '#1e293b'}'>"
            f"<td style='padding:8px; border:1px solid #334155; font-weight:600;'>{ioc['indicator']}</td>"
            f"<td style='padding:8px; border:1px solid #334155; font-family:monospace;'>{ioc['value']}</td>"
            f"<td style='padding:8px; border:1px solid #334155; color:{'#ef4444' if ioc['flagged'] else '#22c55e'}; font-weight:bold;'>{'FLAGGED' if ioc['flagged'] else 'NORMAL'}</td>"
            f"</tr>" for ioc in report["iocs"]
        ])

        vectors_html = "".join([
            f"<div style='background:#1e293b; padding:12px; margin-bottom:8px; border-left:4px solid {color}; border-radius:4px;'>"
            f"<strong style='color:#f8fafc; font-size:15px;'>{v['type']}</strong>"
            f"<p style='color:#94a3b8; margin:4px 0 0 0; font-size:13px;'>{v['detail']}</p>"
            f"</div>" for v in report["attack_vectors"]
        ]) or "<p style='color:#94a3b8;'>No malicious attack vectors detected.</p>"

        shap_html = "".join([
            f"<tr style='background:{'#2e1010' if c['impact'] == 'malicious' else '#0f291e'}'>"
            f"<td style='padding:8px; border:1px solid #334155; font-family:monospace;'>{c['feature']}</td>"
            f"<td style='padding:8px; border:1px solid #334155;'>{c['value']}</td>"
            f"<td style='padding:8px; border:1px solid #334155; font-weight:bold; color:{'#ef4444' if c['impact'] == 'malicious' else '#22c55e'};'>{c['shap_value']:+.4f}</td>"
            f"<td style='padding:8px; border:1px solid #334155; font-size:12px; color:#cbd5e1;'>{c['description']}</td>"
            f"</tr>" for c in report["shap_explanation"]["top_contributors"]
        ])

        user_recs_html = "".join([f"<li style='margin-bottom:6px; color:#e2e8f0;'>{r}</li>" for r in report["user_recommendations"]])
        soc_recs_html = "".join([f"<li style='margin-bottom:6px; color:#e2e8f0;'>{r}</li>" for r in report["soc_recommendations"]])

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cyber Threat Analysis Report - {report['analysis_id']}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0b0f19; color: #f8fafc; margin: 0; padding: 24px; }}
  .container {{ max-width: 900px; margin: 0 auto; background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 32px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
  .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #1f2937; padding-bottom: 16px; margin-bottom: 24px; }}
  .badge {{ display: inline-block; padding: 6px 16px; border-radius: 9999px; font-weight: 800; font-size: 14px; text-transform: uppercase; }}
  h2 {{ color: #38bdf8; border-bottom: 1px solid #334155; padding-bottom: 8px; margin-top: 28px; font-size: 18px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
  .narrative-box {{ background: #0f172a; border-left: 4px solid #38bdf8; padding: 14px; border-radius: 6px; font-size: 14px; line-height: 1.6; color: #cbd5e1; }}
  .footer {{ margin-top: 32px; border-top: 1px solid #1f2937; padding-top: 16px; font-size: 11px; color: #64748b; text-align: center; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1 style="margin:0; font-size:24px; color:#f8fafc;">Cyber Threat Intelligence Report</h1>
      <div style="font-size:12px; color:#94a3b8; margin-top:4px;">ID: {report['analysis_id']} | Generated: {report['timestamp']}</div>
    </div>
    <div style="text-align:right;">
      <div class="badge" style="background:{color}; color:#ffffff;">{severity} RISK ({report['risk_score']}%)</div>
      <div style="font-size:13px; font-weight:bold; margin-top:6px; color:{color};">VERDICT: {verdict}</div>
    </div>
  </div>

  <div style="background:#0f172a; padding:12px 16px; border-radius:8px; margin-bottom:20px;">
    <div style="font-size:12px; color:#94a3b8; text-transform:uppercase; font-weight:bold;">Inspected Target URL:</div>
    <div style="font-family:monospace; font-size:14px; word-break:break-all; color:#38bdf8; margin-top:4px;">{url}</div>
  </div>

  <h2>1. Explainable AI (XAI) Synthesis & Narrative</h2>
  <div class="narrative-box">
    {report['shap_explanation']['narrative']}
  </div>

  <h2>2. SHAP Feature Attribution Waterfall (Local Explainability)</h2>
  <table>
    <thead>
      <tr style="background:#334155; text-align:left; color:#f8fafc;">
        <th style="padding:8px; border:1px solid #475569;">Feature Name</th>
        <th style="padding:8px; border:1px solid #475569;">Observed Value</th>
        <th style="padding:8px; border:1px solid #475569;">SHAP Impact</th>
        <th style="padding:8px; border:1px solid #475569;">Explainable Factor Description</th>
      </tr>
    </thead>
    <tbody>
      {shap_html}
    </tbody>
  </table>

  <h2>3. Detected Attack Vectors & Threat Categorization</h2>
  {vectors_html}

  <h2>4. Technical Indicators of Compromise (IOC) Matrix</h2>
  <table>
    <thead>
      <tr style="background:#334155; text-align:left; color:#f8fafc;">
        <th style="padding:8px; border:1px solid #475569;">Security Parameter</th>
        <th style="padding:8px; border:1px solid #475569;">Measured State</th>
        <th style="padding:8px; border:1px solid #475569;">Threat Status</th>
      </tr>
    </thead>
    <tbody>
      {iocs_html}
    </tbody>
  </table>

  <h2>5. Security Action Plan & Remediation</h2>
  <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
    <div style="background:#1e293b; padding:16px; border-radius:8px;">
      <h3 style="margin-top:0; color:#f8fafc; font-size:15px; border-bottom:1px solid #334155; padding-bottom:6px;">End-User Recommendations</h3>
      <ul style="padding-left:18px; margin:0; font-size:13px;">
        {user_recs_html}
      </ul>
    </div>
    <div style="background:#1e293b; padding:16px; border-radius:8px;">
      <h3 style="margin-top:0; color:#f8fafc; font-size:15px; border-bottom:1px solid #334155; padding-bottom:6px;">SOC / IT Admin Actions</h3>
      <ul style="padding-left:18px; margin:0; font-size:13px;">
        {soc_recs_html}
      </ul>
    </div>
  </div>

  <div class="footer">
    APS Project: Explainable AI and Big Data Framework for Phishing URL Detection & Cyber Threat Analysis<br>
    Powered by PhiUSIIL Dataset, Apache Spark, Scikit-Learn, XGBoost, and SHAP XAI Engine.
  </div>
</div>
</body>
</html>
"""
        return html

if __name__ == "__main__":
    analyzer = ThreatAnalyzer()
    test_url = "http://192.168.1.100/paypal/login.php?cmd=verify&token=4892842"
    rep = analyzer.analyze_url(test_url)
    print("\n" + "=" * 70)
    print(f"THREAT REPORT: {rep['analysis_id']}")
    print(f"URL: {rep['target_url']}")
    print(f"Verdict: {rep['verdict']} | Severity: {rep['severity']} ({rep['risk_score']}%)")
    print(f"Attack Vectors: {[v['type'] for v in rep['attack_vectors']]}")
    print("=" * 70)
