"""
Explainable AI (XAI) Engine powered by SHAP (SHapley Additive exPlanations)
Provides both Global Feature Importance and Local (Instance-Level) Interpretability.
Generates waterfall breakdowns, feature attribution scores, and natural language cyber narratives.
"""

import os
import sys
import json
import joblib
import numpy as np

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure local imports work regardless of cwd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import shap

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
GLOBAL_SHAP_PATH = os.path.join(MODELS_DIR, "global_shap_importance.json")

# Human-readable contextual descriptions for key features in threat analysis
FEATURE_EXPLANATIONS = {
    'IsDomainIP': {
        'pos': "Direct IP host detected instead of a legitimate registered domain name, hiding true ownership.",
        'neg': "Domain uses a standard registered DNS hostname rather than a raw IP address."
    },
    'URLSimilarityIndex': {
        'pos': "Severe brand-name spoofing/typosquatting detected mimicking a known legitimate enterprise.",
        'neg': "URL similarity matches authentic domain naming conventions without deceptive spoofing."
    },
    'URLLength': {
        'pos': "Excessively long URL (>70 chars), commonly used to pack redirect tokens and conceal malicious domains.",
        'neg': "Normal, concise URL length matching standard user-facing web services."
    },
    'NoOfSubDomain': {
        'pos': "Multiple nested subdomains stacked together, a frequent evasion technique to mimic trusted brand paths.",
        'neg': "Normal subdomain hierarchy conforming to standard web architecture."
    },
    'HasObfuscation': {
        'pos': "Detected encoded character strings (%xx, hex, @) intentionally obscuring target destinations.",
        'neg': "Clean, transparent URL parameters without obfuscation symbols."
    },
    'SpacialCharRatioInURL': {
        'pos': "Abnormally high proportion of special characters and delimiters indicating parameter stuffing.",
        'neg': "Balanced lexical distribution typical of legitimate web pages."
    },
    'TLDLegitimateProb': {
        'pos': "Domain utilizes a top-level domain (TLD) with historically high cyber abuse and spam frequency.",
        'neg': "Domain utilizes a reputable top-level domain (.gov, .edu, .org, or established .com)."
    },
    'IsHTTPS': {
        'pos': "Absence of HTTPS/TLS encryption exposes user traffic and signals an untrusted phishing endpoint.",
        'neg': "Valid HTTPS protocol enforced, securing the transport layer."
    },
    'CharContinuationRate': {
        'pos': "High repetitive character sequences indicative of algorithmic URL generation or evasion.",
        'neg': "Natural language lexical variation across domain and path tokens."
    },
    'Bank': {
        'pos': "Targeted banking and financial authentication keywords found in suspicious context.",
        'neg': "No deceptive banking or authorization lures detected."
    },
    'Pay': {
        'pos': "Payment gateway and billing tokens detected, suggesting credential/credit card harvesting.",
        'neg': "No deceptive payment or invoice tokens found."
    },
    'Crypto': {
        'pos': "Cryptocurrency wallet or exchange lure tokens detected, common in drainer campaigns.",
        'neg': "No crypto drainer or token lures identified."
    },
    'HasPasswordField': {
        'pos': "Webpage contains an input password field on an unverified domain, characteristic of credential harvesting.",
        'neg': "No deceptive password harvesting forms identified."
    },
    'HasExternalFormSubmit': {
        'pos': "Form action routes submitted credentials to an external third-party server, a definitive phishing indicator.",
        'neg': "Form actions stay within first-party domain boundary."
    },
    'NoOfiFrame': {
        'pos': "Hidden iframe elements detected, frequently utilized for clickjacking or invisible credential scraping.",
        'neg': "Standard page rendering without suspicious iframe embedding."
    }
}

class XAIEngine:
    def __init__(self, model_name="xgboost"):
        self.model_name = model_name
        self._load_artifacts()
        self._init_explainer()

    def _load_artifacts(self):
        """Loads trained model, scaler, and feature definitions."""
        with open(os.path.join(MODELS_DIR, "feature_names.json"), "r") as f:
            self.feature_names = json.load(f)

        if self.model_name == "xgboost":
            self.model = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
            self.is_scaled = False
        elif self.model_name == "random_forest":
            self.model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
            self.is_scaled = False
        else:
            self.model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
            self.is_scaled = True

        self.scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        
        # Load sample baseline data
        sample_data = joblib.load(os.path.join(MODELS_DIR, "sample_test_data.pkl"))
        self.baseline_X = sample_data["X_test"][:200]
        self.baseline_X_scaled = sample_data["X_test_scaled"][:200]

    def _init_explainer(self):
        """Initializes SHAP TreeExplainer or LinearExplainer."""
        try:
            if self.model_name in ["xgboost", "random_forest"]:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                self.explainer = shap.LinearExplainer(self.model, self.baseline_X_scaled)
        except Exception as e:
            print(f"[!] Warning: SHAP explainer fallback due to: {e}")
            self.explainer = shap.Explainer(self.model, self.baseline_X)

    def precompute_global_importance(self):
        """Precomputes global SHAP summary statistics across baseline sample."""
        print("[*] Precomputing Global SHAP feature importances...")
        X_sample = self.baseline_X_scaled if self.is_scaled else self.baseline_X
        
        shap_values = self.explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            # Binary classification [legitimate, phishing]
            shap_vals = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_vals = shap_values[:, :, 1]
        else:
            shap_vals = shap_values

        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]

        global_features = []
        for idx in sorted_indices:
            feat = self.feature_names[idx]
            global_features.append({
                "feature": feat,
                "importance": round(float(mean_abs_shap[idx]), 4),
                "description": FEATURE_EXPLANATIONS.get(feat, {}).get("pos", f"Metric analyzing {feat} variation across phishing and legitimate datasets.")
            })

        with open(GLOBAL_SHAP_PATH, "w") as f:
            json.dump(global_features, f, indent=2)

        print(f"[+] Global SHAP importance saved to {GLOBAL_SHAP_PATH} (Top feature: {global_features[0]['feature']} = {global_features[0]['importance']})")
        return global_features

    def explain_instance(self, feature_vector: list, top_k=8) -> dict:
        """
        Computes SHAP feature attribution for a single URL feature vector.
        Returns:
            - base_value: baseline model log-odds / expected output
            - prediction_proba: estimated phishing probability
            - top_contributors: ordered list of features pushing towards phishing (+) or legitimate (-)
            - natural_language_summary: plain-English cyber intelligence explanation
        """
        arr = np.array(feature_vector, dtype=float).reshape(1, -1)
        if self.is_scaled:
            arr_eval = self.scaler.transform(arr)
        else:
            arr_eval = arr

        # Compute SHAP values
        raw_shap = self.explainer.shap_values(arr_eval)
        
        if isinstance(raw_shap, list):
            sv = raw_shap[1][0]
        elif len(raw_shap.shape) == 3:
            sv = raw_shap[0, :, 1]
        elif len(raw_shap.shape) == 2:
            sv = raw_shap[0]
        else:
            sv = raw_shap

        base_val = getattr(self.explainer, "expected_value", 0.0)
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[-1])
        else:
            base_val = float(base_val)

        # Predict probability
        proba = float(self.model.predict_proba(arr_eval)[0, 1])

        # Match features with SHAP contributions
        contributions = []
        for i, val in enumerate(sv):
            f_name = self.feature_names[i]
            f_val = round(float(feature_vector[i]), 4)
            shap_val = round(float(val), 4)
            impact = "malicious" if shap_val > 0 else "benign"
            
            explanation_dict = FEATURE_EXPLANATIONS.get(f_name, {})
            text_desc = explanation_dict.get("pos" if shap_val > 0 else "neg", f"{f_name} evaluated at {f_val}")

            contributions.append({
                "feature": f_name,
                "value": f_val,
                "shap_value": shap_val,
                "abs_shap": abs(shap_val),
                "impact": impact,
                "description": text_desc
            })

        # Sort by absolute SHAP impact
        contributions.sort(key=lambda x: x["abs_shap"], reverse=True)
        top_factors = contributions[:top_k]

        # Generate Natural Language Narrative
        pos_factors = [f for f in top_factors if f["impact"] == "malicious"]
        neg_factors = [f for f in top_factors if f["impact"] == "benign"]

        verdict_str = "PHISHING" if proba >= 0.50 else "LEGITIMATE"
        narrative_parts = []
        narrative_parts.append(
            f"The AI framework classifies this URL as **{verdict_str}** with **{proba*100:.1f}% confidence**."
        )

        if verdict_str == "PHISHING":
            if pos_factors:
                key_reasons = "; ".join([f"{f['feature']} ({f['description']})" for f in pos_factors[:3]])
                narrative_parts.append(
                    f"**Primary Malicious Drivers:** The prediction was predominantly driven by: {key_reasons}."
                )
            if neg_factors:
                neg_reasons = ", ".join([f['feature'] for f in neg_factors[:2]])
                narrative_parts.append(
                    f"*Mitigating Legitimate Signals:* Counteracting features ({neg_reasons}) slightly reduced the risk score but were insufficient to overturn the threat verdict."
                )
        else:
            if neg_factors:
                legit_reasons = "; ".join([f"{f['feature']} ({f['description']})" for f in neg_factors[:3]])
                narrative_parts.append(
                    f"**Legitimacy Indicators:** Authentic characteristics verified include: {legit_reasons}."
                )
            if pos_factors:
                warn_reasons = ", ".join([f['feature'] for f in pos_factors[:2]])
                narrative_parts.append(
                    f"*Minor Anomalies Noted:* Minor flags detected ({warn_reasons}), but standard trust metrics remain dominant."
                )

        narrative = " ".join(narrative_parts)

        return {
            "prediction_proba": round(proba, 4),
            "verdict": verdict_str,
            "base_value": round(base_val, 4),
            "top_contributors": top_factors,
            "narrative": narrative
        }

if __name__ == "__main__":
    from feature_extractor import extract_features_vector
    engine = XAIEngine(model_name="xgboost")
    engine.precompute_global_importance()
    
    test_phish = "http://192.168.1.100/paypal/login.php?cmd=verify&token=4892842"
    feats = extract_features_vector(test_phish)
    res = engine.explain_instance(feats)
    print("\n" + "="*70)
    print(f"URL: {test_phish}")
    print(f"Verdict: {res['verdict']} ({res['prediction_proba']*100:.1f}%)")
    print("Top SHAP Contributors:")
    for c in res['top_contributors'][:5]:
        print(f"  {c['feature']} = {c['value']} -> SHAP: {c['shap_value']:+.4f} [{c['impact'].upper()}]")
    print(f"\nNarrative:\n{res['narrative']}")
