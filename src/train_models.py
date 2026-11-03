"""
Machine Learning Pipeline & Model Comparison Suite
Trains, benchmarks, and persists 3 ML models on the full PhiUSIIL dataset:
  1. Logistic Regression (Baseline Model)
  2. Random Forest Classifier (Improved Model 1)
  3. XGBoost Classifier (Improved Model 2)

Uses full dataset (191,459 records) with stratified 80/20 train/test split.
Computes comprehensive metrics: Accuracy, Precision, Recall, F1, ROC-AUC, FPR, FNR,
Confusion Matrices, Training Time, and Inference Latency.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)

# Ensure stdout handles UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "phiusiil_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DROP_COLS = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title']

def load_and_preprocess_data():
    print("=" * 70)
    print(f"[*] Ingesting and preprocessing full dataset from {DATA_PATH}...")
    t0 = time.time()
    
    # Read dataset
    df = pd.read_csv(DATA_PATH, on_bad_lines='skip', low_memory=False)
    
    # Drop identifier and non-predictive metadata columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Clean target label: PhiUSIIL raw has 1=Legitimate, 0=Phishing
    # In cybersecurity convention: 1=Phishing (Malicious), 0=Legitimate (Benign)
    df = df.dropna(subset=['label'])
    df['target'] = (df['label'].astype(float) == 0.0).astype(int)
    df = df.drop(columns=['label'])
    
    # Convert all feature columns to numeric
    feature_cols = [c for c in df.columns if c != 'target']
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    print(f"[+] Total clean dataset: {len(df):,} records")
    print(f"    - Legitimate (0): {sum(df['target']==0):,} ({sum(df['target']==0)/len(df)*100:.2f}%)")
    print(f"    - Phishing   (1): {sum(df['target']==1):,} ({sum(df['target']==1)/len(df)*100:.2f}%)")
    print(f"[+] Clean predictive features: {len(feature_cols)}")
    print(f"[+] Ingestion elapsed time: {time.time()-t0:.2f}s")
    
    X = df[feature_cols].values
    y = df['target'].values
    
    return X, y, feature_cols

def train_and_evaluate_models():
    X, y, feature_cols = load_and_preprocess_data()
    
    # Save feature names
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
        
    # Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[+] Stratified Split:")
    print(f"    - Training Set:   {len(X_train):,} samples (80%)")
    print(f"    - Held-Out Test:  {len(X_test):,} samples (20%)")

    # Fit StandardScaler on training data
    print("[*] Fitting StandardScaler on training set...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # Save representative test slice for quick XAI background
    joblib.dump({
        "X_test": X_test[:1000],
        "X_test_scaled": X_test_scaled[:1000],
        "y_test": y_test[:1000],
        "feature_names": feature_cols
    }, os.path.join(MODELS_DIR, "sample_test_data.pkl"))

    # Define the 3 targeted models: Baseline LR, Improved RF, Improved XGBoost
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "use_scaled": True,
            "filename": "logistic_regression.pkl",
            "type": "Baseline Linear"
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1),
            "use_scaled": False,
            "filename": "random_forest.pkl",
            "type": "Improved Ensemble (Bagging)"
        },
        "XGBoost": {
            "model": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42, n_jobs=-1),
            "use_scaled": False,
            "filename": "xgboost.pkl",
            "type": "Improved Gradient Boosting"
        }
    }

    metrics_results = {}
    curves_data = {}
    conf_matrices = {}

    print("=" * 70)
    print("[*] TRAINING & BENCHMARKING ALGORITHMS (FULL DATASET)")
    print("=" * 70)

    for name, config in models.items():
        print(f"\n---> [{config['type']}] Training: {name}...")
        clf = config["model"]
        use_scaled = config["use_scaled"]
        
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test
        
        # Training time
        t_start = time.time()
        clf.fit(X_tr, y_train)
        train_time = round(time.time() - t_start, 3)
        
        # Save model
        joblib.dump(clf, os.path.join(MODELS_DIR, config["filename"]))
        
        # Inference time benchmark (over 1,000 samples)
        t_infer = time.time()
        _ = clf.predict(X_te[:1000])
        latency_1k = round((time.time() - t_infer) * 1000, 2)
        
        # Evaluation on full held-out test set
        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        cm = confusion_matrix(y_test, y_pred)
        conf_matrices[name] = cm
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        metrics_results[name] = {
            "type": config["type"],
            "accuracy": round(acc * 100, 3),
            "precision": round(prec * 100, 3),
            "recall": round(rec * 100, 3),
            "f1_score": round(f1 * 100, 3),
            "roc_auc": round(auc * 100, 3),
            "fpr": round(fpr * 100, 3),
            "fnr": round(fnr * 100, 3),
            "train_time_sec": train_time,
            "latency_ms_per_1000": latency_1k,
            "test_samples": len(y_test),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
        }
        
        print(f"     Accuracy: {acc*100:.3f}% | Precision: {prec*100:.3f}% | Recall: {rec*100:.3f}% | F1: {f1*100:.3f}%")
        print(f"     ROC-AUC:  {auc*100:.3f}% | FPR: {fpr*100:.3f}% | Train Time: {train_time}s | Latency (1k): {latency_1k}ms")
        
        # Generate ROC and PR curves
        fpr_pts, tpr_pts, _ = roc_curve(y_test, y_proba)
        prec_pts, rec_pts, _ = precision_recall_curve(y_test, y_proba)
        
        # Subsample curve points for clean JSON storage and web charting
        step = max(1, len(fpr_pts) // 100)
        curves_data[name] = {
            "roc": {
                "fpr": [round(float(x), 5) for x in fpr_pts[::step]],
                "tpr": [round(float(x), 5) for x in tpr_pts[::step]]
            },
            "pr": {
                "recall": [round(float(x), 5) for x in rec_pts[::step]],
                "precision": [round(float(x), 5) for x in prec_pts[::step]]
            }
        }

    # Save metrics JSON
    with open(os.path.join(MODELS_DIR, "model_comparison_metrics.json"), "w") as f:
        json.dump(metrics_results, f, indent=2)
    print(f"\n[+] Saved model metrics to: {os.path.join(MODELS_DIR, 'model_comparison_metrics.json')}")

    # Save curve data JSON
    with open(os.path.join(MODELS_DIR, "roc_pr_curves.json"), "w") as f:
        json.dump(curves_data, f, indent=2)
    print(f"[+] Saved ROC/PR curve points to: {os.path.join(MODELS_DIR, 'roc_pr_curves.json')}")

    # Generate and save Confusion Matrix figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    palette_names = ["Blues", "Greens", "Purples"]
    for idx, (m_name, cm) in enumerate(conf_matrices.items()):
        ax = axes[idx]
        sns.heatmap(
            cm, annot=True, fmt=',d', cmap=palette_names[idx], cbar=False, ax=ax,
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'],
            annot_kws={"size": 13, "weight": "bold"}
        )
        ax.set_title(f"{m_name}\nAcc: {metrics_results[m_name]['accuracy']}% | F1: {metrics_results[m_name]['f1_score']}%", fontsize=12, pad=10)
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("True Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "confusion_matrices.png"), dpi=200)
    plt.close()
    print(f"[+] Saved confusion matrix comparison plot to: {os.path.join(MODELS_DIR, 'confusion_matrices.png')}")

    # Precompute global SHAP importance using XGBoost
    print("\n[*] Precomputing Global SHAP feature importance...")
    try:
        import shap
        xgb_model = models["XGBoost"]["model"]
        explainer = shap.TreeExplainer(xgb_model)
        # Use 1,000 samples for swift, high-accuracy global importance
        shap_values = explainer.shap_values(X_test[:1000])
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        
        feature_rationales = {
            "URLSimilarityIndex": "Quantifies structural lexical deviation from benign domain references.",
            "LineOfCode": "Reflects webpage markup depth; phishing clones often exhibit distinct line-count profiles.",
            "IsHTTPS": "Validates TLS/SSL encrypted transport; absence flags elevated attack exposure.",
            "CharContinuationRate": "Measures consecutive alphanumeric sequences used to mask malicious paths.",
            "NoOfSubDomain": "Detects excessive subdomain nesting common in DNS tunneling and brand impersonation.",
            "DomainTitleMatchScore": "Identifies semantic mismatch between target brand name and registration domain.",
            "URLCharProb": "Calculates character distribution probability against English language baseline.",
            "TLDLegitimateProb": "Historical empirical safety probability of the top-level domain.",
            "HasPasswordField": "Detects credential capture forms deployed for credential harvesting.",
            "Bank": "Presence of banking terminology triggering financial fraud monitoring.",
            "Pay": "Payment gateway keywords indicating transaction interception campaigns.",
            "Crypto": "Cryptocurrency wallet keywords associated with drainer attacks.",
            "NoOfObfuscatedChar": "Count of percent-encoded (%xx) and obfuscated payload characters.",
            "SpacialCharRatioInURL": "Abnormal ratio of delimiters, query operators, and punctuation in the path.",
            "IsDomainIP": "Direct IPv4/IPv6 host destination bypassing traditional DNS reputation checks."
        }
        
        shap_records = []
        for rank, idx in enumerate(sorted_indices):
            f_name = feature_cols[idx]
            shap_records.append({
                "rank": rank + 1,
                "feature": f_name,
                "importance": round(float(mean_abs_shap[idx]), 5),
                "description": feature_rationales.get(f_name, f"Predictive feature for classification.")
            })
            
        with open(os.path.join(MODELS_DIR, "global_shap_importance.json"), "w") as f:
            json.dump(shap_records, f, indent=2)
        print(f"[+] Global SHAP importance computed for {len(shap_records)} features and saved.")
    except Exception as e:
        print(f"[-] SHAP computation error: {e}")

    print("\n[+] Full pipeline training completed successfully!")

if __name__ == "__main__":
    train_and_evaluate_models()
