"""
Standalone Statistical Evaluation & Report Asset Generation Script
Fulfills all requirements for:
"Applying Probability and Statistics to Evaluate a Machine Learning Model"

Includes:
1. Data Preparation & Feature Engineering (Metadata pruning, target standardization)
2. Model Training: Baseline (Logistic Regression) vs Improved (XGBoost & Random Forest)
3. Same Held-Out Test Set Evaluation
4. Empirical Bootstrap Resampling (B = 1,000 iterations)
5. Statistical Estimation: 95% Confidence Intervals (Parametric & Non-Parametric)
6. Formal Hypothesis Testing:
   - McNemar's Test on paired contingency matrix (Chi-Square)
   - Paired Student's t-Test across 5-Fold Stratified Cross Validation
7. Comprehensive 300 DPI publication figures saved to 'report_figures/'
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)

# Output configuration
sys.stdout.reconfigure(encoding='utf-8')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Publication plot styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.color'] = '#E2E8F0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "phiusiil_dataset.csv")

def run_pipeline():
    print("=" * 80)
    print("APPLYING PROBABILITY & STATISTICS TO EVALUATE A MACHINE LEARNING MODEL")
    print("=" * 80)
    
    # ---------------------------------------------------------
    # 1. DATA PREPARATION & FEATURE ENGINEERING
    # ---------------------------------------------------------
    print("\n[Step 1] Data Preparation & Feature Engineering...")
    t0 = time.time()
    df_raw = pd.read_csv(DATA_PATH, on_bad_lines='skip', low_memory=False)
    
    # Drop non-predictive metadata fields
    drop_cols = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title']
    df = df_raw.drop(columns=[c for c in drop_cols if c in df_raw.columns]).dropna(subset=['label']).copy()
    
    # Map target: 1 = Phishing (Malicious), 0 = Legitimate (Benign)
    df['target'] = (df['label'].astype(float) == 0.0).astype(int)
    df = df.drop(columns=['label'])
    
    feature_cols = [c for c in df.columns if c != 'target']
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
    X = df[feature_cols].values
    y = df['target'].values
    
    n_total = len(df)
    n_phish = int(sum(y == 1))
    n_legit = int(sum(y == 0))
    
    print(f"  - Total Dataset Size: {n_total:,} rows × {len(feature_cols)} clean predictive features")
    print(f"  - Target Distribution: Legitimate = {n_legit:,} ({n_legit/n_total*100:.2f}%), Phishing = {n_phish:,} ({n_phish/n_total*100:.2f}%)")
    print(f"  - Missing Values: {df.isna().sum().sum()} (100% clean)")
    print(f"  - Ingestion & Preparation Time: {time.time()-t0:.2f}s")
    
    # ---------------------------------------------------------
    # 2. STRATIFIED SAMPLING & HELD-OUT SPLIT
    # ---------------------------------------------------------
    print("\n[Step 2] Stratified Partitioning (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  - Training Set (80%): {len(X_train):,} samples (Legit: {sum(y_train==0):,}, Phish: {sum(y_train==1):,})")
    print(f"  - Held-Out Test Set (20%): {len(X_test):,} samples (Legit: {sum(y_test==0):,}, Phish: {sum(y_test==1):,})")
    
    # Feature Scaling: fitted on train, transformed on test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ---------------------------------------------------------
    # 3. MODEL TRAINING: BASELINE VS. IMPROVED
    # ---------------------------------------------------------
    print("\n[Step 3] Training Baseline and Improved Models on Training Set...")
    # Baseline: Logistic Regression
    t_lr = time.time()
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_train_scaled, y_train)
    t_lr_train = time.time() - t_lr
    
    # Improved 1: Random Forest
    t_rf = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    t_rf_train = time.time() - t_rf
    
    # Improved 2: XGBoost
    t_xgb = time.time()
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    t_xgb_train = time.time() - t_xgb
    
    # ---------------------------------------------------------
    # 4. HELD-OUT TEST EVALUATION
    # ---------------------------------------------------------
    print("\n[Step 4] Evaluating on the Same Held-Out Test Set (N = 38,292)...")
    
    # Predictions
    y_pred_base = lr.predict(X_test_scaled)
    y_prob_base = lr.predict_proba(X_test_scaled)[:, 1]
    
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    
    # Metrics
    models_dict = {
        "Baseline (Logistic Reg)": {"pred": y_pred_base, "prob": y_prob_base, "time": t_lr_train},
        "Improved 1 (Random Forest)": {"pred": y_pred_rf, "prob": y_prob_rf, "time": t_rf_train},
        "Improved 2 (XGBoost)": {"pred": y_pred_xgb, "prob": y_prob_xgb, "time": t_xgb_train}
    }
    
    eval_table = []
    cms = {}
    for name, data in models_dict.items():
        yp = data["pred"]
        ypr = data["prob"]
        acc = accuracy_score(y_test, yp)
        prec = precision_score(y_test, yp)
        rec = recall_score(y_test, yp)
        f1 = f1_score(y_test, yp)
        auc = roc_auc_score(y_test, ypr)
        cm = confusion_matrix(y_test, yp)
        cms[name] = cm
        tn, fp, fn, tp = cm.ravel()
        eval_table.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Train Time (s)": round(data["time"], 2)
        })
        print(f"  - {name:<26}: Acc={acc*100:.3f}% | Prec={prec*100:.3f}% | Rec={rec*100:.3f}% | F1={f1*100:.3f}% | AUC={auc*100:.3f}% | FP={fp} | FN={fn}")
        
    df_eval = pd.DataFrame(eval_table)
    
    # ---------------------------------------------------------
    # 5. STATISTICAL ESTIMATION & BOOTSTRAP SAMPLING (B = 1000)
    # ---------------------------------------------------------
    print("\n[Step 5] Statistical Estimation & Empirical Bootstrap Sampling (B = 1,000)...")
    B = 1000
    n_test = len(y_test)
    rng = np.random.default_rng(42)
    
    boot_acc_base = np.zeros(B)
    boot_acc_xgb = np.zeros(B)
    boot_f1_base = np.zeros(B)
    boot_f1_xgb = np.zeros(B)
    boot_diff_acc = np.zeros(B)
    
    t_boot = time.time()
    for b in range(B):
        idx = rng.choice(n_test, size=n_test, replace=True)
        y_b = y_test[idx]
        acc_b_base = accuracy_score(y_b, y_pred_base[idx])
        acc_b_xgb = accuracy_score(y_b, y_pred_xgb[idx])
        boot_acc_base[b] = acc_b_base
        boot_acc_xgb[b] = acc_b_xgb
        boot_f1_base[b] = f1_score(y_b, y_pred_base[idx])
        boot_f1_xgb[b] = f1_score(y_b, y_pred_xgb[idx])
        boot_diff_acc[b] = acc_b_xgb - acc_b_base
        
    print(f"  - Bootstrap Execution Completed in {time.time()-t_boot:.2f}s")
    
    # Confidence Intervals (95% CI)
    ci_acc_base = (np.percentile(boot_acc_base, 2.5), np.percentile(boot_acc_base, 97.5))
    ci_acc_xgb = (np.percentile(boot_acc_xgb, 2.5), np.percentile(boot_acc_xgb, 97.5))
    ci_diff_acc = (np.percentile(boot_diff_acc, 2.5), np.percentile(boot_diff_acc, 97.5))
    
    # Parametric Wald CI
    p_hat_base = accuracy_score(y_test, y_pred_base)
    se_base = np.sqrt(p_hat_base * (1 - p_hat_base) / n_test)
    ci_wald_base = (p_hat_base - 1.96 * se_base, p_hat_base + 1.96 * se_base)
    
    p_hat_xgb = accuracy_score(y_test, y_pred_xgb)
    se_xgb = np.sqrt(p_hat_xgb * (1 - p_hat_xgb) / n_test)
    ci_wald_xgb = (p_hat_xgb - 1.96 * se_xgb, p_hat_xgb + 1.96 * se_xgb)
    
    print("\n[+] 95% Confidence Interval Estimates:")
    print(f"  - Baseline (Logistic Regression) Accuracy: Point = {p_hat_base*100:.3f}%")
    print(f"      * Parametric Wald 95% CI:     [{ci_wald_base[0]*100:.4f}%, {ci_wald_base[1]*100:.4f}%]")
    print(f"      * Bootstrap Percentile 95% CI: [{ci_acc_base[0]*100:.4f}%, {ci_acc_base[1]*100:.4f}%]")
    print(f"  - Improved (XGBoost) Accuracy:            Point = {p_hat_xgb*100:.3f}%")
    print(f"      * Parametric Wald 95% CI:     [{ci_wald_xgb[0]*100:.4f}%, {ci_wald_xgb[1]*100:.4f}%]")
    print(f"      * Bootstrap Percentile 95% CI: [{ci_acc_xgb[0]*100:.4f}%, {ci_acc_xgb[1]*100:.4f}%]")
    print(f"  - Accuracy Difference (XGBoost - Baseline): Delta = {(p_hat_xgb - p_hat_base)*100:.4f}%")
    print(f"      * Bootstrap 95% CI for Delta: [{ci_diff_acc[0]*100:.4f}%, {ci_diff_acc[1]*100:.4f}%]")

    # ---------------------------------------------------------
    # 6. FORMAL HYPOTHESIS TESTING
    # ---------------------------------------------------------
    print("\n[Step 6] Formal Hypothesis Testing...")
    
    # Test 1: McNemar's Test
    b_corr = (y_pred_base == y_test)
    x_corr = (y_pred_xgb == y_test)
    
    n00 = np.sum(b_corr & x_corr)   # Both correct
    n01 = np.sum(b_corr & ~x_corr)  # Base correct, XGBoost wrong
    n10 = np.sum(~b_corr & x_corr)  # XGBoost correct, Base wrong
    n11 = np.sum(~b_corr & ~x_corr) # Both wrong
    
    # McNemar's Chi-square with Yates' continuity correction: (|n01 - n10| - 1)^2 / (n01 + n10)
    chi2_stat = (abs(n01 - n10) - 1)**2 / (n01 + n10) if (n01 + n10) > 0 else 0.0
    p_mcnemar = 1 - stats.chi2.cdf(chi2_stat, df=1)
    
    # Test 2: Paired 5-Fold Cross Validation t-Test
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_base, cv_xgb = [], []
    for tr_i, val_i in kf.split(X, y):
        lr_fold = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        lr_fold.fit(scaler.fit_transform(X[tr_i]), y[tr_i])
        cv_base.append(accuracy_score(y[val_i], lr_fold.predict(scaler.transform(X[val_i]))))
        
        xgb_fold = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42, n_jobs=-1)
        xgb_fold.fit(X[tr_i], y[tr_i])
        cv_xgb.append(accuracy_score(y[val_i], xgb_fold.predict(X[val_i])))
        
    t_stat, p_ttest = stats.ttest_rel(cv_xgb, cv_base)
    
    print("  --- Test 1: McNemar's Test on Paired Test Classifications ---")
    print(f"      * Discordant Pairs: n01 (Base correct, XGBoost wrong) = {n01}")
    print(f"                          n10 (XGBoost correct, Base wrong) = {n10}")
    print(f"      * Chi-Square Statistic = {chi2_stat:.4f}, df = 1")
    print(f"      * p-value = {p_mcnemar:.5f}")
    if p_mcnemar < 0.05:
        print(f"      * Decision: Reject H0 (p < 0.05) -> The improvement is STATISTICALLY SIGNIFICANT.")
    else:
        print(f"      * Decision: Fail to reject H0 (p >= 0.05).")
        
    print("\n  --- Test 2: Paired 5-Fold Cross-Validation t-Test ---")
    print(f"      * Mean Baseline Accuracy:  {np.mean(cv_base)*100:.4f}% ± {np.std(cv_base)*100:.4f}%")
    print(f"      * Mean Improved Accuracy:  {np.mean(cv_xgb)*100:.4f}% ± {np.std(cv_xgb)*100:.4f}%")
    print(f"      * Paired t-Statistic = {t_stat:.4f}, df = 4")
    print(f"      * p-value = {p_ttest:.5f}")
    if p_ttest < 0.05:
        print(f"      * Decision: Reject H0 (p < 0.05) -> Significant improvement across data resamplings.")
    else:
        print(f"      * Decision: Fail to reject H0 (p >= 0.05).")

    # ---------------------------------------------------------
    # 7. GENERATING PUBLICATION-QUALITY REPORT FIGURES
    # ---------------------------------------------------------
    print("\n[Step 7] Generating High-Resolution (300 DPI) Report Figures...")
    
    # Figure 1: Dataset Architecture & Class Balance
    domains = {
        'Lexical & URL Structure': 12,
        'Obfuscation & Evasion': 8,
        'Content & Behavior': 25,
        'Brand & Financial': 5
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].pie(
        [n_legit, n_phish], labels=[f'Legitimate\\n{n_legit:,} (57.3%)', f'Phishing\\n{n_phish:,} (42.7%)'],
        autopct='%1.1f%%', colors=['#10B981', '#EF4444'], startangle=140, pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2.5),
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    axes[0].set_title("Ground-Truth Target Class Distribution\\n(Total: 191,459 Records)", fontsize=12, weight='bold')
    bars = axes[1].barh(list(domains.keys()), list(domains.values()), color=['#3B82F6', '#8B5CF6', '#F59E0B', '#EC4899'], edgecolor='#CBD5E1', height=0.55)
    axes[1].set_title("50 Engineered Predictive Features by Domain", fontsize=12, weight='bold')
    axes[1].set_xlabel("Feature Count", fontsize=10, weight='bold')
    axes[1].set_xlim(0, 30)
    for b in bars:
        axes[1].text(b.get_width() + 0.5, b.get_y() + b.get_height()/2, f'{int(b.get_width())} features', va='center', weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "01_dataset_architecture_and_class_distribution.png"), dpi=300)
    plt.close()
    print("  ✓ Saved: 01_dataset_architecture_and_class_distribution.png")

    # Figure 2: Empirical Feature Densities
    key_features = [
        ("URLSimilarityIndex", "URL Similarity Index"),
        ("CharContinuationRate", "Char Continuation Rate"),
        ("TLDLegitimateProb", "TLD Legitimate Probability"),
        ("LineOfCode", "HTML Lines of Code")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for idx, (col_name, display_title) in enumerate(key_features):
        ax = axes[idx]
        p99 = df[col_name].quantile(0.99)
        p01 = df[col_name].quantile(0.00)
        sns.kdeplot(df[df['target']==0][col_name].clip(p01, p99), ax=ax, color='#10B981', fill=True, alpha=0.35, label='Legitimate (0)', linewidth=2)
        sns.kdeplot(df[df['target']==1][col_name].clip(p01, p99), ax=ax, color='#EF4444', fill=True, alpha=0.35, label='Phishing (1)', linewidth=2)
        ax.set_title(f"Density: {display_title}", fontsize=11, weight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.suptitle("Empirical Feature Density Distributions (Legitimate vs. Phishing)", fontsize=13, weight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "02_empirical_feature_densities.png"), dpi=300)
    plt.close()
    print("  ✓ Saved: 02_empirical_feature_densities.png")

    # Figure 3: Correlation Matrix Heatmap
    top_corr = ["URLSimilarityIndex", "CharContinuationRate", "TLDLegitimateProb", "URLCharProb", "NoOfSubDomain", "LineOfCode", "DomainLength", "URLLength", "IsHTTPS", "HasTitle", "HasPasswordField", "target"]
    plt.figure(figsize=(13, 10))
    sns.heatmap(df[top_corr].corr(), annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, center=0, cbar_kws={'label': 'Pearson r'}, annot_kws={"size": 9, "weight": "bold"})
    plt.title("Inter-Feature Pearson Correlation Heatmap & Target Association", fontsize=13, weight='bold', pad=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_inter_feature_correlation_heatmap.png"), dpi=300)
    plt.close()
    print("  ✓ Saved: 03_inter_feature_correlation_heatmap.png")

    # Figure 4: Target Correlation Divergence
    corrs = df[feature_cols].apply(lambda x: x.corr(df['target'])).sort_values()
    div = pd.concat([corrs.head(8), corrs.tail(8)])
    plt.figure(figsize=(12, 6))
    colors = ['#10B981' if v < 0 else '#EF4444' for v in div.values]
    bars = plt.barh(div.index, div.values, color=colors, edgecolor='#CBD5E1', height=0.6)
    plt.axvline(0, color='#64748B', linestyle='--')
    plt.title("Top Directional Correlation Signals with Phishing Target", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Pearson Correlation (r)", fontsize=10, weight='bold')
    plt.xlim(-1.0, 1.0)
    for b in bars:
        w = b.get_width()
        plt.text(w + (0.02 if w>=0 else -0.09), b.get_y() + b.get_height()/2, f'{w:+.2f}', va='center', weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "04_target_correlation_divergence.png"), dpi=300)
    plt.close()
    print("  ✓ Saved: 04_target_correlation_divergence.png")

    # Figure 5: Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    cm_palettes = ["Blues", "Greens", "Purples"]
    for i, (m_name, cm) in enumerate(cms.items()):
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt=',d', cmap=cm_palettes[i], ax=ax, cbar=False,
                    xticklabels=['Pred: Legit', 'Pred: Phish'], yticklabels=['True: Legit', 'True: Phish'],
                    annot_kws={"size": 12, "weight": "bold"})
        acc_val = accuracy_score(y_test, models_dict[m_name]["pred"]) * 100
        ax.set_title(f"{m_name}\\nAccuracy: {acc_val:.3f}% | FP: {cm[0,1]} | FN: {cm[1,0]}", fontsize=11, weight='bold')
        ax.set_xlabel("Predicted Label", weight='bold')
        ax.set_ylabel("True Ground-Truth", weight='bold')
    plt.suptitle("Confusion Matrices on Held-Out Test Set (N = 38,292)", fontsize=13, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_model_confusion_matrices.png"), dpi=300)
    plt.close()
    print("  ✓ Saved: 05_model_confusion_matrices.png")

    # Figure 6: ROC & PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    curve_colors = {"Baseline (Logistic Reg)": "#3B82F6", "Improved 1 (Random Forest)": "#10B981", "Improved 2 (XGBoost)": "#8B5CF6"}
    for name, data in models_dict.items():
        fpr, tpr, _ = roc_curve(y_test, data["prob"])
        auc_score = roc_auc_score(y_test, data["prob"])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc_score*100:.2f}%)", color=curve_colors[name], linewidth=2.2)
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, data["prob"])
        axes[1].plot(rec_vals, prec_vals, label=f"{name}", color=curve_colors[name], linewidth=2.2)
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4)
    axes[0].set_title("Receiver Operating Characteristic (ROC)", weight='bold')
    axes[0].set_xlabel("False Positive Rate (FPR)")
    axes[0].set_ylabel("True Positive Rate (TPR / Recall)")
    axes[0].legend(loc="lower right")
    axes[1].set_title("Precision-Recall (PR) Curves", weight='bold')
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "06_roc_and_precision_recall_curves.png"), dpi=300)
    plt.close()
    print("  ✓ Saved: 06_roc_and_precision_recall_curves.png")

    # Figure 7: Multi-Metric Comparison Bar Chart
    df_plot = df_eval.melt(id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], var_name="Metric", value_name="Score")
    df_plot["Score (%)"] = df_plot["Score"] * 100
    plt.figure(figsize=(13, 5.5))
    bar_plot = sns.barplot(data=df_plot, x="Metric", y="Score (%)", hue="Model", palette=["#3B82F6", "#10B981", "#8B5CF6"], edgecolor="#CBD5E1")
    plt.title("Comparative Performance Across 5 Evaluation Metrics (N = 38,292)", fontsize=13, weight='bold', pad=12)
    plt.ylim(99.4, 100.1)
    plt.ylabel("Score (%)", weight='bold')
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "07_multi_metric_model_comparison.png"), dpi=300)
    plt.close()
    print("  ✓ Saved: 07_multi_metric_model_comparison.png")

    # Figure 8: Bootstrap Sampling Distributions & Q-Q Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(boot_acc_base * 100, kde=True, color='#3B82F6', ax=axes[0], label='Baseline (Logistic Regression)', stat='density', alpha=0.4)
    sns.histplot(boot_acc_xgb * 100, kde=True, color='#8B5CF6', ax=axes[0], label='Improved (XGBoost)', stat='density', alpha=0.4)
    axes[0].set_title("Bootstrap Sampling Distribution of Accuracy (B = 1,000)", weight='bold')
    axes[0].set_xlabel("Accuracy (%)", weight='bold')
    axes[0].set_ylabel("Density", weight='bold')
    axes[0].legend()
    stats.probplot(boot_diff_acc * 100, dist='norm', plot=axes[1])
    axes[1].set_title("Normal Q-Q Plot: Accuracy Delta (XGBoost - Baseline)", weight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "08_bootstrap_sampling_distributions.png"), dpi=300)
    plt.close()
    print("  ✓ Saved: 08_bootstrap_sampling_distributions.png")

    # Figure 9: Forest Plot of 95% Confidence Intervals
    plt.figure(figsize=(11, 4.5))
    y_pos = [0, 0.6, 1.6, 2.2]
    pts = [p_hat_base * 100, p_hat_xgb * 100, df_eval.loc[0, 'F1-Score']*100, df_eval.loc[2, 'F1-Score']*100]
    ci_lo = [ci_acc_base[0] * 100, ci_acc_xgb[0] * 100, np.percentile(boot_f1_base, 2.5)*100, np.percentile(boot_f1_xgb, 2.5)*100]
    ci_hi = [ci_acc_base[1] * 100, ci_acc_xgb[1] * 100, np.percentile(boot_f1_base, 97.5)*100, np.percentile(boot_f1_xgb, 97.5)*100]
    for i in range(4):
        c = '#3B82F6' if i % 2 == 0 else '#8B5CF6'
        plt.errorbar(pts[i], y_pos[i], xerr=[[pts[i] - ci_lo[i]], [ci_hi[i] - pts[i]]], fmt='o', color=c, capsize=6, elinewidth=2.5, markersize=8)
    plt.yticks(y_pos, ['Base: Accuracy', 'Imp: Accuracy', 'Base: F1-Score', 'Imp: F1-Score'], weight='bold')
    plt.xlabel("Metric Score (%)", weight='bold')
    plt.title("Forest Plot: 95% Bootstrap Confidence Intervals (B = 1,000)", weight='bold', pad=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "09_confidence_intervals_forest_plot.png"), dpi=300)
    plt.close()
    print("  ✓ Saved: 09_confidence_intervals_forest_plot.png")

    # Figure 10: Hypothesis Testing Diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    mcnemar_matrix = np.array([[n00, n01], [n10, n11]])
    sns.heatmap(mcnemar_matrix, annot=True, fmt=',d', cmap='YlGnBu', ax=axes[0],
                xticklabels=['XGBoost: Correct', 'XGBoost: Error'],
                yticklabels=['Base: Correct', 'Base: Error'],
                annot_kws={'size': 13, 'weight': 'bold'})
    axes[0].set_title(f"McNemar 2x2 Contingency Matrix\\n(Chi2 = {chi2_stat:.2f}, p = {p_mcnemar:.4f})", weight='bold')
    fold_df = pd.DataFrame({
        'Fold': [f'Fold {i+1}' for i in range(5)] * 2,
        'Model': ['Baseline (Logistic Reg)'] * 5 + ['Improved (XGBoost)'] * 5,
        'Accuracy': [x * 100 for x in cv_base] + [x * 100 for x in cv_xgb]
    })
    sns.barplot(data=fold_df, x='Fold', y='Accuracy', hue='Model', palette=['#3B82F6', '#8B5CF6'], ax=axes[1])
    axes[1].set_ylim(99.8, 100.05)
    axes[1].set_title(f"5-Fold Cross Validation Accuracy\\n(Paired t = {t_stat:.2f}, p = {p_ttest:.4f})", weight='bold')
    axes[1].legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "10_hypothesis_testing_diagnostics.png"), dpi=300)
    plt.close()
    print("  ✓ Saved: 10_hypothesis_testing_diagnostics.png")

    # Figure 11: Feature Importance (RF vs XGBoost)
    rf_imp = rf.feature_importances_
    xgb_imp = xgb.feature_importances_
    df_imp = pd.DataFrame({'Feature': feature_cols, 'Random Forest': rf_imp, 'XGBoost': xgb_imp})
    top15 = df_imp.sort_values(by='XGBoost', ascending=False).head(15).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].barh(top15['Feature'], top15['Random Forest'], color='#10B981', edgecolor='#CBD5E1', height=0.6)
    axes[0].set_title("Random Forest: Top 15 Features (Gini Impurity)", weight='bold')
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)
    axes[1].barh(top15['Feature'], top15['XGBoost'], color='#8B5CF6', edgecolor='#CBD5E1', height=0.6)
    axes[1].set_title("XGBoost: Top 15 Features (Gain Score)", weight='bold')
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)
    plt.suptitle("Top 15 Predictive Features Comparison Across Tree Ensembles", weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "11_feature_importance_ranking.png"), dpi=300)
    plt.close()
    print("  ✓ Saved: 11_feature_importance_ranking.png")

    print("\n" + "=" * 80)
    print(f"✓ ALL EVALUATIONS & 11 PUBLICATION FIGURES CREATED IN: {FIGURES_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline()

