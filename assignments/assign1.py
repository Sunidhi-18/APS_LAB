import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import precision_recall_curve, auc, classification_report, confusion_matrix

# Generate an imbalanced dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=2, 
    n_redundant=10, 
    n_clusters_per_class=1, 
    weights=[0.9, 0.1],  # 90% class 0, 10% class 1 
    flip_y=0, 
    random_state=42
)

# Split dataset using stratified sampling to preserve class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Generate predictions and probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Evaluate Precision-Recall and PR-AUC
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

# Plot Precision-Recall Curve
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR Curve (PR-AUC = {pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Imbalanced Classification")
plt.legend(loc="lower left")
plt.grid(True)
plt.show()

# Print Evaluation Metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Area Under Precision-Recall Curve (PR-AUC):", round(pr_auc, 4))
