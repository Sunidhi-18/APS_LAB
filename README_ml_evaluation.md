# Probability and Statistics for ML Model Evaluation

This project compares a baseline Decision Tree with an Extra Trees classifier on the Breast Cancer Wisconsin Diagnostic dataset.

## Run locally

```bash
pip install numpy scipy scikit-learn
python evaluate_ml_models.py
```

The script uses a stratified train/test split, evaluates both models on the same held-out test set, estimates a paired bootstrap confidence interval for the accuracy difference, and runs an exact two-sided McNemar test.

## Files

- `evaluate_ml_models.py` — reproducible experiment
- `ML_Model_Statistical_Evaluation_Report.docx` — report with methodology, observations, decision, limitations, and statistical concepts

The expected result for the fixed random seed is a higher observed accuracy for Extra Trees, but the paired bootstrap interval includes zero and the exact McNemar p-value is above 0.05. Therefore, the result is not statistically significant at the 5% level.
