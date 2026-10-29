"""
Applying Probability and Statistics to Evaluate a Machine Learning Model

Dataset: Breast Cancer Wisconsin Diagnostic dataset bundled with scikit-learn.
Baseline: Decision Tree
Improved model: Extra Trees ensemble
Evaluation: Same stratified held-out test set, paired bootstrap CI, exact McNemar test.

Run:
    pip install numpy scipy scikit-learn
    python evaluate_ml_models.py
"""

import numpy as np
from scipy.stats import binomtest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.25
BOOTSTRAP_REPLICATES = 10_000
ALPHA = 0.05


def paired_bootstrap_ci(correct_baseline, correct_improved, n_bootstrap=10_000,
                        confidence=0.95, random_state=42):
    """
    Percentile bootstrap CI for the paired accuracy difference:
    accuracy(improved) - accuracy(baseline).
    Each resample selects test-row indices once and applies them to both models.
    """
    rng = np.random.default_rng(random_state)
    n = len(correct_baseline)
    paired_differences = correct_improved.astype(float) - correct_baseline.astype(float)
    boot_differences = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_differences[i] = paired_differences[indices].mean()

    tail = (1.0 - confidence) / 2.0
    return tuple(np.quantile(boot_differences, [tail, 1.0 - tail]))


def exact_mcnemar_test(y_true, pred_baseline, pred_improved):
    """
    Exact two-sided McNemar test using the binomial distribution.
    b: baseline correct, improved wrong.
    c: baseline wrong, improved correct.
    """
    baseline_correct = pred_baseline == y_true
    improved_correct = pred_improved == y_true
    b = int(np.sum(baseline_correct & ~improved_correct))
    c = int(np.sum(~baseline_correct & improved_correct))

    if b + c == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(min(b, c), n=b + c, p=0.5,
                                  alternative="two-sided").pvalue)
    return b, c, p_value


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Stratified sampling preserves the class proportions in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Baseline: a single decision tree.
    baseline = DecisionTreeClassifier(random_state=RANDOM_STATE)

    # Improved candidate: an ensemble of randomized trees.
    improved = ExtraTreesClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    baseline.fit(X_train, y_train)
    improved.fit(X_train, y_train)

    # Both models are evaluated on exactly the same held-out examples.
    pred_baseline = baseline.predict(X_test)
    pred_improved = improved.predict(X_test)

    acc_baseline = accuracy_score(y_test, pred_baseline)
    acc_improved = accuracy_score(y_test, pred_improved)
    difference = acc_improved - acc_baseline

    correct_baseline = pred_baseline == y_test
    correct_improved = pred_improved == y_test
    ci_low, ci_high = paired_bootstrap_ci(
        correct_baseline, correct_improved,
        n_bootstrap=BOOTSTRAP_REPLICATES,
        confidence=0.95,
        random_state=RANDOM_STATE
    )

    b, c, p_value = exact_mcnemar_test(y_test, pred_baseline, pred_improved)

    print("=" * 72)
    print("PROBABILITY AND STATISTICS FOR ML MODEL EVALUATION")
    print("=" * 72)
    print(f"Dataset: {data.DESCR.splitlines()[0]}")
    print(f"Total samples: {len(y)}; features: {X.shape[1]}")
    print(f"Train samples: {len(y_train)}; held-out test samples: {len(y_test)}")
    print(f"Random state: {RANDOM_STATE}; test size: {TEST_SIZE:.0%}")
    print()
    print(f"Baseline (Decision Tree) accuracy: {acc_baseline:.4f} ({acc_baseline:.2%})")
    print(f"Improved (Extra Trees) accuracy:  {acc_improved:.4f} ({acc_improved:.2%})")
    print(f"Observed accuracy difference:     {difference:+.4f} ({difference:+.2%})")
    print(f"95% paired bootstrap CI:           [{ci_low:+.4f}, {ci_high:+.4f}]")
    print()
    print("Exact McNemar test (paired predictions)")
    print(f"Baseline correct / improved wrong (b): {b}")
    print(f"Baseline wrong / improved correct (c): {c}")
    print(f"Two-sided p-value: {p_value:.4f}")
    print(f"Significance level alpha: {ALPHA:.2f}")

    print("\nConfusion matrix — baseline (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, pred_baseline))
    print("\nConfusion matrix — improved (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, pred_improved))

    print("\nClassification report — baseline:")
    print(classification_report(y_test, pred_baseline,
                                target_names=data.target_names, zero_division=0))
    print("Classification report — improved:")
    print(classification_report(y_test, pred_improved,
                                target_names=data.target_names, zero_division=0))

    print("\nSTATISTICAL DECISION")
    if p_value < ALPHA and ci_low > 0:
        print("Evidence supports an improvement at the 5% significance level.")
    else:
        print("The improved model has higher observed accuracy, but the evidence")
        print("is not strong enough to establish a reliable improvement at 5%.")
        print("The bootstrap interval includes zero and/or McNemar's p-value")
        print("is not below alpha. Do not claim statistical significance.")

    print("\nInterpretation note: the test set is a sample, so its result has")
    print("sampling uncertainty. The bootstrap CI quantifies uncertainty in")
    print("the paired accuracy difference; McNemar's test compares discordant")
    print("paired predictions. Neither test guarantees performance on every")
    print("future population or dataset.")


if __name__ == "__main__":
    main()
