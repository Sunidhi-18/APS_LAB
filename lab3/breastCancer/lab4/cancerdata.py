import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # Load the dataset
    data = load_breast_cancer()
    x = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series((data.target == 0).astype(int), name="malignant")
    
    print(y.value_counts())
    print("feature matrix shape: ", x.shape)
    print("target shape: ", y.shape)
    print("class names: ", data.target_names)
    
    # Examine the class distribution
    class_counts = y.value_counts().sort_index()
    class_distribution = pd.DataFrame({
        "class": data.target_names,
        "count": class_counts.values,
        "probability": class_counts.values / len(y)
    })
    print("\nClass Distribution:")
    print(class_distribution)
    
    return x, y, class_distribution

def plot_class_distribution(class_distribution):
    class_distribution.plot(
        x="class",
        y="count",
        kind="bar",
        legend=False,
        color=["darkred", "green"]
    )
    plt.ylabel("no.of observations")
    plt.title("class distribution")
    plt.xticks(rotation=0)
    plt.show()

def get_train_test_data(x, y):
    # Create training and testing samples
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print("\ntraining size:", len(y_train))
    print("testing size:", len(y_test))
    print("\nTraining properties: ")
    print(y_train.value_counts(normalize=True).sort_index())
    print("\nTesting properties: ")
    print(y_test.value_counts(normalize=True).sort_index())
    
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x, y, class_dist = load_and_preprocess_data()
    plot_class_distribution(class_dist)
    x_train, x_test, y_train, y_test = get_train_test_data(x, y)