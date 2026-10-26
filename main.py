"""Iris flower classification: a simple end-to-end ML pipeline.

Steps: load data (Pandas) -> explore -> visualise -> train a pre-built
model (scikit-learn RandomForest) -> evaluate -> interpret.
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
SEED = 42


def load_data():
    """Load Iris into a Pandas DataFrame with a readable species column."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris


def explore(df):
    """Print basic summaries and save exploratory plots."""
    print("Shape:", df.shape)
    print("Missing values per column:\n", df.isna().sum(), "\n")
    print(df.describe().round(2), "\n")
    print("Class counts:\n", df["species"].value_counts(), "\n")

    # Histograms of each feature
    df.drop(columns="species").hist(figsize=(8, 6), bins=15)
    plt.suptitle("Feature distributions")
    plt.tight_layout()
    plt.savefig(OUT / "feature_histograms.png", dpi=120)
    plt.close()

    # Scatter plot of petal measurements, coloured by species
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, group in df.groupby("species", observed=True):
        ax.scatter(
            group["petal length (cm)"], group["petal width (cm)"], label=name
        )
    ax.set_xlabel("Petal length (cm)")
    ax.set_ylabel("Petal width (cm)")
    ax.set_title("Petal size by species")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "petal_scatter.png", dpi=120)
    plt.close(fig)


def main():
    df, iris = load_data()
    explore(df)

    X = df[iris.feature_names].to_numpy()
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}\n")
    print(classification_report(y_test, preds, target_names=iris.target_names))

    # Interpretation: which features matter most?
    importances = pd.Series(
        model.feature_importances_, index=iris.feature_names
    ).sort_values()
    print("Feature importances:\n", importances.round(3).to_string(), "\n")
    importances.plot.barh(figsize=(7, 4), title="Feature importance")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=120)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, display_labels=iris.target_names
    )
    plt.title("Iris confusion matrix")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=120)
    plt.close()

    joblib.dump(model, OUT / "model.joblib")
    (OUT / "metrics.json").write_text(
        json.dumps({"accuracy": round(float(acc), 3), "n_test": int(len(y_test))}, indent=2)
    )
    print("Mean petal length (numpy):", round(float(np.mean(X[:, 2])), 2))


if __name__ == "__main__":
    main()
