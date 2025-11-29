import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = "data/heart.csv"
REPORTS_FIG_DIR = "reports/figures"


def load_data(path: str) -> pd.DataFrame:
    """Load the heart disease dataset from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please place heart.csv in the data/ folder."
        )
    df = pd.read_csv(path)
    return df


def train_test_split_data(df: pd.DataFrame):
    """Split the dataset into train and test sets."""
    if "target" not in df.columns:
        raise ValueError("The dataset must contain a 'target' column.")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def build_model() -> Pipeline:
    """Build a pipeline with StandardScaler + LogisticRegression."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    return model


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(y_true)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path: str):
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    X_train, X_test, y_train, y_test = train_test_split_data(df)

    model = build_model()
    print("Training Logistic Regression model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(REPORTS_FIG_DIR, exist_ok=True)
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        save_path=os.path.join(REPORTS_FIG_DIR, "confusion_matrix.png"),
    )

    plot_roc_curve(
        y_true=y_test,
        y_prob=y_prob,
        save_path=os.path.join(REPORTS_FIG_DIR, "roc_curve.png"),
    )

    print(f"Confusion matrix and ROC curve saved to: {REPORTS_FIG_DIR}")


if __name__ == "__main__":
    main()
