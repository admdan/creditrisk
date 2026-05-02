# case9.py
# ---------------------------------------------------------
# This script trains a Multi-Layer Perceptron (MLP)
# on the synthetic_credit_risk_case9.csv dataset.
#
# Case 9 intentionally uses a dataset whose label generation
# is no longer calibrated, producing a much more imbalanced
# target distribution.
# ---------------------------------------------------------

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model


def main():
    repo_root = Path(__file__).resolve().parents[2]
    visuals_dir = repo_root / "final_project" / "visuals"
    visuals_dir.mkdir(exist_ok=True)

    data_path = repo_root / "final_project"/ "cases"/ "data" / "synthetic_credit_risk_case9.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Please run final_project/cases/dataset_creation_case9.py first."
        )

    df = pd.read_csv(data_path)

    target_col = "high_risk"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    print("Dataset label distribution:")
    print(y.value_counts(normalize=True).sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    tf.random.set_seed(42)

    model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.20),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1
    )

    y_prob = model.predict(X_test_scaled).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    print("\n===== Test Results =====")
    print(f"Accuracy: {acc:.3f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("MLP Training and Validation Loss (Case 9)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(visuals_dir / "mlp_loss_curve_case9.png", dpi=200)
    plt.show()

    print("\nLoss curve saved as final_project/visuals/mlp_loss_curve_case9.png")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (MLP Case 9)")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred Low (0)", "Pred High (1)"])
    plt.yticks([0, 1], ["Actual Low (0)", "Actual High (1)"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(visuals_dir / "mlp_confusion_matrix_case9.png", dpi=200)
    plt.show()

    print("Saved: final_project/visuals/mlp_confusion_matrix_case9.png")

    low_probs = y_prob[y_test.values == 0]
    high_probs = y_prob[y_test.values == 1]

    plt.figure(figsize=(8, 5))
    plt.hist(low_probs, bins=30, alpha=0.7, label="Actual Low Risk (0)")
    plt.hist(high_probs, bins=30, alpha=0.7, label="Actual High Risk (1)")
    plt.xlabel("Predicted Probability of High Risk")
    plt.ylabel("Count")
    plt.title("Predicted Probability Distribution by Class (Case 9)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(visuals_dir / "mlp_probability_distribution_case9.png", dpi=200)
    plt.show()

    print("Saved: final_project/visuals/mlp_probability_distribution_case9.png")

    plot_model(
        model,
        to_file=str(visuals_dir / "mlp_architecture_case9.png"),
        show_shapes=True,
        show_layer_names=True,
        dpi=200
    )

    print("Saved: final_project/visuals/mlp_architecture_case9.png")


if __name__ == "__main__":
    main()
