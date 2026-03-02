# train_mlp_v1.py
# ---------------------------------------------------------
# This script trains a Multi-Layer Perceptron (MLP)
# on the synthetic_credit_risk_v1.csv dataset.
#
# It performs:
# 1. Data loading
# 2. Train/test split
# 3. Feature scaling
# 4. MLP model creation
# 5. Training with Binary Cross-Entropy loss
# 6. Evaluation (accuracy, confusion matrix, report)
# 7. Loss curve visualization
# 8. Confusion matrix heatmap visualization
# 9. Probability distribution histogram visualization
# ---------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt

# Traditional ML utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Deep Learning (TensorFlow / Keras)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

def main():
    # ---------------------------------------------------------
    # 0. Ensure output folders exist
    # ---------------------------------------------------------
    os.makedirs("visuals", exist_ok=True)

    # ---------------------------------------------------------
    # 1. Load Dataset
    # ---------------------------------------------------------
    # Rename the path to match the dataset file
    data_path = "data/synthetic_credit_risk_v1.csv"

    # Check if dataset exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Dataset not found. Please run dataset_creation_v1.py first."
        )

    df = pd.read_csv(data_path)

    # ---------------------------------------------------------
    # 2. Define Features (X) and Target (y)
    # ---------------------------------------------------------
    target_col = "high_risk"

    X = df.drop(columns=[target_col])  # All input features
    y = df[target_col].astype(int)     # Binary labels (0 or 1)

    # ---------------------------------------------------------
    # 3. Train/Test Split
    # ---------------------------------------------------------
    # Stratify ensures class ratio (70/30) is preserved
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # ---------------------------------------------------------
    # 4. Feature Scaling
    # ---------------------------------------------------------
    # Neural networks are sensitive to feature scale.
    # StandardScaler transforms features to mean=0, std=1.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------
    # 5. Build the Multi-Layer Perceptron (MLP)
    # ---------------------------------------------------------
    # Architecture:
    # Input Layer -> 64 neurons -> 32 neurons -> 1 output neuron
    # ReLU activation introduces non-linearity
    # Sigmoid outputs probability between 0 and 1

    tf.random.set_seed(42)

    model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),  # input dimension
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.30),  # reduces overfitting
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.20),
        layers.Dense(1, activation="sigmoid")  # binary classification output
    ])

    # ---------------------------------------------------------
    # 6. Compile the Model
    # ---------------------------------------------------------
    # Binary Cross Entropy is used for binary classification
    # Adam optimizer performs adaptive gradient updates

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # ---------------------------------------------------------
    # 7. Train the Model
    # ---------------------------------------------------------
    # validation_split=0.2 reserves 20% of training data for validation
    # Epochs determine how many times the model sees the dataset

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # ---------------------------------------------------------
    # 8. Evaluate on Test Data
    # ---------------------------------------------------------
    # Predict probabilities
    y_prob = model.predict(X_test_scaled).ravel()

    # Convert probabilities to class labels using threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)

    # Compute evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    print("\n===== Test Results =====")
    print(f"Accuracy: {acc:.3f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # ---------------------------------------------------------
    # 9. Plot Training vs Validation Loss
    # ---------------------------------------------------------
    # This shows learning behavior and possible overfitting

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("MLP Training and Validation Loss (v1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("visuals/mlp_loss_curve_v1.png", dpi=200)
    plt.show()

    print("\nLoss curve saved as visuals/mlp_loss_curve_v1.png")

    # ---------------------------------------------------------
    # 10. Confusion Matrix Heatmap (no seaborn required)
    # ---------------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (MLP v1)") # Change for every version
    plt.colorbar()

    # Axis tick labels
    plt.xticks([0, 1], ["Pred Low (0)", "Pred High (1)"])
    plt.yticks([0, 1], ["Actual Low (0)", "Actual High (1)"])

    # Add counts inside the boxes
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("visuals/mlp_confusion_matrix_v1.png", dpi=200)
    plt.show()

    print("Saved: visuals/mlp_confusion_matrix_v1.png")

    # ---------------------------------------------------------
    # 11. Predicted Probability Distribution Histogram
    # ---------------------------------------------------------
    # Split predicted probabilities by true class
    low_probs = y_prob[y_test.values == 0]
    high_probs = y_prob[y_test.values == 1]

    plt.figure(figsize=(8, 5))
    plt.hist(low_probs, bins=30, alpha=0.7, label="Actual Low Risk (0)")
    plt.hist(high_probs, bins=30, alpha=0.7, label="Actual High Risk (1)")
    plt.xlabel("Predicted Probability of High Risk")
    plt.ylabel("Count")
    plt.title("Predicted Probability Distribution by Class (v1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("visuals/mlp_probability_distribution_v1.png", dpi=200)
    plt.show()

    print("Saved: visuals/mlp_probability_distribution_v1.png")

    # ---------------------------------------------------------
    # 12. MLP Architecture
    # ---------------------------------------------------------

    plot_model(
        model,
        to_file="visuals/mlp_architecture_v1.png",
        show_shapes=True,
        show_layer_names=True,
        dpi=200
    )

    print("Saved: visuals/mlp_architecture_v1.png")

# ---------------------------------------------------------
# Run script
# ---------------------------------------------------------
if __name__ == "__main__":
    main()