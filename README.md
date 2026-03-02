# 🧠 Credit Risk Classification with Neural Networks

A structured deep learning project exploring how a Multi-Layer
Perceptron (MLP) behaves under different dataset distributions using
synthetic credit applicant data.

This project demonstrates an important idea:

**A model can look “good” on accuracy while still being wrong in a
meaningful way.** 

## 📌 Project Overview

Banks and lenders need to decide whether an applicant is low risk or
high risk. This project frames that as a binary classification task
using a neural network on tabular financial features.

Because real credit datasets are private, the dataset here is
synthetically generated with:
- Realistic feature ranges (income, credit score, loan amount, etc.)
- Nonlinear risk interactions
- Probabilistic labeling using a sigmoid-based scoring function

## 🔴 Version 1 (Imbalanced Dataset)

**What happened:**
Even though the terminal showed high accuracy (0.906),
the model predicted almost everyone as high risk.

**Why this matters:**
Accuracy was misleading because of severe class
imbalance and biased probability mapping.

**Key Results:** 
- Accuracy: 0.906
- Model predicted nearly all applicants as high risk
- Very poor discrimination for low-risk applicants

This version shows how accuracy alone can hide structural flaws.

## 🟢 Version 2 (Calibrated Dataset)

**What changed:**
The generation logic remained the same, but a probability
calibration step was added to achieve:

> ~70% low risk / ~30% high risk

**Result:**

Accuracy decreased to ~0.750, but predictions became far more meaningful
and balanced.

This version reflects true classification capability rather than
dominance of a majority class.

## 🏗️ Model Architecture (MLP)

The neural network consists of:
- Dense layer (64 neurons, ReLU activation)
- Dropout layer (30%)
- Dense layer (32 neurons, ReLU activation)
- Dropout layer (20%)
- Output layer (1 neuron, Sigmoid activation)

Training Configuration:
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Binary Cross-Entropy
- Epochs: 50
- Batch size: 32
- Validation split: 20%

## 📊 Visual Outputs

Each training script generates and saves:

1.  📈 Training vs Validation Loss Curve
2.  🔲 Confusion Matrix Heatmap
3.  📊 Predicted Probability Distribution Histogram
4.  🧩 MLP Architecture Diagram

All saved inside the **visuals** folder.

## 🚀 How to Run

1)  Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

2)  Generate dataset:
```bash
python dataset_creation_v1.py or python dataset_creation_v2.py
```
3)  Train model:
``` bash
python train_mlp_v1.py or python train_mlp_v2.py
```

## 🧪 Python Libraries Used

- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib

## 🔍 Key Lessons Learned

- Accuracy alone can be misleading in imbalanced datasets.
- Confusion matrices and F1-scores reveal hidden bias.
- Neural networks amplify patterns present in the data.
- Dataset design strongly influences model behavior.
- Improving data quality can matter more than increasing model complexity.

## Prepared By
- Adam Nasir, IST Major, Class of 2026

## 🏁 Final Takeaway

Sometimes the biggest improvement in a neural network model is not
adding more layers.

It is fixing the dataset so the model is forced to learn something real.
