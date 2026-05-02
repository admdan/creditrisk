import os
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
np.random.seed(42)
N = 5000

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -----------------------------
# Generate Features
# -----------------------------
income = np.random.normal(loc=75000, scale=25000, size=N)
income = np.clip(income, 20000, 200000)

credit_score = np.random.normal(loc=680, scale=70, size=N)
credit_score = np.clip(credit_score, 300, 850)

debt_to_income = np.random.uniform(0.0, 1.0, N)

employment_years = np.random.normal(loc=8, scale=5, size=N)
employment_years = np.clip(employment_years, 0, 30)

late_payments = np.random.poisson(lam=1.5, size=N)
late_payments = np.clip(late_payments, 0, 10)

loan_amount = np.random.normal(loc=15000, scale=8000, size=N)
loan_amount = np.clip(loan_amount, 1000, 50000)

age = np.random.normal(loc=40, scale=12, size=N)
age = np.clip(age, 18, 70)

# -----------------------------
# Risk Score Calculation (raw)
# -----------------------------
credit_norm = (850 - credit_score) / 550
loan_income_ratio = loan_amount / (income + 1)

risk_score_raw = (
    2.5 * credit_norm +
    3.0 * debt_to_income +
    1.5 * (late_payments / 10) +
    2.0 * loan_income_ratio -
    1.2 * (employment_years / 30) +
    0.5 * (debt_to_income * credit_norm)
)

# -----------------------------
# Intentional bug for Case 9:
# remove calibration and use the raw sigmoid output directly.
# -----------------------------
risk_probability = sigmoid(risk_score_raw)
risk_probability = np.clip(risk_probability, 0.01, 0.99)

# -----------------------------
# Generate Labels
# -----------------------------
labels = np.random.binomial(1, risk_probability)

# -----------------------------
# Create DataFrame
# -----------------------------
df = pd.DataFrame({
    "income": income,
    "credit_score": credit_score,
    "debt_to_income": debt_to_income,
    "employment_years": employment_years,
    "late_payments": late_payments,
    "loan_amount": loan_amount,
    "age": age,
    "high_risk": labels
})

# -----------------------------
# Check Class Distribution
# -----------------------------
counts = df["high_risk"].value_counts()
rates = df["high_risk"].value_counts(normalize=True)

print("Class Distribution (counts):")
print(counts.to_string())

print("\nClass Distribution (rates):")
print(rates.to_string())

# -----------------------------
# Save to CSV
# -----------------------------
out_path = DATA_DIR / "synthetic_credit_risk_case9.csv"
df.to_csv(out_path, index=False)
print(f"\nDataset saved as {out_path}")
