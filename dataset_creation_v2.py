import os
import numpy as np
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
np.random.seed(42)
N = 5000
TARGET_HIGH_RISK_RATE = 0.30  # want ~30% of labels to be 1 (high risk)

os.makedirs("data", exist_ok=True)

# -----------------------------
# Helper Functions
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calibrate_bias(risk_score, target_rate=0.30, max_iter=60):
    """
    Finds a bias term 'b' such that mean(sigmoid(risk_score + b)) ~= target_rate.
    Uses binary search because sigmoid is monotonic with respect to b.
    """
    lo, hi = -10.0, 10.0  # wide enough range for shifting probabilities

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        probs = sigmoid(risk_score + mid)
        rate = probs.mean()

        if rate > target_rate:
            hi = mid
        else:
            lo = mid

    return (lo + hi) / 2

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

# Raw risk score (same logic as v1)
risk_score_raw = (
    2.5 * credit_norm +
    3.0 * debt_to_income +
    1.5 * (late_payments / 10) +
    2.0 * loan_income_ratio -
    1.2 * (employment_years / 30) +
    0.5 * (debt_to_income * credit_norm)  # nonlinear interaction
)

# -----------------------------
# Calibrate probabilities to hit ~30% high risk
# -----------------------------
bias = calibrate_bias(risk_score_raw, target_rate=TARGET_HIGH_RISK_RATE)
risk_probability = sigmoid(risk_score_raw + bias)

# Optional: clip extreme probs slightly (helps avoid all-0 or all-1 behavior)
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

print("Calibrated bias used:", round(bias, 4))
print("\nClass Distribution (counts):")
print(counts.to_string())

print("\nClass Distribution (rates):")
print(rates.to_string())

# -----------------------------
# Save to CSV
# -----------------------------
out_path = "data/synthetic_credit_risk_v2.csv"
df.to_csv(out_path, index=False)
print(f"\nDataset saved as {out_path}")