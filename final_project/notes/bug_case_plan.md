# AI-100 Final Project: Bug Case Plan

Student name: Adam Nasir  
Project: Deep Learning for Credit Risk Assessment  
Base system: Synthetic credit-risk dataset + MLP binary classifier

## Clean Base Code

The final project reuses the midterm credit-risk AI system. The clean working version is based on:

- `dataset_creation_v2.py`
- `train_mlp_v2.py`
- `data/synthetic_credit_risk_v2.csv`

The bug cases below are separate experiments. They should not all be combined into one final code version. Each case starts from the clean working code, introduces one intentional bug, observes the behavior, and then explains the fix.

## Case 1: Target Column Name Mismatch

Original idea: The training script should use `high_risk` as the target column because that is the label column created by the dataset script.

Buggy change: Change `target_col = "high_risk"` to a name that does not exist in the dataset, such as `target_col = "risk_label"`.

## Case 2: Feature Scaling Removed

Original idea: The model uses `StandardScaler` before training.

Buggy change: Train the neural network directly on raw feature values.

## Case 3: Target Column Included as Input

Original idea: `high_risk` should only be the target label.

Buggy change: Accidentally include `high_risk` in the input features.

## Case 4: Train/Test Split Not Stratified

Original idea: Use `stratify=y` during train/test split.

Buggy change: Remove `stratify=y`.

## Case 5: Prediction Threshold Too High

Original idea: Use a threshold of `0.5`.

Buggy change: Use a threshold like `0.9`.

## Case 6: Output Activation Changed Incorrectly

Original idea: Use sigmoid activation for binary classification.

Buggy change: Replace sigmoid with ReLU.

## Case 7: Wrong Loss Function Used

Original idea: Use `binary_crossentropy`.

Buggy change: Use `categorical_crossentropy`.

## Case 8: Learning Rate Too High

Original idea: Use Adam with learning rate `0.001`.

Buggy change: Increase the learning rate to `0.1`.

## Case 9: Dataset Labels Too Imbalanced

Original idea: The v2 dataset is calibrated to approximately 70% low risk and 30% high risk.

Buggy change: Remove or alter the calibration process.

## Case 10: Dropout Too High

Original idea: Use moderate dropout values like `0.30` and `0.20`.

Buggy change: Increase dropout to something too high, such as `0.80`.
