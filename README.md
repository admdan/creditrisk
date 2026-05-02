# Credit Risk Classification with Neural Networks

This repository contains both the **midterm project** and the **final project**
for an AI/deep-learning course.

- The **midterm project** focuses on building a working synthetic credit-risk
  classification pipeline with a Multi-Layer Perceptron (MLP).
- The **final project** builds on that baseline by creating a set of intentional
  bug cases and analyzing how different failures affect model behavior,
  evaluation, and reflection quality.

The main idea behind the project is simple:

**A model can appear successful on surface metrics like accuracy while still
failing in important ways.**

## Project Structure

### Midterm base system

The clean working versions are:

- `dataset_creation_v1.py`
- `dataset_creation_v2.py`
- `train_mlp_v1.py`
- `train_mlp_v2.py`

Datasets are stored in:

- `data/`

Baseline visual outputs are stored in:

- `visuals/`

### Final project extension

The final project lives in:

- `final_project/`

It includes:

- bug-case training scripts in `final_project/cases/`
- case visuals in `final_project/visuals/`
- case documentation in `final_project/notes/bug_case_plan.md`

## Midterm Overview

The system frames credit risk as a binary classification problem:

- `0` = low risk
- `1` = high risk

Because real lending data is private, the project uses synthetic applicant
profiles with realistic feature ranges such as:

- income
- credit score
- loan amount
- debt-to-income ratio
- employment length

The dataset is generated with nonlinear interactions and a sigmoid-based risk
scoring process.

### Version 1

Version 1 produces a much more imbalanced dataset. It can show high accuracy
while still behaving poorly in a meaningful way because the model tends to
favor the majority class.

### Version 2

Version 2 adds a calibration step during dataset creation so that the class
distribution is closer to:

- about 70% low risk
- about 30% high risk

This makes the evaluation much more meaningful and serves as the clean baseline
for the final project.

## Final Project Overview

The final project starts from the clean `v2` system and creates a series of
intentional bug cases. Each case changes one part of the pipeline and observes
what happens.

Examples include:

- target-column mismatch
- removing feature scaling
- target leakage
- non-stratified train/test split
- threshold misconfiguration
- wrong output activation
- wrong loss function
- learning rate too high
- dataset calibration removed
- dropout too high

The purpose of these cases is not just to show crashes. Several bugs still let
the model run, but they silently damage performance, distort the outputs, or
make evaluation misleading.

## Model Architecture

The clean MLP baseline uses:

- Dense layer with 64 neurons and ReLU activation
- Dropout layer with rate `0.30`
- Dense layer with 32 neurons and ReLU activation
- Dropout layer with rate `0.20`
- Output layer with 1 neuron and sigmoid activation

Training configuration:

- Optimizer: Adam
- Learning rate: `0.001`
- Loss function: `binary_crossentropy`
- Epochs: `50`
- Batch size: `32`
- Validation split: `0.2`

## Python Version

This project is currently configured and tested with:

- **Python 3.13.5**

If you are reproducing the environment, it is safest to use the same Python
version.

## Requirements

Install the pinned dependencies from:

- `requirements.txt`

Current requirements:

```text
numpy==2.4.2
pandas==3.0.1
matplotlib==3.10.8
scikit-learn==1.8.0
tensorflow==2.20.0
pydot==4.0.1
graphviz==0.21
```

Install them with:

```bash
pip install -r requirements.txt
```

## How to Run

### Midterm baseline

Generate a dataset:

```bash
python dataset_creation_v1.py
python dataset_creation_v2.py
```

Train the model:

```bash
python train_mlp_v1.py
python train_mlp_v2.py
```

### Final project bug cases

Run any case from the repo root:

```bash
python final_project/cases/case1.py
python final_project/cases/case2.py
...
python final_project/cases/case10.py
```

Case 9 also includes a separate dataset-generation step:

```bash
python final_project/cases/dataset_creation_case9.py
python final_project/cases/case9.py
```

## Visual Outputs

The training scripts generate:

- training vs validation loss curve
- confusion matrix heatmap
- predicted probability distribution histogram
- MLP architecture diagram

Midterm visuals are saved in:

- `visuals/`

Final-project case visuals are saved in:

- `final_project/visuals/`

## Key Lessons

- Accuracy alone can be misleading.
- Dataset design strongly affects model behavior.
- Preprocessing, loss functions, thresholds, and optimization settings all
  matter.
- An AI system should be treated as a full pipeline, not just a model.
- Silent failures can be as dangerous as runtime errors.

## Author

- Adam Nasir, IST Major, Class of 2026
