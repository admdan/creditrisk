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

Buggy change: Change the target column name in the training script from `high_risk` to a name that does not exist in the dataset, such as `risk_label`.

Case file: `final_project/cases/case1.py`

Original code snippet:

```python
target_col = "high_risk"

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)
```

New code snippet with bugs:

```python
target_col = "risk_label"

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)
```

Observed error/behavior:

- The preprocessing step fails before model training begins.
- Running `final_project/cases/case1.py` produces this error:

```text
Traceback (most recent call last):
  File "C:\Users\Adam Danial\PycharmProjects\creditrisk\final_project\cases\case1.py", line 239, in <module>
    main()
  File "C:\Users\Adam Danial\PycharmProjects\creditrisk\final_project\cases\case1.py", line 67, in main
    X = df.drop(columns=[target_col])  # All input features
  File "C:\Users\Adam Danial\PycharmProjects\creditrisk\.venv\Lib\site-packages\pandas\core\frame.py", line 6288, in drop
    return super().drop(
  File "C:\Users\Adam Danial\PycharmProjects\creditrisk\.venv\Lib\site-packages\pandas\core\generic.py", line 4644, in drop
    obj = obj._drop_axis(labels, axis, level=level, errors=errors)
  File "C:\Users\Adam Danial\PycharmProjects\creditrisk\.venv\Lib\site-packages\pandas\core\generic.py", line 4686, in _drop_axis
    new_axis = axis.drop(labels, errors=errors)
  File "C:\Users\Adam Danial\PycharmProjects\creditrisk\.venv\Lib\site-packages\pandas\core\indexes\base.py", line 7268, in drop
    raise KeyError(f"{labels[mask].tolist()} not found in axis")
KeyError: "['risk_label'] not found in axis"
```

Initial self-reflection:

- The error happens before training starts, so I think the issue is in the preprocessing step where the script separates the input features and target label. My first thought is that the dataset may not be matching the training script correctly, or the label information may not be read in the expected format.

GenAI's label:

- Bad

GenAI's guidance:

- The reflection is on the right track because it notices the error happens before training and connects it to the feature/target split. However, it is still a bit too general. The traceback specifically says KeyError: "['risk_label'] not found in axis", and the uploaded script originally uses target_col = "high_risk" before dropping that column from the dataframe.

- To reason more deeply, ask yourself:
  - What exact column names exist in the CSV after pd.read_csv(data_path)?
  When df.drop(columns=[target_col]) runs, what does pandas expect target_col to match exactly?
  If the script worked with one target column name but failed after renaming it, what does that suggest about the dataset versus the code?

New self-reflection:

- The issue is not a general preprocessing problem, but a mismatch between the dataset schema and the target column expected by the training script. The CSV contains the label column `high_risk`, but the modified code changes `target_col` to `risk_label`. When pandas runs `df.drop(columns=[target_col])`, it expects that column name to exist exactly in the dataframe. Since `risk_label` is not present, pandas raises a `KeyError` before the model can begin training.

## Case 2: Feature Scaling Removed

Original idea: The model uses `StandardScaler` before training.

Buggy change: Train the neural network directly on raw feature values.

Case file: `final_project/cases/case2.py`

Original code snippet:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)
```

New code snippet with bugs:

```python
X_train_raw = X_train
X_test_raw = X_test

history = model.fit(
    X_train_raw,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)
```

Observed error/behavior:

- The script still runs to completion, but model performance becomes noticeably worse than the clean baseline.
- Clean baseline (`train_mlp_v2.py`) test results:

```text
Accuracy: 0.756

Confusion Matrix:
[[797  76]
 [229 148]]
```

- Buggy Case 2 (`final_project/cases/case2.py`) test results:

```text
Accuracy: 0.698

Confusion Matrix:
[[873   0]
 [377   0]]
```

- The buggy model predicts every test sample as class `0` (low risk), which means it completely fails to identify any high-risk applicants.
- The classification report also shows this collapse in behavior:

```text
              precision    recall  f1-score   support

           0      0.698     1.000     0.822       873
           1      0.000     0.000     0.000       377

    accuracy                          0.698      1250
   macro avg      0.349     0.500     0.411      1250
weighted avg      0.488     0.698     0.574      1250
```

- The training logs also show unusually large early loss values without scaling, such as loss values above `2000` in the first epoch, which supports that the raw feature magnitudes are making optimization unstable.

Initial self-reflection:

- The model still runs, but its behavior becomes much worse than the clean baseline. The confusion matrix shows that it predicts only the low-risk class, and the loss curve also looks very different from the original version. This makes me think the problem is happening earlier in the pipeline, most likely in how the input data is being prepared before training. At this point, I suspect the model is no longer receiving the feature information in a form that allows it to learn balanced decision patterns.

GenAI's label:

- Bad

GenAI's guidance:

- "The reflection is partially insightful—you correctly notice that the model behavior degrades (predicting only one class, strange loss curve) and that the issue is tied to the input pipeline. That’s a good start. 
- However, your reasoning is still too high-level and misses a key causal link. You say the model is “no longer receiving the feature information in a form that allows it to learn,” but you don’t dig into what specifically changed in the data representation between the working and broken versions. The code clearly shows that the only modification was removing feature scaling before training. 
- To think more deeply, ask yourself:
  - What does StandardScaler actually do to the feature values (mean, variance)? 
  - How might neural networks behave if some input features have very large magnitudes compared to others? 
  - Looking at the loss values in the logs (very large at the beginning), what does that suggest about the scale of the inputs? 
  - Why would poor scaling lead to predicting only one class instead of learning a boundary? 
  - Try to connect:
    - “change in preprocessing” → “change in numerical properties of inputs” → “effect on optimization (loss/gradients)” → “observed model behavior.”"

New self-reflection:

- After thinking about the feedback more carefully, I think the main issue is that removing StandardScaler changed the numerical scale of the inputs before training. In the original version, the features were normalized so they were on more similar ranges, but in the buggy version the model used raw values like income, loan_amount, and credit_score directly. Because those values are on very different scales, the network seems to train less effectively, which matches the very large loss values at the beginning. I think this is why the model ended up predicting only the low-risk class instead of learning a better boundary between low-risk and high-risk applicants.

## Case 3: Target Column Included as Input

Original idea: `high_risk` should only be the target label.

Buggy change: Accidentally include `high_risk` in the input features.

Case file: `final_project/cases/case3.py`

Original code snippet:

```python
X = df.drop(columns=[target_col])  # All input features
y = df[target_col].astype(int)     # Binary labels (0 or 1)
```

New code snippet with bugs:

```python
X = df.copy()
y = df[target_col].astype(int)
```

Observed error/behavior:

- The script still runs, but the performance becomes unrealistically perfect.
- Buggy Case 3 (`final_project/cases/case3.py`) test results:

```text
Accuracy: 1.000

Confusion Matrix:
[[873   0]
 [  0 377]]
```

- The classification report shows precision, recall, and F1-score of `1.000` for both classes:

```text
              precision    recall  f1-score   support

           0      1.000     1.000     1.000       873
           1      1.000     1.000     1.000       377

    accuracy                          1.000      1250
   macro avg      1.000     1.000     1.000      1250
weighted avg      1.000     1.000     1.000      1250
```

- The loss curve drops extremely quickly toward zero, which makes the task look artificially easy.
- In the first run, the probability-distribution step even crashed because the predicted values were too concentrated for the original histogram settings, which further showed how extreme the leakage effect had become.

Initial self-reflection:

- The model still runs, but if the results become unusually strong compared with the clean baseline, I think the issue may be related to how the input data is being prepared. My first thought is that something in the feature setup may be giving the model information that makes the task easier than it should be.

GenAI's label:

- Bad

GenAI's guidance:

- "You’re noticing the symptom (suspiciously perfect performance) and correctly suspect something is wrong with the feature setup, but your reasoning stops too early and doesn’t pinpoint why the task suddenly became trivial. 
- Right now, your explanation is still vague (“something in the feature setup is making it easier”). To go deeper, you need to question exactly what changed in the data pipeline and how that affects the relationship between inputs and labels. 
- Ask yourself:
  - When you changed X = df.drop(columns=[target_col]) to X = df.copy(), what columns are now included in X that weren’t before? 
  - Is there any column in X that might be directly related to y? 
  - If a model achieves perfect accuracy almost immediately (as seen in the logs and loss going near zero very fast ), what does that usually imply about the difficulty of the task? 
  - Could the model be “seeing” information during training that it normally shouldn’t have access to? 
  - Try to connect:
    - “change in feature definition” → “information available to model” → “why predictions become trivially perfect.”"

New self-reflection:

- After looking at the code change more carefully, I think the main issue is that the target label was accidentally left inside the input features. In the original version, X was supposed to exclude high_risk, but after changing it to X = df.copy(), the model received every column in the dataframe, including the label itself. That means the network was being trained with direct access to the answer it was supposed to predict. This explains why the model reached perfect accuracy so quickly and why the loss dropped close to zero almost immediately, because the task was no longer a real prediction problem.

## Case 4: Train/Test Split Not Stratified

Original idea: Use `stratify=y` during train/test split.

Buggy change: Remove `stratify=y`.

Case file: `final_project/cases/case4.py`

Original code snippet:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

New code snippet with bugs:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=169
)
```

Observed error/behavior:

- The script still runs successfully, but the train/test class balance becomes less consistent than the clean baseline because the split is no longer stratified.
- Buggy Case 4 (`final_project/cases/case4.py`) label distribution:

```text
Train label distribution:
0    0.707733
1    0.292267

Test label distribution:
0    0.6696
1    0.3304
```

- Compared with the baseline target rate of about `30.18%` high risk, the test set becomes noticeably more imbalanced toward high-risk examples (`33.04%`).
- Clean baseline (`train_mlp_v2.py`) test results:

```text
Accuracy: 0.756

Confusion Matrix:
[[797  76]
 [229 148]]
```

- Buggy Case 4 (`final_project/cases/case4.py`) test results:

```text
Accuracy: 0.717

Confusion Matrix:
[[761  76]
 [278 135]]
```

- The classification report shows weaker performance than the clean baseline, especially for the high-risk class:

```text
              precision    recall  f1-score   support

           0      0.732     0.909     0.811       837
           1      0.640     0.327     0.433       413

    accuracy                          0.717      1250
   macro avg      0.686     0.618     0.622      1250
weighted avg      0.702     0.717     0.686      1250
```

- The visual comparisons also show that the buggy split leads to a different probability distribution and weaker confusion-matrix performance than the clean baseline, even though the model architecture and preprocessing steps are otherwise unchanged.

Initial self-reflection:

- The script still runs, but if the split is no longer balanced between classes, the evaluation results may become less reliable than the clean baseline. My first thought is that the problem may be related to how the data is divided before training, which could affect how well the test set represents the full dataset.

GenAI's label:

- Bad

GenAI's guidance:

- You’re pointing in a reasonable direction (data splitting affects results), but your explanation is still too surface-level and doesn’t actually engage with what the outputs are telling you.
- Right now, you’re saying “the split may not be balanced” and “results may be less reliable,” but:
  - The printed distributions show the train and test sets are still fairly similar (around 70/30).
  - The model performance is also still fairly close to the baseline, not dramatically unstable.
- To think more deeply, ask yourself:
  - What exactly did removing `stratify=y` change compared to just changing the random seed?
  - If the class proportions are still close, what other effect might a different random split have?
  - Could specific samples, not just proportions, influence the learned boundary and the evaluation metrics?
  - Why might two splits with similar distributions still lead to weaker recall on the high-risk class?
- Try to connect:
  - “random split change” -> “which specific data points end up in train vs test” -> “impact on learned decision boundary and evaluation metrics.”

New self-reflection:

- After thinking about the split more carefully, I do not think the issue is only that the overall class proportions changed. The train and test sets are still fairly close to the original 70/30 balance, so that alone does not fully explain the difference in results. A better explanation is that removing stratify=y changed which specific samples ended up in the training set and the test set. Even if the proportions stay similar, the model may still see a slightly different mix of easier or harder examples during training and evaluation. I think this is why the overall accuracy only changed moderately, but the confusion matrix and recall for the high-risk class still became worse. This suggests that the model is sensitive not only to class balance, but also to the exact sampling of examples in the split.

## Case 5: Prediction Threshold Too High

Original idea: Use a threshold of `0.5`.

Buggy change: Use a threshold like `0.9`.

Case file: `final_project/cases/case5.py`

Original code snippet:

```python
y_prob = model.predict(X_test_scaled).ravel()
y_pred = (y_prob >= 0.5).astype(int)
```

New code snippet with bugs:

```python
y_prob = model.predict(X_test_scaled).ravel()
y_pred = (y_prob >= 0.9).astype(int)
```

Observed error/behavior:

- The script still runs successfully, but the prediction threshold is so high that the model almost never labels any applicant as high risk.
- Clean baseline (`train_mlp_v2.py`) test results:

```text
Accuracy: 0.756

Confusion Matrix:
[[797  76]
 [229 148]]
```

- Buggy Case 5 (`final_project/cases/case5.py`) test results:

```text
Accuracy: 0.701

Confusion Matrix:
[[873   0]
 [374   3]]
```

- The classification report shows that recall for the high-risk class collapses almost completely:

```text
              precision    recall  f1-score   support

           0      0.700     1.000     0.824       873
           1      1.000     0.008     0.016       377

    accuracy                          0.701      1250
   macro avg      0.850     0.504     0.420      1250
weighted avg      0.791     0.701     0.580      1250
```

- The model still produces a spread of predicted probabilities, but with the threshold raised to `0.9`, almost none of those probabilities are high enough to be converted into class `1`.
- The loss curve remains similar to the clean baseline, which suggests the model training itself is still normal and that the main problem happens at the decision step after probabilities are generated.

Initial self-reflection:

- The model still runs, but the final predictions become much more one-sided than in the clean baseline. Since the training loss and validation loss still look fairly normal, I do not think the main problem is in how the model learned. Instead, I think the issue is happening at the stage where the probability outputs are turned into class labels. My current guess is that the decision rule has become too strict, which makes the model classify far fewer applicants as high risk even when some of them still have moderately high predicted probabilities.

GenAI's label:

- Bad

GenAI's guidance:

- You’re moving in the right direction by focusing on the prediction stage rather than training, but the reasoning is still too vague and doesn’t fully engage with the evidence in the outputs.
- Right now, you say the decision rule is “too strict,” but you do not explain what specifically changed in that rule or how it mathematically affects predictions.
- To think more deeply, ask yourself:
  - What exact condition determines whether a prediction becomes class `1`?
  - How does increasing the threshold from `0.5` to `0.9` change the range of probabilities that qualify as “high risk”?
  - Looking at the probability distributions, how many predictions actually exceed `0.9`?
  - Why does that lead to very low recall for the high-risk class but possibly high precision?
- Try to connect:
  - “threshold change” -> “which probabilities qualify as positive” -> “number of predicted positives” -> “impact on confusion matrix (false negatives vs true positives).”

New self-reflection:

- After thinking about the outputs more carefully, I think the main issue is that increasing the threshold from 0.5 to 0.9 changed the condition for predicting class 1 too drastically. In the original version, any predicted probability of 0.5 or higher was labeled as high risk, but in the buggy version only probabilities of 0.9 or higher counted as class 1. Since very few predictions actually reached that level, almost all applicants were labeled as low risk. This explains why the confusion matrix shows only 3 true positives and 374 false negatives for the high-risk class. The model still learned in a similar way, but the stricter decision rule filtered out almost every positive prediction, which caused recall for the high-risk class to collapse.

## Case 6: Output Activation Changed Incorrectly

Original idea: Use sigmoid activation for binary classification.

Buggy change: Replace sigmoid with ReLU.

Case file: `final_project/cases/case6.py`

Original code snippet:

```python
layers.Dense(1, activation="sigmoid")  # binary classification output
```

New code snippet with bugs:

```python
layers.Dense(1, activation="relu")  # binary classification output
```

Observed error/behavior:

- The script still runs successfully, but changing the output activation from sigmoid to ReLU makes the binary classification behavior worse.
- Clean baseline (`train_mlp_v2.py`) test results:

```text
Accuracy: 0.756

Confusion Matrix:
[[797  76]
 [229 148]]
```

- Buggy Case 6 (`final_project/cases/case6.py`) test results:

```text
Accuracy: 0.741

Confusion Matrix:
[[836  37]
 [287  90]]
```

- The classification report shows noticeably weaker performance on the high-risk class:

```text
              precision    recall  f1-score   support

           0      0.744     0.958     0.838       873
           1      0.709     0.239     0.357       377

    accuracy                          0.741      1250
   macro avg      0.727     0.598     0.597      1250
weighted avg      0.734     0.741     0.693      1250
```

- The output summary also shows that the final layer values are no longer behaving like the usual sigmoid-based probability outputs:

```text
count    1250.000000
mean        0.264590
std         0.158879
min         0.000000
25%         0.139564
50%         0.229694
75%         0.357395
max         0.808924
```

- The loss curve also starts much higher than the clean baseline, which suggests that the output layer is no longer well matched to the binary-classification setup.

Initial self-reflection:

- The model still runs, but if the output activation is changed from sigmoid to ReLU, the predictions may no longer behave like proper probabilities. My first thought is that the issue may not be in the hidden layers, but in the way the final output is being produced for a binary classification task.

GenAI's label:

- Bad

GenAI's guidance:

- It looks like you've noticed the model is still running, but you're a bit stuck on why the results changed.
- Think about the role of the final activation compared with hidden-layer activations.
- `binary_crossentropy` has a very specific contract with the output layer: it expects a probability-like value.
- Ask yourself:
  - What range of numbers should the final layer produce for a binary classifier?
  - How do ReLU outputs differ from sigmoid outputs in the final layer?
  - If the prediction threshold is still `0.5`, does that cutoff mean the same thing when the output activation is no longer sigmoid?
  - Why did the training loss start much higher than normal in the first epoch?

New self-reflection:

- After looking at the results more carefully, I think the main issue is that ReLU is not really appropriate for the final layer of this binary classification model. In the working version, sigmoid keeps the output between 0 and 1, which makes sense because the model is supposed to produce something like a probability before applying the 0.5 threshold. After changing the final activation to ReLU, the output no longer behaves in that same probability-based way. I think this is why the training loss started much higher than normal and why the model ended up missing more high-risk cases.

## Case 7: Wrong Loss Function Used

Original idea: Use `binary_crossentropy`.

Buggy change: Use `categorical_crossentropy`.

Case file: `final_project/cases/case7.py`

Original code snippet:

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

New code snippet with bugs:

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

Observed error/behavior:

- The script still runs, but the training setup becomes inconsistent after replacing `binary_crossentropy` with `categorical_crossentropy`.
- During training, Keras prints this warning:

```text
In loss categorical_crossentropy, expected y_pred.shape to be
(batch_size, num_classes) with num_classes > 1.
Received: y_pred.shape=(None, 1). Consider using
'binary_crossentropy' if you only have 2 classes.
```

- Both training loss and validation loss become `0.0000e+00` almost immediately and stay there, which suggests that the optimizer is no longer receiving a meaningful learning signal.
- Clean baseline (`train_mlp_v2.py`) test results:

```text
Accuracy: 0.756

Confusion Matrix:
[[797  76]
 [229 148]]
```

- Buggy Case 7 (`final_project/cases/case7.py`) test results:

```text
Accuracy: 0.698

Confusion Matrix:
[[873   0]
 [377   0]]
```

- The classification report shows that the model completely fails to identify the high-risk class:

```text
              precision    recall  f1-score   support

           0      0.698     1.000     0.822       873
           1      0.000     0.000     0.000       377

    accuracy                          0.698      1250
   macro avg      0.349     0.500     0.411      1250
weighted avg      0.488     0.698     0.574      1250
```

- The probability distribution also collapses unnaturally into a flat spike near `0`, which matches the confusion matrix and shows that the model is not learning a useful binary decision boundary.

Initial self-reflection:

- The model still has the same architecture and data pipeline, but changing the loss function may break the way training interprets the labels and outputs. My first thought is that the issue may come from a mismatch between the kind of predictions the model makes and what the loss function expects during optimization.

GenAI's label:

- Bad

GenAI's guidance:

- It looks like you're seeing a lot of zeros in your results and you're not sure why the model has completely stopped learning.
- Look closely at three clues:
  - the loss is exactly `0.0000e+00` from the first epoch
  - Keras warns that `categorical_crossentropy` expects `(batch_size, num_classes)` but the model outputs `(None, 1)`
  - the confusion matrix and probability distribution show the model predicting the same thing for every sample
- Ask yourself why a single-output sigmoid classifier should use `binary_crossentropy` instead of a categorical loss that expects multiple class columns.

New self-reflection:

- After looking at the warning and the zero-loss behavior more carefully, I think the main issue is that categorical_crossentropy does not fit a model with only one sigmoid output for binary classification. In the working version, binary_crossentropy matches the single-output setup correctly. After changing the loss function, Keras warns that categorical_crossentropy expects predictions shaped like multiple class probabilities, but the model only produces one output value. Because of that mismatch, the loss becomes zero immediately and the optimizer does not meaningfully update the weights. This explains why the model stays stuck predicting only the low-risk class instead of learning to separate the two classes.

## Case 8: Learning Rate Too High

Original idea: Use Adam with learning rate `0.001`.

Buggy change: Increase the learning rate to `0.1`.

Case file: `final_project/cases/case8.py`

Original code snippet:

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

New code snippet with bugs:

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

Observed error/behavior:

- The script still runs successfully, but increasing the learning rate to `0.1` makes the training process much less effective.
- Clean baseline (`train_mlp_v2.py`) test results:

```text
Accuracy: 0.756

Confusion Matrix:
[[797  76]
 [229 148]]
```

- Buggy Case 8 (`final_project/cases/case8.py`) test results:

```text
Accuracy: 0.698

Confusion Matrix:
[[873   0]
 [377   0]]
```

- The classification report shows complete failure on the high-risk class:

```text
              precision    recall  f1-score   support

           0      0.698     1.000     0.822       873
           1      0.000     0.000     0.000       377

    accuracy                          0.698      1250
   macro avg      0.349     0.500     0.411      1250
weighted avg      0.488     0.698     0.574      1250
```

- The loss curve bounces early and then flattens at a worse level than the clean baseline, which suggests the optimizer is taking steps that are too large to settle into a useful minimum.
- The probability distribution also collapses into a narrow vertical band, showing that the model ends up outputting almost the same value for every input instead of learning a useful decision boundary.

Initial self-reflection:

- The model still runs, but the training behavior becomes worse after increasing the learning rate. My first thought is that the optimizer may now be updating the weights too aggressively, which could make the model less stable and prevent it from settling into a good solution. The results suggest that the problem is not in the dataset or architecture, but in how the training process is moving through the loss landscape.

GenAI's label:

- Bad

GenAI's guidance:

- While you correctly identified the learning rate as the variable being changed, your reflection is superficial.
- The evidence shows a specific failure pattern:
  - the loss “bounces” early, which suggests overshooting
  - the loss and accuracy then flatline, which suggests the model gets pushed into a bad region and stops learning
  - the confusion matrix shows a “lazy” majority-class model that predicts only low risk
  - the probability distribution collapses into one narrow band
- Ask yourself how a single massive update early in training could make the model output the same value for every input.

New self-reflection:

- After looking at the results more carefully, I think the main issue is not just that the learning rate is “less stable,” but that it is so large that the optimizer makes updates that overshoot good solutions early in training. The loss curve bounces noticeably at the beginning, which suggests the updates are too large to settle into a useful minimum. After that, the training becomes almost completely flat, which makes me think the model got pushed into a bad region where it stopped learning anything meaningful. This explains why the final predictions collapsed into a single narrow value and why the model ended up predicting only the low-risk class for every input. So the problem is not simply slower or noisier learning, but that the large learning rate prevented the optimizer from finding a useful decision boundary at all.

## Case 9: Dataset Labels Too Imbalanced

Original idea: The v2 dataset is calibrated to approximately 70% low risk and 30% high risk.

Buggy change: Remove or alter the calibration process.

Case files:
- `final_project/cases/dataset_creation_case9.py`
- `final_project/cases/case9.py`

Original code snippet:

```python
bias = calibrate_bias(risk_score_raw, target_rate=TARGET_HIGH_RISK_RATE)
risk_probability = sigmoid(risk_score_raw + bias)
```

New code snippet with bugs:

```python
risk_probability = sigmoid(risk_score_raw)
```

Observed error/behavior:

- The system still runs successfully, but removing the calibration step makes the generated dataset extremely imbalanced.
- Dataset generation results:

```text
Class Distribution (counts):
1    4535
0     465

Class Distribution (rates):
1    0.907
0    0.093
```

- The training script confirms the label distribution is heavily skewed:

```text
Dataset label distribution:
0    0.093
1    0.907
```

- Buggy Case 9 (`final_project/cases/case9.py`) test results:

```text
Accuracy: 0.906

Confusion Matrix:
[[   0  116]
 [   1 1133]]
```

- The classification report shows that the model almost always predicts the majority class:

```text
              precision    recall  f1-score   support

           0      0.000     0.000     0.000       116
           1      0.907     0.999     0.951      1134

    accuracy                          0.906      1250
   macro avg      0.454     0.500     0.475      1250
weighted avg      0.823     0.906     0.863      1250
```

- The model looks strong on overall accuracy, but that result is misleading because it completely fails to identify low-risk applicants and instead learns to predict high risk for almost everyone.

Initial self-reflection:

- The system may still run, but if the label distribution becomes much more imbalanced than the original version, the model could appear to perform well while actually learning a biased prediction pattern. My first thought is that the problem may come from the dataset design rather than from the model architecture itself.

GenAI's label:

- Bad

GenAI's guidance:

- While you correctly identified that the calibration step is missing and that the model is favoring the majority class, your reasoning remains superficial.
- Look more closely at how removing `calibrate_bias` changes the raw risk scores before the sigmoid.
- Ask yourself:
  - If the raw scores are mostly positive, what does the sigmoid do to them without the bias shift?
  - Why does a dataset with 1,134 high-risk samples and only 116 low-risk samples make “predict high risk for everyone” an easy low-loss strategy?
  - Why is 90.6% accuracy not actually a sign of success if recall for class `0` is `0.000`?
  - Why can the loss curve still look low and stable even when the model is essentially broken by label imbalance?

New self-reflection:

- After looking at the dataset creation logic more carefully, I think the main issue is that removing the calibration step changed the raw risk scores before they were passed into the sigmoid function. In the original version, calibrate_bias shifts those scores so the final label distribution stays closer to the intended balance. After removing that shift, many of the raw scores stay positive, so the sigmoid pushes a very large share of the probabilities toward 1.0. That is why the generated dataset ends up with about 90.7% high-risk labels. Once the training data becomes that imbalanced, the model can get high accuracy by predicting high risk for almost everyone, which is exactly what happened in the confusion matrix. So the problem is not just general class imbalance, but that removing the bias correction changed the dataset mathematically in a way that made the majority class dominate the whole learning process.

## Case 10: Dropout Too High

Original idea: Use moderate dropout values like `0.30` and `0.20`.

Buggy change: Increase dropout to something too high, such as `0.80`.

Case file: `final_project/cases/case10.py`

Original code snippet:

```python
layers.Dense(64, activation="relu"),
layers.Dropout(0.30),  # reduces overfitting
layers.Dense(32, activation="relu"),
layers.Dropout(0.20),
layers.Dense(1, activation="sigmoid")
```

New code snippet with bugs:

```python
layers.Dense(64, activation="relu"),
layers.Dropout(0.80),
layers.Dense(32, activation="relu"),
layers.Dropout(0.80),
layers.Dense(1, activation="sigmoid")
```

Observed error/behavior:

- The script still runs successfully, but using dropout rates of `0.80` and `0.80` makes the network underfit the dataset.
- Clean baseline (`train_mlp_v2.py`) test results:

```text
Accuracy: 0.756

Confusion Matrix:
[[797  76]
 [229 148]]
```

- Buggy Case 10 (`final_project/cases/case10.py`) test results:

```text
Accuracy: 0.742

Confusion Matrix:
[[848  25]
 [298  79]]
```

- The classification report shows that the high-risk class is detected much less effectively:

```text
              precision    recall  f1-score   support

           0      0.740     0.971     0.840       873
           1      0.760     0.210     0.328       377

    accuracy                          0.742      1250
   macro avg      0.750     0.590     0.584      1250
weighted avg      0.746     0.742     0.686      1250
```

- The loss curve stays higher than the clean baseline, and the probability distributions become more compressed and less separated, which suggests the model is not learning strong enough patterns from the data.

Initial self-reflection:

- The model still runs, but if the dropout is set too high, the network may lose too much information during training and fail to learn the patterns in the data properly. My first thought is that the problem may not be in the dataset itself, but in the regularization setting being so strong that the model becomes too weak to fit the task.

GenAI's label:

- Bad

GenAI's guidance:

- Your reflection is on the right track regarding “information loss,” but it stays too high-level.
- To reason more deeply, connect the `0.80` dropout rate to the specific failures in the charts:
  - why does the model still do reasonably well on the majority low-risk class but struggle badly on the high-risk class?
  - why are the predicted probabilities compressed into a narrower range instead of stretching toward strong high-risk confidence?
  - why is the training loss higher than the validation loss for most of the run?
  - if 80% of the neurons are turned off at each step, is there enough active capacity left to learn the harder class patterns?

New self-reflection:

- After looking at the results more carefully, I think the problem is not just general information loss, but that the dropout rate is so high that too much of the network is being turned off during training. With dropout set to 0.80, only a small fraction of the neurons remain active at each step, so the model becomes too weak to learn the more complex patterns that separate high-risk applicants from low-risk ones. That helps explain why the predicted probabilities stay compressed in a narrower range instead of reaching stronger high-risk confidence values. It also explains why the model still does fairly well on the majority low-risk class but misses many high-risk cases, because it falls back on simpler decision patterns that are easier to learn but less useful for the harder class.
