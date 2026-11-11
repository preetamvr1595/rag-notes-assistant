# 01 — Scikit-learn Basics (Preetham's Scikit-learn Mastery Series)

Part 1 of 5:
1. **Basics** (this notebook)
2. Supervised Learning — Classification
3. Supervised Learning — Regression
4. Unsupervised Learning & Preprocessing
5. Model Selection & Real-World Project

Scikit-learn is Python's core machine learning library — consistent API across every algorithm: `fit()`, `predict()`, `transform()`.

## 1. What is Scikit-learn?

- Built on top of NumPy, SciPy, and matplotlib.
- Every model follows the same pattern: **Estimator** (`.fit()`), **Predictor** (`.predict()`), **Transformer** (`.transform()`).
- Comes with built-in toy datasets (`iris`, `wine`, `breast_cancer`, `digits`) — no CSV hunting needed for practice.
- Covers classification, regression, clustering, dimensionality reduction, and model selection.

```python
!pip install scikit-learn --quiet
```

Output:
```text
[1;31merror[0m: [1mexternally-managed-environment[0m

[31m×[0m This environment is externally managed
[31m╰─>[0m To install Python packages system-wide, try apt install
[31m   [0m python3-xyz, where xyz is the package you are trying to
[31m   [0m install.
[31m   [0m 
[31m   [0m If you wish to install a non-Debian-packaged Python package,
[31m   [0m create a virtual environment using python3 -m venv path/to/venv.
[31m   [0m Then use path/to/venv/bin/python and path/to …
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

import sklearn
print('scikit-learn version:', sklearn.__version__)
```

Output:
```text
scikit-learn version: 1.8.0
```

## 2. Loading a Built-in Dataset

```python
iris = datasets.load_iris()
print('Data shape:', iris.data.shape)
print('Feature names:', iris.feature_names)
print('Target names:', iris.target_names)
print('\nFirst 5 rows:\n', iris.data[:5])
```

Output:
```text
Data shape: (150, 4)
Feature names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Target names: ['setosa' 'versicolor' 'virginica']

First 5 rows:
 [[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
```

## 3. Converting to a DataFrame for Easier Exploration

```python
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
iris_df.head()
```

Output:
```text
sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \
0                5.1               3.5                1.4               0.2   
1                4.9               3.0                1.4               0.2   
2                4.7               3.2                1.3               0.2   
3                4.6               3.1                1.5               0.2   
4                5.0               3.6                1.4               0.2   

  species  
0  setosa  
1  s …
```

## 4. The Estimator API — `fit()` and `predict()`

```python
from sklearn.linear_model import LogisticRegression

X = iris.data
y = iris.target

model = LogisticRegression(max_iter=200)
model.fit(X, y)
predictions = model.predict(X[:5])
print('Predictions:', predictions)
print('Actual     :', y[:5])
```

Output:
```text
Predictions: [0 0 0 0 0]
Actual     : [0 0 0 0 0]
```

## 5. Train/Test Split — The First Rule of ML

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print('Train shape:', X_train.shape, '| Test shape:', X_test.shape)
```

Output:
```text
Train shape: (120, 4) | Test shape: (30, 4)
```

## 6. Feature Scaling — `StandardScaler`

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # transform only, never re-fit on test data!

print('Before scaling — mean:', X_train[:, 0].mean().round(2), '| std:', X_train[:, 0].std().round(2))
print('After scaling  — mean:', X_train_scaled[:, 0].mean().round(2), '| std:', X_train_scaled[:, 0].std().round(2))
```

Output:
```text
Before scaling — mean: 5.84 | std: 0.84
After scaling  — mean: -0.0 | std: 1.0
```

## 7. `MinMaxScaler` — Scaling to a [0, 1] Range

```python
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
X_train_minmax = minmax.fit_transform(X_train)
print('Min:', X_train_minmax.min(axis=0).round(2))
print('Max:', X_train_minmax.max(axis=0).round(2))
```

Output:
```text
Min: [0. 0. 0. 0.]
Max: [1. 1. 1. 1.]
```

## 8. Encoding Categorical Labels — `LabelEncoder`

```python
from sklearn.preprocessing import LabelEncoder

categories = np.array(['cat', 'dog', 'dog', 'bird', 'cat'])
le = LabelEncoder()
encoded = le.fit_transform(categories)
print('Original:', categories)
print('Encoded :', encoded)
print('Classes :', le.classes_)
```

Output:
```text
Original: ['cat' 'dog' 'dog' 'bird' 'cat']
Encoded : [1 2 2 0 1]
Classes : ['bird' 'cat' 'dog']
```

## 9. One-Hot Encoding — `OneHotEncoder`

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)
categories_2d = categories.reshape(-1, 1)
onehot = ohe.fit_transform(categories_2d)
print(onehot)
print('Feature names:', ohe.get_feature_names_out())
```

Output:
```text
[[0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]]
Feature names: ['x0_bird' 'x0_cat' 'x0_dog']
```

## 10. Building a Pipeline — Chaining Preprocessing + Model

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=200))
])

pipeline.fit(X_train, y_train)
print('Pipeline test accuracy:', pipeline.score(X_test, y_test))
```

Output:
```text
Pipeline test accuracy: 0.9333333333333333
```

## 11. Why Pipelines Matter — Preventing Data Leakage

```python
# WRONG way: scaling before splitting leaks test-set information into training
X_all_scaled_WRONG = StandardScaler().fit_transform(X)  # fit on everything, including test data

# RIGHT way: fit scaler only on training data (what the Pipeline does automatically)
scaler_right = StandardScaler().fit(X_train)
X_train_right = scaler_right.transform(X_train)
X_test_right = scaler_right.transform(X_test)
print('Pipelines handle this correctly by design — no manual bookkeeping needed.')
```

Output:
```text
Pipelines handle this correctly by design — no manual bookkeeping needed.
```

## 12. Basic Model Evaluation — `.score()`

```python
model_scaled = LogisticRegression(max_iter=200)
model_scaled.fit(X_train_scaled, y_train)
print('Training accuracy:', model_scaled.score(X_train_scaled, y_train))
print('Test accuracy    :', model_scaled.score(X_test_scaled, y_test))
```

Output:
```text
Training accuracy: 0.9583333333333334
Test accuracy    : 0.9333333333333333
```

## 13. Visualizing the Iris Dataset

```python
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 2], c=iris.target, cmap='viridis', s=60)
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[2])
ax.set_title('Iris Dataset — Sepal Length vs Petal Length')
legend = ax.legend(*scatter.legend_elements(), title='Species')
ax.add_artist(legend)
plt.show()
```

## 14. Summary — What You Learned

- The universal Estimator API: `fit()`, `predict()`, `transform()`
- Train/test splitting and why it matters
- Feature scaling (`StandardScaler`, `MinMaxScaler`)
- Categorical encoding (`LabelEncoder`, `OneHotEncoder`)
- Pipelines and why they prevent data leakage

### Next Notebook
Continue to **`02_classification.ipynb`** for KNN, Decision Trees, Random Forests, SVM, and classification metrics.
