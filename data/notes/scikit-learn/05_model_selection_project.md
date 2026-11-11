# 05 — Model Selection & Real-World Project (Preetham's Scikit-learn Mastery Series)

Part 5 of 5 (final notebook). Covers `GridSearchCV`, `RandomizedSearchCV`, `ColumnTransformer`, learning curves, and a complete end-to-end classification project.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_theme(style='whitegrid')
cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

## 1. `GridSearchCV` — Exhaustive Hyperparameter Search

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print('Best parameters:', grid_search.best_params_)
print('Best CV score  :', grid_search.best_score_.round(4))
```

Output:
```text
Best parameters: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}
Best CV score  : 0.9604
```

## 2. Evaluating the Best Model from Grid Search

```python
best_model = grid_search.best_estimator_
test_preds = best_model.predict(X_test)
print('Test accuracy:', round(accuracy_score(y_test, test_preds), 4))
```

Output:
```text
Test accuracy: 0.9561
```

## 3. `RandomizedSearchCV` — Faster Search Over Large Parameter Spaces

```python
from scipy.stats import randint

param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42), param_distributions,
    n_iter=20, cv=5, scoring='accuracy', random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)

print('Best parameters:', random_search.best_params_)
print('Best CV score  :', random_search.best_score_.round(4))
```

Output:
```text
Best parameters: {'max_depth': 17, 'min_samples_split': 4, 'n_estimators': 121}
Best CV score  : 0.9604
```

## 4. Comparing Grid Search vs Randomized Search

```python
print(f'GridSearchCV best score      : {grid_search.best_score_:.4f} ({len(grid_search.cv_results_["params"])} combinations tried)')
print(f'RandomizedSearchCV best score : {random_search.best_score_:.4f} (20 combinations tried)')
print('\nRandomizedSearchCV often finds a comparably good result with far fewer evaluations.')
```

Output:
```text
GridSearchCV best score      : 0.9604 (18 combinations tried)
RandomizedSearchCV best score : 0.9604 (20 combinations tried)

RandomizedSearchCV often finds a comparably good result with far fewer evaluations.
```

## 5. `ColumnTransformer` — Different Preprocessing for Different Columns

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Simulated mixed-type dataset
mixed_df = pd.DataFrame({
    'age': [25, 32, 47, 51, 62],
    'income': [40000, 55000, 72000, 61000, 89000],
    'city': ['Bengaluru', 'Mumbai', 'Delhi', 'Bengaluru', 'Mumbai']
})

numeric_features = ['age', 'income']
categorical_features = ['city']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

transformed = preprocessor.fit_transform(mixed_df)
print(transformed)
```

Output:
```text
[[-1.38287244 -1.42239385  1.          0.          0.        ]
 [-0.85677967 -0.51060292  0.          0.          1.        ]
 [ 0.270562    0.52276013  0.          1.          0.        ]
 [ 0.57118644 -0.14588655  1.          0.          0.        ]
 [ 1.39790367  1.55612318  0.          0.          1.        ]]
```

## 6. Full Pipeline with ColumnTransformer + Model

```python
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
print('Pipeline structure:')
print(full_pipeline)
```

Output:
```text
Pipeline structure:
Pipeline(steps=[('preprocessor',
                 ColumnTransformer(transformers=[('num', StandardScaler(),
                                                  ['age', 'income']),
                                                 ('cat', OneHotEncoder(),
                                                  ['city'])])),
                ('classifier', RandomForestClassifier(random_state=42))])
```

## 7. Learning Curves — Diagnosing Overfitting/Underfitting

```python
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(random_state=42), X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(9, 6))
plt.plot(train_sizes, train_mean, 'o-', label='Training Score')
plt.plot(train_sizes, val_mean, 'o-', label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve — Random Forest')
plt.legend()
plt.show()
```

## 8. Interpreting the Learning Curve

```python
gap = train_mean[-1] - val_mean[-1]
print(f'Final training score  : {train_mean[-1]:.4f}')
print(f'Final validation score: {val_mean[-1]:.4f}')
print(f'Gap: {gap:.4f}')
print('A small gap suggests the model generalizes well; a large gap suggests overfitting.')
```

Output:
```text
Final training score  : 1.0000
Final validation score: 0.9561
Gap: 0.0439
A small gap suggests the model generalizes well; a large gap suggests overfitting.
```

## 9. Real-World Mini Project — Complete Classification Pipeline

```python
# Step 1: Load and split data
wine = datasets.load_wine()
Xw, yw = wine.data, wine.target
Xw_train, Xw_test, yw_train, yw_test = train_test_split(Xw, yw, test_size=0.2, random_state=42, stratify=yw)

# Step 2: Build a pipeline
wine_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Step 3: Tune hyperparameters
wine_param_grid = {'classifier__n_estimators': [50, 100, 150], 'classifier__max_depth': [None, 5, 10]}
wine_grid = GridSearchCV(wine_pipeline, wine_param_grid, cv=5)
wine_grid.fit(Xw_train, yw_train)

print('Best parameters:', wine_grid.best_params_)
print('Best CV score  :', wine_grid.best_score_.round(4))
```

Output:
```text
Best parameters: {'classifier__max_depth': None, 'classifier__n_estimators': 50}
Best CV score  : 0.9862
```

## 10. Final Evaluation on Held-Out Test Set

```python
final_preds = wine_grid.predict(Xw_test)
print('Test accuracy:', round(accuracy_score(yw_test, final_preds), 4))
print('\n', classification_report(yw_test, final_preds, target_names=wine.target_names))
```

Output:
```text
Test accuracy: 1.0

               precision    recall  f1-score   support

     class_0       1.00      1.00      1.00        12
     class_1       1.00      1.00      1.00        14
     class_2       1.00      1.00      1.00        10

    accuracy                           1.00        36
   macro avg       1.00      1.00      1.00        36
weighted avg       1.00      1.00      1.00        36
```

## 11. Confusion Matrix for the Final Model

```python
cm = confusion_matrix(yw_test, final_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title('Final Model — Confusion Matrix (Wine Classification)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

## 12. Saving the Final Model

```python
import joblib
import os

os.makedirs('models', exist_ok=True)
joblib.dump(wine_grid.best_estimator_, 'models/wine_classifier_pipeline.pkl')
print('Model saved to models/wine_classifier_pipeline.pkl')

# Demonstrate loading it back
loaded_model = joblib.load('models/wine_classifier_pipeline.pkl')
print('Loaded model test accuracy:', round(accuracy_score(yw_test, loaded_model.predict(Xw_test)), 4))
```

Output:
```text
Model saved to models/wine_classifier_pipeline.pkl
Loaded model test accuracy: 1.0
```

## 13. Summary of the Complete Series

- **Notebook 1:** The Estimator API, scaling, encoding, pipelines
- **Notebook 2:** 7 classification algorithms + precision/recall/F1/ROC-AUC
- **Notebook 3:** 7 regression algorithms + regularization + residual analysis
- **Notebook 4:** Clustering (K-Means/Hierarchical/DBSCAN), PCA, t-SNE, feature selection
- **Notebook 5:** GridSearchCV, RandomizedSearchCV, ColumnTransformer, learning curves, and a full tuned-and-saved pipeline

This series covers the practical scikit-learn workflow end-to-end: from raw data to a saved, deployable model.
