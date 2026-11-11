# 02 — Supervised Learning: Classification (Preetham's Scikit-learn Mastery Series)

Part 2 of 5. Covers KNN, Decision Trees, Random Forests, SVM, Gradient Boosting, Naive Bayes, and every major classification metric.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

sns.set_theme(style='whitegrid')
cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target
print('Shape:', X.shape, '| Classes:', cancer.target_names)
```

Output:
```text
Shape: (569, 30) | Classes: ['malignant' 'benign']
```

## 1. Train/Test Split + Scaling

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print('Train:', X_train.shape, '| Test:', X_test.shape)
```

Output:
```text
Train: (426, 30) | Test: (143, 30)
```

## 2. K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_s, y_train)
knn_preds = knn.predict(X_test_s)
print('KNN Accuracy:', accuracy_score(y_test, knn_preds))
```

Output:
```text
KNN Accuracy: 0.9790209790209791
```

## 3. Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=5000)
logreg.fit(X_train_s, y_train)
logreg_preds = logreg.predict(X_test_s)
print('Logistic Regression Accuracy:', accuracy_score(y_test, logreg_preds))
```

Output:
```text
Logistic Regression Accuracy: 0.986013986013986
```

## 4. Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
tree_preds = tree.predict(X_test)
print('Decision Tree Accuracy:', accuracy_score(y_test, tree_preds))
```

Output:
```text
Decision Tree Accuracy: 0.9440559440559441
```

## 5. Visualizing the Decision Tree

```python
plt.figure(figsize=(16, 8))
plot_tree(tree, max_depth=2, feature_names=cancer.feature_names, class_names=cancer.target_names, filled=True, fontsize=9)
plt.title('Decision Tree (first 2 levels)')
plt.show()
```

## 6. Random Forest — An Ensemble of Trees

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print('Random Forest Accuracy:', accuracy_score(y_test, rf_preds))
```

Output:
```text
Random Forest Accuracy: 0.958041958041958
```

## 7. Feature Importance from Random Forest

```python
importances = pd.Series(rf.feature_importances_, index=cancer.feature_names).sort_values(ascending=False).head(10)
plt.figure(figsize=(9, 5))
importances.plot(kind='barh')
plt.title('Top 10 Most Important Features')
plt.gca().invert_yaxis()
plt.show()
```

## 8. Support Vector Machine (SVM)

```python
from sklearn.svm import SVC

svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_s, y_train)
svm_preds = svm.predict(X_test_s)
print('SVM Accuracy:', accuracy_score(y_test, svm_preds))
```

Output:
```text
SVM Accuracy: 0.9790209790209791
```

## 9. Gradient Boosting Classifier

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)
print('Gradient Boosting Accuracy:', accuracy_score(y_test, gb_preds))
```

Output:
```text
Gradient Boosting Accuracy: 0.958041958041958
```

## 10. Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)
print('Naive Bayes Accuracy:', accuracy_score(y_test, nb_preds))
```

Output:
```text
Naive Bayes Accuracy: 0.9370629370629371
```

## 11. Comparing All Models

```python
results = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'Gradient Boosting', 'Naive Bayes'],
    'Accuracy': [
        accuracy_score(y_test, knn_preds),
        accuracy_score(y_test, logreg_preds),
        accuracy_score(y_test, tree_preds),
        accuracy_score(y_test, rf_preds),
        accuracy_score(y_test, svm_preds),
        accuracy_score(y_test, gb_preds),
        accuracy_score(y_test, nb_preds)
    ]
}).sort_values('Accuracy', ascending=False)
print(results)
```

Output:
```text
Model  Accuracy
1  Logistic Regression  0.986014
0                  KNN  0.979021
4                  SVM  0.979021
5    Gradient Boosting  0.958042
3        Random Forest  0.958042
2        Decision Tree  0.944056
6          Naive Bayes  0.937063
```

```python
plt.figure(figsize=(9, 5))
plt.barh(results['Model'], results['Accuracy'], color='teal')
plt.xlabel('Accuracy')
plt.title('Model Comparison — Breast Cancer Classification')
plt.xlim(0.85, 1.0)
plt.gca().invert_yaxis()
plt.show()
```

## 12. Precision, Recall, F1-Score

```python
print('Precision:', precision_score(y_test, rf_preds))
print('Recall   :', recall_score(y_test, rf_preds))
print('F1-Score :', f1_score(y_test, rf_preds))
print('\nFull report:\n', classification_report(y_test, rf_preds, target_names=cancer.target_names))
```

Output:
```text
Precision: 0.9565217391304348
Recall   : 0.9777777777777777
F1-Score : 0.967032967032967

Full report:
               precision    recall  f1-score   support

   malignant       0.96      0.92      0.94        53
      benign       0.96      0.98      0.97        90

    accuracy                           0.96       143
   macro avg       0.96      0.95      0.95       143
weighted avg       0.96      0.96      0.96       143
```

## 13. Confusion Matrix

```python
cm = confusion_matrix(y_test, rf_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cancer.target_names, yticklabels=cancer.target_names)
plt.title('Random Forest — Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

## 14. ROC Curve and AUC

```python
rf_proba = rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, rf_proba)
auc = roc_auc_score(y_test, rf_proba)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Random Forest')
plt.legend()
plt.show()
```

## 15. Cross-Validation — More Reliable Than a Single Split

```python
cv_scores = cross_val_score(rf, X, y, cv=5)
print('CV scores per fold:', cv_scores.round(4))
print('Mean CV accuracy  :', cv_scores.mean().round(4))
print('Std CV accuracy   :', cv_scores.std().round(4))
```

Output:
```text
CV scores per fold: [0.9211 0.9386 0.9825 0.9737 0.9735]
Mean CV accuracy  : 0.9578
Std CV accuracy   : 0.0238
```

## 16. Multi-Class Classification — Iris Dataset

```python
iris = datasets.load_iris()
Xi_train, Xi_test, yi_train, yi_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

multi_rf = RandomForestClassifier(random_state=42)
multi_rf.fit(Xi_train, yi_train)
multi_preds = multi_rf.predict(Xi_test)
print('Multi-class accuracy:', accuracy_score(yi_test, multi_preds))
print('\n', classification_report(yi_test, multi_preds, target_names=iris.target_names))
```

Output:
```text
Multi-class accuracy: 1.0

               precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

## 17. Summary — What You Learned

| Model | Best For |
|---|---|
| KNN | Simple baseline, small datasets |
| Logistic Regression | Linear decision boundaries, interpretable |
| Decision Tree | Interpretable, handles non-linear data |
| Random Forest | Strong general-purpose performer |
| SVM | Effective in high-dimensional spaces |
| Gradient Boosting | Often the top performer on tabular data |
| Naive Bayes | Fast, works well with text/high dimensions |

Plus: precision/recall/F1, confusion matrices, ROC-AUC, and cross-validation.

### Next Notebook
Continue to **`03_regression.ipynb`** for predicting continuous values.
