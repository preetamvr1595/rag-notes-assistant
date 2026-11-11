# 04 — Unsupervised Learning & Advanced Preprocessing (Preetham's Scikit-learn Mastery Series)

Part 4 of 5. Covers K-Means, Hierarchical Clustering, DBSCAN, PCA, t-SNE, and feature selection.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

wine = datasets.load_wine()
X, y = wine.data, wine.target
X_scaled = StandardScaler().fit_transform(X)
print('Shape:', X.shape, '| Classes:', wine.target_names)
```

Output:
```text
Shape: (178, 13) | Classes: ['class_0' 'class_1' 'class_2']
```

## 1. K-Means Clustering

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
print('Cluster assignments (first 20):', clusters[:20])
print('Cluster centers shape:', kmeans.cluster_centers_.shape)
```

Output:
```text
Cluster assignments (first 20): [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
Cluster centers shape: (3, 13)
```

## 2. Finding the Right Number of Clusters — Elbow Method

```python
inertias = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()
```

## 3. Evaluating Clustering — Silhouette Score

```python
from sklearn.metrics import silhouette_score

for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    score = silhouette_score(X_scaled, km.labels_)
    print(f'k={k}: silhouette score = {score:.4f}')
```

Output:
```text
k=2: silhouette score = 0.2593
k=3: silhouette score = 0.2849
k=4: silhouette score = 0.2602
k=5: silhouette score = 0.2016
```

## 4. Comparing Clusters to True Labels

```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(y, clusters)
print(f'Adjusted Rand Index (cluster vs true labels): {ari:.4f}')
print('(1.0 = perfect match, 0.0 = random)')
```

Output:
```text
Adjusted Rand Index (cluster vs true labels): 0.8975
(1.0 = perfect match, 0.0 = random)
```

## 5. Hierarchical (Agglomerative) Clustering

```python
from sklearn.cluster import AgglomerativeClustering

hierarchical = AgglomerativeClustering(n_clusters=3)
hier_labels = hierarchical.fit_predict(X_scaled)
print('Hierarchical cluster sizes:', np.bincount(hier_labels))
```

Output:
```text
Hierarchical cluster sizes: [58 56 64]
```

## 6. Dendrogram — Visualizing Hierarchical Clustering

```python
from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(X_scaled[:30], method='ward')  # subset for a readable plot
plt.figure(figsize=(12, 6))
dendrogram(linked)
plt.title('Hierarchical Clustering Dendrogram (first 30 samples)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
```

## 7. DBSCAN — Density-Based Clustering (Finds Outliers Too)

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
print('Cluster labels:', np.unique(dbscan_labels))
print('Noise points (-1):', (dbscan_labels == -1).sum())
```

Output:
```text
Cluster labels: [-1]
Noise points (-1): 178
```

## 8. Principal Component Analysis (PCA) — Dimensionality Reduction

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print('Original shape:', X_scaled.shape, '-> Reduced shape:', X_pca.shape)
print('Variance explained by 2 components:', pca.explained_variance_ratio_.sum().round(4))
```

Output:
```text
Original shape: (178, 13) -> Reduced shape: (178, 2)
Variance explained by 2 components: 0.5541
```

## 9. Visualizing PCA-Reduced Data

```python
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=60)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Wine Dataset After PCA (2D)')
plt.legend(*scatter.legend_elements(), title='Wine Class')
plt.show()
```

## 10. Choosing the Number of Components — Explained Variance

```python
pca_full = PCA().fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
plt.axhline(0.95, color='red', linestyle='--', label='95% variance threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA — How Many Components Do We Need?')
plt.legend()
plt.show()
```

## 11. t-SNE — Non-Linear Dimensionality Reduction for Visualization

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='plasma', s=60)
plt.title('Wine Dataset After t-SNE')
plt.legend(*scatter.legend_elements(), title='Wine Class')
plt.show()
```

## 12. Feature Selection — `SelectKBest`

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)
selected_features = np.array(wine.feature_names)[selector.get_support()]
print('Top 5 features selected:', selected_features)
```

Output:
```text
Top 5 features selected: ['alcohol' 'flavanoids' 'color_intensity' 'od280/od315_of_diluted_wines'
 'proline']
```

## 13. Feature Selection — Recursive Feature Elimination (RFE)

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=5)
rfe.fit(X, y)
rfe_features = np.array(wine.feature_names)[rfe.support_]
print('RFE-selected features:', rfe_features)
```

Output:
```text
RFE-selected features: ['alcohol' 'flavanoids' 'color_intensity' 'od280/od315_of_diluted_wines'
 'proline']
```

## 14. Handling Missing Values — `SimpleImputer`

```python
from sklearn.impute import SimpleImputer

data_with_nan = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, np.nan], [5.0, 6.0]])
imputer = SimpleImputer(strategy='mean')
imputed = imputer.fit_transform(data_with_nan)
print('Before:\n', data_with_nan)
print('\nAfter:\n', imputed)
```

Output:
```text
Before:
 [[ 1.  2.]
 [nan  3.]
 [ 4. nan]
 [ 5.  6.]]

After:
 [[1.         2.        ]
 [3.33333333 3.        ]
 [4.         3.66666667]
 [5.         6.        ]]
```

## 15. Summary — What You Learned

| Technique | Purpose |
|---|---|
| K-Means | Partition data into k clusters |
| Elbow method + Silhouette score | Choose the right number of clusters |
| Hierarchical clustering | Tree-structured clustering, no k needed upfront |
| DBSCAN | Density-based, finds outliers naturally |
| PCA | Linear dimensionality reduction |
| t-SNE | Non-linear reduction, great for visualization |
| SelectKBest / RFE | Feature selection |
| SimpleImputer | Handling missing values |

### Next Notebook
Continue to **`05_model_selection_project.ipynb`** for hyperparameter tuning and a complete real-world project.
