# 03 — Distribution Plots (Preetham's Seaborn Mastery Series)

Part 3 of 5. Covers `histplot`, `kdeplot`, `ecdfplot`, `rugplot`, `displot`, `jointplot`, and `pairplot`.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style='whitegrid')
tips = sns.load_dataset('tips')
penguins = sns.load_dataset('penguins')
penguins.head()
```

Output:
```text
species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  \
0  Adelie  Torgersen            39.1           18.7              181.0   
1  Adelie  Torgersen            39.5           17.4              186.0   
2  Adelie  Torgersen            40.3           18.0              195.0   
3  Adelie  Torgersen             NaN            NaN                NaN   
4  Adelie  Torgersen            36.7           19.3              193.0   

   body_mass_g     sex  
0       3750.0    Male  
1       …
```

## 1. `histplot()` — Histogram with Optional KDE Overlay

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=tips, x='total_bill', bins=25, kde=True, ax=ax, color='steelblue')
ax.set_title('Bill Amount Distribution')
plt.show()
```

## 2. `histplot()` with `hue` — Comparing Groups

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=tips, x='total_bill', hue='time', kde=True, ax=ax, alpha=0.5)
ax.set_title('Bill Distribution by Time of Day')
plt.show()
```

## 3. `kdeplot()` — Smooth Density Curve

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.kdeplot(data=tips, x='total_bill', hue='time', fill=True, ax=ax, alpha=0.4)
ax.set_title('Density Curve by Time')
plt.show()
```

## 4. 2-D `kdeplot()` — Bivariate Density

```python
fig, ax = plt.subplots(figsize=(8, 6))
sns.kdeplot(data=tips, x='total_bill', y='tip', fill=True, cmap='Blues', ax=ax)
ax.set_title('2D Density: Bill vs Tip')
plt.show()
```

## 5. `ecdfplot()` — Empirical Cumulative Distribution

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.ecdfplot(data=tips, x='total_bill', hue='time', ax=ax)
ax.set_title('Cumulative Distribution of Bill Amount')
plt.show()
```

## 6. `rugplot()` — Individual Data Points as Tick Marks

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=tips, x='total_bill', ax=ax, color='lightgreen')
sns.rugplot(data=tips, x='total_bill', ax=ax, color='darkgreen')
ax.set_title('Histogram with Rug Plot')
plt.show()
```

## 7. `displot()` — Figure-Level Distribution Plot with Facets

```python
g = sns.displot(data=tips, x='total_bill', col='time', kde=True, height=4)
g.fig.suptitle('Distribution Faceted by Time', y=1.05)
plt.show()
```

## 8. `jointplot()` — Scatter + Marginal Distributions Combined

```python
g = sns.jointplot(data=tips, x='total_bill', y='tip', kind='scatter', height=6)
g.fig.suptitle('Bill vs Tip with Marginal Distributions', y=1.02)
plt.show()
```

## 9. `jointplot()` with `kind='hex'` — For Dense Data

```python
g = sns.jointplot(data=tips, x='total_bill', y='tip', kind='hex', height=6)
plt.show()
```

## 10. `jointplot()` with `kind='kde'` — Density Version

```python
g = sns.jointplot(data=tips, x='total_bill', y='tip', kind='kde', height=6, fill=True)
plt.show()
```

## 11. `pairplot()` — Every Pairwise Relationship at Once

```python
penguins_clean = penguins.dropna()
g = sns.pairplot(penguins_clean, hue='species', height=2.2)
g.fig.suptitle('Penguin Measurements — Pairwise Relationships', y=1.02)
plt.show()
```

## 12. `pairplot()` with Custom Diagonal (KDE)

```python
g = sns.pairplot(penguins_clean, hue='species', diag_kind='kde', height=2.2, vars=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm'])
plt.show()
```

## 13. Summary — What You Learned

| Plot | Purpose |
|---|---|
| `histplot()` | Binned frequency, optional KDE overlay |
| `kdeplot()` | Smooth density curve (1D or 2D) |
| `ecdfplot()` | Cumulative distribution |
| `rugplot()` | Individual points as tick marks |
| `displot()` | Figure-level version with facets |
| `jointplot()` | Scatter/hex/KDE + marginal distributions |
| `pairplot()` | All pairwise relationships in one grid |

### Next Notebook
Continue to **`04_regression_matrix.ipynb`** for regression plots and correlation heatmaps.
