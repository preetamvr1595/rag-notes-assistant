# 02 — Categorical Plots (Preetham's Seaborn Mastery Series)

Part 2 of 5. Covers `barplot`, `countplot`, `boxplot`, `violinplot`, `stripplot`, `swarmplot`, `boxenplot`, `pointplot`, and `catplot`.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style='whitegrid')
tips = sns.load_dataset('tips')
titanic = sns.load_dataset('titanic')
tips.head()
```

Output:
```text
total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3
3       23.68  3.31    Male     No  Sun  Dinner     2
4       24.59  3.61  Female     No  Sun  Dinner     4
```

## 1. `barplot()` — Mean + Confidence Interval by Category

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x='day', y='total_bill', data=tips, hue='sex', ax=ax, errorbar='sd')
ax.set_title('Average Bill by Day and Sex')
plt.show()
```

## 2. `countplot()` — Frequency Counts

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x='day', data=tips, hue='smoker', ax=ax)
ax.set_title('Count of Visits by Day and Smoker Status')
plt.show()
```

## 3. `boxplot()` — Distribution Summary with Outliers

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x='day', y='total_bill', data=tips, hue='smoker', ax=ax)
ax.set_title('Bill Distribution by Day')
plt.show()
```

## 4. `violinplot()` — Distribution Shape + Box Summary Combined

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', split=True, ax=ax)
ax.set_title('Bill Distribution Shape by Day and Sex')
plt.show()
```

## 5. `stripplot()` — Every Individual Data Point

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.stripplot(x='day', y='total_bill', data=tips, hue='sex', dodge=True, alpha=0.6, ax=ax)
ax.set_title('Every Individual Bill (Strip Plot)')
plt.show()
```

## 6. `swarmplot()` — Non-Overlapping Individual Points

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.swarmplot(x='day', y='total_bill', data=tips, hue='sex', dodge=True, ax=ax, size=4)
ax.set_title('Non-Overlapping Points (Swarm Plot)')
plt.show()
```

## 7. `boxenplot()` — Letter-Value Plot for Larger Datasets

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxenplot(x='day', y='total_bill', data=tips, ax=ax)
ax.set_title('Boxen Plot (Handles Large Datasets Better than Box Plot)')
plt.show()
```

## 8. `pointplot()` — Mean Trend Comparison

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.pointplot(x='day', y='total_bill', data=tips, hue='sex', ax=ax, dodge=True)
ax.set_title('Mean Bill Trend by Day and Sex')
plt.show()
```

## 9. `catplot()` — Figure-Level Categorical Plot with Facets

```python
g = sns.catplot(x='day', y='total_bill', data=tips, col='time', kind='box', height=4)
g.fig.suptitle('Bill Distribution by Day, Faceted by Time', y=1.05)
plt.show()
```

## 10. Combining Box + Strip — Showing Distribution AND Raw Points

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x='day', y='total_bill', data=tips, ax=ax, showfliers=False, color='lightblue')
sns.stripplot(x='day', y='total_bill', data=tips, ax=ax, color='black', alpha=0.4, size=3)
ax.set_title('Box + Strip Combined')
plt.show()
```

## 11. Real-World Application — Titanic Survival by Class

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x='class', y='survived', data=titanic, hue='sex', ax=ax)
ax.set_title('Titanic Survival Rate by Class and Sex')
ax.set_ylabel('Survival Rate')
plt.show()
```

## 12. Titanic Age Distribution by Survival — Violin Plot

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(x='survived', y='age', data=titanic, hue='sex', split=True, ax=ax)
ax.set_title('Age Distribution by Survival and Sex')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Did Not Survive', 'Survived'])
plt.show()
```

## 13. Summary — What You Learned

| Plot | Purpose |
|---|---|
| `barplot()` | Mean + confidence interval by category |
| `countplot()` | Frequency counts |
| `boxplot()` | Distribution summary + outliers |
| `violinplot()` | Distribution shape + box summary |
| `stripplot()` | All individual points (may overlap) |
| `swarmplot()` | All individual points (non-overlapping) |
| `boxenplot()` | Letter-value plot for large data |
| `pointplot()` | Mean trend comparison |
| `catplot()` | Figure-level version with facets |

### Next Notebook
Continue to **`03_distribution_plots.ipynb`** for histograms, KDE, ECDF, and joint/pair plots.
