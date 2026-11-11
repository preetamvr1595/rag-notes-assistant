# 02 — Plot Types

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

## Bar Plots

```python
categories = ['HR', 'IT', 'Sales', 'Finance']
salaries = [45000, 78000, 52000, 65000]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(categories, salaries, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
ax.set_title('Department Salaries')
ax.set_ylabel('Salary ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

```python
# Horizontal bar plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(categories, salaries, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
ax.set_title('Departments by Salary')
ax.set_xlabel('Salary ($)')
plt.tight_layout()
plt.show()
```

## Histograms

```python
np.random.seed(42)
data = np.random.normal(100, 15, 1000)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax.set_title('Distribution')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
plt.show()
```

## Pie Charts

```python
sizes = [30, 25, 20, 25]
labels = ['HR', 'IT', 'Sales', 'Finance']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax.set_title('Department Distribution')
plt.show()
```

## Box Plots

```python
np.random.seed(42)
hr_sal = np.random.normal(45000, 5000, 50)
it_sal = np.random.normal(78000, 8000, 50)
fin_sal = np.random.normal(65000, 6000, 50)
fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot([hr_sal, it_sal, fin_sal], labels=['HR', 'IT', 'Finance'])
ax.set_title('Salary Distribution by Department')
ax.set_ylabel('Salary ($)')
plt.show()
```

Output:
```text
/tmp/ipykernel_529/4003854856.py:6: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  ax.boxplot([hr_sal, it_sal, fin_sal], labels=['HR', 'IT', 'Finance'])
```

## Violin Plots

```python
fig, ax = plt.subplots(figsize=(8, 5))
parts = ax.violinplot([hr_sal, it_sal, fin_sal], positions=[1,2,3])
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['HR', 'IT', 'Finance'])
ax.set_title('Salary Distribution (Violin Plot)')
ax.set_ylabel('Salary ($)')
plt.show()
```

## Heatmaps

```python
data = np.random.rand(5, 5) * 100
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
ax.set_title('Heatmap')
plt.colorbar(im, ax=ax)
plt.show()
```

## Stem Plots

```python
x = np.linspace(0, 10, 20)
y = np.sin(x)
fig, ax = plt.subplots(figsize=(8, 5))
ax.stem(x, y, basefmt=' ')
ax.set_title('Stem Plot')
plt.show()
```

## 2D Histograms

```python
x = np.random.randn(1000)
y = np.random.randn(1000)
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist2d(x, y, bins=20, cmap='Blues')
ax.set_title('2D Histogram')
plt.colorbar(ax.collections[0], ax=ax)
plt.show()
```

## Grouped Bar Charts — Comparing Multiple Series

```python
depts = ['HR', 'IT', 'Sales', 'Finance']
x = np.arange(len(depts))
width = 0.35
year1 = [45000, 78000, 52000, 65000]
year2 = [48000, 82000, 55000, 68000]
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width/2, year1, width, label='2025', color='#4ECDC4')
ax.bar(x + width/2, year2, width, label='2026', color='#FF6B6B')
ax.set_xticks(x)
ax.set_xticklabels(depts)
ax.set_title('Grouped Bar Chart — Year-over-Year Salary')
ax.legend()
plt.show()
```

## Stacked Bar Charts

```python
base_sal = [45000, 78000, 52000, 65000]
bonus = [5000, 15000, 8000, 12000]
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(depts, base_sal, label='Base Salary', color='#2E86AB')
ax.bar(depts, bonus, bottom=base_sal, label='Bonus', color='#F18F01')
ax.set_title('Stacked Bar — Salary + Bonus')
ax.legend()
plt.show()
```

## Bar Plot with `align='edge'` (Your Original Style)

```python
ages = [22, 23, 24, 25, 26, 27]
salary = [40000, 42000, 45000, 47000, 50000, 52000]
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(ages, salary, color='green', align='edge', width=0.8)
ax.set_title('Salary by Age (align=edge)')
ax.set_xlabel('Age')
ax.set_ylabel('Salary')
plt.show()
```

## Box Plot Comparing Departments (Your Original Style, Cleaned)

```python
np.random.seed(3)
hr_sal = np.random.normal(45000, 4000, 30)
it_sal = np.random.normal(78000, 6000, 30)
fin_sal = np.random.normal(65000, 5000, 30)
fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot([hr_sal, it_sal, fin_sal], labels=['HR', 'IT', 'Finance'])
ax.set_title('Salary Distribution by Department')
plt.show()
```

Output:
```text
/tmp/ipykernel_529/3251780053.py:6: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  ax.boxplot([hr_sal, it_sal, fin_sal], labels=['HR', 'IT', 'Finance'])
```

## Pie Chart with Explode (Your Original Style, Fixed)

```python
salary_by_dept = pd.Series([45000, 78000, 52000, 65000], index=['HR', 'IT', 'Sales', 'Finance'])
fig, ax = plt.subplots(figsize=(7, 7))
explode = [0, 0.1, 0, 0]  # highlight IT slice
ax.pie(salary_by_dept, labels=salary_by_dept.index, autopct='%1.2f%%', explode=explode, shadow=True)
ax.set_title('Salary Distribution (Exploded Pie)')
plt.show()
```

## Count Plot Equivalent (Bar of `value_counts()`)

```python
dept_list = ['hr', 'it', 'sales', 'hr', 'it', 'hr', 'finance', 'sales'] 
counts = pd.Series(dept_list).value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(counts.index, counts.values, color='teal')
ax.set_title('Department Frequency (Count Plot Style)')
plt.show()
```

## Density / Normalized Histogram

```python
data = np.random.normal(100, 15, 1000)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data, bins=30, density=True, color='lightgreen', edgecolor='black', alpha=0.7)
ax.set_title('Normalized (Density) Histogram')
plt.show()
```

## Cumulative Histogram

```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data, bins=30, cumulative=True, color='coral', edgecolor='black')
ax.set_title('Cumulative Histogram')
plt.show()
```

## Hexbin Plot — Dense Scatter Alternative

```python
x = np.random.randn(5000)
y = np.random.randn(5000)
fig, ax = plt.subplots(figsize=(8, 6))
hb = ax.hexbin(x, y, gridsize=30, cmap='Blues')
ax.set_title('Hexbin Plot (Dense Scatter)')
plt.colorbar(hb, ax=ax)
plt.show()
```

## Broken Bar (Horizontal Timeline / Gantt-Style)

```python
fig, ax = plt.subplots(figsize=(9, 4))
ax.broken_barh([(10, 20), (40, 15), (60, 10)], (5, 8), facecolors=('#4ECDC4', '#FF6B6B', '#F18F01'))
ax.set_ylim(0, 20)
ax.set_xlim(0, 80)
ax.set_xlabel('Days')
ax.set_yticks([9])
ax.set_yticklabels(['Project'])
ax.set_title('Broken Bar (Gantt-Style Timeline)')
plt.show()
```

## Radar / Spider Chart

```python
categories = ['Speed', 'Reliability', 'Comfort', 'Safety', 'Price']
values = [4, 3, 5, 4, 2]
values += values[:1]
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
ax.plot(angles, values, linewidth=2)
ax.fill(angles, values, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_title('Radar Chart — Product Ratings')
plt.show()
```

## Summary — Notebook 2

Bar (grouped/stacked/edge-aligned), histograms (density/cumulative), pie (with explode), box plots, count plots, hexbin, broken bar, radar/spider chart, violin plots, heatmaps, stem plots, 2D histograms.

### Next: `03_subplots_styling.ipynb` for multi-plot layouts and styling.
