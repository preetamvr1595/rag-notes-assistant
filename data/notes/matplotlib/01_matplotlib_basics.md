# 01 — Matplotlib Basics (Preetham's Matplotlib Mastery)

Figure/Axes, line plots, scatter, colors, markers, legends, grids, saving.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
print('Matplotlib:', plt.matplotlib.__version__)
```

Output:
```text
Matplotlib: 3.10.8
```

## Simple Line Plot

```python
x = [1,2,3,4,5]
y = [2,4,6,8,10]
plt.plot(x, y)
plt.title('Simple Plot')
plt.show()
```

## Using Figure & Axes

```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, linewidth=2, color='blue', marker='o', label='Data')
ax.set_title('With Axes Control')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

## Multiple Lines

```python
x = np.linspace(0, 10, 100)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, np.sin(x), label='sin(x)', linewidth=2)
ax.plot(x, np.cos(x), label='cos(x)', linewidth=2)
ax.plot(x, x*0.1, label='0.1*x', linestyle='--', linewidth=2)
ax.legend()
ax.set_title('Multiple Lines')
ax.grid(True)
plt.show()
```

## Scatter Plot

```python
np.random.seed(42)
x_scatter = np.random.randn(100)
y_scatter = np.random.randn(100)
colors = np.random.rand(100)
sizes = np.random.randint(20, 200, 100)
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(x_scatter, y_scatter, c=colors, s=sizes, alpha=0.6, cmap='viridis')
ax.set_title('Scatter with Color & Size')
plt.colorbar(sc, ax=ax)
plt.show()
```

## Color Control

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].plot([1,2,3], [1,4,9], color='red', linewidth=3)
axes[0].set_title('Named Color')
axes[1].plot([1,2,3], [1,4,9], color='#FF5733', linewidth=3)
axes[1].set_title('Hex Color')
axes[2].plot([1,2,3], [1,4,9], color=(0.2,0.8,0.5), linewidth=3)
axes[2].set_title('RGB')
plt.tight_layout()
plt.show()
```

## Marker Styles

```python
markers = ['o', 's', '^', 'D', 'v', 'p', '*']
fig, ax = plt.subplots(figsize=(10, 5))
for i, m in enumerate(markers):
    ax.plot([i], [i**2], marker=m, markersize=12, label=m)
ax.set_title('Marker Styles')
ax.legend()
plt.show()
```

## Line Styles

```python
fig, ax = plt.subplots(figsize=(10, 5))
styles = ['-', '--', '-.', ':']
for i, style in enumerate(styles):
    ax.plot([0, 10], [i, i], linestyle=style, linewidth=3, label=style)
ax.set_title('Line Styles')
ax.set_ylim(-1, 4)
ax.legend()
plt.show()
```

## Text & Annotations

```python
fig, ax = plt.subplots(figsize=(9, 5))
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
ax.plot(x, y, linewidth=2)
ax.annotate('Peak', xy=(np.pi/2, 1), xytext=(np.pi, 0.5),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=12)
ax.text(3, -0.5, 'Text Label', fontsize=10, ha='center')
ax.set_title('Annotations & Text')
ax.grid(True)
plt.show()
```

## Legends & Positioning

```python
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot([1,2,3], [1,4,9], label='Series A')
ax.plot([1,2,3], [3,2,1], label='Series B')
ax.legend(loc='upper left', fontsize=11)
ax.set_title('Legend Positioning')
ax.grid(True, alpha=0.3)
plt.show()
```

## Saving Figures

```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([1,2,3,4], [1,4,2,3], marker='o')
ax.set_title('Plot to Save')
fig.savefig('/tmp/plot.png', dpi=300, bbox_inches='tight')
print('Saved: /tmp/plot.png')
plt.show()
```

Output:
```text
Saved: /tmp/plot.png
```

## Figure Size & DPI

```python
fig1, ax1 = plt.subplots(figsize=(4, 3))
ax1.plot([1,2,3], [1,2,3])
ax1.set_title('Small (4x3)')
plt.show()
fig2, ax2 = plt.subplots(figsize=(12, 8), dpi=150)
ax2.plot([1,2,3], [1,2,3])
ax2.set_title('Large (12x8, 150 DPI)')
plt.show()
```

## Axis Scales

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
x = np.linspace(0, 10, 100)
y = np.exp(x/5)
axes[0].plot(x, y)
axes[0].set_title('Linear Scale')
axes[1].plot(x, y, color='red')
axes[1].set_yscale('log')
axes[1].set_title('Log Scale')
plt.tight_layout()
plt.show()
```

## Grid Customization

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, ax in enumerate(axes):
    ax.plot([1,2,3,4], [1,4,2,3], marker='o')
    if i==1:
        ax.grid(True, which='major', linestyle='-', alpha=0.7)
    elif i==2:
        ax.grid(True, which='both', alpha=0.3)
        ax.minorticks_on()
    ax.set_title(f'Grid Style {i+1}')
plt.tight_layout()
plt.show()
```

## Fill Between (Area Under Curve)

```python
x = np.linspace(0, 10, 100)
y = np.sin(x)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, y, color='navy', linewidth=2)
ax.fill_between(x, y, 0, alpha=0.3, color='skyblue')
ax.set_title('Fill Between (Area Under Curve)')
ax.axhline(0, color='black', linewidth=0.5)
plt.show()
```

## Step Plots — For Discrete/Stepped Data

```python
x = np.arange(10)
y = np.random.randint(1, 10, 10)
fig, ax = plt.subplots(figsize=(9, 5))
ax.step(x, y, where='mid', linewidth=2, marker='o')
ax.set_title('Step Plot')
ax.grid(True, alpha=0.3)
plt.show()
```

## Stack Plots — Cumulative Category Comparison

```python
days = np.arange(1, 8)
team_a = np.random.randint(5, 20, 7)
team_b = np.random.randint(5, 20, 7)
team_c = np.random.randint(5, 20, 7)
fig, ax = plt.subplots(figsize=(9, 5))
ax.stackplot(days, team_a, team_b, team_c, labels=['Team A', 'Team B', 'Team C'], alpha=0.8)
ax.set_title('Stacked Area Chart')
ax.legend(loc='upper left')
plt.show()
```

## Twin Axes — Two Different Y-Scales on One Plot

```python
x = np.arange(1, 13)
sales = np.random.randint(100, 500, 12)
temp = np.random.randint(15, 40, 12)
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(x, sales, color='blue', marker='o', label='Sales')
ax1.set_xlabel('Month')
ax1.set_ylabel('Sales', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax2 = ax1.twinx()
ax2.plot(x, temp, color='red', marker='s', label='Temperature')
ax2.set_ylabel('Temperature (C)', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax1.set_title('Sales vs Temperature (Dual Axis)')
plt.show()
```

## Spines Control — Customizing Plot Borders

```python
fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(-5, 5, 100)
ax.plot(x, x**2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.set_title('Math-Style Axes (Spines at Zero)')
plt.show()
```

## Global Style Settings — `rcParams`

```python
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 2

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_title('Using rcParams Global Settings')
plt.show()

plt.rcdefaults()  # reset to defaults afterward
```

## Polar Plots

```python
theta = np.linspace(0, 2*np.pi, 100)
r = np.abs(np.sin(3*theta))
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
ax.plot(theta, r, linewidth=2)
ax.set_title('Polar Plot')
plt.show()
```

## `subplot_mosaic()` — Modern Flexible Layouts

```python
fig, axd = plt.subplot_mosaic([['left', 'right_top'], ['left', 'right_bottom']], figsize=(10, 6))
axd['left'].plot([1, 2, 3], [1, 4, 9])
axd['left'].set_title('Left (spans 2 rows)')
axd['right_top'].bar(['A', 'B'], [3, 5])
axd['right_top'].set_title('Right Top')
axd['right_bottom'].scatter([1, 2, 3], [3, 1, 2])
axd['right_bottom'].set_title('Right Bottom')
plt.tight_layout()
plt.show()
```

## Summary — Notebook 1

Figure/Axes fundamentals, line & scatter plots, colors, markers, line styles, annotations, legends, saving, DPI control, scales, grids, fill_between, step/stack plots, twin axes, spines, rcParams, polar plots, and subplot_mosaic.

### Next: `02_plot_types.ipynb` for bar, histogram, pie, box, violin, and more.
