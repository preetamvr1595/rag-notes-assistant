# 03 — Subplots & Styling

```python
import matplotlib.pyplot as plt
import numpy as np
```

## Basic Subplots

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('sin(x)')
axes[0, 1].plot(x, np.cos(x))
axes[0, 1].set_title('cos(x)')
axes[1, 0].plot(x, np.exp(-x/5))
axes[1, 0].set_title('Decay')
axes[1, 1].plot(x, x**2)
axes[1, 1].set_title('Quadratic')
plt.tight_layout()
plt.show()
```

## GridSpec for Complex Layouts

```python
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(3, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :-1])
ax3 = fig.add_subplot(gs[1:, -1])
ax1.plot([1,2,3], [1,2,3])
ax1.set_title('Span Full Width')
ax2.plot([1,2,3], [3,2,1])
ax2.set_title('Partial Subplot')
ax3.plot([1,2,3], [1,4,9])
ax3.set_title('Right Column')
plt.tight_layout()
plt.show()
```

## Shared Axes

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x))
ax1.set_title('sin(x)')
ax2.plot(x, np.cos(x))
ax2.set_title('cos(x)')
plt.tight_layout()
plt.show()
```

## Styles

```python
styles = ['default', 'seaborn-v0_8-darkgrid', 'grayscale']
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, style in zip(axes, styles):
    with plt.style.context(style):
        ax.plot(np.random.randn(20).cumsum())
        ax.set_title(f'Style: {style}')
plt.tight_layout()
plt.show()
```

## Font Control

```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([1,2,3], [1,4,9])
ax.set_title('Title', fontsize=16, fontweight='bold', fontfamily='serif')
ax.set_xlabel('X Label', fontsize=12)
ax.set_ylabel('Y Label', fontsize=12)
for tick in ax.get_xticklabels():
    tick.set_fontsize(10)
plt.show()
```

## Figure Spacing

```python
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for ax in axes.flat:
    ax.plot([1,2,3], [1,2,3])
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
plt.show()
```

## Inset Axes — Zoomed-In View Within a Plot

```python
fig, ax = plt.subplots(figsize=(9, 6))
x = np.linspace(0, 10, 200)
y = np.sin(x) * np.exp(-x/10)
ax.plot(x, y, linewidth=2)
ax.set_title('Main Plot with Inset Zoom')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(ax, width='30%', height='30%', loc='upper right')
axins.plot(x, y, linewidth=2, color='red')
axins.set_xlim(0, 2)
axins.set_ylim(0, 1)
axins.set_title('Zoomed', fontsize=9)
plt.show()
```

## Colorbar Customization

```python
data = np.random.rand(10, 10)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(data, cmap='plasma')
cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
cbar.set_label('Intensity', fontsize=12)
cbar.set_ticks([0, 0.5, 1])
ax.set_title('Custom Colorbar')
plt.show()
```

## `constrained_layout` vs `tight_layout`

```python
fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
for ax in axes.flat:
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title('Constrained Layout Auto-Spaces')
plt.show()
```

## Creating a Custom Colormap

```python
from matplotlib.colors import LinearSegmentedColormap
colors_list = ['#2E86AB', '#F18F01', '#A23B72']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors_list)

data = np.random.rand(10, 10)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(data, cmap=custom_cmap)
plt.colorbar(im, ax=ax)
ax.set_title('Custom Colormap')
plt.show()
```

## Date Axis Formatting

```python
import matplotlib.dates as mdates
import datetime

dates = [datetime.date(2026, 1, 1) + datetime.timedelta(days=i) for i in range(30)]
values = np.random.randint(50, 150, 30)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dates, values, marker='o', markersize=3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator())
plt.xticks(rotation=45)
ax.set_title('Date-Formatted X-Axis')
plt.tight_layout()
plt.show()
```

## Shared Legend for Multiple Subplots

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
line1, = axes[0].plot([1,2,3], [1,4,9], label='Series A', color='blue')
line2, = axes[1].plot([1,2,3], [3,2,1], label='Series B', color='red')
fig.legend(handles=[line1, line2], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.05))
plt.tight_layout()
plt.show()
```

## Summary — Notebook 3

Basic/complex subplots, GridSpec, shared axes, plot styles, font control, spacing, inset axes (zoom), colorbar customization, constrained_layout, custom colormaps, date axis formatting, shared legends.

### Next: `04_advanced.ipynb` for 3D plots, contours, vector fields, and more.
