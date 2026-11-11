# 04 — Advanced Matplotlib

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
```

## 3D Line Plot

```python
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
t = np.linspace(0, 10, 100)
z = t
x = np.sin(t)
y = np.cos(t)
ax.plot(x, y, z, linewidth=2)
ax.set_title('3D Line')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
```

## 3D Surface

```python
X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('3D Surface')
plt.colorbar(surf, ax=ax)
plt.show()
```

## Contour Plot

```python
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contour(X, Y, Z, levels=15, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)
ax.set_title('Contour Plot')
plt.colorbar(contour, ax=ax)
plt.show()
```

## Quiver (Vector) Plot

```python
x = np.linspace(0, 5, 10)
y = np.linspace(0, 5, 10)
X, Y = np.meshgrid(x, y)
U = np.cos(X)
V = np.sin(Y)
fig, ax = plt.subplots(figsize=(8, 6))
ax.quiver(X, Y, U, V)
ax.set_title('Vector Field')
plt.show()
```

## Image Display

```python
img_data = np.random.rand(10, 10) * 255
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(img_data, cmap='gray')
ax.set_title('Image')
plt.colorbar(im, ax=ax)
plt.show()
```

## Error Bars

```python
x = [1, 2, 3, 4]
y = [2, 4, 6, 8]
errors = [0.2, 0.3, 0.25, 0.4]
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(x, y, yerr=errors, fmt='o-', capsize=5, capthick=2)
ax.set_title('With Error Bars')
plt.show()
```

## Filled Area

```python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(x, y1, y2, alpha=0.3, color='blue', label='Between sin & cos')
ax.plot(x, y1, label='sin(x)')
ax.plot(x, y2, label='cos(x)')
ax.legend()
plt.show()
```

## Stream Plot — Flow Field Visualization

```python
y, x = np.mgrid[-3:3:100j, -3:3:100j]
U = -1 - x**2 + y
V = 1 + x - y**2
fig, ax = plt.subplots(figsize=(8, 6))
strm = ax.streamplot(x, y, U, V, color=U, cmap='autumn', linewidth=1)
ax.set_title('Stream Plot — Flow Field')
plt.colorbar(strm.lines, ax=ax)
plt.show()
```

## Patches — Rectangle, Circle, Polygon

```python
import matplotlib.patches as patches
fig, ax = plt.subplots(figsize=(8, 6))
rect = patches.Rectangle((1, 1), 2, 3, linewidth=2, edgecolor='blue', facecolor='lightblue')
circle = patches.Circle((5, 5), 1.5, edgecolor='red', facecolor='pink')
polygon = patches.Polygon([[7, 1], [8, 3], [9, 1]], closed=True, edgecolor='green', facecolor='lightgreen')
ax.add_patch(rect)
ax.add_patch(circle)
ax.add_patch(polygon)
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_title('Custom Shapes: Rectangle, Circle, Polygon')
ax.set_aspect('equal')
plt.show()
```

## Filled Contour Plot — `contourf()`

```python
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))
fig, ax = plt.subplots(figsize=(8, 6))
cf = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu')
ax.set_title('Filled Contour Plot')
plt.colorbar(cf, ax=ax)
plt.show()
```

## Broken/Split Axis — Showing Data with a Gap

```python
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 3]})
x = np.arange(20)
y = np.concatenate([np.random.randint(1, 10, 5), np.random.randint(90, 100, 15)])
ax1.bar(x[:5], y[:5], color='steelblue')
ax2.bar(x[5:], y[5:], color='steelblue')
ax1.set_xlim(-1, 5)
ax2.set_xlim(4, 20)
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax1.yaxis.tick_left()
ax2.yaxis.tick_right()
fig.suptitle('Broken Axis Example')
plt.show()
```

## Table Embedded in a Plot

```python
fig, ax = plt.subplots(figsize=(8, 5))
months = ['Jan', 'Feb', 'Mar']
sales = [100, 150, 130]
ax.bar(months, sales, color='#2E86AB')
table_data = [[m, s] for m, s in zip(months, sales)]
ax.table(cellText=table_data, colLabels=['Month', 'Sales'], loc='bottom', bbox=[0, -0.35, 1, 0.2])
ax.set_xticks([])
ax.set_title('Bar Chart with Embedded Table')
plt.subplots_adjust(bottom=0.3)
plt.show()
```

## Basic Animation Setup (FuncAnimation)

Animations render as video/gif in a live Jupyter session — here we show the setup code and save a GIF to disk instead of displaying inline.

```python
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(7, 5))
x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x))
ax.set_ylim(-1.5, 1.5)
ax.set_title('Animated Sine Wave')

def update(frame):
    line.set_ydata(np.sin(x + frame * 0.1))
    return line,

ani = animation.FuncAnimation(fig, update, frames=30, interval=50, blit=True)
ani.save('/tmp/sine_animation.gif', writer='pillow', fps=20)
plt.close(fig)
print('Saved animation: /tmp/sine_animation.gif')
```

Output:
```text
Saved animation: /tmp/sine_animation.gif
```

## Summary — Notebook 4

3D line/surface plots, contour and filled contour, quiver (vector fields), stream plots, image display, error bars, filled areas, custom patches (Rectangle/Circle/Polygon), broken axes, embedded tables, and basic animation with `FuncAnimation`.

### Next: `05_real_world_weather_eda.ipynb` — the full applied project using a real 96,000-row weather dataset.
