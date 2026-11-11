# 04 — Reshaping & Stacking in NumPy (Preetham's NumPy Mastery Series)

Part 4 of 5. This notebook covers `reshape`, `ravel`, `flatten`, `transpose`, `swapaxes`, `squeeze`, `expand_dims`, joining (`concatenate`, `stack`, `hstack`, `vstack`, `dstack`) and splitting (`split`, `array_split`, `hsplit`, `vsplit`, `dsplit`).

## 1. `reshape()` — Changing Array Shape Without Changing Data

```python
import numpy as np

# 12 daily temperature readings
temps = np.array([21, 23, 19, 25, 22, 24, 20, 26, 23, 21, 27, 22])
print('Original shape:', temps.shape)

weekly_view = temps.reshape(3, 4)   # 3 weeks x 4 days
print('\nReshaped to (3,4):\n', weekly_view)

column_view = temps.reshape(12, 1)
print('\nReshaped to (12,1):\n', column_view)
```

Output:
```text
Original shape: (12,)

Reshaped to (3,4):
 [[21 23 19 25]
 [22 24 20 26]
 [23 21 27 22]]

Reshaped to (12,1):
 [[21]
 [23]
 [19]
 [25]
 [22]
 [24]
 [20]
 [26]
 [23]
 [21]
 [27]
 [22]]
```

```python
# Reshape into 3-D: 2 blocks x 2 rows x 3 columns
cubed = temps.reshape(2, 2, 3)
print(cubed)
```

Output:
```text
[[[21 23 19]
  [25 22 24]]

 [[20 26 23]
  [21 27 22]]]
```

## 2. Using `-1` for Unknown Dimension

```python
sensor_data = np.arange(24)

auto_reshape = sensor_data.reshape(4, -1)   # numpy figures out columns
print('Auto columns (4, -1):\n', auto_reshape)

auto_reshape2 = sensor_data.reshape(-1, 6)  # numpy figures out rows
print('\nAuto rows (-1, 6):\n', auto_reshape2)
```

Output:
```text
Auto columns (4, -1):
 [[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]]

Auto rows (-1, 6):
 [[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]]
```

## 3. `ravel()` vs `flatten()` — Flattening to 1-D

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])

raveled = matrix.ravel()     # returns a VIEW (fast, shares memory)
flattened = matrix.flatten() # returns a COPY (safe, independent)

print('ravel():  ', raveled)
print('flatten():', flattened)

# Proof of the difference
raveled[0] = 999
print('\nAfter modifying raveled[0], original matrix also changed:\n', matrix)
```

Output:
```text
ravel():   [1 2 3 4 5 6]
flatten(): [1 2 3 4 5 6]

After modifying raveled[0], original matrix also changed:
 [[999   2   3]
 [  4   5   6]]
```

## 4. `transpose()` and `.T` — Flipping Rows and Columns

```python
sales_by_region = np.array([
    [100, 200, 300],   # North
    [150, 250, 350],   # South
    [120, 220, 320]     # East
])
print('Original (regions x months):\n', sales_by_region)

transposed = sales_by_region.T
print('\nTransposed (months x regions):\n', transposed)

# Same effect using transpose() function
print('\nUsing np.transpose():\n', np.transpose(sales_by_region))
```

Output:
```text
Original (regions x months):
 [[100 200 300]
 [150 250 350]
 [120 220 320]]

Transposed (months x regions):
 [[100 150 120]
 [200 250 220]
 [300 350 320]]

Using np.transpose():
 [[100 150 120]
 [200 250 220]
 [300 350 320]]
```

## 5. `swapaxes()` — Swapping Any Two Axes

```python
cube = np.arange(24).reshape(2, 3, 4)
print('Original shape:', cube.shape)

swapped = cube.swapaxes(0, 2)
print('Swapped axes (0,2) shape:', swapped.shape)
```

Output:
```text
Original shape: (2, 3, 4)
Swapped axes (0,2) shape: (4, 3, 2)
```

## 6. `squeeze()` — Removing Single-Dimension Axes

```python
padded = np.array([[[1, 2, 3]]])
print('Before squeeze, shape:', padded.shape)

squeezed = padded.squeeze()
print('After squeeze, shape :', squeezed.shape)
print(squeezed)
```

Output:
```text
Before squeeze, shape: (1, 1, 3)
After squeeze, shape : (3,)
[1 2 3]
```

## 7. `expand_dims()` — Adding a New Axis

```python
single_row = np.array([10, 20, 30])
print('Original shape:', single_row.shape)

expanded_0 = np.expand_dims(single_row, axis=0)
expanded_1 = np.expand_dims(single_row, axis=1)

print('expand_dims axis=0, shape:', expanded_0.shape)
print('expand_dims axis=1, shape:', expanded_1.shape)
```

Output:
```text
Original shape: (3,)
expand_dims axis=0, shape: (1, 3)
expand_dims axis=1, shape: (3, 1)
```

## 8. Joining Arrays — `concatenate()`

```python
team_a_scores = np.array([80, 85, 90])
team_b_scores = np.array([70, 75, 95])

combined = np.concatenate((team_a_scores, team_b_scores))
print('Combined 1-D:', combined)
```

Output:
```text
Combined 1-D: [80 85 90 70 75 95]
```

```python
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])

concat_rows = np.concatenate((matrix_a, matrix_b), axis=0)
concat_cols = np.concatenate((matrix_a, matrix_b), axis=1)

print('Concatenate along rows (axis=0):\n', concat_rows)
print('\nConcatenate along columns (axis=1):\n', concat_cols)
```

Output:
```text
Concatenate along rows (axis=0):
 [[1 2]
 [3 4]
 [5 6]
 [7 8]]

Concatenate along columns (axis=1):
 [[1 2 5 6]
 [3 4 7 8]]
```

## 9. `stack()` — Joining Along a New Axis

```python
day1_prices = np.array([100, 200, 300])
day2_prices = np.array([110, 190, 310])

stacked_axis0 = np.stack((day1_prices, day2_prices), axis=0)
stacked_axis1 = np.stack((day1_prices, day2_prices), axis=1)

print('Stacked axis=0:\n', stacked_axis0)
print('\nStacked axis=1:\n', stacked_axis1)
```

Output:
```text
Stacked axis=0:
 [[100 200 300]
 [110 190 310]]

Stacked axis=1:
 [[100 110]
 [200 190]
 [300 310]]
```

## 10. `hstack()`, `vstack()`, `dstack()`

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print('hstack (horizontal):', np.hstack((a, b)))
print('vstack (vertical):\n', np.vstack((a, b)))
print('dstack (depth-wise):\n', np.dstack((a, b)))
```

Output:
```text
hstack (horizontal): [1 2 3 4 5 6]
vstack (vertical):
 [[1 2 3]
 [4 5 6]]
dstack (depth-wise):
 [[[1 4]
  [2 5]
  [3 6]]]
```

## 11. Splitting Arrays — `split()` and `array_split()`

```python
class_scores = np.array([55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 45, 50])

equal_groups = np.split(class_scores, 4)   # must divide evenly
print('Equal split into 4 groups:')
for grp in equal_groups:
    print(' ', grp)
```

Output:
```text
Equal split into 4 groups:
  [55 60 65]
  [70 75 80]
  [85 90 95]
  [100  45  50]
```

```python
uneven_data = np.array([1, 2, 3, 4, 5, 6, 7])
uneven_groups = np.array_split(uneven_data, 3)  # handles uneven sizes
print('Uneven split into 3 groups:')
for grp in uneven_groups:
    print(' ', grp)
```

Output:
```text
Uneven split into 3 groups:
  [1 2 3]
  [4 5]
  [6 7]
```

## 12. `hsplit()`, `vsplit()`, `dsplit()`

```python
grid = np.arange(16).reshape(4, 4)
print('Original grid:\n', grid)

h_parts = np.hsplit(grid, 2)
print('\nhsplit into 2 (column-wise):')
for p in h_parts:
    print(p)

v_parts = np.vsplit(grid, 2)
print('\nvsplit into 2 (row-wise):')
for p in v_parts:
    print(p)
```

Output:
```text
Original grid:
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]

hsplit into 2 (column-wise):
[[ 0  1]
 [ 4  5]
 [ 8  9]
 [12 13]]
[[ 2  3]
 [ 6  7]
 [10 11]
 [14 15]]

vsplit into 2 (row-wise):
[[0 1 2 3]
 [4 5 6 7]]
[[ 8  9 10 11]
 [12 13 14 15]]
```

## 13. `resize()` — Reshape That Can Change Total Size

```python
small = np.array([1, 2, 3])
resized = np.resize(small, (2, 4))  # repeats values to fill new shape
print(resized)
```

Output:
```text
[[1 2 3 1]
 [2 3 1 2]]
```

## 14. Real-World Mini Task

Given a flat list of 30 days of temperature readings, reshape it into a (5 weeks x 6 days) grid, then find the transpose to view it as (6 days x 5 weeks).

```python
np.random.seed(1)
daily_temps = np.random.randint(15, 40, size=30)
print('Flat readings:', daily_temps)

weekly_grid = daily_temps.reshape(5, 6)
print('\n5 weeks x 6 days:\n', weekly_grid)

day_first_view = weekly_grid.T
print('\nTransposed (6 days x 5 weeks):\n', day_first_view)
```

Output:
```text
Flat readings: [20 26 27 23 24 26 20 30 15 31 16 27 22 28 21 33 35 20 33 35 26 25 29 33
 19 38 38 24 32 38]

5 weeks x 6 days:
 [[20 26 27 23 24 26]
 [20 30 15 31 16 27]
 [22 28 21 33 35 20]
 [33 35 26 25 29 33]
 [19 38 38 24 32 38]]

Transposed (6 days x 5 weeks):
 [[20 20 22 33 19]
 [26 30 28 35 38]
 [27 15 21 26 38]
 [23 31 33 25 24]
 [24 16 35 29 32]
 [26 27 20 33 38]]
```

## 15. Broadcasting Rules — The Deep Dive

Two arrays are broadcast-compatible if, comparing shapes from the **right**, each pair of dimensions is either equal, or one of them is 1.

```python
a = np.ones((3, 1))       # shape (3,1)
b = np.ones((1, 4))       # shape (1,4)
result = a + b             # broadcasts to shape (3,4)
print('a shape:', a.shape, '| b shape:', b.shape, '| result shape:', result.shape)
print(result)
```

Output:
```text
a shape: (3, 1) | b shape: (1, 4) | result shape: (3, 4)
[[2. 2. 2. 2.]
 [2. 2. 2. 2.]
 [2. 2. 2. 2.]]
```

```python
# Incompatible shapes raise an error — demonstrating the rule
try:
    bad = np.ones((3, 4)) + np.ones((2, 5))
except ValueError as e:
    print('Broadcasting error (expected):', e)
```

Output:
```text
Broadcasting error (expected): operands could not be broadcast together with shapes (3,4) (2,5)
```

## 16. `np.moveaxis()` and `np.rollaxis()` — Relocating Axes

```python
video_frames = np.zeros((10, 64, 64, 3))  # (frames, height, width, channels)
print('Original shape:', video_frames.shape)

channels_first = np.moveaxis(video_frames, -1, 1)  # move channels to axis 1
print('Channels-first shape:', channels_first.shape)
```

Output:
```text
Original shape: (10, 64, 64, 3)
Channels-first shape: (10, 3, 64, 64)
```

## 17. `np.flip()`, `fliplr()`, `flipud()` — Reversing Arrays

```python
photo_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print('Original:\n', photo_matrix)
print('\nflipud (upside down):\n', np.flipud(photo_matrix))
print('\nfliplr (left-right mirror):\n', np.fliplr(photo_matrix))
print('\nflip both axes:\n', np.flip(photo_matrix))
```

Output:
```text
Original:
 [[1 2 3]
 [4 5 6]
 [7 8 9]]

flipud (upside down):
 [[7 8 9]
 [4 5 6]
 [1 2 3]]

fliplr (left-right mirror):
 [[3 2 1]
 [6 5 4]
 [9 8 7]]

flip both axes:
 [[9 8 7]
 [6 5 4]
 [3 2 1]]
```

## 18. `np.roll()` — Circular Shifting of Elements

```python
playlist_queue = np.array(['Song1', 'Song2', 'Song3', 'Song4', 'Song5'])
shifted = np.roll(playlist_queue, 2)
print('Original:', playlist_queue)
print('Rolled by 2:', shifted)
```

Output:
```text
Original: ['Song1' 'Song2' 'Song3' 'Song4' 'Song5']
Rolled by 2: ['Song4' 'Song5' 'Song1' 'Song2' 'Song3']
```

## 19. `np.pad()` — Adding Borders/Padding

```python
small_image = np.array([[1, 2], [3, 4]])
padded = np.pad(small_image, pad_width=1, mode='constant', constant_values=0)
print(padded)
```

Output:
```text
[[0 0 0 0]
 [0 1 2 0]
 [0 3 4 0]
 [0 0 0 0]]
```

## 20. `np.block()` — Assembling Arrays from Sub-Blocks

```python
top_left = np.ones((2, 2))
top_right = np.zeros((2, 2))
bottom = np.full((1, 4), 9)

assembled = np.block([[top_left, top_right], [bottom]])
print(assembled)
```

Output:
```text
[[1. 1. 0. 0.]
 [1. 1. 0. 0.]
 [9. 9. 9. 9.]]
```

## 21. `np.atleast_1d/2d/3d()` — Guaranteeing Minimum Dimensions

```python
scalar_val = 5
one_d = np.atleast_1d(scalar_val)
two_d = np.atleast_2d(scalar_val)
three_d = np.atleast_3d(scalar_val)

print('1D:', one_d, one_d.shape)
print('2D:', two_d, two_d.shape)
print('3D:', three_d, three_d.shape)
```

Output:
```text
1D: [5] (1,)
2D: [[5]] (1, 1)
3D: [[[5]]] (1, 1, 1)
```

## 22. `np.repeat()` vs `np.tile()` in Reshaping Context (Comparison)

```python
base = np.array([[1, 2], [3, 4]])
print('Original:\n', base)
print('\nnp.tile (whole array repeated):\n', np.tile(base, (2, 1)))
print('\nnp.repeat (each row repeated):\n', np.repeat(base, 2, axis=0))
```

Output:
```text
Original:
 [[1 2]
 [3 4]]

np.tile (whole array repeated):
 [[1 2]
 [3 4]
 [1 2]
 [3 4]]

np.repeat (each row repeated):
 [[1 2]
 [1 2]
 [3 4]
 [3 4]]
```

## 23. `np.dstack` + `np.stack` Round Trip Example (Image Channels)

```python
red_channel = np.full((2, 2), 255)
green_channel = np.full((2, 2), 128)
blue_channel = np.full((2, 2), 0)

rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
print('RGB image shape:', rgb_image.shape)
print(rgb_image)
```

Output:
```text
RGB image shape: (2, 2, 3)
[[[255 128   0]
  [255 128   0]]

 [[255 128   0]
  [255 128   0]]]
```

## 25b. `reshape()` with `order='F'` vs `order='C'`

```python
flat = np.arange(1, 7)
print('C order (row-major, default):\n', flat.reshape(2, 3, order='C'))
print('\nF order (column-major):\n', flat.reshape(2, 3, order='F'))
```

Output:
```text
C order (row-major, default):
 [[1 2 3]
 [4 5 6]]

F order (column-major):
 [[1 3 5]
 [2 4 6]]
```

## 25c. `np.column_stack()` and `np.row_stack()`

```python
col1 = np.array([1, 2, 3])
col2 = np.array([4, 5, 6])

print('column_stack:\n', np.column_stack((col1, col2)))
print('\nrow_stack (same as vstack):\n', np.vstack((col1, col2)))
```

Output:
```text
column_stack:
 [[1 4]
 [2 5]
 [3 6]]

row_stack (same as vstack):
 [[1 2 3]
 [4 5 6]]
```

## 25d. `np.append()` — Adding Elements (Returns a New Array)

```python
base = np.array([1, 2, 3])
appended = np.append(base, [4, 5])
print('Appended:', appended)

# Appending to a 2-D array along an axis
matrix = np.array([[1, 2], [3, 4]])
appended_row = np.append(matrix, [[5, 6]], axis=0)
print('\nWith new row:\n', appended_row)
```

Output:
```text
Appended: [1 2 3 4 5]

With new row:
 [[1 2]
 [3 4]
 [5 6]]
```

## 25e. `np.insert()` and `np.delete()`

```python
arr = np.array([10, 20, 30, 40])
inserted = np.insert(arr, 2, 999)
print('After insert at index 2:', inserted)

deleted = np.delete(arr, 1)
print('After deleting index 1:', deleted)
```

Output:
```text
After insert at index 2: [ 10  20 999  30  40]
After deleting index 1: [10 30 40]
```

## 25f. `np.trim_zeros()` — Stripping Leading/Trailing Zeros

```python
padded_signal = np.array([0, 0, 3, 5, 8, 0, 0, 0])
trimmed = np.trim_zeros(padded_signal)
print('Original:', padded_signal)
print('Trimmed :', trimmed)
```

Output:
```text
Original: [0 0 3 5 8 0 0 0]
Trimmed : [3 5 8]
```

## 25g. `np.kron()` — Kronecker Product

```python
a = np.array([1, 0])
b = np.array([1, 1])
print(np.kron(a, b))
```

Output:
```text
[1 1 0 0]
```

## 25h. `np.outer()` — Outer Product

```python
v1 = np.array([1, 2, 3])
v2 = np.array([10, 20])
print(np.outer(v1, v2))
```

Output:
```text
[[10 20]
 [20 40]
 [30 60]]
```

## 25i. Real-World Mini Task — Rebuilding a Damaged Image Grid

```python
np.random.seed(9)
damaged_pixels = np.random.randint(0, 256, size=(4, 4))
print('Original 4x4:\n', damaged_pixels)

# Pad with a 1-pixel border, then remove the noisy first row
padded = np.pad(damaged_pixels, pad_width=1, mode='edge')
cleaned = np.delete(padded, 0, axis=0)
print('\nPadded then first row removed, shape:', cleaned.shape)
```

Output:
```text
Original 4x4:
 [[126  92 126 245]
 [182  56 251 150]
 [125  65  22 180]
 [ 59  40 250 241]]

Padded then first row removed, shape: (5, 6)
```

## 26. Summary — What You Learned

| Function | Purpose |
|---|---|
| `reshape()` | Change shape, same data |
| `ravel()` | Flatten to 1D (view) |
| `flatten()` | Flatten to 1D (copy) |
| `transpose()` / `.T` | Swap rows & columns |
| `swapaxes()` | Swap any two axes |
| `squeeze()` | Remove size-1 axes |
| `expand_dims()` | Add a new axis |
| `concatenate()` | Join along existing axis |
| `stack()` | Join along a new axis |
| `hstack/vstack/dstack` | Directional joins |
| `split/array_split` | Divide into equal/uneven parts |
| `hsplit/vsplit/dsplit` | Directional splits |
| `resize()` | Reshape with size change |

### Next Notebook
Continue to **`05_searching_sorting_filtering.ipynb`** — the final notebook covering `where`, `sort`, `argsort`, `unique`, and boolean filtering.
