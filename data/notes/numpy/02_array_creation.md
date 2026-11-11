# 02 — Array Creation in NumPy (Preetham's NumPy Mastery Series)

Part 2 of 5. This notebook covers **every method** to create NumPy arrays: from Python objects, using built-in generator functions, and using the random module.

## 1. Creating Arrays from Python Objects

```python
import numpy as np

# From a list
fruits_count = np.array([12, 5, 8, 20])
print('From list:', fruits_count)

# From a tuple
coordinates = np.array((10.5, 20.3, 30.7))
print('From tuple:', coordinates)

# From a list of lists (2D)
seating_chart = np.array([[1, 2, 3], [4, 5, 6]])
print('From nested list:\n', seating_chart)
```

Output:
```text
From list: [12  5  8 20]
From tuple: [10.5 20.3 30.7]
From nested list:
 [[1 2 3]
 [4 5 6]]
```

```python
# From a set (order not guaranteed, generally avoided but possible via list conversion)
unique_ids = {101, 102, 103, 104}
ids_array = np.array(list(unique_ids))
print('From set:', ids_array)
```

Output:
```text
From set: [104 101 102 103]
```

## 2. `np.zeros()` — Arrays Filled with Zero

```python
empty_cart = np.zeros(5)
print('1-D zeros:', empty_cart)

blank_grid = np.zeros((3, 4))
print('2-D zeros:\n', blank_grid)

zeros_int = np.zeros((2, 2), dtype='int')
print('Integer zeros:\n', zeros_int)
```

Output:
```text
1-D zeros: [0. 0. 0. 0. 0.]
2-D zeros:
 [[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
Integer zeros:
 [[0 0]
 [0 0]]
```

## 3. `np.ones()` — Arrays Filled with One

```python
full_attendance = np.ones(6)
print('1-D ones:', full_attendance)

weights_init = np.ones((3, 3))
print('2-D ones:\n', weights_init)
```

Output:
```text
1-D ones: [1. 1. 1. 1. 1. 1.]
2-D ones:
 [[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
```

## 4. `np.full()` — Arrays Filled with a Custom Value

```python
default_price = np.full((2, 3), 499)
print(default_price)

temperature_grid = np.full((3, 3), 36.6, dtype='float')
print(temperature_grid)
```

Output:
```text
[[499 499 499]
 [499 499 499]]
[[36.6 36.6 36.6]
 [36.6 36.6 36.6]
 [36.6 36.6 36.6]]
```

## 5. `np.empty()` — Uninitialized Array (Fast but Garbage Values)

```python
uninitialized = np.empty((2, 2))
print(uninitialized)
print('Note: values are whatever was already in memory — always overwrite before use!')
```

Output:
```text
[[1.68731927e-315 0.00000000e+000]
 [1.69376470e-315 1.69232148e-315]]
Note: values are whatever was already in memory — always overwrite before use!
```

## 6. `np.eye()` and `np.identity()` — Identity Matrices

```python
identity_5 = np.eye(5)
print('5x5 Identity:\n', identity_5)

shifted_eye = np.eye(5, 5, k=1)  # diagonal shifted by 1
print('\nShifted diagonal (k=1):\n', shifted_eye)

identity_alt = np.identity(4)
print('\nUsing identity():\n', identity_alt)
```

Output:
```text
5x5 Identity:
 [[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]

Shifted diagonal (k=1):
 [[0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0.]]

Using identity():
 [[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
```

## 7. `np.diag()` — Diagonal Matrices

```python
loan_interest_rates = np.diag([2.5, 3.0, 4.5, 5.0])
print(loan_interest_rates)
```

Output:
```text
[[2.5 0.  0.  0. ]
 [0.  3.  0.  0. ]
 [0.  0.  4.5 0. ]
 [0.  0.  0.  5. ]]
```

```python
# Extracting the diagonal from an existing matrix
matrix = np.array([[10, 1, 2], [3, 20, 4], [5, 6, 30]])
diagonal_values = np.diag(matrix)
print('Original matrix:\n', matrix)
print('Extracted diagonal:', diagonal_values)
```

Output:
```text
Original matrix:
 [[10  1  2]
 [ 3 20  4]
 [ 5  6 30]]
Extracted diagonal: [10 20 30]
```

## 8. `np.arange()` — Range-Based Arrays

```python
seq1 = np.arange(10)              # 0 to 9
seq2 = np.arange(5, 15)           # start=5, end=15 (exclusive)
seq3 = np.arange(0, 50, 5)        # start, end, step
seq4 = np.arange(10, 0, -2)       # negative step (countdown)

print('seq1:', seq1)
print('seq2:', seq2)
print('seq3:', seq3)
print('seq4:', seq4)
```

Output:
```text
seq1: [0 1 2 3 4 5 6 7 8 9]
seq2: [ 5  6  7  8  9 10 11 12 13 14]
seq3: [ 0  5 10 15 20 25 30 35 40 45]
seq4: [10  8  6  4  2]
```

## 9. `np.linspace()` — Evenly Spaced Values

```python
# 10 evenly spaced points between 0 and 100 (inclusive)
exam_percentile_bins = np.linspace(0, 100, 10)
print(exam_percentile_bins)

# Exclude the endpoint
angles = np.linspace(0, 360, 8, endpoint=False)
print('\nAngles (endpoint excluded):', angles)
```

Output:
```text
[  0.          11.11111111  22.22222222  33.33333333  44.44444444
  55.55555556  66.66666667  77.77777778  88.88888889 100.        ]

Angles (endpoint excluded): [  0.  45.  90. 135. 180. 225. 270. 315.]
```

## 10. `np.logspace()` — Logarithmically Spaced Values

```python
# 5 points between 10^1 and 10^4
frequency_bands = np.logspace(1, 4, 5)
print(frequency_bands)
```

Output:
```text
[   10.            56.23413252   316.22776602  1778.27941004
 10000.        ]
```

## 11. `np.fromfunction()` — Build Arrays from a Rule

```python
multiplication_table = np.fromfunction(lambda i, j: (i+1)*(j+1), (5, 5), dtype=int)
print(multiplication_table)
```

Output:
```text
[[ 1  2  3  4  5]
 [ 2  4  6  8 10]
 [ 3  6  9 12 15]
 [ 4  8 12 16 20]
 [ 5 10 15 20 25]]
```

## 12. `np.meshgrid()` — Coordinate Grids

```python
x = np.array([1, 2, 3])
y = np.array([10, 20])

X, Y = np.meshgrid(x, y)
print('X grid:\n', X)
print('Y grid:\n', Y)
```

Output:
```text
X grid:
 [[1 2 3]
 [1 2 3]]
Y grid:
 [[10 10 10]
 [20 20 20]]
```

## 13. Random Array Creation — `np.random` Module

```python
np.random.seed(42)  # reproducibility

# Uniform random floats in [0, 1)
dice_probabilities = np.random.random((2, 3))
print('random():\n', dice_probabilities)
```

Output:
```text
random():
 [[0.37454012 0.95071431 0.73199394]
 [0.59865848 0.15601864 0.15599452]]
```

```python
# Random integers between a range
lottery_numbers = np.random.randint(1, 100, size=6)
print('randint():', lottery_numbers)
```

Output:
```text
randint(): [75 75 88 24  3 22]
```

```python
# Standard normal distribution (mean=0, std=1)
sensor_noise = np.random.randn(4, 4)
print('randn():\n', sensor_noise)
```

Output:
```text
randn():
 [[ 0.07983181 -0.1595165   0.02222183 -0.42779291]
 [-0.53181741 -0.1174755   0.2220789  -0.7679765 ]
 [ 0.1424646  -0.03465218  1.13433927 -0.10474555]
 [-0.52512285  1.91277127 -2.02671962  1.11942361]]
```

```python
# Uniform distribution over [0, 1) using rand()
weights_random = np.random.rand(3, 3)
print('rand():\n', weights_random)
```

Output:
```text
rand():
 [[0.01596625 0.23089383 0.24102547]
 [0.68326352 0.60999666 0.83319491]
 [0.17336465 0.39106061 0.18223609]]
```

```python
# Random choice from an existing array (like picking a winner)
players = np.array(['Rahul', 'Aisha', 'Kabir', 'Meena', 'Zoya'])
winner = np.random.choice(players)
top_three = np.random.choice(players, size=3, replace=False)
print('Random winner:', winner)
print('Top three (no repeat):', top_three)
```

Output:
```text
Random winner: Meena
Top three (no repeat): ['Rahul' 'Kabir' 'Meena']
```

```python
# Shuffle an array in place
deck_positions = np.arange(1, 11)
np.random.shuffle(deck_positions)
print('Shuffled deck order:', deck_positions)
```

Output:
```text
Shuffled deck order: [ 1  3  7  9  8  2  5  6  4 10]
```

## 14. `np.copy()` at Creation Time — Ensuring Independence

```python
base_prices = np.array([100, 200, 300])
discounted_prices = base_prices.copy()
discounted_prices[0] = 50

print('Base prices unaffected:', base_prices)
print('Discounted (independent) copy:', discounted_prices)
```

Output:
```text
Base prices unaffected: [100 200 300]
Discounted (independent) copy: [ 50 200 300]
```

## 15. `np.tile()` and `np.repeat()` — Repeating Data

```python
pattern = np.array([1, 2, 3])

tiled = np.tile(pattern, 3)              # repeats the whole array
repeated = np.repeat(pattern, 3)         # repeats each element
tiled_2d = np.tile(pattern, (2, 1))      # tile into 2 rows

print('tile()  :', tiled)
print('repeat():', repeated)
print('tile 2D:\n', tiled_2d)
```

Output:
```text
tile()  : [1 2 3 1 2 3 1 2 3]
repeat(): [1 1 1 2 2 2 3 3 3]
tile 2D:
 [[1 2 3]
 [1 2 3]]
```

## 16. Structured Arrays — Rows with Named, Mixed-Type Fields

```python
employee_dtype = np.dtype([('name', 'U20'), ('age', 'i4'), ('salary', 'f8')])

employees = np.array([
    ('Rahul', 28, 55000.0),
    ('Aisha', 34, 72000.5),
    ('Kabir', 25, 48000.0)
], dtype=employee_dtype)

print(employees)
print('\nAll names  :', employees['name'])
print('All salaries:', employees['salary'])
print('First employee:', employees[0])
```

Output:
```text
[('Rahul', 28, 55000. ) ('Aisha', 34, 72000.5) ('Kabir', 25, 48000. )]

All names  : ['Rahul' 'Aisha' 'Kabir']
All salaries: [55000.  72000.5 48000. ]
First employee: ('Rahul', 28, 55000.0)
```

## 17. Record Arrays — Attribute-Style Access

```python
records = employees.view(np.recarray)
print('Access via attribute:', records.name)
print('Single record age   :', records[1].age)
```

Output:
```text
Access via attribute: ['Rahul' 'Aisha' 'Kabir']
Single record age   : 34
```

## 18. `datetime64` and `timedelta64` — Date Arrays

```python
order_dates = np.array(['2026-01-15', '2026-02-20', '2026-03-05'], dtype='datetime64')
print(order_dates)
print('dtype:', order_dates.dtype)

# Adding days
delivery_dates = order_dates + np.timedelta64(5, 'D')
print('\nDelivery dates (order + 5 days):', delivery_dates)

# Difference between dates
gap = order_dates[1] - order_dates[0]
print('\nGap between order 1 and 2:', gap)
```

Output:
```text
['2026-01-15' '2026-02-20' '2026-03-05']
dtype: datetime64[D]

Delivery dates (order + 5 days): ['2026-01-20' '2026-02-25' '2026-03-10']

Gap between order 1 and 2: 36 days
```

```python
# Generating a date range
week_range = np.arange('2026-07-01', '2026-07-08', dtype='datetime64[D]')
print(week_range)
```

Output:
```text
['2026-07-01' '2026-07-02' '2026-07-03' '2026-07-04' '2026-07-05'
 '2026-07-06' '2026-07-07']
```

## 19. `np.mgrid` and `np.ogrid` — Alternative Grid Creation

```python
dense_grid = np.mgrid[0:3, 0:3]
print('mgrid (dense):\n', dense_grid)

open_grid_x, open_grid_y = np.ogrid[0:3, 0:3]
print('\nogrid x (open, memory-efficient):\n', open_grid_x)
print('ogrid y (open, memory-efficient):\n', open_grid_y)
```

Output:
```text
mgrid (dense):
 [[[0 0 0]
  [1 1 1]
  [2 2 2]]

 [[0 1 2]
  [0 1 2]
  [0 1 2]]]

ogrid x (open, memory-efficient):
 [[0]
 [1]
 [2]]
ogrid y (open, memory-efficient):
 [[0 1 2]]
```

## 20. `np.vander()` — Vandermonde Matrix (useful for polynomial fitting)

```python
x_values = np.array([1, 2, 3, 4])
vander_matrix = np.vander(x_values, 3)
print(vander_matrix)
```

Output:
```text
[[ 1  1  1]
 [ 4  2  1]
 [ 9  3  1]
 [16  4  1]]
```

## 21. `np.geomspace()` — Geometrically Spaced Values

```python
geo_seq = np.geomspace(1, 1000, num=4)
print(geo_seq)
```

Output:
```text
[   1.   10.  100. 1000.]
```

## 22. Triangular Matrices — `np.triu()` and `np.tril()`

```python
square = np.arange(1, 17).reshape(4, 4)
print('Original:\n', square)
print('\nUpper triangle (triu):\n', np.triu(square))
print('\nLower triangle (tril):\n', np.tril(square))
```

Output:
```text
Original:
 [[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]]

Upper triangle (triu):
 [[ 1  2  3  4]
 [ 0  6  7  8]
 [ 0  0 11 12]
 [ 0  0  0 16]]

Lower triangle (tril):
 [[ 1  0  0  0]
 [ 5  6  0  0]
 [ 9 10 11  0]
 [13 14 15 16]]
```

## 23. Masked Arrays — Handling Missing/Invalid Data

```python
import numpy.ma as ma

sensor_readings = np.array([23.5, -999, 25.1, -999, 22.8])  # -999 = sensor error
masked_readings = ma.masked_equal(sensor_readings, -999)

print('Masked array   :', masked_readings)
print('Valid mean     :', masked_readings.mean())
print('Filled version :', masked_readings.filled(0))
```

Output:
```text
Masked array   : [23.5 -- 25.1 -- 22.8]
Valid mean     : 23.8
Filled version : [23.5  0.  25.1  0.  22.8]
```

## 24b. `zeros_like()`, `ones_like()`, `full_like()`, `empty_like()`

```python
template = np.array([[1, 2, 3], [4, 5, 6]])

print('zeros_like:\n', np.zeros_like(template))
print('\nones_like:\n', np.ones_like(template))
print('\nfull_like (fill=9):\n', np.full_like(template, 9))
```

Output:
```text
zeros_like:
 [[0 0 0]
 [0 0 0]]

ones_like:
 [[1 1 1]
 [1 1 1]]

full_like (fill=9):
 [[9 9 9]
 [9 9 9]]
```

## 24c. The Modern Random API — `np.random.default_rng()`

NumPy recommends the `Generator` API over the legacy `np.random.seed()` style for new code — better statistical properties and explicit reproducibility.

```python
rng = np.random.default_rng(seed=42)

print('Random floats  :', rng.random(4))
print('Random integers:', rng.integers(1, 100, 5))
print('Random normal  :', rng.normal(loc=0, scale=1, size=4))
```

Output:
```text
Random floats  : [0.77395605 0.43887844 0.85859792 0.69736803]
Random integers: [20 10 53 97 73]
Random normal  : [-0.31624259 -0.01680116 -0.85304393  0.87939797]
```

## 24d. Sampling from Different Distributions

```python
rng = np.random.default_rng(1)

print('Binomial (coin flips)     :', rng.binomial(n=10, p=0.5, size=5))
print('Poisson (event counts)    :', rng.poisson(lam=3, size=5))
print('Uniform (custom range)    :', rng.uniform(low=10, high=20, size=5))
print('Exponential (wait times)  :', rng.exponential(scale=2.0, size=5))
```

Output:
```text
Binomial (coin flips)     : [5 8 3 8 4]
Poisson (event counts)    : [4 4 2 2 6]
Uniform (custom range)    : [11.60652009 19.69925413 15.16068586 11.15865612 16.23489756]
Exponential (wait times)  : [3.80450291 0.63983452 3.60248986 0.09563025 2.71135013]
```

## 24e. `np.fromiter()` — Building an Array from a Generator/Iterator

```python
squares_gen = (x**2 for x in range(10))
squares_array = np.fromiter(squares_gen, dtype=int)
print(squares_array)
```

Output:
```text
[ 0  1  4  9 16 25 36 49 64 81]
```

## 24f. `np.indices()` — Generating Index Grids

```python
grid_indices = np.indices((3, 3))
print('Row indices:\n', grid_indices[0])
print('\nColumn indices:\n', grid_indices[1])
```

Output:
```text
Row indices:
 [[0 0 0]
 [1 1 1]
 [2 2 2]]

Column indices:
 [[0 1 2]
 [0 1 2]
 [0 1 2]]
```

## 24g. Saving and Loading Arrays — `.npy` and `.npz`

```python
sample_array = np.array([[1, 2, 3], [4, 5, 6]])

np.save('/tmp/my_array.npy', sample_array)
loaded = np.load('/tmp/my_array.npy')
print('Loaded back:\n', loaded)
print('Same data?', np.array_equal(sample_array, loaded))
```

Output:
```text
Loaded back:
 [[1 2 3]
 [4 5 6]]
Same data? True
```

```python
# Saving multiple arrays together with savez
arr_a = np.arange(5)
arr_b = np.arange(5, 10)
np.savez('/tmp/multi_arrays.npz', first=arr_a, second=arr_b)

loaded_bundle = np.load('/tmp/multi_arrays.npz')
print('Keys:', list(loaded_bundle.keys()))
print('first :', loaded_bundle['first'])
print('second:', loaded_bundle['second'])
```

Output:
```text
Keys: ['first', 'second']
first : [0 1 2 3 4]
second: [5 6 7 8 9]
```

## 24h. Saving/Loading Plain Text — `savetxt()` and `loadtxt()`

```python
text_data = np.array([[1.5, 2.5], [3.5, 4.5]])
np.savetxt('/tmp/data.csv', text_data, delimiter=',', header='colA,colB', comments='')

loaded_text = np.loadtxt('/tmp/data.csv', delimiter=',', skiprows=1)
print(loaded_text)
```

Output:
```text
[[1.5 2.5]
 [3.5 4.5]]
```

## 24i. `np.char` — Vectorized String Operations

```python
names = np.array(['rahul', 'AISHA', 'Kabir'])
print('Upper :', np.char.upper(names))
print('Title :', np.char.title(names))
print('Length:', np.char.str_len(names))
print('Starts with a:', np.char.startswith(np.char.lower(names), 'a'))
```

Output:
```text
Upper : ['RAHUL' 'AISHA' 'KABIR']
Title : ['Rahul' 'Aisha' 'Kabir']
Length: [5 5 5]
Starts with a: [False  True False]
```

## 25. Summary — What You Learned

| Function | Purpose |
|---|---|
| `np.array()` | Convert Python list/tuple/set to ndarray |
| `np.zeros()` | Array filled with 0 |
| `np.ones()` | Array filled with 1 |
| `np.full()` | Array filled with custom value |
| `np.empty()` | Uninitialized array (fast) |
| `np.eye()` / `np.identity()` | Identity matrix |
| `np.diag()` | Diagonal matrix / extract diagonal |
| `np.arange()` | Range-based sequence |
| `np.linspace()` | Evenly spaced values |
| `np.logspace()` | Log-spaced values |
| `np.fromfunction()` | Rule-based array generation |
| `np.meshgrid()` | Coordinate grid generation |
| `np.random.*` | Random array generation |

### Next Notebook
Continue to **`03_indexing_slicing.ipynb`** to master accessing and slicing array elements across 1D, 2D, and 3D arrays.
