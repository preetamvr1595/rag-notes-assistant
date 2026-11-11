# 01 — NumPy Basics (Preetham's NumPy Mastery Series)

This notebook is Part 1 of a 5-part end-to-end NumPy series:
1. **NumPy Basics** (this notebook)
2. Array Creation
3. Indexing & Slicing
4. Reshaping & Stacking
5. Searching, Sorting & Filtering

By the end of this notebook you will understand what NumPy is, why it exists, how it compares to Python lists, and all the core attributes of an `ndarray`.

## 1. What is NumPy?

**NumPy** stands for **Numerical Python**. It is the fundamental package for scientific and numerical computing in Python.

- Created by **Travis Oliphant** in **2005**, combining the earlier `Numeric` and `Numarray` packages.
- Core object: the **`ndarray`** (N-dimensional array).
- Internally implemented in **C**, which makes it dramatically faster than native Python lists for numerical work.
- Forms the base layer for Pandas, Scikit-learn, TensorFlow, PyTorch, SciPy, and Matplotlib.

Official source code: https://github.com/numpy/numpy

## 2. Installing & Importing NumPy

Run this in your terminal (not inside the notebook, no `!` needed there):
```
pip install numpy
```
Inside a notebook or script, you can use `!` to run shell commands directly.

```python
# Install (safe to run inside notebook/colab too)
!pip install numpy --quiet
```

Output:
```text
[1;31merror[0m: [1mexternally-managed-environment[0m

[31m×[0m This environment is externally managed
[31m╰─>[0m To install Python packages system-wide, try apt install
[31m   [0m python3-xyz, where xyz is the package you are trying to
[31m   [0m install.
[31m   [0m 
[31m   [0m If you wish to install a non-Debian-packaged Python package,
[31m   [0m create a virtual environment using python3 -m venv path/to/venv.
[31m   [0m Then use path/to/venv/bin/python and path/to …
```

```python
import numpy as np
print('NumPy version:', np.__version__)
```

Output:
```text
NumPy version: 2.4.4
```

## 3. Why NumPy instead of Python Lists?

| Feature | Python List | NumPy Array |
|---|---|---|
| Speed | Slow (loops in pure Python) | Very fast (vectorized C loops) |
| Memory | Stores pointers to objects | Stores raw values contiguously |
| Type | Can mix types | Homogeneous (single dtype) |
| Operations | Needs manual loops | Element-wise built in |

Let's prove the speed difference ourselves with a timing experiment.

```python
import time

size = 2_000_000

# Pure Python list approach
py_list1 = list(range(size))
py_list2 = list(range(size))

start = time.time()
py_result = [py_list1[i] + py_list2[i] for i in range(size)]
py_time = time.time() - start

# NumPy array approach
np_arr1 = np.arange(size)
np_arr2 = np.arange(size)

start = time.time()
np_result = np_arr1 + np_arr2
np_time = time.time() - start

print(f'Pure Python time : {py_time:.4f} sec')
print(f'NumPy time       : {np_time:.4f} sec')
print(f'NumPy is roughly {py_time/np_time:.1f}x faster')
```

Output:
```text
Pure Python time : 0.1264 sec
NumPy time       : 0.0221 sec
NumPy is roughly 5.7x faster
```

## 4. Your First NumPy Array

```python
marks = np.array([88, 92, 79, 95, 60])
print(marks)
print(type(marks))
```

Output:
```text
[88 92 79 95 60]
<class 'numpy.ndarray'>
```

## 5. `ndarray` — The Core Object

Every NumPy array is an instance of the `ndarray` class. Let's explore every important attribute using a fresh, unique example: a small **student score sheet**.

```python
scores = np.array([
    [85, 90, 78],   # Student 1: Math, Science, English
    [70, 88, 95],   # Student 2
    [60, 75, 80],   # Student 3
    [99, 100, 97]   # Student 4
])
print(scores)
```

Output:
```text
[[ 85  90  78]
 [ 70  88  95]
 [ 60  75  80]
 [ 99 100  97]]
```

```python
print('ndim  (dimensions)  :', scores.ndim)
print('shape (rows, cols)   :', scores.shape)
print('size  (total elements):', scores.size)
print('dtype (data type)    :', scores.dtype)
print('itemsize (bytes/elem):', scores.itemsize)
print('nbytes (total bytes) :', scores.nbytes)
```

Output:
```text
ndim  (dimensions)  : 2
shape (rows, cols)   : (4, 3)
size  (total elements): 12
dtype (data type)    : int64
itemsize (bytes/elem): 8
nbytes (total bytes) : 96
```

## 6. Dimensions Explained with Unique Examples

```python
# 0-D array : a single scalar wrapped as ndarray
temperature_today = np.array(37)
print('0-D array:', temperature_today)
print('Dimensions:', temperature_today.ndim)
```

Output:
```text
0-D array: 37
Dimensions: 0
```

```python
# 1-D array : a simple list of weekly sales
weekly_sales = np.array([1200, 1500, 900, 1700, 2100, 1300, 1000])
print('1-D array:', weekly_sales)
print('Dimensions:', weekly_sales.ndim)
```

Output:
```text
1-D array: [1200 1500  900 1700 2100 1300 1000]
Dimensions: 1
```

```python
# 2-D array : attendance of 3 employees across 4 weeks
attendance = np.array([
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 1]
])
print('2-D array:\n', attendance)
print('Dimensions:', attendance.ndim)
```

Output:
```text
2-D array:
 [[1 1 0 1]
 [1 0 1 1]
 [0 1 1 1]]
Dimensions: 2
```

```python
# 3-D array : RGB-like tiny image data (2 images, 2x2 pixels, 3 channels)
image_batch = np.array([
    [[255, 0, 0], [0, 255, 0]],
    [[0, 0, 255], [255, 255, 0]]
])
print('3-D array:\n', image_batch)
print('Dimensions:', image_batch.ndim)
print('Shape:', image_batch.shape)
```

Output:
```text
3-D array:
 [[[255   0   0]
  [  0 255   0]]

 [[  0   0 255]
  [255 255   0]]]
Dimensions: 3
Shape: (2, 2, 3)
```

```python
# N-Dimensional array using ndmin
forced_5d = np.array([10, 20, 30], ndmin=5)
print(forced_5d)
print('Dimensions:', forced_5d.ndim)
```

Output:
```text
[[[[[10 20 30]]]]]
Dimensions: 5
```

## 7. Data Types (`dtype`) in NumPy

| Character Code | Meaning |
|---|---|
| i | integer |
| b | boolean |
| u | unsigned integer |
| f | float |
| c | complex float |
| m | timedelta |
| M | datetime |
| O | object |
| S | byte string |
| U | unicode string |
| V | void (fixed memory chunk) |

```python
int_arr = np.array([1, 2, 3])
float_arr = np.array([1.5, 2.5, 3.5])
bool_arr = np.array([True, False, True])
str_arr = np.array(['red', 'green', 'blue'])
complex_arr = np.array([1+2j, 3+4j])

for name, arr in [('int', int_arr), ('float', float_arr), ('bool', bool_arr),
                   ('str', str_arr), ('complex', complex_arr)]:
    print(f'{name:8s} -> dtype: {arr.dtype}')
```

Output:
```text
int      -> dtype: int64
float    -> dtype: float64
bool     -> dtype: bool
str      -> dtype: <U5
complex  -> dtype: complex128
```

```python
# Explicitly setting a dtype at creation time
prices = np.array([199, 299, 499], dtype='float32')
print(prices, prices.dtype)
```

Output:
```text
[199. 299. 499.] float32
```

```python
# Type conversion using astype()
temps_celsius = np.array([36.6, 37.2, 38.9, 39.5])
temps_rounded = temps_celsius.astype('int')
print('Original :', temps_celsius)
print('Converted:', temps_rounded, temps_rounded.dtype)
```

Output:
```text
Original : [36.6 37.2 38.9 39.5]
Converted: [36 37 38 39] int64
```

## 8. Array vs List — Mutability Behaviour

```python
# Lists can hold mixed types and grow dynamically; arrays cannot mix types easily
mixed_list = [1, 'two', 3.0, True]
print('List with mixed types:', mixed_list)

# NumPy will upcast everything to a common type (here, string)
mixed_array = np.array([1, 'two', 3.0, True])
print('NumPy forces common dtype:', mixed_array, mixed_array.dtype)
```

Output:
```text
List with mixed types: [1, 'two', 3.0, True]
NumPy forces common dtype: ['1' 'two' '3.0' 'True'] <U32
```

## 9. Checking Memory Efficiency

```python
import sys

py_list = list(range(1000))
np_array = np.arange(1000)

print('Python list size in memory (approx):', sys.getsizeof(py_list) + sum(sys.getsizeof(x) for x in py_list[:5])*200, 'bytes (rough)')
print('NumPy array size in memory          :', np_array.nbytes, 'bytes (exact)')
```

Output:
```text
Python list size in memory (approx): 36056 bytes (rough)
NumPy array size in memory          : 8000 bytes (exact)
```

## 10. Memory Layout — Strides & Contiguity

Every ndarray stores its data as one continuous memory block. **Strides** tell NumPy how many bytes to jump to move to the next element along each axis.

```python
grid = np.arange(12).reshape(3, 4)
print(grid)
print('Strides (bytes to jump per axis):', grid.strides)
print('Is C-contiguous (row-major)?    :', grid.flags['C_CONTIGUOUS'])
print('Is F-contiguous (column-major)? :', grid.flags['F_CONTIGUOUS'])
```

Output:
```text
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
Strides (bytes to jump per axis): (32, 8)
Is C-contiguous (row-major)?    : True
Is F-contiguous (column-major)? : False
```

```python
# Transposing does NOT copy data, it just changes the strides!
transposed = grid.T
print('Transposed strides:', transposed.strides)
print('Shares memory with original:', np.shares_memory(grid, transposed))
```

Output:
```text
Transposed strides: (8, 32)
Shares memory with original: True
```

## 11. `np.info()` — Built-in Documentation Lookup

```python
np.info(np.mean)
```

Output:
```text
Compute the arithmetic mean along the specified axis.

Returns the average of the array elements.  The average is taken over
the flattened array by default, otherwise over the specified axis.
`float64` intermediate and return values are used for integer inputs.

Parameters
----------
a : array_like
    Array containing numbers whose mean is desired. If `a` is not an
    array, a conversion is attempted.
axis : None or int or tuple of ints, optional
    Axis or axes along which the means are comp …
```

## 12. Universal Constants in NumPy

```python
print('pi   :', np.pi)
print('e    :', np.e)
print('inf  :', np.inf)
print('-inf :', -np.inf)
print('nan  :', np.nan)
print('Is nan == nan?', np.nan == np.nan)  # famous NumPy gotcha
print('Use np.isnan() instead:', np.isnan(np.nan))
```

Output:
```text
pi   : 3.141592653589793
e    : 2.718281828459045
inf  : inf
-inf : -inf
nan  : nan
Is nan == nan? False
Use np.isnan() instead: True
```

## 13. Comparing Arrays Correctly

```python
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
c = np.array([1, 2, 4])

print('a == b (element-wise):', a == b)
print('np.array_equal(a, b) :', np.array_equal(a, b))
print('np.array_equal(a, c) :', np.array_equal(a, c))
print('a is b (identity, not equality):', a is b)
```

Output:
```text
a == b (element-wise): [ True  True  True]
np.array_equal(a, b) : True
np.array_equal(a, c) : False
a is b (identity, not equality): False
```

## 14. Universal Functions (ufuncs) — The Building Blocks of Speed

```python
prices = np.array([100, 250, 400, 999])

print('sqrt   :', np.sqrt(prices))
print('log    :', np.log(prices))
print('exp    :', np.exp(np.array([1, 2, 3])))
print('abs    :', np.abs(np.array([-5, 3, -8, 2])))
print('power  :', np.power(prices, 2))
print('mod    :', np.mod(prices, 7))
```

Output:
```text
sqrt   : [10.         15.8113883  20.         31.60696126]
log    : [4.60517019 5.52146092 5.99146455 6.90675478]
exp    : [ 2.71828183  7.3890561  20.08553692]
abs    : [5 3 8 2]
power  : [ 10000  62500 160000 998001]
mod    : [2 5 1 5]
```

## 15. `np.vectorize()` — Turning a Python Function into a ufunc

```python
def grade_letter(score):
    if score >= 90: return 'A'
    elif score >= 75: return 'B'
    elif score >= 60: return 'C'
    else: return 'D'

vectorized_grader = np.vectorize(grade_letter)
student_scores = np.array([95, 72, 58, 88, 63])
print(vectorized_grader(student_scores))
```

Output:
```text
['A' 'C' 'D' 'B' 'C']
```

## 16. Broadcasting — The Core Speed Trick (Introductory)

Broadcasting lets NumPy operate on arrays of **different shapes** without writing explicit loops, by virtually 'stretching' the smaller array.

```python
prices = np.array([100, 200, 300])
gst_rate = 0.18  # a single scalar

# The scalar is broadcast across every element automatically
price_with_gst = prices + (prices * gst_rate)
print(price_with_gst)
```

Output:
```text
[118. 236. 354.]
```

```python
# Broadcasting a 1-D array against a 2-D array
monthly_bills = np.array([
    [1200, 800, 300],
    [1100, 750, 280],
    [1300, 900, 320]
])
discount_per_category = np.array([100, 50, 20])  # shape (3,)

final_bills = monthly_bills - discount_per_category  # broadcast across each row
print(final_bills)
```

Output:
```text
[[1100  750  280]
 [1000  700  260]
 [1200  850  300]]
```

## 17. Aggregate Functions — `sum`, `mean`, `std`, `var`, `min`, `max`, `median`

```python
gym_weights_kg = np.array([65, 70, 55, 80, 60, 75])

print('Sum   :', gym_weights_kg.sum())
print('Mean  :', gym_weights_kg.mean())
print('Std   :', gym_weights_kg.std())
print('Var   :', gym_weights_kg.var())
print('Min   :', gym_weights_kg.min())
print('Max   :', gym_weights_kg.max())
print('Median:', np.median(gym_weights_kg))
```

Output:
```text
Sum   : 405
Mean  : 67.5
Std   : 8.539125638299666
Var   : 72.91666666666667
Min   : 55
Max   : 80
Median: 67.5
```

## 18. Aggregates Along an Axis (2-D)

```python
quarterly_sales = np.array([
    [200, 300, 250],  # Product A: Q1, Q2, Q3
    [150, 400, 320],  # Product B
    [500, 180, 220]   # Product C
])
print('Total per product (axis=1):', quarterly_sales.sum(axis=1))
print('Total per quarter (axis=0):', quarterly_sales.sum(axis=0))
print('Grand total               :', quarterly_sales.sum())
```

Output:
```text
Total per product (axis=1): [750 870 900]
Total per quarter (axis=0): [850 880 790]
Grand total               : 2520
```

## 19. Type Hierarchy — Checking with `np.issubdtype`

```python
arr_int = np.array([1, 2, 3], dtype='int32')
arr_float = np.array([1.0, 2.0], dtype='float64')

print('Is int32 a subtype of integer?', np.issubdtype(arr_int.dtype, np.integer))
print('Is float64 a subtype of floating?', np.issubdtype(arr_float.dtype, np.floating))
```

Output:
```text
Is int32 a subtype of integer? True
Is float64 a subtype of floating? True
```

## 19b. Rounding Functions — `round`, `floor`, `ceil`, `trunc`

```python
prices_raw = np.array([19.499, 20.501, -5.5, 3.14159])
print('round :', np.round(prices_raw, 2))
print('floor :', np.floor(prices_raw))
print('ceil  :', np.ceil(prices_raw))
print('trunc :', np.trunc(prices_raw))
```

Output:
```text
round : [19.5  20.5  -5.5   3.14]
floor : [19. 20. -6.  3.]
ceil  : [20. 21. -5.  4.]
trunc : [19. 20. -5.  3.]
```

## 19c. `np.clip()` Preview (full coverage in Notebook 5)

```python
raw_pulse = np.array([45, 200, 75, -10, 130])
safe_pulse = np.clip(raw_pulse, 40, 180)
print(safe_pulse)
```

Output:
```text
[ 45 180  75  40 130]
```

## 19d. `np.errstate()` — Controlling Warnings for Edge-Case Math

```python
with np.errstate(divide='ignore', invalid='ignore'):
    result = np.array([1, 0, -1]) / 0
print(result)  # inf, nan, -inf — no warning printed because we suppressed it
```

Output:
```text
[ inf  nan -inf]
```

## 19e. Integer Overflow — A Common Silent Bug

```python
small_int_a = np.array([100], dtype='int8')
small_int_b = np.array([50], dtype='int8')
overflowed = small_int_a + small_int_b  # 150 exceeds int8 max (127), wraps around
print('100 + 50 as int8 (wraps around):', overflowed)
print('int8 range:', np.iinfo('int8').min, 'to', np.iinfo('int8').max)
```

Output:
```text
100 + 50 as int8 (wraps around): [-106]
int8 range: -128 to 127
```

## 19f. Floating Point Precision Gotchas

```python
print('0.1 + 0.2 == 0.3?', 0.1 + 0.2 == 0.3)
print('Actual value:', 0.1 + 0.2)
print('Correct comparison with np.isclose:', np.isclose(0.1 + 0.2, 0.3))
print('np.allclose on arrays:', np.allclose([0.1+0.2, 1.0000001], [0.3, 1.0]))
```

Output:
```text
0.1 + 0.2 == 0.3? False
Actual value: 0.30000000000000004
Correct comparison with np.isclose: True
np.allclose on arrays: True
```

## 19g. Complex Numbers in NumPy

```python
z = np.array([2+3j, 1-1j, 0+5j])
print('Array         :', z)
print('Real parts    :', z.real)
print('Imaginary parts:', z.imag)
print('Magnitude(abs):', np.abs(z))
print('Conjugate     :', np.conj(z))
```

Output:
```text
Array         : [2.+3.j 1.-1.j 0.+5.j]
Real parts    : [2. 1. 0.]
Imaginary parts: [ 3. -1.  5.]
Magnitude(abs): [3.60555128 1.41421356 5.        ]
Conjugate     : [2.-3.j 1.+1.j 0.-5.j]
```

## 19h. Boolean Array Reductions — `any()` and `all()`

```python
attendance_check = np.array([True, True, False, True])
print('Did EVERYONE attend? (all):', attendance_check.all())
print('Did ANYONE attend?    (any):', attendance_check.any())

matrix_check = np.array([[True, True], [False, True]])
print('\nPer-column all():', matrix_check.all(axis=0))
print('Per-row any()   :', matrix_check.any(axis=1))
```

Output:
```text
Did EVERYONE attend? (all): False
Did ANYONE attend?    (any): True

Per-column all(): [False  True]
Per-row any()   : [ True  True]
```

## 19i. Controlling Print Output — `np.set_printoptions()`

```python
big_array = np.random.random(20)
np.set_printoptions(precision=3, suppress=True)
print(big_array)

np.set_printoptions()  # reset to defaults
```

Output:
```text
[0.754 0.128 0.229 0.016 0.107 0.628 0.607 0.07  0.998 0.944 0.822 0.943
 0.267 0.795 0.266 0.525 0.46  0.235 0.474 0.877]
```

## 19j. `np.where()` with Multiple Conditions (Preview — full coverage in Notebook 5)

```python
temps = np.array([15, 25, 35, 45])
label = np.where(temps < 20, 'Cold', np.where(temps < 35, 'Mild', 'Hot'))
print(label)
```

Output:
```text
['Cold' 'Mild' 'Hot' 'Hot']
```

## 19k. Checking Array Flags in Detail

```python
sample = np.arange(6).reshape(2, 3)
print(sample.flags)
```

Output:
```text
C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : False
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False
```

## 20. Summary — What You Learned

- NumPy = Numerical Python, built for fast numerical computation
- `ndarray` is the core data structure
- Key attributes: `ndim`, `shape`, `size`, `dtype`, `itemsize`, `nbytes`
- Arrays are homogeneous (single dtype), unlike Python lists
- NumPy is dramatically faster than pure Python for numeric operations
- 0-D, 1-D, 2-D, 3-D, and N-D arrays, each with real-world style examples

### Next Notebook
Continue to **`02_array_creation.ipynb`** to learn every way of creating NumPy arrays: `zeros`, `ones`, `full`, `eye`, `diag`, `arange`, `linspace`, `logspace`, and random array generation.
