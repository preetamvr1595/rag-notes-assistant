# 03 — Indexing & Slicing in NumPy (Preetham's NumPy Mastery Series)

Part 3 of 5. This notebook covers indexing/slicing on 1D, 2D, 3D arrays, negative indices, step slicing, boolean/fancy indexing, `newaxis`, and `Ellipsis`.

## 1. Basic Indexing on 1-D Arrays

```python
import numpy as np

monthly_expenses = np.array([4500, 5200, 3900, 6100, 4800, 5000, 4700, 5300, 6000, 5100, 4600, 5900])
print('All months:', monthly_expenses)
print('January (index 0):', monthly_expenses[0])
print('July (index 6)   :', monthly_expenses[6])
print('December (index -1):', monthly_expenses[-1])
```

Output:
```text
All months: [4500 5200 3900 6100 4800 5000 4700 5300 6000 5100 4600 5900]
January (index 0): 4500
July (index 6)   : 4700
December (index -1): 5900
```

## 2. Slicing 1-D Arrays: `[start:end:step]`

```python
print('Q1 (Jan-Mar):', monthly_expenses[0:3])
print('Last quarter:', monthly_expenses[-3:])
print('Every alternate month:', monthly_expenses[::2])
print('Reversed order:', monthly_expenses[::-1])
print('From May to end:', monthly_expenses[4:])
print('Up to August:', monthly_expenses[:8])
```

Output:
```text
Q1 (Jan-Mar): [4500 5200 3900]
Last quarter: [5100 4600 5900]
Every alternate month: [4500 3900 4800 4700 6000 4600]
Reversed order: [5900 4600 5100 6000 5300 4700 5000 4800 6100 3900 5200 4500]
From May to end: [4800 5000 4700 5300 6000 5100 4600 5900]
Up to August: [4500 5200 3900 6100 4800 5000 4700 5300]
```

## 3. Negative Indexing Explained

```python
letters = np.array(['A','B','C','D','E','F','G'])
print('Array          :', letters)
print('Index -1 (last):', letters[-1])
print('Index -7 (first):', letters[-7])
print('Slice [-4:-1]  :', letters[-4:-1])
print('Slice [-5:]    :', letters[-5:])
```

Output:
```text
Array          : ['A' 'B' 'C' 'D' 'E' 'F' 'G']
Index -1 (last): G
Index -7 (first): A
Slice [-4:-1]  : ['D' 'E' 'F']
Slice [-5:]    : ['C' 'D' 'E' 'F' 'G']
```

## 4. Indexing 2-D Arrays — Row & Column Access

```python
classroom_grades = np.array([
    [88, 92, 79, 95],   # Row 0: Student A
    [70, 88, 95, 60],   # Row 1: Student B
    [60, 75, 80, 90],   # Row 2: Student C
    [99, 100, 97, 85]   # Row 3: Student D
])
print(classroom_grades)
```

Output:
```text
[[ 88  92  79  95]
 [ 70  88  95  60]
 [ 60  75  80  90]
 [ 99 100  97  85]]
```

```python
print('Entire row 2 (Student C):', classroom_grades[2])
print('Single cell [1,2]        :', classroom_grades[1, 2])
print('Single cell alt syntax   :', classroom_grades[1][2])
print('Entire column 0 (Subj 1) :', classroom_grades[:, 0])
print('Entire column -1 (last)  :', classroom_grades[:, -1])
```

Output:
```text
Entire row 2 (Student C): [60 75 80 90]
Single cell [1,2]        : 95
Single cell alt syntax   : 95
Entire column 0 (Subj 1) : [88 70 60 99]
Entire column -1 (last)  : [95 60 90 85]
```

## 5. Slicing 2-D Arrays

```python
print('Top-left 2x2 block:\n', classroom_grades[0:2, 0:2])
print('\nBottom-right 2x2 block:\n', classroom_grades[2:4, 2:4])
print('\nAll students, first two subjects:\n', classroom_grades[:, 0:2])
print('\nFirst two students, all subjects:\n', classroom_grades[0:2, :])
print('\nEvery other row, every other column:\n', classroom_grades[::2, ::2])
```

Output:
```text
Top-left 2x2 block:
 [[88 92]
 [70 88]]

Bottom-right 2x2 block:
 [[80 90]
 [97 85]]

All students, first two subjects:
 [[ 88  92]
 [ 70  88]
 [ 60  75]
 [ 99 100]]

First two students, all subjects:
 [[88 92 79 95]
 [70 88 95 60]]

Every other row, every other column:
 [[88 79]
 [60 80]]
```

## 6. Indexing & Slicing 3-D Arrays

```python
# 2 warehouses x 3 shelves x 4 products (stock counts)
warehouse_stock = np.array([
    [[10, 20, 30, 40], [15, 25, 35, 45], [5, 10, 15, 20]],
    [[12, 22, 32, 42], [17, 27, 37, 47], [7, 12, 17, 22]]
])
print('Shape:', warehouse_stock.shape)
print(warehouse_stock)
```

Output:
```text
Shape: (2, 3, 4)
[[[10 20 30 40]
  [15 25 35 45]
  [ 5 10 15 20]]

 [[12 22 32 42]
  [17 27 37 47]
  [ 7 12 17 22]]]
```

```python
print('Warehouse 0, all shelves:\n', warehouse_stock[0])
print('\nWarehouse 1, shelf 2:', warehouse_stock[1, 2])
print('\nWarehouse 0, shelf 1, product 3:', warehouse_stock[0, 1, 3])
print('\nAll warehouses, shelf 0, all products:\n', warehouse_stock[:, 0, :])
```

Output:
```text
Warehouse 0, all shelves:
 [[10 20 30 40]
 [15 25 35 45]
 [ 5 10 15 20]]

Warehouse 1, shelf 2: [ 7 12 17 22]

Warehouse 0, shelf 1, product 3: 45

All warehouses, shelf 0, all products:
 [[10 20 30 40]
 [12 22 32 42]]
```

## 7. Boolean (Mask) Indexing

```python
temperatures = np.array([28, 34, 19, 41, 25, 37, 22, 30])

hot_days_mask = temperatures > 30
print('Mask:', hot_days_mask)
print('Hot day temperatures:', temperatures[hot_days_mask])

# Directly, without storing mask separately
print('Cold days (<25):', temperatures[temperatures < 25])
```

Output:
```text
Mask: [False  True False  True False  True False False]
Hot day temperatures: [34 41 37]
Cold days (<25): [19 22]
```

```python
# Combine multiple conditions with & and |
mild_days = temperatures[(temperatures >= 20) & (temperatures <= 30)]
print('Mild days (20-30):', mild_days)

extreme_days = temperatures[(temperatures < 20) | (temperatures > 38)]
print('Extreme days (<20 or >38):', extreme_days)
```

Output:
```text
Mild days (20-30): [28 25 22 30]
Extreme days (<20 or >38): [19 41]
```

## 8. Fancy Indexing — Selecting with a List/Array of Indices

```python
product_prices = np.array([250, 100, 750, 400, 999, 150, 300])

wishlist_positions = [4, 0, 6]
print('Wishlist items:', product_prices[wishlist_positions])

# Fancy indexing on 2-D array (select specific rows)
selected_students = classroom_grades[[0, 3]]
print('\nSelected students (0 and 3):\n', selected_students)
```

Output:
```text
Wishlist items: [999 250 300]

Selected students (0 and 3):
 [[ 88  92  79  95]
 [ 99 100  97  85]]
```

## 9. `np.newaxis` — Adding a New Dimension

```python
sales_1d = np.array([100, 200, 300])
print('Original shape:', sales_1d.shape)

sales_row = sales_1d[np.newaxis, :]
print('As a row vector, shape:', sales_row.shape)
print(sales_row)

sales_col = sales_1d[:, np.newaxis]
print('\nAs a column vector, shape:', sales_col.shape)
print(sales_col)
```

Output:
```text
Original shape: (3,)
As a row vector, shape: (1, 3)
[[100 200 300]]

As a column vector, shape: (3, 1)
[[100]
 [200]
 [300]]
```

## 10. `Ellipsis (...)` — Skipping Dimensions Automatically

```python
big_array = np.arange(2*3*4).reshape(2, 3, 4)
print('Shape:', big_array.shape)

# Instead of writing big_array[0, :, :]
print('Using ellipsis:\n', big_array[0, ...])

# Instead of writing big_array[:, :, 2]
print('\nLast axis index 2 via ellipsis:\n', big_array[..., 2])
```

Output:
```text
Shape: (2, 3, 4)
Using ellipsis:
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]

Last axis index 2 via ellipsis:
 [[ 2  6 10]
 [14 18 22]]
```

## 11. Modifying Values via Indexing/Slicing

```python
inventory = np.array([50, 30, 20, 10, 5])
print('Before:', inventory)

inventory[0] = 100          # single element
inventory[1:3] = [0, 0]     # slice assignment
inventory[inventory < 10] = 999  # boolean assignment (restock alert)

print('After :', inventory)
```

Output:
```text
Before: [50 30 20 10  5]
After : [100 999 999  10 999]
```

## 12. Views vs Copies When Slicing (Important!)

```python
original = np.array([1, 2, 3, 4, 5])
view_slice = original[1:4]     # This is a VIEW, not a copy!
view_slice[0] = 999

print('Original changed too!:', original)
print('Because slicing returns a view, not a copy.')

# To avoid this, use .copy()
safe_slice = original[1:4].copy()
safe_slice[0] = -1
print('\nOriginal unaffected  :', original)
print('Safe (copied) slice  :', safe_slice)
```

Output:
```text
Original changed too!: [  1 999   3   4   5]
Because slicing returns a view, not a copy.

Original unaffected  : [  1 999   3   4   5]
Safe (copied) slice  : [-1  3  4]
```

## 13. `np.take()` — Indexing as a Function

```python
song_ratings = np.array([3, 5, 2, 4, 1, 5])
favorite_positions = [1, 3, 5]
print(np.take(song_ratings, favorite_positions))
```

Output:
```text
[5 4 5]
```

## 14. `np.put()` — Setting Values at Specific Indices

```python
seats = np.array([0, 0, 0, 0, 0, 0])
np.put(seats, [1, 3, 4], [1, 1, 1])  # mark seats 1,3,4 as booked
print(seats)
```

Output:
```text
[0 1 0 1 1 0]
```

## 15. `np.select()` — Multi-Condition Choice (Vectorized if-elif-else)

```python
marks = np.array([95, 72, 58, 40, 88])
conditions = [marks >= 90, marks >= 75, marks >= 60, marks >= 40]
grades = ['A', 'B', 'C', 'D']

result = np.select(conditions, grades, default='F')
print(result)
```

Output:
```text
['A' 'C' 'D' 'D' 'B']
```

## 16. `np.ix_()` — Building an Open Mesh for Cross Indexing

```python
matrix = np.arange(1, 26).reshape(5, 5)
print('Original:\n', matrix)

rows = [0, 2, 4]
cols = [1, 3]
selected = matrix[np.ix_(rows, cols)]
print('\nSelected rows [0,2,4] x cols [1,3]:\n', selected)
```

Output:
```text
Original:
 [[ 1  2  3  4  5]
 [ 6  7  8  9 10]
 [11 12 13 14 15]
 [16 17 18 19 20]
 [21 22 23 24 25]]

Selected rows [0,2,4] x cols [1,3]:
 [[ 2  4]
 [12 14]
 [22 24]]
```

## 17. `np.take_along_axis()` — Index-Aligned Selection

```python
scores_matrix = np.array([[50, 90, 70], [80, 40, 95], [60, 85, 30]])
top_index_per_row = np.argmax(scores_matrix, axis=1, keepdims=True)

top_scores = np.take_along_axis(scores_matrix, top_index_per_row, axis=1)
print('Top index per row:\n', top_index_per_row)
print('Top score per row :\n', top_scores)
```

Output:
```text
Top index per row:
 [[1]
 [2]
 [1]]
Top score per row :
 [[90]
 [95]
 [85]]
```

## 18. Structured Array Field Indexing (from Array Creation notebook, reused here)

```python
car_dtype = np.dtype([('model', 'U15'), ('year', 'i4'), ('price', 'f8')])
cars = np.array([('Swift', 2021, 650000.0), ('Nexon', 2023, 1450000.0)], dtype=car_dtype)

print('All years :', cars['year'])
print('First car :', cars[0])
print('Price of Nexon:', cars[cars['model'] == 'Nexon']['price'])
```

Output:
```text
All years : [2021 2023]
First car : ('Swift', 2021, 650000.0)
Price of Nexon: [1450000.]
```

## 19. Conditional Element Replacement with `np.where` (Indexing Context)

```python
portfolio_returns = np.array([5.2, -3.1, 8.4, -1.2, 0.0, 6.7])
labelled = np.where(portfolio_returns > 0, 'Gain', np.where(portfolio_returns < 0, 'Loss', 'Flat'))
print(labelled)
```

Output:
```text
['Gain' 'Loss' 'Gain' 'Loss' 'Flat' 'Gain']
```

## 20. Diagonal Indexing on 2-D Arrays

```python
matrix2 = np.arange(1, 17).reshape(4, 4)
print('Matrix:\n', matrix2)
print('\nMain diagonal:', matrix2.diagonal())
print('Diagonal offset +1:', matrix2.diagonal(offset=1))
print('Diagonal offset -1:', matrix2.diagonal(offset=-1))
```

Output:
```text
Matrix:
 [[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]]

Main diagonal: [ 1  6 11 16]
Diagonal offset +1: [ 2  7 12]
Diagonal offset -1: [ 5 10 15]
```

## 21. Slicing with Assignment Broadcasting

```python
seating_grid = np.zeros((4, 4), dtype=int)
seating_grid[1:3, 1:3] = 1   # mark VIP block
print(seating_grid)
```

Output:
```text
[[0 0 0 0]
 [0 1 1 0]
 [0 1 1 0]
 [0 0 0 0]]
```

## 21b. Fancy Indexing to Reorder Rows

```python
leaderboard_names = np.array(['Zoya', 'Kabir', 'Aisha', 'Rahul'])
leaderboard_scores = np.array([88, 95, 72, 90])

rank_order = np.argsort(leaderboard_scores)[::-1]
print('Ranked names :', leaderboard_names[rank_order])
print('Ranked scores:', leaderboard_scores[rank_order])
```

Output:
```text
Ranked names : ['Kabir' 'Rahul' 'Zoya' 'Aisha']
Ranked scores: [95 90 88 72]
```

## 21c. Combining Fancy + Boolean Indexing

```python
inventory_ids = np.array([101, 102, 103, 104, 105])
stock_levels = np.array([0, 15, 3, 0, 22])

low_or_out_of_stock = inventory_ids[stock_levels <= 3]
print('Needs restocking:', low_or_out_of_stock)
```

Output:
```text
Needs restocking: [101 103 104]
```

## 22b. Indexing with `None` — Same Effect as `np.newaxis`

```python
flat = np.array([10, 20, 30])
print('Using None:', flat[None, :].shape)
print('Using newaxis:', flat[np.newaxis, :].shape)  # identical result
```

Output:
```text
Using None: (1, 3)
Using newaxis: (1, 3)
```

## 22c. Negative Step Slicing on 2-D Arrays

```python
grid = np.arange(1, 17).reshape(4, 4)
print('Original:\n', grid)
print('\nReversed rows:\n', grid[::-1])
print('\nReversed columns:\n', grid[:, ::-1])
print('\nFully reversed:\n', grid[::-1, ::-1])
```

Output:
```text
Original:
 [[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]]

Reversed rows:
 [[13 14 15 16]
 [ 9 10 11 12]
 [ 5  6  7  8]
 [ 1  2  3  4]]

Reversed columns:
 [[ 4  3  2  1]
 [ 8  7  6  5]
 [12 11 10  9]
 [16 15 14 13]]

Fully reversed:
 [[16 15 14 13]
 [12 11 10  9]
 [ 8  7  6  5]
 [ 4  3  2  1]]
```

## 22d. `np.ndindex()` — Iterating Over All Index Combinations

```python
small_grid = np.arange(6).reshape(2, 3)
for idx in np.ndindex(small_grid.shape):
    print(idx, '->', small_grid[idx])
```

Output:
```text
(0, 0) -> 0
(0, 1) -> 1
(0, 2) -> 2
(1, 0) -> 3
(1, 1) -> 4
(1, 2) -> 5
```

## 22e. Setting Values with a 2-D Boolean Mask

```python
temperature_grid = np.array([[15, 25, 35], [45, 10, 28], [33, 12, 40]])
temperature_grid[temperature_grid > 30] = 30  # cap all values above 30
print(temperature_grid)
```

Output:
```text
[[15 25 30]
 [30 10 28]
 [30 12 30]]
```

## 22f. The View-vs-Copy Trap with Chained Indexing

A classic pandas/NumPy gotcha: chained indexing can silently fail to modify the original array.

```python
data = np.zeros((3, 3))

# This might not always update 'data' as expected, depending on whether
# the first index returns a view or a copy — always prefer a single combined index:
data[1, 1] = 5  # SAFE: single combined index, always works
print(data)
```

Output:
```text
[[0. 0. 0.]
 [0. 5. 0.]
 [0. 0. 0.]]
```

## 22g. Structured Array Multi-Field Indexing

```python
student_dtype = np.dtype([('name', 'U10'), ('score', 'i4')])
students = np.array([('Zoya', 88), ('Kabir', 92), ('Meena', 79)], dtype=student_dtype)

print('Names only :', students['name'])
high_scorers = students[students['score'] > 80]
print('High scorers:', high_scorers)
```

Output:
```text
Names only : ['Zoya' 'Kabir' 'Meena']
High scorers: [('Zoya', 88) ('Kabir', 92)]
```

## 23. Summary — What You Learned

- 1-D, 2-D, 3-D indexing and slicing with real-world examples
- Negative indexing rules
- Boolean masking with single & combined conditions
- Fancy indexing with lists of indices
- `np.newaxis` for reshaping into row/column vectors
- `Ellipsis (...)` for skipping dimensions
- Modifying arrays via indexing
- The critical difference between **views** and **copies**

### Next Notebook
Continue to **`04_reshaping_stacking.ipynb`** to learn reshaping, flattening, transposing, joining, and splitting arrays.
