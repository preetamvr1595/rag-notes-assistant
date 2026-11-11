# 05 — Searching, Sorting & Filtering in NumPy (Preetham's NumPy Mastery Series)

Part 5 of 5 (final notebook). This covers `where`, `searchsorted`, `sort`, `argsort`, `argmax/argmin`, `unique`, `nonzero`, `extract`, `count_nonzero`, `clip`, and boolean filtering — all with unique real-world examples.

## 1. `np.where()` — Finding Indices That Match a Condition

```python
import numpy as np

exam_scores = np.array([45, 78, 92, 33, 88, 55, 67, 91, 40, 100])

passed_indices = np.where(exam_scores >= 50)
print('Indices of passing scores:', passed_indices)
print('Actual passing scores    :', exam_scores[passed_indices])
```

Output:
```text
Indices of passing scores: (array([1, 2, 4, 5, 6, 7, 9]),)
Actual passing scores    : [ 78  92  88  55  67  91 100]
```

```python
# where() as a conditional replacement (like an if-else across the array)
grade_label = np.where(exam_scores >= 50, 'Pass', 'Fail')
print(grade_label)
```

Output:
```text
['Fail' 'Pass' 'Pass' 'Fail' 'Pass' 'Pass' 'Pass' 'Pass' 'Fail' 'Pass']
```

## 2. `np.searchsorted()` — Binary Search Insert Position

```python
sorted_ages = np.array([18, 21, 25, 30, 35, 40])

# Where should age 28 be inserted to keep the array sorted?
position = np.searchsorted(sorted_ages, 28)
print('Insert position for 28:', position)

# Multiple values at once
positions = np.searchsorted(sorted_ages, [20, 33, 41])
print('Insert positions for [20, 33, 41]:', positions)

# side='right' vs 'left'
print('side=left :', np.searchsorted(sorted_ages, 25, side='left'))
print('side=right:', np.searchsorted(sorted_ages, 25, side='right'))
```

Output:
```text
Insert position for 28: 3
Insert positions for [20, 33, 41]: [1 4 6]
side=left : 2
side=right: 3
```

## 3. `np.sort()` — Sorting Arrays

```python
runner_times = np.array([15.2, 12.8, 14.1, 11.9, 13.5])
sorted_times = np.sort(runner_times)
print('Original:', runner_times)
print('Sorted  :', sorted_times)
```

Output:
```text
Original: [15.2 12.8 14.1 11.9 13.5]
Sorted  : [11.9 12.8 13.5 14.1 15.2]
```

```python
# Sorting strings alphabetically
city_names = np.array(['Bengaluru', 'Chennai', 'Ahmedabad', 'Delhi'])
print('Sorted cities:', np.sort(city_names))
```

Output:
```text
Sorted cities: ['Ahmedabad' 'Bengaluru' 'Chennai' 'Delhi']
```

```python
# Sorting a 2-D array along each axis
leaderboard = np.array([
    [90, 40, 70],
    [30, 95, 60],
    [85, 20, 99]
])
print('Original:\n', leaderboard)
print('\nSorted along axis=1 (each row):\n', np.sort(leaderboard, axis=1))
print('\nSorted along axis=0 (each column):\n', np.sort(leaderboard, axis=0))
```

Output:
```text
Original:
 [[90 40 70]
 [30 95 60]
 [85 20 99]]

Sorted along axis=1 (each row):
 [[40 70 90]
 [30 60 95]
 [20 85 99]]

Sorted along axis=0 (each column):
 [[30 20 60]
 [85 40 70]
 [90 95 99]]
```

## 4. `np.argsort()` — Indices That Would Sort the Array

```python
product_ratings = np.array([3.5, 4.8, 2.1, 4.9, 3.9])
ranking_indices = np.argsort(product_ratings)[::-1]  # descending
print('Ratings          :', product_ratings)
print('Best to worst idx:', ranking_indices)
print('Best to worst val:', product_ratings[ranking_indices])
```

Output:
```text
Ratings          : [3.5 4.8 2.1 4.9 3.9]
Best to worst idx: [3 1 4 0 2]
Best to worst val: [4.9 4.8 3.9 3.5 2.1]
```

## 5. `argmax()` and `argmin()` — Index of Max/Min Value

```python
stock_prices = np.array([102, 108, 95, 120, 99, 115])
print('Highest price index:', stock_prices.argmax(), '-> value:', stock_prices.max())
print('Lowest price index :', stock_prices.argmin(), '-> value:', stock_prices.min())
```

Output:
```text
Highest price index: 3 -> value: 120
Lowest price index : 2 -> value: 95
```

```python
# argmax/argmin on a 2-D array with axis
matrix = np.array([[5, 9, 2], [8, 1, 7]])
print('Matrix:\n', matrix)
print('argmax per column (axis=0):', matrix.argmax(axis=0))
print('argmax per row    (axis=1):', matrix.argmax(axis=1))
```

Output:
```text
Matrix:
 [[5 9 2]
 [8 1 7]]
argmax per column (axis=0): [1 0 1]
argmax per row    (axis=1): [1 0]
```

## 6. `np.unique()` — Finding Distinct Values

```python
survey_responses = np.array(['Yes', 'No', 'Yes', 'Maybe', 'No', 'Yes', 'Maybe'])

unique_vals = np.unique(survey_responses)
print('Unique responses:', unique_vals)

# With counts
unique_vals, counts = np.unique(survey_responses, return_counts=True)
for val, cnt in zip(unique_vals, counts):
    print(f'  {val}: {cnt} times')
```

Output:
```text
Unique responses: ['Maybe' 'No' 'Yes']
  Maybe: 2 times
  No: 2 times
  Yes: 3 times
```

```python
# return_index gives first occurrence position
vals, first_idx = np.unique(survey_responses, return_index=True)
print('First occurrence indices:', first_idx)
```

Output:
```text
First occurrence indices: [3 1 0]
```

## 7. `np.nonzero()` — Locating Non-Zero Elements

```python
seat_availability = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0])
available_seats = np.nonzero(seat_availability)
print('Available seat positions:', available_seats[0])
```

Output:
```text
Available seat positions: [1 2 5 6 7]
```

## 8. `np.count_nonzero()` — Counting Matches

```python
attendance_week = np.array([1, 1, 0, 1, 1, 0, 1])
present_days = np.count_nonzero(attendance_week)
print('Days present this week:', present_days)

# Also works with conditions directly
high_scorers = np.count_nonzero(exam_scores > 80)
print('Students scoring above 80:', high_scorers)
```

Output:
```text
Days present this week: 5
Students scoring above 80: 4
```

## 9. `np.extract()` — Extract Elements Matching a Condition

```python
donations = np.array([100, 500, 50, 1000, 250, 20])
big_donations = np.extract(donations >= 250, donations)
print('Donations >= 250:', big_donations)
```

Output:
```text
Donations >= 250: [ 500 1000  250]
```

## 10. `np.clip()` — Limiting Values to a Range

```python
raw_sensor_readings = np.array([-5, 15, 102, 47, -20, 99, 5])
safe_readings = np.clip(raw_sensor_readings, 0, 100)
print('Raw   :', raw_sensor_readings)
print('Clipped to [0,100]:', safe_readings)
```

Output:
```text
Raw   : [ -5  15 102  47 -20  99   5]
Clipped to [0,100]: [  0  15 100  47   0  99   5]
```

## 11. Boolean Filtering — Building Custom Filters

```python
employee_salaries = np.array([25000, 45000, 60000, 30000, 75000, 52000])

# Manual method with a loop (educational)
mask = []
for s in employee_salaries:
    mask.append(s > 40000)
print('Manual mask:', mask)
print('Filtered (manual):', employee_salaries[mask])
```

Output:
```text
Manual mask: [np.False_, np.True_, np.True_, np.False_, np.True_, np.True_]
Filtered (manual): [45000 60000 75000 52000]
```

```python
# Vectorized method (preferred, much faster)
vectorized_mask = employee_salaries > 40000
print('Vectorized mask:', vectorized_mask)
print('Filtered (vectorized):', employee_salaries[vectorized_mask])
```

Output:
```text
Vectorized mask: [False  True  True False  True  True]
Filtered (vectorized): [45000 60000 75000 52000]
```

```python
# Combined filter: salary between 30000 and 60000
mid_range = employee_salaries[(employee_salaries >= 30000) & (employee_salaries <= 60000)]
print('Mid-range salaries:', mid_range)
```

Output:
```text
Mid-range salaries: [45000 60000 30000 52000]
```

## 12. `np.isin()` — Checking Membership

```python
cart_items = np.array([101, 205, 310, 415, 520])
discount_eligible_ids = np.array([205, 415, 999])

eligible_mask = np.isin(cart_items, discount_eligible_ids)
print('Eligible mask:', eligible_mask)
print('Eligible items:', cart_items[eligible_mask])
```

Output:
```text
Eligible mask: [False  True False  True False]
Eligible items: [205 415]
```

## 13. Combining Search + Sort + Filter — Real Mini Project

Given a list of product prices, find the **top 3 cheapest** products that are still above a minimum quality-price threshold of 100.

```python
np.random.seed(7)
product_prices = np.random.randint(50, 500, size=15)
print('All prices:', product_prices)

# Step 1: filter prices above threshold
above_threshold = product_prices[product_prices > 100]
print('\nAbove threshold:', above_threshold)

# Step 2: sort ascending
sorted_prices = np.sort(above_threshold)
print('\nSorted ascending:', sorted_prices)

# Step 3: pick cheapest 3
top_3_cheapest = sorted_prices[:3]
print('\nTop 3 cheapest (above threshold):', top_3_cheapest)
```

Output:
```text
All prices: [225 246  75 117 261 457 153 398 235 448  73 122 395 416  92]

Above threshold: [225 246 117 261 457 153 398 235 448 122 395 416]

Sorted ascending: [117 122 153 225 235 246 261 395 398 416 448 457]

Top 3 cheapest (above threshold): [117 122 153]
```

## 14. `np.lexsort()` — Sorting by Multiple Keys

```python
last_names = np.array(['Sharma', 'Iyer', 'Sharma', 'Khan'])
ages = np.array([28, 34, 22, 40])

# Sort primarily by last name, then by age within same last name
order = np.lexsort((ages, last_names))
print('Sort order indices:', order)
for i in order:
    print(f'  {last_names[i]}, age {ages[i]}')
```

Output:
```text
Sort order indices: [1 3 2 0]
  Iyer, age 34
  Khan, age 40
  Sharma, age 22
  Sharma, age 28
```

## 15. `np.partition()` and `np.argpartition()` — Fast Top-K Selection

```python
response_times_ms = np.array([120, 45, 300, 80, 15, 220, 60])

# Get the 3 smallest values without fully sorting (faster for large arrays)
k = 3
partitioned = np.partition(response_times_ms, k)
print('Partitioned (first k are the k smallest, unordered among themselves):', partitioned)
print('The 3 smallest response times:', np.sort(partitioned[:k]))
```

Output:
```text
Partitioned (first k are the k smallest, unordered among themselves): [ 15  45  60  80 120 220 300]
The 3 smallest response times: [15 45 60]
```

## 16. Set Operations — `unique`, `intersect1d`, `union1d`, `setdiff1d`, `in1d`

```python
team_a = np.array([1, 2, 3, 4, 5])
team_b = np.array([4, 5, 6, 7])

print('Common members (intersect1d)  :', np.intersect1d(team_a, team_b))
print('All members combined (union1d):', np.union1d(team_a, team_b))
print('Only in Team A (setdiff1d)    :', np.setdiff1d(team_a, team_b))
print('Only in Team B (setdiff1d)    :', np.setdiff1d(team_b, team_a))
```

Output:
```text
Common members (intersect1d)  : [4 5]
All members combined (union1d): [1 2 3 4 5 6 7]
Only in Team A (setdiff1d)    : [1 2 3]
Only in Team B (setdiff1d)    : [6 7]
```

## 17. Statistics — `percentile`, `median`, `histogram`, `bincount`, `digitize`

```python
np.random.seed(3)
salaries = np.random.randint(20000, 100000, size=20)

print('Median         :', np.median(salaries))
print('25th percentile:', np.percentile(salaries, 25))
print('75th percentile:', np.percentile(salaries, 75))
print('Standard dev   :', np.std(salaries))
print('Variance       :', np.var(salaries))
```

Output:
```text
Median         : 80824.5
25th percentile: 59253.5
75th percentile: 87043.25
Standard dev   : 21653.8230972616
Variance       : 468888054.72749996
```

```python
counts, bin_edges = np.histogram(salaries, bins=4)
print('Histogram counts   :', counts)
print('Histogram bin edges:', bin_edges)
```

Output:
```text
Histogram counts   : [ 3  2  3 12]
Histogram bin edges: [23610.   41969.75 60329.5  78689.25 97049.  ]
```

```python
# bincount: frequency count of small non-negative integers
dice_rolls = np.array([1, 3, 3, 6, 2, 1, 1, 4, 6, 6, 6])
freq = np.bincount(dice_rolls)
print('Frequency of each face (index = face value):', freq)
```

Output:
```text
Frequency of each face (index = face value): [0 3 1 2 1 0 4]
```

```python
# digitize: assign each value to a bin
exam_scores2 = np.array([45, 67, 82, 91, 55, 78])
bins = np.array([0, 40, 60, 80, 100])  # F, D, C, B/A boundaries
bin_indices = np.digitize(exam_scores2, bins)
print('Bin index for each score:', bin_indices)
```

Output:
```text
Bin index for each score: [2 3 4 4 2 3]
```

## 18. Cumulative Functions — `cumsum`, `cumprod`, `diff`, `gradient`

```python
daily_revenue = np.array([1000, 1500, 1200, 1800, 2000])

print('Running total (cumsum) :', np.cumsum(daily_revenue))
print('Running product(cumprod):', np.cumprod(np.array([1, 2, 3, 4])))
print('Day-to-day change (diff):', np.diff(daily_revenue))
print('Rate of change (gradient):', np.gradient(daily_revenue))
```

Output:
```text
Running total (cumsum) : [1000 2500 3700 5500 7500]
Running product(cumprod): [ 1  2  6 24]
Day-to-day change (diff): [ 500 -300  600  200]
Rate of change (gradient): [500. 100. 150. 400. 200.]
```

## 19. Linear Algebra with `np.linalg`

```python
A = np.array([[2, 1], [1, 3]])
b_vec = np.array([8, 13])

print('Matrix A:\n', A)
print('Determinant  :', np.linalg.det(A))
print('Inverse:\n', np.linalg.inv(A))
print('Solve Ax = b :', np.linalg.solve(A, b_vec))
```

Output:
```text
Matrix A:
 [[2 1]
 [1 3]]
Determinant  : 5.000000000000001
Inverse:
 [[ 0.6 -0.2]
 [-0.2  0.4]]
Solve Ax = b : [2.2 3.6]
```

```python
eigenvalues, eigenvectors = np.linalg.eig(A)
print('Eigenvalues :', eigenvalues)
print('Eigenvectors:\n', eigenvectors)

print('\nMatrix rank :', np.linalg.matrix_rank(A))
print('Trace       :', np.trace(A))
print('Norm        :', np.linalg.norm(A))
```

Output:
```text
Eigenvalues : [1.38196601 3.61803399]
Eigenvectors:
 [[-0.85065081 -0.52573111]
 [ 0.52573111 -0.85065081]]

Matrix rank : 2
Trace       : 5
Norm        : 3.872983346207417
```

## 20b. `np.count_nonzero()` and `np.flatnonzero()` with Axis

```python
attendance_matrix = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]])
print('Present per day (axis=0):', np.count_nonzero(attendance_matrix, axis=0))
print('Present per student (axis=1):', np.count_nonzero(attendance_matrix, axis=1))
print('Flat indices of all non-zero:', np.flatnonzero(attendance_matrix))
```

Output:
```text
Present per day (axis=0): [2 1 2]
Present per student (axis=1): [2 2 1]
Flat indices of all non-zero: [0 2 3 4 8]
```

## 20c. Sorting Structured Arrays by a Field

```python
player_dtype = np.dtype([('name', 'U10'), ('score', 'i4')])
players = np.array([('Zoya', 88), ('Kabir', 95), ('Meena', 79)], dtype=player_dtype)

sorted_players = np.sort(players, order='score')
print(sorted_players)
```

Output:
```text
[('Meena', 79) ('Zoya', 88) ('Kabir', 95)]
```

## 20d. `np.unique()` with `axis` — Finding Unique Rows

```python
matrix_with_dupes = np.array([[1, 2], [3, 4], [1, 2], [5, 6]])
unique_rows = np.unique(matrix_with_dupes, axis=0)
print(unique_rows)
```

Output:
```text
[[1 2]
 [3 4]
 [5 6]]
```

## 20e. `np.searchsorted()` with a Custom `sorter`

```python
unsorted_ages = np.array([34, 18, 42, 25])
sort_order = np.argsort(unsorted_ages)

position = np.searchsorted(unsorted_ages, 30, sorter=sort_order)
print('Would insert age 30 at sorted-position:', position)
```

Output:
```text
Would insert age 30 at sorted-position: 2
```

## 20f. `np.correlate()` and `np.convolve()` — Signal Processing Basics

```python
signal = np.array([1, 2, 3, 4, 5])
kernel = np.array([0, 1, 0.5])

print('Convolution:', np.convolve(signal, kernel, mode='valid'))
print('Correlation:', np.correlate(signal, kernel, mode='valid'))
```

Output:
```text
Convolution: [2.5 4.  5.5]
Correlation: [3.5 5.  6.5]
```

## 20g. `np.polyfit()` and `np.polyval()` — Polynomial Curve Fitting

```python
x_points = np.array([1, 2, 3, 4, 5])
y_points = np.array([2.1, 3.9, 9.2, 16.1, 24.8])

coefficients = np.polyfit(x_points, y_points, deg=2)  # fit a quadratic
print('Fitted coefficients (a, b, c):', coefficients)

predicted = np.polyval(coefficients, 6)
print('Predicted value at x=6:', predicted)
```

Output:
```text
Fitted coefficients (a, b, c): [ 1.1  -0.84  1.64]
Predicted value at x=6: 36.20000000000004
```

## 20h. Weighted Average — `np.average()`

```python
exam_scores3 = np.array([70, 85, 90])
weights = np.array([0.2, 0.3, 0.5])  # different exam weightings
weighted_avg = np.average(exam_scores3, weights=weights)
print('Weighted average:', weighted_avg)
```

Output:
```text
Weighted average: 84.5
```

## 20i. Real-World Mini Project #2 — Complete Sales Analysis Pipeline

```python
np.random.seed(11)
daily_sales = np.random.randint(500, 5000, size=30)

print('Total month sales     :', daily_sales.sum())
print('Best sales day (index):', daily_sales.argmax(), '-> value:', daily_sales.max())
print('Worst sales day (index):', daily_sales.argmin(), '-> value:', daily_sales.min())
print('Days above average    :', np.count_nonzero(daily_sales > daily_sales.mean()))
print('75th percentile       :', np.percentile(daily_sales, 75))
print('3-day rolling sum (first 5):', np.convolve(daily_sales, np.ones(3), mode='valid')[:5])
```

Output:
```text
Total month sales     : 80643
Best sales day (index): 19 -> value: 4838
Worst sales day (index): 10 -> value: 640
Days above average    : 15
75th percentile       : 3804.75
3-day rolling sum (first 5): [11243. 10591.  7148.  3708.  6279.]
```

## 21. Summary — What You Learned

| Function | Purpose |
|---|---|
| `where()` | Conditional index lookup / replacement |
| `searchsorted()` | Binary search insert position |
| `sort()` | Sort array values |
| `argsort()` | Indices that would sort the array |
| `argmax()/argmin()` | Index of max/min value |
| `unique()` | Distinct values, with counts/indices |
| `nonzero()` | Locate non-zero elements |
| `count_nonzero()` | Count matches |
| `extract()` | Extract elements matching condition |
| `clip()` | Bound values to a range |
| Boolean filtering | Custom conditions, combined with `&` / `\|` |
| `isin()` | Membership testing |

## 🎉 Series Complete!

You've now covered the **entire NumPy fundamentals series** end-to-end:
1. Basics
2. Array Creation
3. Indexing & Slicing
4. Reshaping & Stacking
5. Searching, Sorting & Filtering

Next up in the larger Data Science Mastery repo: **Pandas**, **Matplotlib**, **Seaborn**, **Scikit-learn**, **TensorFlow**.
