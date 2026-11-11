# 04 — Data Cleaning in Pandas (Preetham's Pandas Mastery Series)

Part 4 of 5. Covers missing data handling, duplicates, type conversion, string cleaning, outlier handling, and applying custom functions.

```python
import pandas as pd
import numpy as np

messy_orders = pd.DataFrame({
    'OrderID': [101, 102, 103, 104, 105, 105],
    'Customer': [' Rahul ', 'AISHA', 'kabir', None, 'Zoya', 'Zoya'],
    'Amount': [1500, np.nan, 2200, 1800, -50, -50],
    'Status': ['Delivered', 'delivered', 'PENDING', 'Cancelled', 'Delivered', 'Delivered']
})
print(messy_orders)
```

Output:
```text
OrderID Customer  Amount     Status
0      101   Rahul   1500.0  Delivered
1      102    AISHA     NaN  delivered
2      103    kabir  2200.0    PENDING
3      104      NaN  1800.0  Cancelled
4      105     Zoya   -50.0  Delivered
5      105     Zoya   -50.0  Delivered
```

## 1. Detecting Missing Values

```python
print('Missing per column:\n', messy_orders.isna().sum())
print('\nAny missing at all?', messy_orders.isna().any().any())
```

Output:
```text
Missing per column:
 OrderID     0
Customer    1
Amount      1
Status      0
dtype: int64

Any missing at all? True
```

## 2. Dropping Missing Values — `dropna()`

```python
dropped_rows = messy_orders.dropna()
print(dropped_rows)
```

Output:
```text
OrderID Customer  Amount     Status
0      101   Rahul   1500.0  Delivered
2      103    kabir  2200.0    PENDING
4      105     Zoya   -50.0  Delivered
5      105     Zoya   -50.0  Delivered
```

```python
# Drop only if a SPECIFIC column has missing values
dropped_specific = messy_orders.dropna(subset=['Customer'])
print(dropped_specific)
```

Output:
```text
OrderID Customer  Amount     Status
0      101   Rahul   1500.0  Delivered
1      102    AISHA     NaN  delivered
2      103    kabir  2200.0    PENDING
4      105     Zoya   -50.0  Delivered
5      105     Zoya   -50.0  Delivered
```

## 3. Filling Missing Values — `fillna()`

```python
filled_amount = messy_orders.copy()
filled_amount['Amount'] = filled_amount['Amount'].fillna(filled_amount['Amount'].mean())
print(filled_amount)
```

Output:
```text
OrderID Customer  Amount     Status
0      101   Rahul   1500.0  Delivered
1      102    AISHA  1080.0  delivered
2      103    kabir  2200.0    PENDING
3      104      NaN  1800.0  Cancelled
4      105     Zoya   -50.0  Delivered
5      105     Zoya   -50.0  Delivered
```

```python
# Forward-fill and backward-fill
time_series = pd.Series([10, np.nan, np.nan, 25, np.nan, 30])
print('Original    :', time_series.tolist())
print('Forward fill:', time_series.ffill().tolist())
print('Backward fill:', time_series.bfill().tolist())
```

Output:
```text
Original    : [10.0, nan, nan, 25.0, nan, 30.0]
Forward fill: [10.0, 10.0, 10.0, 25.0, 25.0, 30.0]
Backward fill: [10.0, 25.0, 25.0, 25.0, 30.0, 30.0]
```

## 4. Removing Duplicate Rows

```python
print('Duplicated flag:\n', messy_orders.duplicated())
print('\nAfter drop_duplicates:\n', messy_orders.drop_duplicates())
```

Output:
```text
Duplicated flag:
 0    False
1    False
2    False
3    False
4    False
5     True
dtype: bool

After drop_duplicates:
    OrderID Customer  Amount     Status
0      101   Rahul   1500.0  Delivered
1      102    AISHA     NaN  delivered
2      103    kabir  2200.0    PENDING
3      104      NaN  1800.0  Cancelled
4      105     Zoya   -50.0  Delivered
```

```python
# Duplicate check on a specific column only
dup_by_id = messy_orders.drop_duplicates(subset=['OrderID'], keep='first')
print(dup_by_id)
```

Output:
```text
OrderID Customer  Amount     Status
0      101   Rahul   1500.0  Delivered
1      102    AISHA     NaN  delivered
2      103    kabir  2200.0    PENDING
3      104      NaN  1800.0  Cancelled
4      105     Zoya   -50.0  Delivered
```

## 5. Cleaning Text — `str` Accessor Methods

```python
text_df = messy_orders.dropna(subset=['Customer']).copy()
text_df['Customer_clean'] = text_df['Customer'].str.strip().str.title()
print(text_df[['Customer', 'Customer_clean']])
```

Output:
```text
Customer Customer_clean
0   Rahul           Rahul
1    AISHA          Aisha
2    kabir          Kabir
4     Zoya           Zoya
5     Zoya           Zoya
```

```python
print('Lowercase   :', text_df['Customer'].str.lower().tolist())
print('Uppercase   :', text_df['Customer'].str.upper().tolist())
print('Length      :', text_df['Customer'].str.len().tolist())
print('Contains a  :', text_df['Customer'].str.contains('a', case=False).tolist())
```

Output:
```text
Lowercase   : [' rahul ', 'aisha', 'kabir', 'zoya', 'zoya']
Uppercase   : [' RAHUL ', 'AISHA', 'KABIR', 'ZOYA', 'ZOYA']
Length      : [7, 5, 5, 4, 4]
Contains a  : [True, True, True, True, True]
```

## 6. Standardizing Categorical Text (Status Column)

```python
messy_orders['Status_clean'] = messy_orders['Status'].str.strip().str.capitalize()
print(messy_orders[['Status', 'Status_clean']])
```

Output:
```text
Status Status_clean
0  Delivered    Delivered
1  delivered    Delivered
2    PENDING      Pending
3  Cancelled    Cancelled
4  Delivered    Delivered
5  Delivered    Delivered
```

## 7. Handling Invalid / Out-of-Range Values

```python
print('Negative amounts (data errors):\n', messy_orders[messy_orders['Amount'] < 0])

# Replace negative amounts with NaN, then decide how to handle
messy_orders.loc[messy_orders['Amount'] < 0, 'Amount'] = np.nan
print('\nAfter marking invalid as NaN:\n', messy_orders[['OrderID', 'Amount']])
```

Output:
```text
Negative amounts (data errors):
    OrderID Customer  Amount     Status Status_clean
4      105     Zoya   -50.0  Delivered    Delivered
5      105     Zoya   -50.0  Delivered    Delivered

After marking invalid as NaN:
    OrderID  Amount
0      101  1500.0
1      102     NaN
2      103  2200.0
3      104  1800.0
4      105     NaN
5      105     NaN
```

## 8. Type Conversion — `astype()`

```python
mixed_types = pd.DataFrame({'Code': ['101', '102', '103'], 'Score': [88.0, 92.5, 79.0]})
print('Before:\n', mixed_types.dtypes)

mixed_types['Code'] = mixed_types['Code'].astype(int)
mixed_types['Score'] = mixed_types['Score'].astype(int)
print('\nAfter:\n', mixed_types.dtypes)
print(mixed_types)
```

Output:
```text
Before:
 Code         str
Score    float64
dtype: object

After:
 Code     int64
Score    int64
dtype: object
   Code  Score
0   101     88
1   102     92
2   103     79
```

## 9. Converting to Datetime — `pd.to_datetime()`

```python
date_strings = pd.Series(['2026-01-15', '15/02/2026', 'March 3, 2026'])
parsed_dates = pd.to_datetime(date_strings, format='mixed')
print(parsed_dates)
```

Output:
```text
0   2026-01-15
1   2026-02-15
2   2026-03-03
dtype: datetime64[us]
```

## 10. Replacing Specific Values — `replace()`

```python
cleaned_status = messy_orders['Status_clean'].replace({'Pending': 'In Progress'})
print(cleaned_status)
```

Output:
```text
0      Delivered
1      Delivered
2    In Progress
3      Cancelled
4      Delivered
5      Delivered
Name: Status_clean, dtype: str
```

## 11. Detecting Outliers with IQR Method

```python
np.random.seed(5)
salaries = pd.Series(np.append(np.random.randint(30000, 70000, 20), [500000, -10000]))

Q1 = salaries.quantile(0.25)
Q3 = salaries.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = salaries[(salaries < lower_bound) | (salaries > upper_bound)]
print('Outliers detected:', outliers.tolist())
```

Output:
```text
Outliers detected: [500000, -10000]
```

## 12. Clipping Outliers to a Valid Range

```python
clipped_salaries = salaries.clip(lower=lower_bound, upper=upper_bound)
print('Original max:', salaries.max(), '| Clipped max:', clipped_salaries.max())
```

Output:
```text
Original max: 500000 | Clipped max: 92664.25
```

## 13. `apply()` and `map()` for Custom Cleaning Logic

```python
def clean_amount(val):
    if pd.isna(val):
        return 0
    return round(val, 2)

messy_orders['Amount_final'] = messy_orders['Amount'].apply(clean_amount)
print(messy_orders[['OrderID', 'Amount', 'Amount_final']])
```

Output:
```text
OrderID  Amount  Amount_final
0      101  1500.0        1500.0
1      102     NaN           0.0
2      103  2200.0        2200.0
3      104  1800.0        1800.0
4      105     NaN           0.0
5      105     NaN           0.0
```

```python
status_map = {'Delivered': 1, 'Pending': 0, 'Cancelled': -1, 'In progress': 0}
messy_orders['Status_code'] = messy_orders['Status_clean'].map(status_map)
print(messy_orders[['Status_clean', 'Status_code']])
```

Output:
```text
Status_clean  Status_code
0    Delivered            1
1    Delivered            1
2      Pending            0
3    Cancelled           -1
4    Delivered            1
5    Delivered            1
```

## 14. Renaming and Reordering Columns

```python
final_orders = messy_orders[['OrderID', 'Customer', 'Amount_final', 'Status_clean']].rename(
    columns={'Amount_final': 'Amount', 'Status_clean': 'Status'}
)
print(final_orders)
```

Output:
```text
OrderID Customer  Amount     Status
0      101   Rahul   1500.0  Delivered
1      102    AISHA     0.0  Delivered
2      103    kabir  2200.0    Pending
3      104      NaN  1800.0  Cancelled
4      105     Zoya     0.0  Delivered
5      105     Zoya     0.0  Delivered
```

## 15. Summary — What You Learned

| Method | Purpose |
|---|---|
| `isna()`/`dropna()`/`fillna()` | Detect and handle missing data |
| `ffill()`/`bfill()` | Forward/backward fill |
| `duplicated()`/`drop_duplicates()` | Remove duplicate rows |
| `.str.*` accessors | Clean text data |
| `astype()` | Type conversion |
| `pd.to_datetime()` | Parse date strings |
| `replace()` | Replace specific values |
| IQR method + `clip()` | Detect and handle outliers |
| `apply()`/`map()` | Custom cleaning logic |

### Next Notebook
Continue to **`05_groupby_merging.ipynb`** — the final notebook covering `groupby`, aggregation, pivot tables, and merging/joining DataFrames.
