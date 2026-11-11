# 02 — DataFrame Creation & I/O (Preetham's Pandas Mastery Series)

Part 2 of 5. Covers every way to build DataFrames and read/write CSV, Excel, JSON files.

```python
import pandas as pd
import numpy as np
```

## 1. Creating DataFrames — Every Method Recap

```python
# From dict of lists (most common)
sales_df = pd.DataFrame({
    'Region': ['North', 'South', 'East', 'West'],
    'Revenue': [50000, 62000, 48000, 71000]
})
print(sales_df)
```

Output:
```text
Region  Revenue
0  North    50000
1  South    62000
2   East    48000
3   West    71000
```

```python
# From dict of Series (auto-aligns on index!)
q1 = pd.Series({'North': 100, 'South': 150})
q2 = pd.Series({'North': 120, 'South': 140, 'East': 90})
combined = pd.DataFrame({'Q1': q1, 'Q2': q2})
print(combined)  # East gets NaN for Q1 since it wasn't in q1
```

Output:
```text
Q1   Q2
East     NaN   90
North  100.0  120
South  150.0  140
```

```python
# Empty DataFrame with defined columns, filled later
empty_orders = pd.DataFrame(columns=['OrderID', 'Amount'])
print(empty_orders)
empty_orders.loc[0] = [1001, 2500]
empty_orders.loc[1] = [1002, 1800]
print('\nAfter adding rows:\n', empty_orders)
```

Output:
```text
Empty DataFrame
Columns: [OrderID, Amount]
Index: []

After adding rows:
    OrderID  Amount
0     1001    2500
1     1002    1800
```

## 2. Writing to CSV — `to_csv()`

```python
sales_df.to_csv('/tmp/sales_data.csv', index=False)
print('Saved to /tmp/sales_data.csv')
```

Output:
```text
Saved to /tmp/sales_data.csv
```

## 3. Reading from CSV — `read_csv()`

```python
loaded_df = pd.read_csv('/tmp/sales_data.csv')
print(loaded_df)
print('\ndtypes:\n', loaded_df.dtypes)
```

Output:
```text
Region  Revenue
0  North    50000
1  South    62000
2   East    48000
3   West    71000

dtypes:
 Region       str
Revenue    int64
dtype: object
```

## 4. Useful `read_csv()` Parameters

```python
# Simulate a messier CSV with extra header rows and a specific separator
with open('/tmp/messy_data.csv', 'w') as f:
    f.write('# This is a comment line\n')
    f.write('Name;Score;City\n')
    f.write('Rahul;88;Bengaluru\n')
    f.write('Aisha;92;Chennai\n')

messy_df = pd.read_csv('/tmp/messy_data.csv', sep=';', comment='#')
print(messy_df)
```

Output:
```text
Name  Score       City
0  Rahul     88  Bengaluru
1  Aisha     92    Chennai
```

```python
# Reading only specific columns, and setting an index column
selective_df = pd.read_csv('/tmp/sales_data.csv', usecols=['Region', 'Revenue'], index_col='Region')
print(selective_df)
```

Output:
```text
Revenue
Region         
North     50000
South     62000
East      48000
West      71000
```

## 5. Writing & Reading JSON

```python
sales_df.to_json('/tmp/sales_data.json', orient='records')
with open('/tmp/sales_data.json') as f:
    print(f.read())
```

Output:
```text
[{"Region":"North","Revenue":50000},{"Region":"South","Revenue":62000},{"Region":"East","Revenue":48000},{"Region":"West","Revenue":71000}]
```

```python
loaded_json_df = pd.read_json('/tmp/sales_data.json')
print(loaded_json_df)
```

Output:
```text
Region  Revenue
0  North    50000
1  South    62000
2   East    48000
3   West    71000
```

## 6. Writing & Reading Excel — `to_excel()` / `read_excel()`

```python
!pip install openpyxl --quiet
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
sales_df.to_excel('/tmp/sales_data.xlsx', index=False, sheet_name='Sales')
loaded_excel_df = pd.read_excel('/tmp/sales_data.xlsx', sheet_name='Sales')
print(loaded_excel_df)
```

Output:
```text
Region  Revenue
0  North    50000
1  South    62000
2   East    48000
3   West    71000
```

## 7. Reading Only the First N Rows — `nrows`

```python
large_sample = pd.DataFrame({'ID': range(1, 101), 'Value': np.random.randint(1, 100, 100)})
large_sample.to_csv('/tmp/large_sample.csv', index=False)

first_5 = pd.read_csv('/tmp/large_sample.csv', nrows=5)
print(first_5)
```

Output:
```text
ID  Value
0   1     67
1   2     48
2   3     22
3   4     17
4   5     96
```

## 8. Handling Missing Values While Reading

```python
with open('/tmp/with_missing.csv', 'w') as f:
    f.write('Name,Score\nRahul,88\nAisha,NA\nKabir,--\n')

cleaned_on_read = pd.read_csv('/tmp/with_missing.csv', na_values=['NA', '--'])
print(cleaned_on_read)
print('\nMissing detected:\n', cleaned_on_read.isna())
```

Output:
```text
Name  Score
0  Rahul   88.0
1  Aisha    NaN
2  Kabir    NaN

Missing detected:
     Name  Score
0  False  False
1  False   True
2  False   True
```

## 9. `pd.date_range()` — Generating Date Sequences

```python
date_index = pd.date_range(start='2026-01-01', periods=7, freq='D')
print(date_index)

daily_log = pd.DataFrame({'Date': date_index, 'Visitors': [120, 150, 90, 200, 180, 220, 95]})
print('\n', daily_log)
```

Output:
```text
DatetimeIndex(['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04',
               '2026-01-05', '2026-01-06', '2026-01-07'],
              dtype='datetime64[us]', freq='D')

         Date  Visitors
0 2026-01-01       120
1 2026-01-02       150
2 2026-01-03        90
3 2026-01-04       200
4 2026-01-05       180
5 2026-01-06       220
6 2026-01-07        95
```

## 10. Creating a DataFrame with a Custom Range Index

```python
custom_indexed = pd.DataFrame(
    {'Score': [70, 85, 90]},
    index=pd.RangeIndex(start=100, stop=103, name='StudentID')
)
print(custom_indexed)
```

Output:
```text
Score
StudentID       
100           70
101           85
102           90
```

## 11. `pd.concat()` — Combining Multiple DataFrames Read from Files

```python
batch1 = pd.DataFrame({'ID': [1, 2], 'Score': [80, 90]})
batch2 = pd.DataFrame({'ID': [3, 4], 'Score': [70, 60]})

combined_batches = pd.concat([batch1, batch2], ignore_index=True)
print(combined_batches)
```

Output:
```text
ID  Score
0   1     80
1   2     90
2   3     70
3   4     60
```

## 12. Copying a DataFrame Safely

```python
original_df = pd.DataFrame({'A': [1, 2, 3]})
shallow_ref = original_df          # same object!
true_copy = original_df.copy()     # independent copy

true_copy.loc[0, 'A'] = 999
print('Original unaffected by copy:\n', original_df)
print('\nTrue copy modified:\n', true_copy)
```

Output:
```text
Original unaffected by copy:
    A
0  1
1  2
2  3

True copy modified:
      A
0  999
1    2
2    3
```

## 13. Summary — What You Learned

| Method | Purpose |
|---|---|
| `pd.DataFrame()` | Create from dict/list/Series/array |
| `to_csv()` / `read_csv()` | CSV I/O |
| `to_json()` / `read_json()` | JSON I/O |
| `to_excel()` / `read_excel()` | Excel I/O |
| `na_values`, `usecols`, `nrows`, `sep`, `comment` | Common read_csv parameters |
| `pd.date_range()` | Generate date sequences |
| `pd.concat()` | Combine multiple DataFrames |
| `.copy()` | Independent DataFrame copy |

### Next Notebook
Continue to **`03_indexing_selecting.ipynb`** for `loc`, `iloc`, boolean filtering, and `query()`.
