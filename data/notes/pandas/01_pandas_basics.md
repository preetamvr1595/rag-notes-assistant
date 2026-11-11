# 01 — Pandas Basics (Preetham's Pandas Mastery Series)

Part 1 of 5:
1. **Pandas Basics** (this notebook)
2. DataFrame Creation & I/O
3. Indexing & Selecting Data
4. Data Cleaning
5. GroupBy, Merging & Aggregation

Pandas is built on top of NumPy — it adds labeled, tabular data structures (`Series` and `DataFrame`) designed for real-world messy data.

## 1. What is Pandas?

- **Pandas** = **Pan**el **Da**ta, created by **Wes McKinney** in 2008.
- Two core structures: **`Series`** (1-D labeled array) and **`DataFrame`** (2-D labeled table).
- Built on top of NumPy — every column in a DataFrame is internally backed by a NumPy array.
- The standard tool for reading, cleaning, transforming, and analyzing structured data (CSV, Excel, SQL, JSON).

```python
!pip install pandas --quiet
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
import pandas as pd
import numpy as np
print('Pandas version:', pd.__version__)
```

Output:
```text
Pandas version: 3.0.2
```

## 2. Creating a `Series` — 1-D Labeled Array

```python
weekly_temps = pd.Series([28, 30, 27, 32, 29, 31, 26])
print(weekly_temps)
print('\nType:', type(weekly_temps))
```

Output:
```text
0    28
1    30
2    27
3    32
4    29
5    31
6    26
dtype: int64

Type: <class 'pandas.Series'>
```

```python
# Series with custom labels (index)
weekly_temps_labeled = pd.Series(
    [28, 30, 27, 32, 29, 31, 26],
    index=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
)
print(weekly_temps_labeled)
```

Output:
```text
Mon    28
Tue    30
Wed    27
Thu    32
Fri    29
Sat    31
Sun    26
dtype: int64
```

```python
# Series from a dictionary (keys become the index automatically)
city_population = pd.Series({'Bengaluru': 13.6, 'Chennai': 11.5, 'Delhi': 32.9, 'Mumbai': 20.7})
print(city_population)
```

Output:
```text
Bengaluru    13.6
Chennai      11.5
Delhi        32.9
Mumbai       20.7
dtype: float64
```

## 3. Series Attributes

```python
print('Values:', weekly_temps_labeled.values)
print('Index :', weekly_temps_labeled.index.tolist())
print('dtype :', weekly_temps_labeled.dtype)
print('shape :', weekly_temps_labeled.shape)
print('size  :', weekly_temps_labeled.size)
print('name  :', weekly_temps_labeled.name)
```

Output:
```text
Values: [28 30 27 32 29 31 26]
Index : ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dtype : int64
shape : (7,)
size  : 7
name  : None
```

## 4. Accessing Series Elements

```python
print('By label       :', weekly_temps_labeled['Wed'])
print('By position    :', weekly_temps_labeled.iloc[2])
print('Multiple labels:', weekly_temps_labeled[['Mon', 'Fri']].tolist())
print('Slice by label :\n', weekly_temps_labeled['Tue':'Thu'])
```

Output:
```text
By label       : 27
By position    : 27
Multiple labels: [28, 29]
Slice by label :
 Tue    30
Wed    27
Thu    32
dtype: int64
```

## 5. Creating a `DataFrame` — 2-D Labeled Table

```python
student_data = pd.DataFrame({
    'Name': ['Rahul', 'Aisha', 'Kabir', 'Meena'],
    'Age': [21, 22, 20, 23],
    'Marks': [88, 92, 79, 95],
    'City': ['Bengaluru', 'Chennai', 'Delhi', 'Mumbai']
})
print(student_data)
```

Output:
```text
Name  Age  Marks       City
0  Rahul   21     88  Bengaluru
1  Aisha   22     92    Chennai
2  Kabir   20     79      Delhi
3  Meena   23     95     Mumbai
```

## 6. DataFrame Attributes

```python
print('Shape  :', student_data.shape)
print('Columns:', student_data.columns.tolist())
print('Index  :', student_data.index.tolist())
print('dtypes :\n', student_data.dtypes)
print('\nSize (total cells):', student_data.size)
print('Number of dimensions:', student_data.ndim)
```

Output:
```text
Shape  : (4, 4)
Columns: ['Name', 'Age', 'Marks', 'City']
Index  : [0, 1, 2, 3]
dtypes :
 Name       str
Age      int64
Marks    int64
City       str
dtype: object

Size (total cells): 16
Number of dimensions: 2
```

## 7. Quick Overview Methods — `head`, `tail`, `info`, `describe`

```python
print('First 2 rows:\n', student_data.head(2))
```

Output:
```text
First 2 rows:
     Name  Age  Marks       City
0  Rahul   21     88  Bengaluru
1  Aisha   22     92    Chennai
```

```python
print('Last 2 rows:\n', student_data.tail(2))
```

Output:
```text
Last 2 rows:
     Name  Age  Marks    City
2  Kabir   20     79   Delhi
3  Meena   23     95  Mumbai
```

```python
student_data.info()
```

Output:
```text
<class 'pandas.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   Name    4 non-null      str  
 1   Age     4 non-null      int64
 2   Marks   4 non-null      int64
 3   City    4 non-null      str  
dtypes: int64(2), str(2)
memory usage: 260.0 bytes
```

```python
print(student_data.describe())
```

Output:
```text
Age      Marks
count   4.000000   4.000000
mean   21.500000  88.500000
std     1.290994   6.952218
min    20.000000  79.000000
25%    20.750000  85.750000
50%    21.500000  90.000000
75%    22.250000  92.750000
max    23.000000  95.000000
```

## 8. Creating a DataFrame from Other Structures

```python
# From a list of dictionaries (like JSON records)
records = [
    {'Product': 'Laptop', 'Price': 55000},
    {'Product': 'Mouse', 'Price': 500},
    {'Product': 'Keyboard', 'Price': 1200}
]
products_df = pd.DataFrame(records)
print(products_df)
```

Output:
```text
Product  Price
0    Laptop  55000
1     Mouse    500
2  Keyboard   1200
```

```python
# From a NumPy 2-D array
matrix_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
from_numpy_df = pd.DataFrame(matrix_data, columns=['A', 'B', 'C'])
print(from_numpy_df)
```

Output:
```text
A  B  C
0  1  2  3
1  4  5  6
2  7  8  9
```

```python
# From a list of lists
rows = [['Chai', 20], ['Coffee', 40], ['Juice', 60]]
menu_df = pd.DataFrame(rows, columns=['Item', 'Price'])
print(menu_df)
```

Output:
```text
Item  Price
0    Chai     20
1  Coffee     40
2   Juice     60
```

## 9. Series vs DataFrame — Selecting a Single Column

```python
name_column = student_data['Name']
print(name_column)
print('\nType of a single column:', type(name_column))  # It's a Series!
```

Output:
```text
0    Rahul
1    Aisha
2    Kabir
3    Meena
Name: Name, dtype: str

Type of a single column: <class 'pandas.Series'>
```

```python
# Selecting multiple columns returns a DataFrame
subset = student_data[['Name', 'Marks']]
print(subset)
print('\nType:', type(subset))
```

Output:
```text
Name  Marks
0  Rahul     88
1  Aisha     92
2  Kabir     79
3  Meena     95

Type: <class 'pandas.DataFrame'>
```

## 10. Setting a Custom Index

```python
indexed_students = student_data.set_index('Name')
print(indexed_students)
```

Output:
```text
Age  Marks       City
Name                        
Rahul   21     88  Bengaluru
Aisha   22     92    Chennai
Kabir   20     79      Delhi
Meena   23     95     Mumbai
```

```python
# Resetting back to default numeric index
back_to_default = indexed_students.reset_index()
print(back_to_default)
```

Output:
```text
Name  Age  Marks       City
0  Rahul   21     88  Bengaluru
1  Aisha   22     92    Chennai
2  Kabir   20     79      Delhi
3  Meena   23     95     Mumbai
```

## 11. Renaming Columns

```python
renamed = student_data.rename(columns={'Marks': 'Score', 'City': 'Location'})
print(renamed)
```

Output:
```text
Name  Age  Score   Location
0  Rahul   21     88  Bengaluru
1  Aisha   22     92    Chennai
2  Kabir   20     79      Delhi
3  Meena   23     95     Mumbai
```

## 12. Basic Statistics on Columns

```python
print('Mean marks  :', student_data['Marks'].mean())
print('Max marks   :', student_data['Marks'].max())
print('Min marks   :', student_data['Marks'].min())
print('Sum of marks:', student_data['Marks'].sum())
print('Std dev     :', student_data['Marks'].std())
```

Output:
```text
Mean marks  : 88.5
Max marks   : 95
Min marks   : 79
Sum of marks: 354
Std dev     : 6.95221787153807
```

## 13. `value_counts()` — Frequency of Each Value

```python
feedback = pd.Series(['Good', 'Excellent', 'Good', 'Average', 'Excellent', 'Good'])
print(feedback.value_counts())
```

Output:
```text
Good         3
Excellent    2
Average      1
Name: count, dtype: int64
```

## 14. Checking Uniqueness

```python
print('Unique cities:', student_data['City'].unique())
print('Number of unique cities:', student_data['City'].nunique())
```

Output:
```text
Unique cities: <StringArray>
['Bengaluru', 'Chennai', 'Delhi', 'Mumbai']
Length: 4, dtype: str
Number of unique cities: 4
```

## 15. Sorting a DataFrame

```python
sorted_by_marks = student_data.sort_values('Marks', ascending=False)
print(sorted_by_marks)
```

Output:
```text
Name  Age  Marks       City
3  Meena   23     95     Mumbai
1  Aisha   22     92    Chennai
0  Rahul   21     88  Bengaluru
2  Kabir   20     79      Delhi
```

## 16. Adding a New Column (Derived)

```python
student_data['Grade'] = np.where(student_data['Marks'] >= 90, 'A', 'B')
print(student_data)
```

Output:
```text
Name  Age  Marks       City Grade
0  Rahul   21     88  Bengaluru     B
1  Aisha   22     92    Chennai     A
2  Kabir   20     79      Delhi     B
3  Meena   23     95     Mumbai     A
```

## 17. Series Arithmetic — Vectorized Operations (Like NumPy)

```python
prices = pd.Series([100, 200, 300, 400])
gst = prices * 0.18
total = prices + gst
print('GST   :', gst.tolist())
print('Total :', total.tolist())
```

Output:
```text
GST   : [18.0, 36.0, 54.0, 72.0]
Total : [118.0, 236.0, 354.0, 472.0]
```

## 18. Checking for Duplicate Rows

```python
dup_check_df = pd.DataFrame({'Item': ['Pen', 'Pen', 'Book'], 'Price': [10, 10, 200]})
print('Duplicated rows:\n', dup_check_df.duplicated())
print('\nDropped duplicates:\n', dup_check_df.drop_duplicates())
```

Output:
```text
Duplicated rows:
 0    False
1     True
2    False
dtype: bool

Dropped duplicates:
    Item  Price
0   Pen     10
2  Book    200
```

## 19. `apply()` — Applying a Custom Function to a Column

```python
def grade_letter(marks):
    if marks >= 90: return 'A'
    elif marks >= 75: return 'B'
    else: return 'C'

student_data['Letter'] = student_data['Marks'].apply(grade_letter)
print(student_data)
```

Output:
```text
Name  Age  Marks       City Grade Letter
0  Rahul   21     88  Bengaluru     B      B
1  Aisha   22     92    Chennai     A      A
2  Kabir   20     79      Delhi     B      B
3  Meena   23     95     Mumbai     A      A
```

## 20. Summary — What You Learned

- Pandas is built on NumPy, adds labeled Series/DataFrame structures
- Creating Series and DataFrames from lists, dicts, NumPy arrays, records
- Core attributes: `shape`, `columns`, `index`, `dtypes`
- Overview methods: `head`, `tail`, `info`, `describe`
- Column selection, renaming, indexing, sorting
- Basic stats, `value_counts()`, duplicates, `apply()`

### Next Notebook
Continue to **`02_dataframe_creation.ipynb`** for every method of building and loading DataFrames, including CSV/Excel/JSON I/O.
