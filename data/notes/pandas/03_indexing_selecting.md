# 03 — Indexing & Selecting Data in Pandas (Preetham's Pandas Mastery Series)

Part 3 of 5. Covers `loc`, `iloc`, `at`/`iat`, boolean filtering, multi-condition selection, `query()`, and MultiIndex basics.

```python
import pandas as pd
import numpy as np

employees = pd.DataFrame({
    'Name': ['Rahul', 'Aisha', 'Kabir', 'Meena', 'Zoya', 'Varun'],
    'Department': ['Sales', 'IT', 'IT', 'HR', 'Sales', 'IT'],
    'Salary': [45000, 72000, 68000, 51000, 47000, 80000],
    'Experience': [2, 5, 4, 3, 1, 7]
}, index=['E1', 'E2', 'E3', 'E4', 'E5', 'E6'])
print(employees)
```

Output:
```text
Name Department  Salary  Experience
E1  Rahul      Sales   45000           2
E2  Aisha         IT   72000           5
E3  Kabir         IT   68000           4
E4  Meena         HR   51000           3
E5   Zoya      Sales   47000           1
E6  Varun         IT   80000           7
```

## 1. `loc[]` — Label-Based Selection

```python
print('Single row by label:\n', employees.loc['E2'])
```

Output:
```text
Single row by label:
 Name          Aisha
Department       IT
Salary        72000
Experience        5
Name: E2, dtype: object
```

```python
print('Multiple rows:\n', employees.loc[['E1', 'E4']])
```

Output:
```text
Multiple rows:
      Name Department  Salary  Experience
E1  Rahul      Sales   45000           2
E4  Meena         HR   51000           3
```

```python
print('Row + column selection:\n', employees.loc['E3', 'Salary'])
```

Output:
```text
Row + column selection:
 68000
```

```python
print('Row range + specific columns:\n', employees.loc['E1':'E3', ['Name', 'Salary']])
```

Output:
```text
Row range + specific columns:
      Name  Salary
E1  Rahul   45000
E2  Aisha   72000
E3  Kabir   68000
```

## 2. `iloc[]` — Position-Based Selection

```python
print('Single row by position:\n', employees.iloc[0])
```

Output:
```text
Single row by position:
 Name          Rahul
Department    Sales
Salary        45000
Experience        2
Name: E1, dtype: object
```

```python
print('Row range (position):\n', employees.iloc[1:4])
```

Output:
```text
Row range (position):
      Name Department  Salary  Experience
E2  Aisha         IT   72000           5
E3  Kabir         IT   68000           4
E4  Meena         HR   51000           3
```

```python
print('Specific rows & columns by position:\n', employees.iloc[[0, 2, 4], [0, 2]])
```

Output:
```text
Specific rows & columns by position:
      Name  Salary
E1  Rahul   45000
E3  Kabir   68000
E5   Zoya   47000
```

## 3. `at[]` and `iat[]` — Fast Scalar Access

```python
print('Single value via at  :', employees.at['E2', 'Salary'])
print('Single value via iat :', employees.iat[1, 2])
```

Output:
```text
Single value via at  : 72000
Single value via iat : 72000
```

## 4. Boolean Filtering — Single Condition

```python
high_earners = employees[employees['Salary'] > 60000]
print(high_earners)
```

Output:
```text
Name Department  Salary  Experience
E2  Aisha         IT   72000           5
E3  Kabir         IT   68000           4
E6  Varun         IT   80000           7
```

## 5. Boolean Filtering — Multiple Conditions

```python
senior_it = employees[(employees['Department'] == 'IT') & (employees['Experience'] >= 4)]
print(senior_it)
```

Output:
```text
Name Department  Salary  Experience
E2  Aisha         IT   72000           5
E3  Kabir         IT   68000           4
E6  Varun         IT   80000           7
```

```python
sales_or_hr = employees[(employees['Department'] == 'Sales') | (employees['Department'] == 'HR')]
print(sales_or_hr)
```

Output:
```text
Name Department  Salary  Experience
E1  Rahul      Sales   45000           2
E4  Meena         HR   51000           3
E5   Zoya      Sales   47000           1
```

```python
not_it = employees[~(employees['Department'] == 'IT')]
print(not_it)
```

Output:
```text
Name Department  Salary  Experience
E1  Rahul      Sales   45000           2
E4  Meena         HR   51000           3
E5   Zoya      Sales   47000           1
```

## 6. `isin()` — Filtering by Membership

```python
target_depts = employees[employees['Department'].isin(['Sales', 'HR'])]
print(target_depts)
```

Output:
```text
Name Department  Salary  Experience
E1  Rahul      Sales   45000           2
E4  Meena         HR   51000           3
E5   Zoya      Sales   47000           1
```

## 7. `query()` — SQL-Like Filtering Syntax

```python
result = employees.query('Salary > 50000 and Experience < 6')
print(result)
```

Output:
```text
Name Department  Salary  Experience
E2  Aisha         IT   72000           5
E3  Kabir         IT   68000           4
E4  Meena         HR   51000           3
```

## 8. `between()` — Range Filtering

```python
mid_range_salary = employees[employees['Salary'].between(45000, 70000)]
print(mid_range_salary)
```

Output:
```text
Name Department  Salary  Experience
E1  Rahul      Sales   45000           2
E3  Kabir         IT   68000           4
E4  Meena         HR   51000           3
E5   Zoya      Sales   47000           1
```

## 9. Selecting Columns by dtype

```python
numeric_only = employees.select_dtypes(include='number')
print(numeric_only)
```

Output:
```text
Salary  Experience
E1   45000           2
E2   72000           5
E3   68000           4
E4   51000           3
E5   47000           1
E6   80000           7
```

## 10. Conditional Column Creation with `loc`

```python
employees.loc[employees['Salary'] > 60000, 'Tier'] = 'Senior'
employees.loc[employees['Salary'] <= 60000, 'Tier'] = 'Junior'
print(employees)
```

Output:
```text
Name Department  Salary  Experience    Tier
E1  Rahul      Sales   45000           2  Junior
E2  Aisha         IT   72000           5  Senior
E3  Kabir         IT   68000           4  Senior
E4  Meena         HR   51000           3  Junior
E5   Zoya      Sales   47000           1  Junior
E6  Varun         IT   80000           7  Senior
```

## 11. Modifying Values with `loc`/`iloc`

```python
employees.loc['E1', 'Salary'] = 48000   # give Rahul a raise
employees.iloc[3, 2] = 55000              # update Meena's salary by position
print(employees)
```

Output:
```text
Name Department  Salary  Experience    Tier
E1  Rahul      Sales   48000           2  Junior
E2  Aisha         IT   72000           5  Senior
E3  Kabir         IT   68000           4  Senior
E4  Meena         HR   55000           3  Junior
E5   Zoya      Sales   47000           1  Junior
E6  Varun         IT   80000           7  Senior
```

## 12. MultiIndex Basics — Hierarchical Indexing

```python
multi_df = pd.DataFrame({
    'Sales': [200, 300, 150, 400]
}, index=pd.MultiIndex.from_tuples(
    [('North', 'Q1'), ('North', 'Q2'), ('South', 'Q1'), ('South', 'Q2')],
    names=['Region', 'Quarter']
))
print(multi_df)
```

Output:
```text
Sales
Region Quarter       
North  Q1         200
       Q2         300
South  Q1         150
       Q2         400
```

```python
print('All North rows:\n', multi_df.loc['North'])
print('\nSpecific combination:\n', multi_df.loc[('South', 'Q2')])
```

Output:
```text
All North rows:
          Sales
Quarter       
Q1         200
Q2         300

Specific combination:
 Sales    400
Name: (South, Q2), dtype: int64
```

## 13. `xs()` — Cross-Section Selection on MultiIndex

```python
print(multi_df.xs('Q1', level='Quarter'))
```

Output:
```text
Sales
Region       
North     200
South     150
```

## 14. Filtering with String Methods

```python
name_starts_with_A = employees[employees['Name'].str.startswith('A')]
print(name_starts_with_A)
```

Output:
```text
Name Department  Salary  Experience    Tier
E2  Aisha         IT   72000           5  Senior
```

## 15. Getting a Random Sample of Rows

```python
sample_rows = employees.sample(n=2, random_state=1)
print(sample_rows)
```

Output:
```text
Name Department  Salary  Experience    Tier
E3  Kabir         IT   68000           4  Senior
E2  Aisha         IT   72000           5  Senior
```

## 16. `nlargest()` and `nsmallest()`

```python
print('Top 2 salaries:\n', employees.nlargest(2, 'Salary'))
print('\nBottom 2 salaries:\n', employees.nsmallest(2, 'Salary'))
```

Output:
```text
Top 2 salaries:
      Name Department  Salary  Experience    Tier
E6  Varun         IT   80000           7  Senior
E2  Aisha         IT   72000           5  Senior

Bottom 2 salaries:
      Name Department  Salary  Experience    Tier
E5   Zoya      Sales   47000           1  Junior
E1  Rahul      Sales   48000           2  Junior
```

## 17. Summary — What You Learned

| Method | Purpose |
|---|---|
| `loc[]` | Label-based row/column selection |
| `iloc[]` | Position-based row/column selection |
| `at[]` / `iat[]` | Fast single-value access |
| Boolean filtering | `df[condition]`, combined with `&`/`\|`/`~` |
| `isin()` | Membership filtering |
| `query()` | SQL-like filter syntax |
| `between()` | Range filtering |
| MultiIndex + `xs()` | Hierarchical indexing |
| `nlargest()`/`nsmallest()` | Top/bottom N rows |

### Next Notebook
Continue to **`04_data_cleaning.ipynb`** for handling missing data, duplicates, type conversion, and string cleaning.
