# Pandas Interview Q&A — Practice Notebook

A set of classic pandas exercises, cleaned up from the original draft (fixed a stray duplicate line, completed one unfinished question, and added a few extra questions). Fully self-contained — no external CSV needed.

### Q1. Import pandas under the alias `pd`

```python
import pandas as pd
import numpy as np
```

### Q2. Print the version of pandas that has been imported

```python
pd.__version__
```

Output:
```text
'3.0.2'
```

### Q3. Print out version information for pandas' dependencies

```python
pd.show_versions()
```

Output:
```text
INSTALLED VERSIONS
------------------
commit                : ab90747e3dae0e69b1bdbf083820b8075689b34b
python                : 3.12.3
python-bits           : 64
OS                    : Linux
OS-release            : 6.18.5
Version               : #1 SMP PREEMPT_DYNAMIC @0
machine               : x86_64
processor             : x86_64
byteorder             : little
LC_ALL                : None
LANG                  : None
LOCALE                : C.UTF-8

pandas                : 3.0.2
numpy          …
```

## Working DataFrame
A few of the fundamental routines for selecting, sorting, adding, and aggregating data in DataFrames.

```python
data = {
    'animal': ['cow', 'cat', 'snake', 'dog', 'dog', 'cat', 'cow', 'snake', 'dog', 'dog'],
    'age': [2, 3, 4, 2, np.nan, 5, 3, np.nan, 3, 9],
    'visites': [1, 2, 3, 4, 5, 6, 7, 8, 4, 10],
    'priority': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'no', 'yes']
}
labels = list('abcdefghij')
df = pd.DataFrame(data, index=labels)
df
```

Output:
```text
animal  age  visites priority
a    cow  2.0        1      yes
b    cat  3.0        2       no
c  snake  4.0        3      yes
d    dog  2.0        4       no
e    dog  NaN        5      yes
f    cat  5.0        6       no
g    cow  3.0        7      yes
h  snake  NaN        8       no
i    dog  3.0        4       no
j    dog  9.0       10      yes
```

### Q4. Display a summary of the information about this DataFrame and its data

```python
df.info()
```

Output:
```text
<class 'pandas.DataFrame'>
Index: 10 entries, a to j
Data columns (total 4 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   animal    10 non-null     str    
 1   age       8 non-null      float64
 2   visites   10 non-null     int64  
 3   priority  10 non-null     str    
dtypes: float64(1), int64(1), str(2)
memory usage: 400.0+ bytes
```

### Q5. Return the first 3 rows of the DataFrame `df`

```python
df.head(3)
```

Output:
```text
animal  age  visites priority
a    cow  2.0        1      yes
b    cat  3.0        2       no
c  snake  4.0        3      yes
```

### Q6. Select just the `animal` and `age` columns from `df`

```python
df[['animal', 'age']]
```

Output:
```text
animal  age
a    cow  2.0
b    cat  3.0
c  snake  4.0
d    dog  2.0
e    dog  NaN
f    cat  5.0
g    cow  3.0
h  snake  NaN
i    dog  3.0
j    dog  9.0
```

### Q7. Select the data in rows [3, 4, 5] and columns `['animal', 'age']`

```python
df.loc[df.index[[3, 4, 5]], ['animal', 'age']]
```

Output:
```text
animal  age
d    dog  2.0
e    dog  NaN
f    cat  5.0
```

### Q8. Select only the rows where the number of visits is greater than 3

```python
df[df['visites'] > 3]
```

Output:
```text
animal  age  visites priority
d    dog  2.0        4       no
e    dog  NaN        5      yes
f    cat  5.0        6       no
g    cow  3.0        7      yes
h  snake  NaN        8       no
i    dog  3.0        4       no
j    dog  9.0       10      yes
```

### Q9. Select the rows where age is missing (NaN)

```python
df[df['age'].isna()]
```

Output:
```text
animal  age  visites priority
e    dog  NaN        5      yes
h  snake  NaN        8       no
```

### Q10. Select the rows where the animal is a cat AND age is less than 3

```python
df[(df['animal'] == 'cat') & (df['age'] < 3)]
```

Output:
```text
Empty DataFrame
Columns: [animal, age, visites, priority]
Index: []
```

### Q11. Select the rows where age is between 2 and 4 (inclusive)

```python
df[(df['age'] >= 2) & (df['age'] <= 4)]
```

Output:
```text
animal  age  visites priority
a    cow  2.0        1      yes
b    cat  3.0        2       no
c  snake  4.0        3      yes
d    dog  2.0        4       no
g    cow  3.0        7      yes
i    dog  3.0        4       no
```

### Q12. Change the age in row 'f' to 1.5

```python
df.loc['f', 'age'] = 1.5
df
```

Output:
```text
animal  age  visites priority
a    cow  2.0        1      yes
b    cat  3.0        2       no
c  snake  4.0        3      yes
d    dog  2.0        4       no
e    dog  NaN        5      yes
f    cat  1.5        6       no
g    cow  3.0        7      yes
h  snake  NaN        8       no
i    dog  3.0        4       no
j    dog  9.0       10      yes
```

### Q13. Calculate the sum of all visits (the total number of visits)

```python
df['visites'].sum()
```

Output:
```text
np.int64(50)
```

### Q14. Calculate the mean age for each different animal in `df`

```python
df.groupby('animal')['age'].mean()
```

Output:
```text
animal
cat      2.250000
cow      2.500000
dog      4.666667
snake    4.000000
Name: age, dtype: float64
```

### Q15. Append a new row 'k' with your choice of values, then delete it to return the original DataFrame

```python
df.loc['k'] = ['monkey', 4, 4, 'yes']
print(df)

df = df.drop(index='k')
df
```

Output:
```text
animal  age  visites priority
a     cow  2.0        1      yes
b     cat  3.0        2       no
c   snake  4.0        3      yes
d     dog  2.0        4       no
e     dog  NaN        5      yes
f     cat  1.5        6       no
g     cow  3.0        7      yes
h   snake  NaN        8       no
i     dog  3.0        4       no
j     dog  9.0       10      yes
k  monkey  4.0        4      yes
  animal  age  visites priority
a    cow  2.0        1      yes
b    cat  3.0        2       no
c  snake  4 …
```

### Q16. Count the number of each type of animal in `df`

```python
df['animal'].value_counts()
```

Output:
```text
animal
dog      4
cow      2
cat      2
snake    2
Name: count, dtype: int64
```

### Q17. Sort `df` first by `age` in descending order, then by `visites` in ascending order

```python
df.sort_values(by=['age', 'visites'], ascending=[False, True])
```

Output:
```text
animal  age  visites priority
j    dog  9.0       10      yes
c  snake  4.0        3      yes
b    cat  3.0        2       no
i    dog  3.0        4       no
g    cow  3.0        7      yes
a    cow  2.0        1      yes
d    dog  2.0        4       no
f    cat  1.5        6       no
e    dog  NaN        5      yes
h  snake  NaN        8       no
```

### Q18. The `priority` column contains 'yes'/'no'. Replace it with boolean values (True/False)

```python
df['priority'] = df['priority'].map({'yes': True, 'no': False})
df
```

Output:
```text
animal  age  visites  priority
a    cow  2.0        1      True
b    cat  3.0        2     False
c  snake  4.0        3      True
d    dog  2.0        4     False
e    dog  NaN        5      True
f    cat  1.5        6     False
g    cow  3.0        7      True
h  snake  NaN        8     False
i    dog  3.0        4     False
j    dog  9.0       10      True
```

### Q19. In the `animal` column, change every 'snake' entry to 'python'

```python
df['animal'] = df['animal'].replace('snake', 'python')
df
```

Output:
```text
animal  age  visites  priority
a     cow  2.0        1      True
b     cat  3.0        2     False
c  python  4.0        3      True
d     dog  2.0        4     False
e     dog  NaN        5      True
f     cat  1.5        6     False
g     cow  3.0        7      True
h  python  NaN        8     False
i     dog  3.0        4     False
j     dog  9.0       10      True
```

### Q20. For each animal type and number of visits, find the mean age
(Each row = an animal, each column = a number of visits, values = mean age.)

```python
df.pivot_table(values='age', index='animal', columns='visites', aggfunc='mean')
```

Output:
```text
visites   1    2    3    4    6    7    10
animal                                    
cat      NaN  3.0  NaN  NaN  1.5  NaN  NaN
cow      2.0  NaN  NaN  NaN  NaN  3.0  NaN
dog      NaN  NaN  NaN  2.5  NaN  NaN  9.0
python   NaN  NaN  4.0  NaN  NaN  NaN  NaN
```

### Q21. How do you subtract the row mean from each element in the row?

```python
df_numeric = pd.DataFrame(np.random.RandomState(0).random(size=(5, 8)))
result = df_numeric.sub(df_numeric.mean(axis=1), axis=0)
result
```

Output:
```text
0         1         2         3         4         5         6  \
0 -0.052506  0.113870  0.001444 -0.056437 -0.177665  0.044574 -0.163733   
1  0.423721 -0.156500  0.251784 -0.011046  0.028103  0.385655 -0.468905   
2 -0.669881  0.142521  0.088058  0.179913  0.288519  0.109059 -0.228620   
3 -0.359415  0.162231 -0.334336  0.466979  0.044159 -0.063028 -0.213134   
4 -0.108301  0.003983 -0.545661  0.053185  0.047645  0.052483  0.379297   

          7  
0  0.290453  
1 -0.452812  
2  0.090430  
3   …
```

### Q22. Given a DataFrame of 10 real-number columns, which column has the smallest sum? Return its label.

(Completed — the original draft built the DataFrame but never showed a genuinely varied example. Here we use random data so the answer isn't trivially obvious.)

```python
df_challenge = pd.DataFrame(np.random.RandomState(1).random(size=(5, 10)), columns=list('abcdefghij'))
smallest_sum_col = df_challenge.sum(axis=0).sort_values().index[0]
print('Column sums:\n', df_challenge.sum(axis=0))
print('\nColumn with smallest sum:', smallest_sum_col)
```

Output:
```text
Column sums:
 a    2.724169
b    3.543079
c    1.756324
d    3.195217
e    1.845636
f    2.420822
g    2.283706
h    2.071545
i    1.013048
j    2.495234
dtype: float64

Column with smallest sum: i
```

### Q23. (Bonus) How many unique values are in each column of `df_challenge`?

```python
df_challenge.nunique()
```

Output:
```text
a    5
b    5
c    5
d    5
e    5
f    5
g    5
h    5
i    5
j    5
dtype: int64
```

### Q24. (Bonus) Find and remove duplicate rows from a DataFrame

```python
dup_demo = pd.DataFrame({'x': [1, 1, 2, 3], 'y': ['a', 'a', 'b', 'c']})
print('Before:\n', dup_demo)
print('\nDuplicated flags:\n', dup_demo.duplicated())
print('\nAfter drop_duplicates:\n', dup_demo.drop_duplicates())
```

Output:
```text
Before:
    x  y
0  1  a
1  1  a
2  2  b
3  3  c

Duplicated flags:
 0    False
1     True
2    False
3    False
dtype: bool

After drop_duplicates:
    x  y
0  1  a
2  2  b
3  3  c
```

### Q25. (Bonus) Create a new column that is the cumulative sum of `visites`

```python
df['cumulative_visites'] = df['visites'].cumsum()
df[['animal', 'visites', 'cumulative_visites']]
```

Output:
```text
animal  visites  cumulative_visites
a     cow        1                   1
b     cat        2                   3
c  python        3                   6
d     dog        4                  10
e     dog        5                  15
f     cat        6                  21
g     cow        7                  28
h  python        8                  36
i     dog        4                  40
j     dog       10                  50
```

## Summary

This notebook covers 25 classic pandas interview-style questions: selection, filtering, sorting, grouping, pivoting, replacing values, and row-wise operations — all on a single small, self-contained dataset. No external files needed; runs anywhere pandas is installed.
