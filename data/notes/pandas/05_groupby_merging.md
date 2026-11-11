# 05 — GroupBy, Merging & Aggregation in Pandas (Preetham's Pandas Mastery Series)

Part 5 of 5 (final notebook). Covers `groupby()`, aggregation functions, `pivot_table()`, `merge()`, `join()`, and `concat()`.

```python
import pandas as pd
import numpy as np

sales = pd.DataFrame({
    'Region': ['North', 'South', 'North', 'East', 'South', 'East', 'North'],
    'Product': ['Laptop', 'Mouse', 'Mouse', 'Laptop', 'Laptop', 'Mouse', 'Keyboard'],
    'Units': [5, 20, 15, 3, 8, 12, 10],
    'Revenue': [250000, 4000, 3000, 150000, 400000, 2400, 15000]
})
print(sales)
```

Output:
```text
Region   Product  Units  Revenue
0  North    Laptop      5   250000
1  South     Mouse     20     4000
2  North     Mouse     15     3000
3   East    Laptop      3   150000
4  South    Laptop      8   400000
5   East     Mouse     12     2400
6  North  Keyboard     10    15000
```

## 1. `groupby()` — Splitting Data into Groups

```python
region_groups = sales.groupby('Region')
print('Group sizes:\n', region_groups.size())
```

Output:
```text
Group sizes:
 Region
East     2
North    3
South    2
dtype: int64
```

## 2. Aggregating a Single Column

```python
total_revenue_by_region = sales.groupby('Region')['Revenue'].sum()
print(total_revenue_by_region)
```

Output:
```text
Region
East     152400
North    268000
South    404000
Name: Revenue, dtype: int64
```

## 3. Aggregating Multiple Statistics — `agg()`

```python
revenue_stats = sales.groupby('Region')['Revenue'].agg(['sum', 'mean', 'max', 'min', 'count'])
print(revenue_stats)
```

Output:
```text
sum           mean     max   min  count
Region                                            
East    152400   76200.000000  150000  2400      2
North   268000   89333.333333  250000  3000      3
South   404000  202000.000000  400000  4000      2
```

## 4. Grouping by Multiple Columns

```python
region_product_summary = sales.groupby(['Region', 'Product'])['Revenue'].sum()
print(region_product_summary)
```

Output:
```text
Region  Product 
East    Laptop      150000
        Mouse         2400
North   Keyboard     15000
        Laptop      250000
        Mouse         3000
South   Laptop      400000
        Mouse         4000
Name: Revenue, dtype: int64
```

## 5. Different Aggregations per Column

```python
custom_agg = sales.groupby('Region').agg({
    'Units': 'sum',
    'Revenue': ['sum', 'mean']
})
print(custom_agg)
```

Output:
```text
Units Revenue               
         sum     sum           mean
Region                             
East      15  152400   76200.000000
North     30  268000   89333.333333
South     28  404000  202000.000000
```

## 6. `transform()` — Group-wise Calculation Kept at Original Shape

```python
sales['Region_Avg_Revenue'] = sales.groupby('Region')['Revenue'].transform('mean')
print(sales)
```

Output:
```text
Region   Product  Units  Revenue  Region_Avg_Revenue
0  North    Laptop      5   250000        89333.333333
1  South     Mouse     20     4000       202000.000000
2  North     Mouse     15     3000        89333.333333
3   East    Laptop      3   150000        76200.000000
4  South    Laptop      8   400000       202000.000000
5   East     Mouse     12     2400        76200.000000
6  North  Keyboard     10    15000        89333.333333
```

## 7. `filter()` — Keeping Only Groups That Match a Condition

```python
high_volume_regions = sales.groupby('Region').filter(lambda g: g['Units'].sum() > 20)
print(high_volume_regions)
```

Output:
```text
Region   Product  Units  Revenue  Region_Avg_Revenue
0  North    Laptop      5   250000        89333.333333
1  South     Mouse     20     4000       202000.000000
2  North     Mouse     15     3000        89333.333333
4  South    Laptop      8   400000       202000.000000
6  North  Keyboard     10    15000        89333.333333
```

## 8. Custom Aggregation Function

```python
def revenue_range(group):
    return group.max() - group.min()

custom_result = sales.groupby('Region')['Revenue'].agg(revenue_range)
print(custom_result)
```

Output:
```text
Region
East     147600
North    247000
South    396000
Name: Revenue, dtype: int64
```

## 9. `pivot_table()` — Excel-Style Summary Tables

```python
pivot = sales.pivot_table(values='Revenue', index='Region', columns='Product', aggfunc='sum', fill_value=0)
print(pivot)
```

Output:
```text
Product  Keyboard  Laptop  Mouse
Region                          
East            0  150000   2400
North       15000  250000   3000
South           0  400000   4000
```

```python
pivot_with_margins = sales.pivot_table(values='Revenue', index='Region', columns='Product', aggfunc='sum', fill_value=0, margins=True)
print(pivot_with_margins)
```

Output:
```text
Product  Keyboard  Laptop  Mouse     All
Region                                  
East            0  150000   2400  152400
North       15000  250000   3000  268000
South           0  400000   4000  404000
All         15000  800000   9400  824400
```

## 10. `pivot()` — Simple Reshape (No Aggregation)

```python
simple_data = pd.DataFrame({'Date': ['D1','D1','D2','D2'], 'City': ['BLR','MUM','BLR','MUM'], 'Temp': [28, 32, 27, 31]})
reshaped = simple_data.pivot(index='Date', columns='City', values='Temp')
print(reshaped)
```

Output:
```text
City  BLR  MUM
Date          
D1     28   32
D2     27   31
```

## 11. `merge()` — SQL-Style Joins

```python
customers = pd.DataFrame({'CustomerID': [1, 2, 3, 4], 'Name': ['Rahul', 'Aisha', 'Kabir', 'Zoya']})
orders = pd.DataFrame({'OrderID': [101, 102, 103, 104], 'CustomerID': [1, 2, 2, 5], 'Amount': [500, 1200, 800, 300]})

print('Customers:\n', customers)
print('\nOrders:\n', orders)
```

Output:
```text
Customers:
    CustomerID   Name
0           1  Rahul
1           2  Aisha
2           3  Kabir
3           4   Zoya

Orders:
    OrderID  CustomerID  Amount
0      101           1     500
1      102           2    1200
2      103           2     800
3      104           5     300
```

```python
inner_join = pd.merge(customers, orders, on='CustomerID', how='inner')
print('INNER JOIN (only matches):\n', inner_join)
```

Output:
```text
INNER JOIN (only matches):
    CustomerID   Name  OrderID  Amount
0           1  Rahul      101     500
1           2  Aisha      102    1200
2           2  Aisha      103     800
```

```python
left_join = pd.merge(customers, orders, on='CustomerID', how='left')
print('LEFT JOIN (all customers):\n', left_join)
```

Output:
```text
LEFT JOIN (all customers):
    CustomerID   Name  OrderID  Amount
0           1  Rahul    101.0   500.0
1           2  Aisha    102.0  1200.0
2           2  Aisha    103.0   800.0
3           3  Kabir      NaN     NaN
4           4   Zoya      NaN     NaN
```

```python
right_join = pd.merge(customers, orders, on='CustomerID', how='right')
print('RIGHT JOIN (all orders):\n', right_join)
```

Output:
```text
RIGHT JOIN (all orders):
    CustomerID   Name  OrderID  Amount
0           1  Rahul      101     500
1           2  Aisha      102    1200
2           2  Aisha      103     800
3           5    NaN      104     300
```

```python
outer_join = pd.merge(customers, orders, on='CustomerID', how='outer')
print('OUTER JOIN (everything):\n', outer_join)
```

Output:
```text
OUTER JOIN (everything):
    CustomerID   Name  OrderID  Amount
0           1  Rahul    101.0   500.0
1           2  Aisha    102.0  1200.0
2           2  Aisha    103.0   800.0
3           3  Kabir      NaN     NaN
4           4   Zoya      NaN     NaN
5           5    NaN    104.0   300.0
```

## 12. `join()` — Merge on Index

```python
left_indexed = customers.set_index('CustomerID')
right_indexed = orders.set_index('CustomerID')

joined = left_indexed.join(right_indexed, how='left')
print(joined)
```

Output:
```text
Name  OrderID  Amount
CustomerID                        
1           Rahul    101.0   500.0
2           Aisha    102.0  1200.0
2           Aisha    103.0   800.0
3           Kabir      NaN     NaN
4            Zoya      NaN     NaN
```

## 13. `concat()` — Stacking DataFrames

```python
jan_sales = pd.DataFrame({'Product': ['Laptop', 'Mouse'], 'Units': [5, 20]})
feb_sales = pd.DataFrame({'Product': ['Laptop', 'Mouse'], 'Units': [7, 18]})

stacked_vertically = pd.concat([jan_sales, feb_sales], keys=['Jan', 'Feb'])
print(stacked_vertically)
```

Output:
```text
Product  Units
Jan 0  Laptop      5
    1   Mouse     20
Feb 0  Laptop      7
    1   Mouse     18
```

```python
side_by_side = pd.concat([jan_sales, feb_sales], axis=1, keys=['Jan', 'Feb'])
print(side_by_side)
```

Output:
```text
Jan           Feb      
  Product Units Product Units
0  Laptop     5  Laptop     7
1   Mouse    20   Mouse    18
```

## 14. Sorting Grouped Results

```python
top_regions = sales.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
print(top_regions)
```

Output:
```text
Region
South    404000
North    268000
East     152400
Name: Revenue, dtype: int64
```

## 15. Real-World Mini Project — Full Sales Report

```python
report = (
    sales.groupby(['Region', 'Product'])
    .agg(Total_Units=('Units', 'sum'), Total_Revenue=('Revenue', 'sum'))
    .reset_index()
    .sort_values('Total_Revenue', ascending=False)
)
print(report)
```

Output:
```text
Region   Product  Total_Units  Total_Revenue
5  South    Laptop            8         400000
3  North    Laptop            5         250000
0   East    Laptop            3         150000
2  North  Keyboard           10          15000
6  South     Mouse           20           4000
4  North     Mouse           15           3000
1   East     Mouse           12           2400
```

## 16. Summary — What You Learned

| Method | Purpose |
|---|---|
| `groupby()` | Split data into groups |
| `agg()` | Multiple/custom aggregations |
| `transform()` | Group stats mapped back to original shape |
| `filter()` | Keep only groups matching a condition |
| `pivot_table()` | Excel-style summary with aggregation |
| `pivot()` | Simple reshape without aggregation |
| `merge()` | SQL-style joins (inner/left/right/outer) |
| `join()` | Merge on index |
| `concat()` | Stack DataFrames vertically/horizontally |

## 🎉 Series Complete!

You've now covered the **entire Pandas fundamentals series** end-to-end:
1. Basics (Series & DataFrame)
2. DataFrame Creation & I/O
3. Indexing & Selecting Data
4. Data Cleaning
5. GroupBy, Merging & Aggregation

Next up in your Data Science Mastery repo: **Matplotlib**, **Seaborn**, **Scikit-learn**, **TensorFlow**.
