# SQL Subqueries & Common Table Expressions (CTEs)

## Common Table Expression (WITH clause)
CTEs improve readability and modularity:

```sql
WITH regional_sales AS (
    SELECT region, SUM(amount) AS total_sales
    FROM orders
    GROUP BY region
)
SELECT region, total_sales
FROM regional_sales
WHERE total_sales > 10000;
```
