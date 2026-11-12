# SQL Fundamentals & Query Execution Order

## Logical Query Processing Order
Unlike procedural programming, SQL queries are evaluated in a specific logical order:
1. `FROM` & `JOIN` (determine data sources)
2. `WHERE` (filter rows)
3. `GROUP BY` (aggregate groups)
4. `HAVING` (filter aggregated groups)
5. `SELECT` (project expressions and aliases)
6. `DISTINCT` (eliminate duplicates)
7. `ORDER BY` (sort results)
8. `LIMIT` / `OFFSET` (restrict output count)
