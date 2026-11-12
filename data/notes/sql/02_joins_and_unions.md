# SQL Joins & Set Operations

## Join Types
- `INNER JOIN`: Returns rows that have matching values in both tables.
- `LEFT JOIN`: Returns all rows from left table, plus matched rows from right table (NULL if no match).
- `RIGHT JOIN`: Returns all rows from right table, plus matched from left.
- `FULL OUTER JOIN`: Combines results of both LEFT and RIGHT joins.
- `CROSS JOIN`: Cartesian product of two tables.

## UNION vs UNION ALL
- `UNION`: Combines result sets of two queries and removes duplicates.
- `UNION ALL`: Combines result sets preserving all duplicate rows (significantly faster).
