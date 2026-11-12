# SQL Window Functions

Window functions perform calculations across a set of table rows related to the current row without collapsing them into a single row.

```sql
SELECT 
    employee_id,
    department_id,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) as rank_in_dept,
    AVG(salary) OVER (PARTITION BY department_id) as dept_avg_salary
FROM employees;
```

Common window functions: `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`, `LEAD()`, `LAG()`.
