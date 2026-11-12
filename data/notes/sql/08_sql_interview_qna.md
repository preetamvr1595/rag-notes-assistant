# Top SQL Interview Questions & Answers

### Q1: What is the difference between RANK() and DENSE_RANK()?
`RANK()` leaves gaps in rank numbering after ties (e.g., 1, 2, 2, 4).
`DENSE_RANK()` does not leave gaps (e.g., 1, 2, 2, 3).

### Q2: How do you find the second highest salary?
```sql
SELECT MAX(salary) FROM employees 
WHERE salary < (SELECT MAX(salary) FROM employees);
```
