# SQL Indexing & Performance Tuning

## B-Tree Indexes
- Default index structure for relational databases.
- Efficient for range scans, exact lookups, and sorting.

## Best Practices
- Index columns used in `WHERE`, `JOIN`, and `ORDER BY`.
- Avoid leading wildcards in `LIKE '%keyword'` queries (invalidates index usage).
- Analyze query execution plans with `EXPLAIN ANALYZE`.
