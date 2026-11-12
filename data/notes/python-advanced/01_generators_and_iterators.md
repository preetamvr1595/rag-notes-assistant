# Python Generators & The Iterator Protocol

An iterator implements `__iter__()` and `__next__()`.
A generator is a function containing the `yield` keyword that produces values lazily on demand, minimizing memory footprint.

```python
def fibonacci_gen():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
```
