# Python Decorators & Closures

A decorator wraps a function to extend its behavior without modifying the original code.

```python
from functools import wraps
import time

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter() - t0:.4f}s")
        return result
    return wrapper
```
