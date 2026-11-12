# Python Context Managers & contextlib

Context managers manage resource setup and teardown using `__enter__` and `__exit__`.

```python
from contextlib import contextmanager

@contextmanager
def managed_resource():
    print("Acquiring resource")
    yield "RESOURCE"
    print("Releasing resource")
```
