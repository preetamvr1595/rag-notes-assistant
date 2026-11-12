# Python Asyncio & Asynchronous Programming

Asyncio uses an event loop on a single thread to achieve cooperative multitasking. Ideal for I/O-bound operations (network requests, database queries).

```python
import asyncio

async def fetch_data(delay: int):
    await asyncio.sleep(delay)
    return {"status": "ok"}
```
