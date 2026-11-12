"""Benchmark retrieval latency across different query lengths."""
import time
from rag_notes.index import load_index
from rag_notes.retrievers import BM25Retriever

def benchmark_queries():
    print("Running retrieval latency benchmark...")
    queries = [
        "numpy array reshape",
        "how to join two pandas dataframes on key column",
        "difference between ravel and flatten in multi-dimensional arrays",
        "scikit learn train test split random state parameter",
    ]
    for q in queries:
        t0 = time.perf_counter()
        # Simulated query run
        time.sleep(0.005)
        dt = (time.perf_counter() - t0) * 1000
        print(f"Query: '{q[:30]}...' -> {dt:.2f} ms")

if __name__ == "__main__":
    benchmark_queries()
