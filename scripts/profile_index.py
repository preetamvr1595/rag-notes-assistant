"""Profile memory usage and serialized index footprint."""
import os

def profile_index():
    path = "models/index.joblib"
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Index size on disk: {size_mb:.2f} MB")
    else:
        print("No index file found. Run 'python -m rag_notes.cli build' first.")

if __name__ == "__main__":
    profile_index()
