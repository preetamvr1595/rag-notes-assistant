"""Report corpus statistics, file breakdown, and word counts."""
import os
import glob

def compute_corpus_stats():
    base_dir = "data/notes"
    if not os.path.exists(base_dir):
        print("data/notes directory not found.")
        return
    files = glob.glob(f"{base_dir}/**/*.md", recursive=True)
    total_words = 0
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            total_words += len(fh.read().split())
    print(f"Total Markdown Documents: {len(files)}")
    print(f"Total Word Count: {total_words:,}")

if __name__ == "__main__":
    compute_corpus_stats()
