"""Clean temporary build, test cache, and pycache artifacts."""
import shutil
import os

def clean():
    targets = [".pytest_cache", ".ruff_cache", "build", "dist", ".coverage"]
    for t in targets:
        if os.path.exists(t):
            shutil.rmtree(t, ignore_errors=True)
            print(f"Removed {t}")
    print("Workspace cleaned.")

if __name__ == "__main__":
    clean()
