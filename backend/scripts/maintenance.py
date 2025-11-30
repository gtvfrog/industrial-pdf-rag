import os
import shutil
import sys
from pathlib import Path

def clean():
    print("Cleaning up...")
    root = Path(".")
    
    for p in root.rglob("__pycache__"):
        if p.is_dir():
            print(f"   Removing {p}")
            shutil.rmtree(p)
            
    for p in root.rglob("*.pyc"):
        if p.is_file():
            p.unlink()
            
    print("Clean complete")

def reset_cache():
    print("Resetting cache...")
    
    dirs_to_remove = ["models_cache", "data", "tmp"]
    
    for d in dirs_to_remove:
        p = Path(d)
        if p.exists() and p.is_dir():
            print(f"   Removing {p}")
            try:
                shutil.rmtree(p)
            except Exception as e:
                print(f"   Failed to remove {p}: {e}")
        else:
            print(f"   Skipping {p} (not found)")
            
    print("Cache reset complete")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "clean":
            clean()
        elif cmd == "reset-cache":
            reset_cache()
        else:
            print(f"Unknown command: {cmd}")
    else:
        print("Usage: python maintenance.py [clean|reset-cache]")
