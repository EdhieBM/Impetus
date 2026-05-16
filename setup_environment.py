#!/usr/bin/env python3
"""
Phase 0 — Environment Setup
Verifies all dependencies and hardware are ready for experimentation.
Run: python setup_environment.py
"""

import importlib
import subprocess
import sys
from pathlib import Path


REQUIREMENTS = [
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "evaluate",
    "sentencepiece",
    "bitsandbytes",
]


def check_imports():
    print("=" * 60)
    print("Checking Python package imports...")
    print("=" * 60)
    for pkg in REQUIREMENTS:
        try:
            importlib.import_module(pkg)
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [FAIL] {pkg} — not installed")
    print()


def check_hardware():
    print("=" * 60)
    print("Hardware check")
    print("=" * 60)
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    [{i}] {torch.cuda.get_device_name(i)}")
                print(f"         Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
        print(f"  MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    print()


def check_transformers():
    print("=" * 60)
    print("Transformers + Accelerate check")
    print("=" * 60)
    try:
        import transformers
        import accelerate
        print(f"  Transformers version: {transformers.__version__}")
        print(f"  Accelerate version: {accelerate.__version__}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    print()


def check_bnb():
    print("=" * 60)
    print("BitsAndBytes quantization check")
    print("=" * 60)
    try:
        import bitsandbytes
        print(f"  bitsandbytes version: {bitsandbytes.__version__}")
        print(f"  CUMA setup: {bitsandbytes.cuda_setup.main()}")
    except Exception as e:
        print(f"  [WARN] bitsandbytes not fully configured: {e}")
    print()


def create_dirs():
    dirs = ["models", "benchmarks", "energy_module", "reranker",
            "latent_optimization", "notebooks", "experiments",
            "logs", "results", "reports"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("  [OK] All directories exist")


if __name__ == "__main__":
    check_imports()
    check_hardware()
    check_transformers()
    check_bnb()
    create_dirs()
    print("=" * 60)
    print("Environment setup complete.")
    print("=" * 60)
