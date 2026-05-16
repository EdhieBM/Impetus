#!/usr/bin/env python3
"""
Compute accuracy/metrix from baseline CSV results.
Extracts final numeric answer from GSM8K predictions and compares to reference.

Usage:
    python benchmarks/evaluate_results.py results/baseline_Qwen2.5-3B-Instruct_20250401_120000.csv
"""

import argparse
import csv
import re
from pathlib import Path


def extract_answer_gsm8k(text: str) -> str:
    """Extract the final numeric answer from a GSM8K-style response."""
    # Look for "#### <number>" pattern used in GSM8K
    m = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if m:
        return m.group(1)
    # Fallback: last number in the text
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else ""


def extract_answer_arc(text: str) -> str:
    """Extract letter answer (A, B, C, D) from ARC response."""
    m = re.search(r"\b([A-D])\b", text.strip())
    return m.group(1) if m else ""


def normalize_answer(text: str) -> str:
    """Clean and normalize for comparison."""
    return text.strip().lower()


def evaluate_gsm8k(results: list[dict]) -> dict:
    correct = 0
    total = len(results)
    for r in results:
        pred = extract_answer_gsm8k(r["prediction"])
        ref = extract_answer_gsm8k(r["reference"])
        if pred == ref:
            correct += 1
    accuracy = correct / total if total > 0 else 0
    latencies = [r["latency_s"] for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "avg_latency_s": round(avg_latency, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline benchmark results")
    parser.add_argument("csv_path", type=str, help="Path to results CSV")
    args = parser.parse_args()

    with open(args.csv_path, newline="") as f:
        reader = csv.DictReader(f)
        results = list(reader)

    benchmarks = {}
    for r in results:
        b = r["benchmark"]
        if b not in benchmarks:
            benchmarks[b] = []
        benchmarks[b].append(r)

    print(f"File: {args.csv_path}")
    print(f"Total samples: {len(results)}")
    print()

    for bname, bresults in benchmarks.items():
        if bname == "gsm8k":
            stats = evaluate_gsm8k(bresults)
            print(f"GSM8K  | Acc: {stats['accuracy']:.2%}  ({stats['correct']}/{stats['total']})  | "
                  f"Avg latency: {stats['avg_latency_s']:.2f}s")
        else:
            print(f"{bname} | {len(bresults)} samples (no evaluator yet)")

    print()


if __name__ == "__main__":
    main()
