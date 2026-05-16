#!/usr/bin/env python3
"""
Phase 1 — Baseline Benchmarking
Measures model performance BEFORE any EBM modifications.

Usage:
    python benchmarks/run_baseline.py --model Qwen/Qwen2.5-3B-Instruct
    python benchmarks/run_baseline.py --model Qwen/Qwen2.5-3B-Instruct --benchmarks gsm8k arc
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, device: str = "auto"):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - t0
    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return answer.strip(), latency


def load_gsm8k(split: str = "test", max_samples: int = None):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def load_arc(split: str = "test", max_samples: int = None):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def load_bbh(subset: str = "logic", max_samples: int = None):
    from datasets import load_dataset
    ds = load_dataset("lukaemon/bbh", subset, split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def format_gsm8k_prompt(question: str) -> str:
    return f"Solve the following math problem step by step.\n\nQuestion: {question}\n\nAnswer:"


def format_arc_prompt(question: str, choices: list) -> str:
    choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    return f"Question: {question}\n\n{choices_str}\n\nAnswer (letter only):"


def run_gsm8k(model, tokenizer, max_samples: int = 200) -> list[dict]:
    ds = load_gsm8k("test", max_samples)
    results = []
    for i, example in enumerate(ds):
        prompt = format_gsm8k_prompt(example["question"])
        answer, latency = generate(model, tokenizer, prompt)
        results.append({
            "benchmark": "gsm8k",
            "question": example["question"],
            "reference": example["answer"],
            "prediction": answer,
            "latency_s": round(latency, 3),
        })
        if (i + 1) % 50 == 0:
            print(f"  GSM8K: {i+1}/{len(ds)}")
    return results


def save_results(results: list[dict], model_name: str, output_dir: str = "results"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_tag = model_name.split("/")[-1] if "/" in model_name else model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"baseline_{model_tag}_{timestamp}.csv"
    if results:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"Results saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Baseline benchmark runner")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "arc", "bbh"])
    parser.add_argument("--max-samples", type=int, default=200, help="samples per benchmark")
    parser.add_argument("--output", default="results")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.device)
    all_results = []

    if "gsm8k" in args.benchmarks:
        print(f"\n--- GSM8K ---")
        all_results.extend(run_gsm8k(model, tokenizer, args.max_samples))

    if "arc" in args.benchmarks:
        print(f"\n--- ARC (not yet implemented) ---")

    if "bbh" in args.benchmarks:
        print(f"\n--- BBH (not yet implemented) ---")

    save_results(all_results, args.model, args.output)
    print(f"\nDone. Total samples: {len(all_results)}")


if __name__ == "__main__":
    main()
