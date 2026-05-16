#!/usr/bin/env python3
"""
Phase 1 — Baseline Benchmarking (batched for dual-GPU)

Usage:
    python benchmarks/run_baseline.py --model Qwen/Qwen2.5-3B-Instruct --batch-size 8
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, device: str = "auto"):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def batch_generate(
    model, tokenizer, prompts: list[str],
    max_new_tokens: int = 512, batch_size: int = 8
) -> list[tuple[str, float]]:
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        batch_latency = time.perf_counter() - t0
        per_sample_latency = batch_latency / len(batch)
        input_len = inputs.input_ids.shape[-1]
        for j, out in enumerate(outputs):
            answer = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
            results.append((answer, round(per_sample_latency, 3)))
        if (i + batch_size) % 50 == 0 or (i + len(batch)) >= len(prompts):
            print(f"  GSM8K: {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return results


def load_gsm8k(split: str = "test", max_samples: int = None):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def format_gsm8k_prompt(question: str) -> str:
    return f"Solve the following math problem step by step.\n\nQuestion: {question}\n\nAnswer:"


def run_gsm8k(model, tokenizer, max_samples: int = 200, batch_size: int = 8) -> list[dict]:
    ds = load_gsm8k("test", max_samples)
    prompts = [format_gsm8k_prompt(ex["question"]) for ex in ds]
    answers = batch_generate(model, tokenizer, prompts, batch_size=batch_size)
    results = []
    for i, example in enumerate(ds):
        results.append({
            "benchmark": "gsm8k",
            "question": example["question"],
            "reference": example["answer"],
            "prediction": answers[i][0],
            "latency_s": answers[i][1],
        })
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
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k"])
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="results")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.device)
    all_results = []

    if "gsm8k" in args.benchmarks:
        print(f"\n--- GSM8K (batch_size={args.batch_size}) ---")
        all_results.extend(run_gsm8k(model, tokenizer, args.max_samples, args.batch_size))

    save_results(all_results, args.model, args.output)
    print(f"\nDone. Total samples: {len(all_results)}")


if __name__ == "__main__":
    main()
