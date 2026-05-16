#!/usr/bin/env python3
"""
Impetus — One-shot benchmark runner
Usage: python run.py [--model Qwen/Qwen2.5-3B-Instruct] [--samples 50]
"""

import argparse, os, sys, subprocess, re, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ─── Utils ───────────────────────────────────────────────

def extract_number(text: str) -> str:
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else ""


def batch_generate(model, tokenizer, prompts, max_new_tokens=256, batch_size=32, do_sample=False, temperature=1.0):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=0.9 if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs.input_ids.shape[-1]
        for out in outputs:
            results.append(tokenizer.decode(out[input_len:], skip_special_tokens=True).strip())
    return results


# ─── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=20, help="max candidates per question")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--scorer", default="majority_voting")
    args = parser.parse_args()

    # GPU info
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    dtype = torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7 else torch.bfloat16

    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Loaded.\n")

    # Load GSM8K
    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.select(range(min(args.samples, len(ds))))
    questions = [ex["question"] for ex in ds]
    refs = [extract_number(ex["answer"]) for ex in ds]
    print(f"GSM8K: {len(questions)} samples\n")

    # ── Step 1: Baseline ──
    t0 = time.time()
    baseline_outs = batch_generate(
        model, tokenizer,
        [f"Solve the following math problem step by step.\n\nQuestion: {q}\n\nAnswer:" for q in questions],
        do_sample=False, batch_size=args.batch_size
    )
    baseline_nums = [extract_number(o) for o in baseline_outs]
    baseline_acc = sum(1 for i in range(len(questions)) if baseline_nums[i] == refs[i]) / len(questions)
    baseline_time = time.time() - t0

    # ── Step 2: Generate candidates ──
    t0 = time.time()
    all_prompts, q_idx = [], []
    for i, q in enumerate(questions):
        p = f"Solve the following math problem step by step.\n\nQuestion: {q}\n\nAnswer:"
        for _ in range(args.candidates):
            all_prompts.append(p)
            q_idx.append(i)

    all_outs = batch_generate(
        model, tokenizer, all_prompts,
        do_sample=True, temperature=0.7, batch_size=args.batch_size
    )
    candidates = {i: [] for i in range(len(questions))}
    for idx, out in zip(q_idx, all_outs):
        candidates[idx].append(out)
    gen_time = time.time() - t0

    # ── Step 3: Score & Rerank ──
    t0 = time.time()
    print(f"{'N':>5}  {'Correct':>7}  {'Accuracy':>8}  {'Δ':>7}  {'Time':>6}")
    print("-" * 40)
    print(f"{'greedy':>5}  {int(baseline_acc*len(questions)):>7}  {baseline_acc:>7.2%}  {'—':>7}  {baseline_time:>5.1f}s")

    for N in [5, 10, 20]:
        correct = 0
        for i in range(len(questions)):
            pool = candidates[i][:N]
            # Majority voting: pick most common extracted number
            nums = [extract_number(a) for a in pool]
            counts = {}
            for n in nums:
                counts[n] = counts.get(n, 0) + 1
            best_num = max(counts, key=counts.get)
            if best_num == refs[i]:
                correct += 1
        acc = correct / len(questions)
        delta = acc - baseline_acc
        print(f"{N:>5}  {correct:>7}  {acc:>7.2%}  {delta:>+6.2%}  {gen_time:>5.1f}s")
    print("-" * 40)

    # Save results
    from datetime import datetime
    Path("results").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = f"""Impetus Benchmark — {ts}
Model: {args.model}
GPU: {torch.cuda.get_device_name(0)}
Samples: {len(questions)} | Candidates: {args.candidates}

Baseline (greedy):  {baseline_acc:.2%} ({int(baseline_acc*len(questions))}/{len(questions)})
N=5:                {sum(1 for i in range(len(questions)) if [extract_number(c) for c in candidates[i][:5]].count(max(set([extract_number(c) for c in candidates[i][:5]]), key=[extract_number(c) for c in candidates[i][:5]].count)) == refs[i])/len(questions):.2%}
"""
    with open(f"results/benchmark_{ts}.txt", "w") as f:
        f.write(summary)
    print(f"\nSaved: results/benchmark_{ts}.txt")


if __name__ == "__main__":
    main()
