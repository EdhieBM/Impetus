#!/usr/bin/env python3
"""
Phase 2 — KONA Verifier Benchmark on GSM8K
Compares: baseline accuracy vs reranked accuracy (N candidates → score → best).

Usage:
    python benchmarks/run_reranker.py --model Qwen/Qwen2.5-3B-Instruct
                                      --scorer majority_voting
                                      --n-candidates 5
                                      --max-samples 50
                                      --batch-size 8
"""

import argparse
import csv
import time
import re
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from energy_module.scorers import MajorityVotingScorer, SelfConsistencyScorer


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
    max_new_tokens: int = 256,
    batch_size: int = 8,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> list[str]:
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=0.9 if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs.input_ids.shape[-1]
        for out in outputs:
            answer = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
            results.append(answer)
    return results


def load_gsm8k(split: str = "test", max_samples: int = None):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def format_gsm8k_prompt(question: str) -> str:
    return f"Solve: {question}\nAnswer:"


def extract_answer_number(text: str) -> str:
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else ""


def run_reranker_gsm8k(
    model, tokenizer, scorer, max_samples: int = 50,
    n_candidates: int = 5, batch_size: int = 8
) -> list[dict]:
    ds = load_gsm8k("test", max_samples)
    results = []

    for i, example in enumerate(ds):
        question = example["question"]
        ref_answer = extract_answer_number(example["answer"])

        # --- Baseline (greedy) ---
        prompt = format_gsm8k_prompt(question)
        baseline_answers = batch_generate(
            model, tokenizer, [prompt],
            do_sample=False, batch_size=1
        )
        baseline_pred = baseline_answers[0]
        baseline_num = extract_answer_number(baseline_pred)

        # --- Reranker: generate N candidates ---
        candidate_prompts = [prompt] * n_candidates
        candidate_answers = batch_generate(
            model, tokenizer, candidate_prompts,
            do_sample=True, temperature=0.7, batch_size=n_candidates
        )
        t0 = time.perf_counter()
        scores = scorer.score_batch(question, candidate_answers)
        score_time = time.perf_counter() - t0

        scored = list(zip(candidate_answers, scores))
        best_answer = min(scored, key=lambda x: x[1])[0]
        best_num = extract_answer_number(best_answer)

        results.append({
            "benchmark": "gsm8k_reranker",
            "question": question,
            "reference": example["answer"],
            "ref_number": ref_answer,
            "baseline_prediction": baseline_pred,
            "baseline_number": baseline_num,
            "baseline_correct": str(baseline_num == ref_answer),
            "reranked_prediction": best_answer,
            "reranked_number": best_num,
            "reranked_correct": str(best_num == ref_answer),
            "n_candidates": str(n_candidates),
            "score_time_s": round(score_time, 3),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(ds)}] baseline_correct={baseline_num == ref_answer}  reranked_correct={best_num == ref_answer}")

    return results


def save_results(results: list[dict], model_name: str, scorer_name: str, output_dir: str = "results"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_tag = model_name.split("/")[-1] if "/" in model_name else model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"reranker_{model_tag}_{scorer_name}_{timestamp}.csv"
    if results:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"Results saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Reranker benchmark on GSM8K")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--scorer", default="majority_voting",
                        choices=["majority_voting", "self_consistency"])
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="results")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.device)

    # Init scorer
    if args.scorer == "majority_voting":
        scorer = MajorityVotingScorer()
    elif args.scorer == "self_consistency":
        scorer = SelfConsistencyScorer(model, tokenizer)

    print(f"\n--- GSM8K Reranker ({args.scorer}, n_candidates={args.n_candidates}) ---")
    results = run_reranker_gsm8k(
        model, tokenizer, scorer,
        max_samples=args.max_samples,
        n_candidates=args.n_candidates,
        batch_size=args.batch_size,
    )

    # Summarize
    baseline_correct = sum(1 for r in results if r["baseline_correct"] == "True")
    reranked_correct = sum(1 for r in results if r["reranked_correct"] == "True")
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Baseline (greedy):  {baseline_correct}/{total} = {baseline_correct/total:.2%}")
    print(f"Reranked ({args.scorer}, N={args.n_candidates}): {reranked_correct}/{total} = {reranked_correct/total:.2%}")
    delta = (reranked_correct - baseline_correct) / total
    print(f"Delta: {delta:+.2%}")
    print(f"{'='*50}")

    save_results(results, args.model, args.scorer, args.output)


if __name__ == "__main__":
    main()
