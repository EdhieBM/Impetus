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
    max_new_tokens: int = 512,
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
    return f"Solve the following math problem step by step.\n\nQuestion: {question}\n\nAnswer:"


def extract_answer_number(text: str) -> str:
    # Remove commas inside numbers (e.g., "70,000" -> "70000")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else ""


def run_reranker_gsm8k(
    model, tokenizer, scorer, max_samples: int = 50,
    n_candidates: int = 5, batch_size: int = 8
) -> list[dict]:
    ds = load_gsm8k("test", max_samples)
    questions = [ex["question"] for ex in ds]
    ref_answers = [extract_answer_number(ex["answer"]) for ex in ds]

    # --- Phase 1: All baselines at once ---
    baseline_prompts = [format_gsm8k_prompt(q) for q in questions]
    print("Generating baselines (greedy)...")
    baseline_outputs = batch_generate(
        model, tokenizer, baseline_prompts,
        do_sample=False, batch_size=batch_size
    )
    baseline_nums = [extract_answer_number(o) for o in baseline_outputs]

    # --- Phase 2: All candidates at once ---
    # Flatten: for each question, n_candidates copies
    candidate_prompts = []
    question_idx_for_candidate = []
    for i, q in enumerate(questions):
        prompt = format_gsm8k_prompt(q)
        for _ in range(n_candidates):
            candidate_prompts.append(prompt)
            question_idx_for_candidate.append(i)

    print(f"Generating {len(candidate_prompts)} candidates ({len(questions)} questions x {n_candidates})...")
    all_candidate_outputs = batch_generate(
        model, tokenizer, candidate_prompts,
        do_sample=True, temperature=0.7,
        batch_size=max(batch_size, n_candidates)
    )

    # Group outputs back by question
    candidates_by_question = {i: [] for i in range(len(questions))}
    for idx, output in zip(question_idx_for_candidate, all_candidate_outputs):
        candidates_by_question[idx].append(output)

    # --- Phase 3: Score and pick best ---
    results = []
    for i, question in enumerate(questions):
        candidate_answers = candidates_by_question[i]
        t0 = time.perf_counter()
        scores = scorer.score_batch(question, candidate_answers)
        score_time = time.perf_counter() - t0

        scored = list(zip(candidate_answers, scores))
        best_answer = min(scored, key=lambda x: x[1])[0]
        best_num = extract_answer_number(best_answer)

        results.append({
            "benchmark": "gsm8k_reranker",
            "question": question,
            "reference": ds[i]["answer"],
            "ref_number": ref_answers[i],
            "baseline_prediction": baseline_outputs[i],
            "baseline_number": baseline_nums[i],
            "baseline_correct": str(baseline_nums[i] == ref_answers[i]),
            "reranked_prediction": best_answer,
            "reranked_number": best_num,
            "reranked_correct": str(best_num == ref_answers[i]),
            "n_candidates": str(n_candidates),
            "score_time_s": round(score_time, 3),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(questions)}] baseline_correct={baseline_nums[i] == ref_answers[i]}  reranked_correct={best_num == ref_answers[i]}")

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
