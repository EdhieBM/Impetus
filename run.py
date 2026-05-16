#!/usr/bin/env python3
"""
Impetus — Multi-benchmark runner: GSM8K, ARC, BBH
Usage: python run.py --benchmark gsm8k --samples 50
       python run.py --benchmark arc --samples 50
       python run.py --benchmark all --samples 50
"""

import argparse, re, time, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ─── Utils ───────────────────────────────────────────────

def extract_number(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1].rstrip(".") if nums else ""


def extract_letter(text: str) -> str:
    m = re.search(r"\b([A-D])\b", text.strip())
    return m.group(1) if m else ""


def batch_generate(model, tokenizer, prompts, max_new_tokens=512, batch_size=32,
                   do_sample=False, temperature=1.0):
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


def majority_vote(items: list) -> str:
    counts = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    return max(counts, key=counts.get)


# ─── Benchmark-specific logic ───────────────────────────

def make_chat_prompt(tokenizer, question):
    msgs = [
        {"role": "system", "content": "You are a helpful assistant. Solve accurately."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_and_run_gsm8k(model, tokenizer, args):
    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.select(range(min(args.samples, len(ds))))
    questions = [ex["question"] for ex in ds]
    refs = [extract_number(ex["answer"]) for ex in ds]
    extract_fn = extract_number
    prompt_template = "Solve the following math problem step by step.\n\nQuestion: {q}\n\nAnswer:"
    return questions, refs, extract_fn, prompt_template


def load_and_run_arc(model, tokenizer, args):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    ds = ds.select(range(min(args.samples, len(ds))))
    questions = []
    refs = []
    for ex in ds:
        choices = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(ex["choices"]["text"])])
        questions.append(f"{ex['question']}\n\n{choices}")
        refs.append(ex["answerKey"])
    extract_fn = extract_letter
    prompt_template = "Question: {q}\n\nAnswer with the letter only (A, B, C, or D):"
    return questions, refs, extract_fn, prompt_template


def load_and_run_bbh(model, tokenizer, args):
    ds = load_dataset("lukaemon/bbh", "temporal_sequences", split="test")
    ds = ds.select(range(min(args.samples, len(ds))))
    # BBH uses multiple choice; extract letter answer
    questions = []
    refs = []
    for ex in ds:
        questions.append(ex["input"])
        refs.append(ex["target"])
    extract_fn = extract_letter
    prompt_template = "{q}\nAnswer:"
    return questions, refs, extract_fn, prompt_template


BENCHMARKS = {
    "gsm8k": load_and_run_gsm8k,
    "arc": load_and_run_arc,
    "bbh": load_and_run_bbh,
}


# ─── Main ─────────────────────────────────────────────────

def run_benchmark(model, tokenizer, benchmark_name, args, scorer=None):
    print(f"\n{'='*50}")
    print(f"Benchmark: {benchmark_name.upper()}")
    print(f"{'='*50}")

    loader = BENCHMARKS[benchmark_name]
    questions, refs, extract_fn, prompt_template = loader(model, tokenizer, args)
    print(f"Samples: {len(questions)}")

    # Baseline (greedy)
    t0 = time.time()
    baseline_outs = batch_generate(model, tokenizer,
        [make_chat_prompt(tokenizer, prompt_template.format(q=q)) for q in questions],
        do_sample=False, batch_size=args.batch_size)
    baseline_extracted = [extract_fn(o) for o in baseline_outs]
    baseline_acc = sum(1 for i in range(len(questions)) if baseline_extracted[i] == refs[i]) / len(questions)
    baseline_time = time.time() - t0

    # Generate N candidates per question
    t0 = time.time()
    all_prompts, q_idx = [], []
    for i, q in enumerate(questions):
        p = make_chat_prompt(tokenizer, prompt_template.format(q=q))
        for _ in range(args.candidates):
            all_prompts.append(p)
            q_idx.append(i)

    all_outs = batch_generate(model, tokenizer, all_prompts,
        do_sample=True, temperature=0.7, batch_size=args.batch_size)
    candidates = {i: [] for i in range(len(questions))}
    for idx, out in zip(q_idx, all_outs):
        candidates[idx].append(out)
    gen_time = time.time() - t0

    # Score & report
    print(f"\n{'N':>5}  {'Correct':>7}  {'Accuracy':>8}  {'Δ':>8}  {'Time':>6}")
    print("-" * 45)
    print(f"{'1(greedy)':>9}  {int(baseline_acc*len(questions)):>7}  {baseline_acc:>7.2%}  {'—':>8}  {baseline_time:>5.1f}s")

    results = {"baseline": baseline_acc, "baseline_correct": int(baseline_acc*len(questions)), "total": len(questions)}
    for N in [5, 10, 20]:
        correct = 0
        for i in range(len(questions)):
            pool = candidates[i][:N]
            if scorer is not None:
                # Neural EBM: score each candidate, pick lowest energy
                energies = scorer.score_batch(questions[i], pool)
                best_idx = min(range(len(energies)), key=lambda j: energies[j])
                best = extract_fn(pool[best_idx])
            else:
                # Majority voting
                extracted = [extract_fn(a) for a in pool]
                best = majority_vote(extracted)
            if best == refs[i]:
                correct += 1
        acc = correct / len(questions)
        delta = acc - baseline_acc
        results[f"N={N}"] = acc
        results[f"N={N}_correct"] = correct
        print(f"{N:>5}  {correct:>7}  {acc:>7.2%}  {delta:>+8.2%}  {gen_time:>5.1f}s")
    print("-" * 45)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "arc", "bbh", "all"])
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--scorer", default="majority_voting",
                        choices=["majority_voting", "neural_ebm"])
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    dtype = torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7 else torch.bfloat16

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="auto", trust_remote_code=True)
    model.eval()
    print("Loaded.\n")

    # Init scorer if neural_ebm
    ebm_scorer = None
    if args.scorer == "neural_ebm":
        from energy_module.scorers import NeuralEBMScorer
        ebm_scorer = NeuralEBMScorer(model, tokenizer)

    benchmarks = ["gsm8k", "arc", "bbh"] if args.benchmark == "all" else [args.benchmark]

    all_results = {}
    for b in benchmarks:
        all_results[b] = run_benchmark(model, tokenizer, b, args, scorer=ebm_scorer)

    # Summary
    print("\n\n" + "=" * 55)
    print("FINAL SUMMARY")
    print("=" * 55)
    print(f"{'Benchmark':<10} {'Baseline':<10} {'Best N':<10} {'Δ':<10}")
    print("-" * 40)
    for b, r in all_results.items():
        best_n = max([k for k in r if k.startswith("N=") and "_correct" not in k], key=lambda k: r[k])
        delta = r[best_n] - r["baseline"]
        print(f"{b:<10} {r['baseline']:<8.2%}  {best_n:<8} {delta:>+7.2%}")
    print("=" * 55)

    # Save
    from datetime import datetime
    Path("results").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"results/benchmark_{ts}.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: results/benchmark_{ts}.json")


if __name__ == "__main__":
    main()
