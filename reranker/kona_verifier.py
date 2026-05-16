#!/usr/bin/env python3
"""
Phase 2 — KONA-style Verifier (V1)
LLM generates N candidate answers → energy scorer → best response selected.

Usage:
    python reranker/kona_verifier.py --prompt "Solve: 2 + 2 = ?"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from energy_module.scorers import MajorityVotingScorer, SelfConsistencyScorer, EmbeddingConsistencyScorer


class KONAVerifier:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct",
                 scorer: str = "majority_voting", device: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        if scorer == "majority_voting":
            self.scorer = MajorityVotingScorer()
        elif scorer == "self_consistency":
            self.scorer = SelfConsistencyScorer(self.model, self.tokenizer)
        elif scorer == "embedding":
            self.scorer = EmbeddingConsistencyScorer(self.model, self.tokenizer)
        else:
            raise ValueError(f"Unknown scorer: {scorer}")

    def generate_candidates(self, prompt: str, n: int = 5,
                            max_new_tokens: int = 256, temperature: float = 0.7) -> list[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        candidates = []
        for _ in range(n):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            answer = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            candidates.append(answer.strip())
        return candidates

    def rerank(self, prompt: str, question: str = None, n: int = 5,
               **gen_kwargs) -> tuple[str, float, list[tuple[str, float]]]:
        candidates = self.generate_candidates(prompt, n=n, **gen_kwargs)
        q = question or prompt
        scores = self.scorer.score_batch(q, candidates)
        scored = list(zip(candidates, scores))
        # argmin energy
        best = min(scored, key=lambda x: x[1])
        return best[0], best[1], scored


def main():
    parser = argparse.ArgumentParser(description="KONA-style verifier")
    parser.add_argument("--prompt", default="Solve: 2 + 2 = ?")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--scorer", default="majority_voting",
                        choices=["majority_voting", "self_consistency", "embedding"])
    args = parser.parse_args()

    verifier = KONAVerifier(args.model, args.scorer)
    best, score, all_candidates = verifier.rerank(args.prompt, n=args.n_candidates)

    print(f"Scorer: {args.scorer}")
    print(f"Prompt: {args.prompt}\n")
    print(f"Best answer (energy={score:.4f}): {best}\n")
    print("All candidates:")
    for i, (ans, s) in enumerate(all_candidates):
        print(f"  [{i}] energy={s:.4f}: {ans[:100]}...")


if __name__ == "__main__":
    main()
