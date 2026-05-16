"""
Phase 3 — Energy Scoring Methods

Given a question + N candidate answers, each scorer returns energy values.
Low energy = good (consistent, logical, correct).
High energy = bad (contradictory, wrong).
"""

import re
import torch
import torch.nn as nn


class MajorityVotingScorer:
    """
    For math: extract final number from each answer.
    Energy = 1 - (frequency of that answer among all candidates).
    Most common answer → lowest energy (best).

    This is a strong baseline for GSM8K-style tasks.
    """

    def score_batch(self, question: str, candidates: list[str]) -> list[float]:
        extracted = []
        for ans in candidates:
            nums = re.findall(r"-?\d+\.?\d*", ans)
            extracted.append(nums[-1] if nums else "")

        freq = {}
        for e in extracted:
            freq[e] = freq.get(e, 0) + 1

        n = len(candidates)
        scores = []
        for e in extracted:
            prob = freq[e] / n
            energy = 1.0 - prob
            scores.append(energy)

        return scores


class SelfConsistencyScorer:
    """
    Ask the model to evaluate each candidate.
    Requires an additional forward pass per candidate (expensive but model-aware).
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._eval_prompt = (
            "Question: {question}\n"
            "Answer: {answer}\n\n"
            "Is the answer correct and logically consistent? "
            "Respond with a single number 0-10 where 0=completely wrong, 10=perfect. "
            "Score:"
        )

    @torch.no_grad()
    def score_batch(self, question: str, candidates: list[str]) -> list[float]:
        scores = []
        for ans in candidates:
            prompt = self._eval_prompt.format(question=question, answer=ans)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            nums = re.findall(r"\d+", text)
            score = float(nums[0]) / 10.0 if nums else 0.5
            # energy = 1 - score (higher confidence = lower energy)
            energy = 1.0 - min(max(score, 0.0), 1.0)
            scores.append(energy)
        return scores


class EmbeddingConsistencyScorer:
    """
    Cosine similarity between question embedding and answer embedding.
    Low similarity → high energy (potential hallucination/contradiction).
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def _embed(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.model.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1][0, -1, :]

    @torch.no_grad()
    def score_batch(self, question: str, candidates: list[str]) -> list[float]:
        q_emb = self._embed(question)
        scores = []
        for ans in candidates:
            a_emb = self._embed(ans)
            cos = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), a_emb.unsqueeze(0))
            energy = 1.0 - cos.item()
            scores.append(max(0.0, min(energy, 2.0)))
        return scores


class NeuralEBMScorer(nn.Module):
    """Method C: Lightweight neural energy model — for Phase 3 training."""

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, prompt_emb: torch.Tensor, answer_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([prompt_emb, answer_emb], dim=-1)
        return self.fc(x).squeeze(-1)
