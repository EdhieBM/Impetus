#!/usr/bin/env python3
"""
Phase 3 — Energy Scoring Methods

Experiment with multiple scoring strategies:
    A. Self-consistency (model critiques itself)
    B. Embedding consistency (semantic similarity)
    C. Lightweight neural EBM
"""

import torch
import torch.nn as nn


class SelfConsistencyScorer:
    """Method A: Ask model to evaluate its own answer."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._critique_prompt = (
            "Evaluate the logical correctness and consistency of the following answer. "
            "Return a score from 0 (perfect) to 100 (completely wrong).\n\n"
            "Question: {question}\nAnswer: {answer}\n\nScore (0-100):"
        )

    def score(self, question: str, answer: str) -> float:
        prompt = self._critique_prompt.format(question=question, answer=answer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        # Parse score from output
        import re
        nums = re.findall(r"\d+", text)
        score = float(nums[0]) / 100.0 if nums else 0.5
        return min(max(score, 0.0), 1.0)


class EmbeddingConsistencyScorer:
    """Method B: Measure semantic consistency between question, reasoning, answer."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _embed(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        # Use last token hidden state as embedding
        return outputs.hidden_states[-1][0, -1, :]

    @torch.no_grad()
    def score(self, question: str, answer: str) -> float:
        q_emb = self._embed(question)
        a_emb = self._embed(answer)
        cos = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), a_emb.unsqueeze(0))
        # Low similarity -> high energy (bad)
        energy = 1.0 - cos.item()
        return max(0.0, min(energy, 2.0))


class NeuralEBMScorer(nn.Module):
    """Method C: Lightweight neural energy model.

    Input: (prompt_embedding, answer_embedding)
    Output: energy scalar

    Train on: good reasoning -> low energy, bad reasoning -> high energy
    """

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
