#!/usr/bin/env python3
"""
Phase 3 — Train Neural EBM Scorer

Trains a small MLP that scores (question, answer) pairs:
  Low energy  = correct/logical → target 0
  High energy = wrong/illogical  → target 1

Usage:
    python energy_module/train_ebm.py --model Qwen/Qwen2.5-3B-Instruct
                                      --train-samples 500
                                      --epochs 10
                                      --lr 1e-3
"""

import argparse, os, re, sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Tiny MLP Energy Scorer ────────────────────────────

class NeuralEBM(nn.Module):
    """Lightweight energy model: (prompt_emb, answer_emb) → energy scalar."""

    def __init__(self, hidden_dim: int = 1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, prompt_emb: torch.Tensor, answer_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([prompt_emb, answer_emb], dim=-1)
        return self.net(x).squeeze(-1)


# ─── Dataset ────────────────────────────────────────────

class EBMdataset(Dataset):
    """Creates (question_emb, answer_emb, label) triples from GSM8K."""

    def __init__(self, model, tokenizer, split="train", max_samples=500,
                 candidates_per_q=5, temperature=0.7, hidden_dim=1536):
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim

        ds = load_dataset("gsm8k", "main", split=split)
        ds = ds.select(range(min(max_samples, len(ds))))

        self.prompt_embs = []
        self.answer_embs = []
        self.labels = []

        print(f"Generating training data from {len(ds)} questions...")
        for i, ex in enumerate(ds):
            question = ex["question"]
            ref = self._extract_number(ex["answer"])

            q_emb = self._embed(question)

            # Generate candidates
            prompt = f"Solve step by step: {question}"
            msgs = [
                {"role": "system", "content": "Solve math accurately."},
                {"role": "user", "content": prompt},
            ]
            full_prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

            for _ in range(candidates_per_q):
                inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=128,
                        do_sample=True, temperature=temperature, top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                answer = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
                pred = self._extract_number(answer)
                label = 0.0 if pred == ref else 1.0  # 0=good, 1=bad

                a_emb = self._embed(answer)
                self.prompt_embs.append(q_emb)
                self.answer_embs.append(a_emb)
                self.labels.append(label)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(ds)}] {sum(self.labels[-candidates_per_q*50:]):.0f}/{candidates_per_q*50} incorrect")

        print(f"Total pairs: {len(self.labels)} ({sum(self.labels):.0f} bad / {len(self.labels)-sum(self.labels):.0f} good)")

    def _extract_number(self, text):
        text = re.sub(r"(?<=\d),(?=\d)", "", text)
        nums = re.findall(r"-?\d+\.?\d*", text)
        return nums[-1].rstrip(".") if nums else ""

    @torch.no_grad()
    def _embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.model.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0, -1, :]  # last token, last layer
        return hidden.cpu()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.prompt_embs[idx], self.answer_embs[idx], torch.tensor(self.labels[idx])


# ─── Training ────────────────────────────────────────────

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train-samples", type=int, default=300,
                        help="questions to generate training data from")
    parser.add_argument("--candidates", type=int, default=5, help="candidates per question")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save", default="energy_module/ebm_scorer.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load LLM (for embeddings only, no gradients needed)
    print(f"Loading {args.model} for embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    print("LLM loaded.\n")

    # Generate dataset
    dataset = EBMdataset(model, tokenizer, "train", args.train_samples, args.candidates)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Init neural EBM
    ebm = NeuralEBM(hidden_dim=1536).to(device)
    optimizer = torch.optim.AdamW(ebm.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"\nTraining Neural EBM ({sum(p.numel() for p in ebm.parameters()):,} params)...")
    for epoch in range(args.epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for p_emb, a_emb, labels in loader:
            p_emb, a_emb, labels = p_emb.to(device), a_emb.to(device), labels.to(device)
            preds = ebm(p_emb, a_emb)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # Binary accuracy: threshold at 0.5
            pred_bin = (preds > 0.5).float()
            correct += (pred_bin == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"  Epoch {epoch+1:2d}/{args.epochs} | Loss: {total_loss/len(loader):.4f} | Acc: {acc:.2%}")

    # Save
    torch.save(ebm.state_dict(), args.save)
    print(f"\nSaved: {args.save}")


if __name__ == "__main__":
    train()
