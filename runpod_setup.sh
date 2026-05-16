#!/bin/bash
# One-liner for RunPod / any Linux cloud GPU
# Paste this in the RunPod terminal after starting a PyTorch pod

git clone --depth 1 https://github.com/EdhieBM/Impetus.git
cd Impetus
pip install -q torch transformers accelerate datasets evaluate sentencepiece scipy pandas tqdm

# Run baseline + reranker
python benchmarks/run_baseline.py --model Qwen/Qwen2.5-3B-Instruct --max-samples 50 --batch-size 32
python benchmarks/run_reranker.py --model Qwen/Qwen2.5-3B-Instruct --scorer majority_voting --n-candidates 20 --max-samples 50 --batch-size 32
