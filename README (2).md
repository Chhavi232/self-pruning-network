# Tredence AI Intern Case Study – Self Pruning Neural Network

Implementation of a feedforward neural network that prunes itself during training using learnable gates + L1 sparsity loss.

## How to run

```bash
pip install torch torchvision matplotlib numpy
python self_pruning_network.py
```

Downloads CIFAR-10 automatically. Runs 3 experiments with different lambda values and saves a plot of gate distributions.

## What's in here

- `self_pruning_network.py` – everything in one file (PrunableLinear layer, model, training loop, eval, plotting)
- `REPORT.md` – writeup with results and analysis
- `gate_distributions.png` – generated after running the script

## Quick summary of results

| Lambda | Test Acc | Sparsity |
|--------|----------|----------|
| 1e-5   | ~54.8%   | ~12%     |
| 1e-4   | ~52.1%   | ~42%     |
| 1e-3   | ~47.6%   | ~79%     |
