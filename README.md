# LoTiC: Parameter-Efficient Temporal Domain Generalization

**Code for ICLR 2027 Review**

LoTiC models the evolution of a large pretrained model within a compact low-rank
core. This repository provides end-to-end experiments on Rotating 2-Moons and
Rotated MNIST, including processed data, executable notebooks, trained
checkpoints, and per-domain records.

## Repository structure

```text
.
├── notebooks/
│   ├── moons.ipynb
│   └── mnist.ipynb
├── data/
│   ├── moons.pkl
│   └── mnist.pkl
├── checkpoints/
│   ├── moons_distilbert.pt
│   ├── moons_qwen3_0_6b.pt
│   ├── moons_llama3_8b.pt
│   ├── mnist_dinov2_b.pt
│   └── mnist_dinov2_l.pt
├── records/
├── requirements.txt
└── README.md
```

Each notebook contains the model definition, training procedure, future-domain
inference, evaluation, checkpoint export, and visual analysis. The Moons
notebook evaluates DistilBERT, Qwen3-0.6B, and LLaMA-3-8B. The MNIST notebook
evaluates DINOv2-B and DINOv2-L. All five unseen future domains are reported and
visualized after training on the preceding 15 domains. The committed notebooks
retain the console, tabular, and figure outputs of the reported runs.

## Installation

Python 3.10 or later and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The notebooks obtain frozen pretrained weights from Hugging Face. Running the
LLaMA-3-8B experiment requires access to
`meta-llama/Meta-Llama-3-8B` under its upstream license.

## Running the experiments

Start Jupyter from the repository root and run either notebook from top to
bottom:

```bash
jupyter lab
```

Both notebooks train LoTiC from the supplied processed data. A completed run
writes the learned LoTiC and task-head tensors to `checkpoints/` and the full
per-domain metrics and loss history to `records/`. Frozen backbone weights are
not duplicated in the checkpoint files.

## Recorded results

| Dataset | Backbone | Mean future-domain accuracy |
|---|---|---:|
| Rotating 2-Moons | DistilBERT | 99.60% |
| Rotating 2-Moons | Qwen3-0.6B | 98.30% |
| Rotating 2-Moons | LLaMA-3-8B | 99.30% |
| Rotated MNIST | DINOv2-B | 92.30% |
| Rotated MNIST | DINOv2-L | 93.80% |

The values above are produced by the executed notebooks in this repository.
The corresponding domain-level results and training histories are available in
`records/`.

Code and data for the remaining benchmarks will be released upon acceptance.
