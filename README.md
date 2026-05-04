# Train Your Own LLM From Scratch

A GPT-style language model built entirely from scratch — **no pretrained models, no HuggingFace tokenizers, no external libraries beyond PyTorch**.

Everything is hand-written:
- **BPE tokenizer** (`tokenizer.py`) — byte-pair encoding trained on your corpus
- **GPT architecture** (`model.py`) — multi-head attention, feed-forward, residual connections
- **Training loop** (`train.py`) — cosine LR, gradient clipping, checkpointing, val loss
- **Dataset** (`dataset.py`) — sliding-window token dataset
- **Inference** (`generate.py`) — temperature, top-k, top-p, nucleus sampling

---

## Repository layout

```
├── tokenizer.py        # BPE tokenizer (from scratch)
├── model.py            # GPT model architecture (from scratch)
├── dataset.py          # Dataset + dataloader
├── train.py            # Training loop
├── generate.py         # Interactive inference
├── train_colab.ipynb   # Google Colab notebook (recommended for training)
├── requirements.txt    # Only dependency: torch
└── README.md
```

---

## Option A — Train on Google Colab (recommended)

Colab gives you a free T4 GPU. No local setup required.

1. **Push this repo to GitHub** (see below)
2. Open `train_colab.ipynb` on GitHub, click the **Open in Colab** badge (or go to [colab.research.google.com](https://colab.research.google.com) → File → Open notebook → GitHub)
3. In Colab: **Runtime → Change runtime type → T4 GPU**
4. Run all cells top to bottom — the notebook will:
   - Clone your repo
   - Install PyTorch
   - Ask you to upload a `.txt` corpus
   - Train the tokenizer and model
   - Plot the loss curve
   - Let you generate text
   - Download the checkpoint as a zip

---

## Option B — Train locally

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Install
pip install -r requirements.txt

# 3. Provide a plain-text corpus (any .txt file)
#    Good sources: Project Gutenberg, Wikipedia dumps, your own text

# 4. Train (tokenizer + model)
python train.py \
    --corpus     my_corpus.txt \
    --save_dir   checkpoints \
    --vocab_size 4096 \
    --max_steps  5000

# 5. Generate text
python generate.py --checkpoint checkpoints
```

---

## Pushing to GitHub

```bash
# Inside your project folder
git init
git branch -M main

# Stage all source files (NOT the corpus or checkpoints)
git add tokenizer.py model.py dataset.py train.py generate.py
git add train_colab.ipynb requirements.txt README.md .gitignore

git commit -m "Initial commit: from-scratch LLM"

# Create the repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

---

## .gitignore

```
# Large binary files — never commit these
checkpoints/
*.pt
*.bin

# Corpus files
*.txt

# Python
__pycache__/
*.pyc
.eggs/
dist/
build/

# OS / IDE
.DS_Store
.vscode/
.idea/
```

---

## Scale guide

| Corpus | vocab_size | d_model | n_layers | batch | steps | Params |
|--------|-----------|---------|----------|-------|-------|--------|
| < 1 MB | 1000 | 256 | 4 | 16 | 2000 | ~5M |
| 1–10 MB | 4096 | 512 | 6 | 32 | 5000 | ~25M |
| 10–100 MB | 8000 | 768 | 8 | 64 | 20000 | ~85M |
| > 100 MB | 16000 | 1024 | 12 | 128 | 50000+ | ~350M |

---

## How it works

### BPE Tokenizer
1. Split corpus into characters
2. Iteratively merge the most frequent adjacent pair into a new token
3. Save the merge rules — applied in the same order at inference time

### GPT Model
```
Input IDs
  └─ Token Embedding + Position Embedding
       └─ Dropout
            └─ [TransformerBlock × N]
                 ├─ LayerNorm → CausalSelfAttention → Residual
                 └─ LayerNorm → FeedForward (GELU) → Residual
            └─ LayerNorm
            └─ Linear → vocab_size logits
```

### Training objective
Cross-entropy loss on next-token prediction:
given tokens `[t0, t1, ..., tN]`, predict `[t1, t2, ..., tN+1]` at every position simultaneously.
