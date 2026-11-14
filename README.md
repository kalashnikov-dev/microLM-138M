<div align="center">

# microLM-138M

**A 138M-parameter language model written and trained from scratch — 27.5B tokens of pretraining plus instruction tuning.**

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-microLM--138M-FFD21E)](https://huggingface.co/kalashnikov-dev/microLM-138M)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

The goal was to understand how modern LLMs work end to end — not only the model code, but the whole pipeline around it: streaming data, mixed precision, gradient accumulation, checkpointing, surviving spot-instance preemptions, and instruction tuning.

No training framework was used. The architecture, training loop, data pipeline and inference code are all in this repo, in plain PyTorch.

Everything ran on **one NVIDIA L4 (24GB)** on Google Cloud spot instances and fit into **~$180 of cloud credits**.

## Highlights

- Modern decoder-only architecture: **GQA**, **SwiGLU**, **RMSNorm**, **RoPE**, **QK-norm**, tied embeddings
- **WSD** learning-rate schedule with a `1 - √p` decay instead of the usual linear one
- Memory-efficient loss via **Liger fused linear cross-entropy** — this is what makes a 49k vocab fit in 24GB at batch size 64
- Full **LoRA** instruction tuning with loss masked on user turns, then merged back into the base weights

## Samples

Real output, temperature 0.7, top-p 0.9. The base model only continues text; the tuned one answers.

```
Why is the sky blue?

BASE  A. The sky is blue because the sun is the center of the universe.
      B. The sun is the center of the universe because the earth is made of
         the same material.
      C. The sun is the center of the universe because the earth is made of...

SFT   The sky is blue because sunlight reflects off the blue sky, creating a
      warm and soothing atmosphere. This is because sunlight is an
      electromagnetic wave that is emitted by the Earth's atmosphere.
```

```
Explain photosynthesis in two sentences.

SFT   Photosynthesis is the process by which plants convert light energy from
      the sun into chemical energy. It occurs in two stages: the
      light-dependent reactions, which involve photosynthesis, and the
      light-independent reactions, which involve respiration, glycolysis, and
      oxidative phosphorylation.
```

Instruction tuning clearly worked — the model stops rambling and starts answering, and the format is right. Factual errors remain common, which is unsurprising for a model with 138M parameters trained on 27.5B tokens.

## Model

| | |
|---|---|
| Parameters | 138.4M (100.7M non-embedding) |
| Layers | 16 |
| Model dim | 768 |
| Attention | 12 query heads / 4 KV heads (GQA), head dim 64 |
| Feed-forward | SwiGLU, hidden dim 2048 |
| Normalization | RMSNorm (pre-norm) + QK-norm, eps 1e-5 |
| Positions | RoPE |
| Vocab | 49,152 (SmolLM2 tokenizer) |
| Context | 2048 |
| Embeddings | tied with the output projection |

Residual projections (`o` and `w2`) are initialized with `std / √(2·n_layers)` so that the variance added by the residual stream does not grow with depth.

## Training

### Pretraining

| | |
|---|---|
| Data | FineWeb-Edu `sample-100BT`, streamed |
| Tokens | ~27.5B |
| Steps | 105,000 × 262,144 tokens (batch 64 × 2 accumulation × 2048) |
| Optimizer | AdamW, lr 2e-3, betas (0.9, 0.95), eps 1e-8 |
| Weight decay | 0.1, applied only to 2D parameters |
| Schedule | WSD — 2,000 warmup, stable, 21,000 decay as `1 - √p` |
| Precision | bf16 autocast, TF32 matmuls |
| Other | grad clip 1.0, gradient checkpointing, `torch.compile` |
| Throughput | ~31,900 tok/s |
| Compute | ~240 GPU-hours on one L4 24GB |

Loss, from the training log:

| Step | 1 | 1k | 5k | 10k | 20k | 40k | 105k |
|---|---|---|---|---|---|---|---|
| Loss | 10.95 | 3.92 | 3.29 | 3.11 | 3.11 | 3.05 | ~2.6 |

Step 1 at 10.95 matches `ln(49152) = 10.80` almost exactly, which is a good sign that the initialization is correct — the model starts at a uniform distribution over the vocabulary and nothing is silently broken. The loss then sits around 3.05–3.11 through the long stable phase, and the final drop comes from the decay phase, which is the whole point of a WSD schedule.

### Instruction tuning

| | |
|---|---|
| Data | `HuggingFaceTB/smol-smoltalk` |
| Method | LoRA, rank 16, alpha 32, on all 7 projections |
| Steps | 12,000 × 32,768 tokens (~393M tokens, 2 epochs) |
| Optimizer | AdamW, lr 2e-4, weight decay 0.01 |
| Schedule | cosine with 10% warmup |
| Loss | 2.17 → 1.52 |
| Compute | ~13 hours |

Only assistant tokens contribute to the loss — user and system turns are masked with `-100`, so the model learns to answer rather than to predict the questions. LoRA adapters are merged into the base weights at the end with `merge_and_unload()`, so the released checkpoint is a normal dense model.

## Engineering notes

The parts that took the most time were not the model.

**Fitting a 49k vocab into 24GB.** A naive `F.cross_entropy` materializes the full logits tensor, then upcasts it to fp32 internally — at batch 64 that is tens of gigabytes just for the loss. Liger's fused linear cross-entropy computes the projection and the loss in chunks and never materializes the full tensor, which is what made a usable batch size possible at all.

**Gradient checkpointing made training *faster*.** It normally costs ~30% throughput in exchange for memory. Here it freed enough memory to raise the batch size from 8 to 64, and the better GPU utilization more than paid for the recomputation — about 15% net gain in tokens/s.

**Spot instances.** They are ~1.7× cheaper and get taken away without warning. Checkpoints are written atomically (temp file plus `os.replace`) so a kill mid-write cannot corrupt them, old ones are rotated, and every 5,000 steps one is uploaded to the Hub in a background thread. A PowerShell watchdog (`scripts/`) polls the instance and restarts it as soon as capacity frees up.

## Limitations

- No benchmark numbers yet — the loss curve is all I can currently claim
- No KV cache, so generation recomputes the full prefix every token and is slow
- 2048-token context, English only
- Factual accuracy is poor, as expected at this scale

## Roadmap

- [ ] KV cache for usable inference speed
- [ ] Standard benchmarks (HellaSwag, ARC, MMLU) to put real numbers on the model
- [ ] Deployment — a small inference API and a demo
- [ ] RAG on top of the tuned model
- [ ] Export to the HF `transformers` format for easier reuse

## Repo

```
model.py        architecture (config, transformer block, model)
pre-train.py    pretraining loop
sft.py          LoRA instruction tuning with loss masking
storage.py      atomic checkpointing, rotation, Hub uploads
scripts/        spot-instance watchdog
```

## A note on AI assistance

An AI assistant helped mostly with three things: resolving tensor dimension mismatches in function calls, the CUDA and kernel-level optimizations, and the PowerShell watchdog in `scripts/`. The architecture, the training loop and the data pipeline I wrote myself, even inline code comlpetion was turned off whole time as a challange.

## Acknowledgements

Tokenizer and SFT dataset from [HuggingFaceTB](https://huggingface.co/HuggingFaceTB), pretraining data from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), fused cross-entropy from [Liger-Kernel](https://github.com/linkedin/Liger-Kernel), RoPE from [torchtune](https://github.com/pytorch/torchtune).

## License

MIT — see [LICENSE](LICENSE).
