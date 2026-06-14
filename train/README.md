# Phase 3 — Fine-Tuning (Modal QLoRA)

QLoRA fine-tune of **Qwen2.5-Coder-7B-Instruct** on the Excalidraw DSL dataset, run on
**Modal** (A100-40GB). The model learns `user prompt -> compact DSL`; the deterministic
converter (Phase 1) turns that DSL into `.excalidraw`.

| File | What it does | Where it runs |
|---|---|---|
| `prepare_data.py` | Pre-flight sanity gate on `train.jsonl` / `validation.jsonl` | local |
| `modal_train.py`  | QLoRA train + (optional) merge | Modal A100-40GB |
| `eval_finetune.py`| Validity-rate of fine-tuned vs base on held-out prompts | Modal A100-40GB |

The training/eval scripts consume the dataset in its existing chat-`messages` shape — no
reformatting. The base model is **not gated**; the HF token only avoids download throttling.

---

## 0. One-time prerequisites

```bash
pip install modal                 # if not already (you install libs)
modal token new                   # authenticate the CLI to your workspace
modal secret create huggingface HF_TOKEN=hf_xxxxxxxx   # read-scope token
modal secret list                 # confirm `huggingface` is listed
```

Modal Volumes (`excali-hf-cache`, `excali-out`) auto-create on first run — nothing to do.
The model cache Volume means only the *first* run pays the ~15 GB Qwen download.

---

## 1. Sanity-check the dataset (local, after generation finishes)

```bash
python3 train/prepare_data.py
# or: python3 train/prepare_data.py --train train.jsonl --val validation.jsonl --max-seq-len 2048
```

Verifies: chat structure, one fixed system prompt, every assistant DSL still passes
`validate_dsl.py`, token-length histogram (flags truncation at `--max-seq-len`), no
train/val prompt leakage, diagram-type balance. **Exit code is non-zero on a blocking
issue** — fix before spending GPU.

> If `transformers` isn't in your local venv, token counts fall back to a `chars/4`
> *estimate* (labeled as such). Exact counts need `transformers` locally; the Modal side
> uses the real tokenizer regardless.

---

## 2. Train

```bash
# from the repo root
modal run train/modal_train.py
# tweak anything:
modal run train/modal_train.py --epochs 4 --lr 1e-4 --lora-r 32 --run-name qwen7b-qlora-v2
```

Defaults: 3 epochs, lr 2e-4 cosine, LoRA r16/α32, eff. batch 16, `max_seq_len` 2048,
completion-only loss (prompt masked), eval+save per epoch, best checkpoint kept.

**Smoke test first (optional, ~couple min):** `--epochs 0.05` to surface any image/version
issue cheaply before the real run.

The dataset is uploaded at `modal run` time, so whatever is on disk then is what trains —
run this only after generation completes.

Pull the adapter down when finished:

```bash
modal volume get excali-out qwen7b-qlora-v1/final-adapter ./adapters/qwen7b-qlora-v1
```

Optional — merge adapter into a standalone fp16 model (for vLLM serving / faster eval):

```bash
modal run train/modal_train.py::merge --run-name qwen7b-qlora-v1
modal volume get excali-out qwen7b-qlora-v1/merged-fp16 ./adapters/qwen7b-qlora-v1-merged
```

---

## 3. Evaluate (the win condition)

```bash
modal run train/eval_finetune.py --run-name qwen7b-qlora-v1
# base model only (skip adapter): --base-only
```

Generates DSL (greedy/reproducible) on the held-out `validation.jsonl` with both base and
base+adapter, scores each through `json.loads -> validate_dsl -> converter`, and prints a
delta. Success = **valid_rate / convert_rate** rising for the fine-tuned model. This is the
25%-validity metric from the project brief.

---

## Hyperparameters (starting point, ~1300 rows)

| Knob | Value | Note |
|---|---|---|
| epochs | 3 | small dataset; watch eval_loss for overfit |
| lr | 2e-4 | cosine, 3% warmup |
| LoRA r / α / dropout | 16 / 32 / 0.05 | all attn + MLP proj |
| eff. batch | 16 | bs 8 × grad_accum 2 |
| max_seq_len | 2048 | covers observed lengths with headroom |
| quant | nf4 + double-quant, bf16 compute | QLoRA |
| optim | paged_adamw_8bit | + gradient checkpointing |

First lever if validity underwhelms: bump epochs to 4 and/or LoRA `r` to 32. If eval_loss
turns up while train_loss keeps falling, that's overfit — drop epochs.

---

## Cost

~1300 rows → ~80 steps/epoch → ~240 steps for 3 epochs. On A100-40GB that's ~15–25 min,
roughly **$1–2/run**. Eval is a few minutes more. The $30 budget covers many experiment
runs; the $250 credit leaves ample room for later serving.

---

## Troubleshooting

- **Image build fails on a version** — the pinned stack (torch 2.5.1 / transformers 4.46.3 /
  trl 0.12.2 / peft 0.13.2 / bnb 0.44.1) is known-good; if a resolver hiccup occurs it shows
  at build time. Catch it with the smoke test.
- **`valid_rate` is 0 but JSON looks fine** — check for the model wrapping output in
  ```` ```json ```` fences or adding prose; the system prompt forbids it, but if it slips
  through, the validator parses the *whole* string. The inference pipeline's repair/retry
  step handles this in production; eval measures raw output deliberately.
- **OOM** — drop `--batch-size` to 4 and raise `--grad-accum` to 4 (same effective batch).
- **Completion-only mask not firing** — the response template is passed as token IDs for
  Qwen's `<|im_start|>assistant\n`; if you change the base model, update that in
  `modal_train.py`.
- **Re-download every run** — confirm both runs use the same `excali-hf-cache` Volume and
  `HF_HOME=/cache/hf` (set in the image).
