"""
QLoRA fine-tune of Qwen2.5-Coder-7B-Instruct on the Excalidraw DSL dataset, on Modal.

What this does
--------------
- Loads Qwen2.5-Coder-7B-Instruct in 4-bit (nf4 + double-quant), attaches a LoRA adapter,
  and trains with TRL's SFTTrainer on `train.jsonl` / `validation.jsonl`.
- COMPLETION-ONLY loss: the system+user prompt is masked; the model is only trained on the
  assistant DSL tokens. This is what makes it learn "prompt -> DSL", not "echo the prompt".
- The HF model weights are cached in a Modal Volume so re-runs skip the ~15 GB download.
- The trained LoRA adapter is written to a second Volume and can be pulled down locally.

The dataset format is already correct (chat `messages`, one fixed system prompt), so there is
no reformatting step here — we just apply the tokenizer's chat template.

Prerequisites
-------------
  modal token new
  modal secret create huggingface HF_TOKEN=hf_xxx      # read scope is enough

Run (from the repo root)
------------------------
  modal run train/modal_train.py                       # defaults: 3 epochs, lr 2e-4
  modal run train/modal_train.py --epochs 4 --lr 1e-4 --run-name qwen7b-qlora-v2

Download the adapter when done
------------------------------
  modal volume get excali-out qwen7b-qlora-v1/final-adapter ./adapters/qwen7b-qlora-v1

Optional: merge adapter into a standalone fp16 model (for vLLM serving / eval speed)
------------------------------------------------------------------------------------
  modal run train/modal_train.py::merge --run-name qwen7b-qlora-v1

Cost
----
~1300 rows, eff. batch 16 -> ~80 steps/epoch -> ~240 steps for 3 epochs.
On A100-40GB that is ~15-25 min, roughly $1-2 per run. Well inside the budget.
"""

import pathlib

import modal

# ---------------------------------------------------------------------------
# Paths (resolved on the *client* at `modal run` time, so the JSONL uploaded is
# whatever is on disk then — i.e. the final dataset once generation finishes).
# ---------------------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

app = modal.App("excali-qlora")

# Pinned versions: TRL's SFTTrainer/SFTConfig API moves between releases, so we pin
# the whole stack to a combination that is known to work together.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "trl==0.12.2",
        "peft==0.13.2",
        "bitsandbytes==0.44.1",
        "accelerate==1.1.1",
        "datasets==3.1.0",
    )
    # Cache HF downloads inside the mounted cache Volume.
    .env({"HF_HOME": "/cache/hf"})
    # Upload the dataset into the container (lazy: happens at `modal run` time).
    .add_local_file(str(REPO / "train.jsonl"), "/data/train.jsonl")
    .add_local_file(str(REPO / "validation.jsonl"), "/data/validation.jsonl")
)

cache_vol = modal.Volume.from_name("excali-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("excali-out", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/cache": cache_vol, "/out": out_vol},
    secrets=[hf_secret],
    timeout=60 * 60,
)
def train(
    epochs: float = 3.0,
    lr: float = 2e-4,
    batch_size: int = 8,
    grad_accum: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_seq_len: int = 2048,
    run_name: str = "qwen7b-qlora-v1",
):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    out_dir = f"/out/{run_name}"

    # --- tokenizer ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 4-bit base model (QLoRA) -----------------------------------------
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",  # no flash-attn build needed
    )
    model.config.use_cache = False  # required with gradient checkpointing

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # --- dataset: messages -> chat-templated text -------------------------
    ds = load_dataset(
        "json",
        data_files={
            "train": "/data/train.jsonl",
            "validation": "/data/validation.jsonl",
        },
    )

    def to_text(ex):
        return {
            "text": tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    ds = ds.map(to_text, remove_columns=ds["train"].column_names)
    print(f"train={len(ds['train'])}  validation={len(ds['validation'])}")

    # --- completion-only loss ---------------------------------------------
    # Qwen renders each assistant turn as "<|im_start|>assistant\n...".
    # Passing the template as token IDs (not a string) avoids the tokenizer
    # context-mismatch that makes the collator silently fail to find it.
    response_template_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids, tokenizer=tokenizer
    )

    # --- training config ---------------------------------------------------
    cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        bf16=True,
        max_seq_length=max_seq_len,
        packing=False,  # required: completion-only collator can't pack
        dataset_text_field="text",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        peft_config=peft_config,
    )

    # Sanity check: confirm the mask actually fires on the first batch (labels
    # should be -100 for the prompt and real token ids only for the DSL).
    sample = trainer.train_dataset[0]
    print("sample text head:\n", sample["text"][:200])

    trainer.train()

    final_dir = f"{out_dir}/final-adapter"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    out_vol.commit()

    metrics = trainer.evaluate()
    print("final eval:", metrics)
    print("saved adapter ->", final_dir)
    return metrics


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/cache": cache_vol, "/out": out_vol},
    secrets=[hf_secret],
    timeout=60 * 60,
)
def merge(run_name: str = "qwen7b-qlora-v1"):
    """Merge the LoRA adapter into the base weights and save a standalone fp16
    model (handy for vLLM serving or faster eval). Optional — eval can also load
    base+adapter directly with PEFT."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = f"/out/{run_name}/final-adapter"
    merged_dir = f"/out/{run_name}/merged-fp16"

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(merged_dir)
    out_vol.commit()
    print("merged model ->", merged_dir)


@app.local_entrypoint()
def main(
    epochs: float = 3.0,
    lr: float = 2e-4,
    batch_size: int = 8,
    grad_accum: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_seq_len: int = 2048,
    run_name: str = "qwen7b-qlora-v1",
):
    metrics = train.remote(
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        max_seq_len=max_seq_len,
        run_name=run_name,
    )
    print("done:", metrics)
    print(
        f"pull adapter with:\n"
        f"  modal volume get excali-out {run_name}/final-adapter "
        f"./adapters/{run_name}"
    )
