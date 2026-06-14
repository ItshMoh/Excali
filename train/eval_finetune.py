"""
Evaluate the fine-tuned adapter against the base model on held-out prompts — the Phase-3
"did it work?" check. Runs on Modal (loading a 7B model needs the GPU).

For each held-out prompt in validation.jsonl it generates DSL with BOTH the base model and
base+adapter, then scores each generation through the real pipeline:

  generated text -> json.loads -> validate_dsl.py -> convert_dsl_to_excalidraw.py

and reports, per model:
  - json_rate      : output parses as JSON
  - valid_rate     : passes validate_dsl (schema + semantic) — the 25%-weight metric
  - convert_rate   : the validated DSL converts to .excalidraw without error
  - type_match     : generated `diagram` type matches the reference
  - node_recall    : avg fraction of reference node labels present in the generation

The win condition is valid_rate / convert_rate climbing for the fine-tuned model vs base.

Run (from repo root, after training has produced an adapter)
------------------------------------------------------------
  modal run train/eval_finetune.py --run-name qwen7b-qlora-v1
  modal run train/eval_finetune.py --run-name qwen7b-qlora-v1 --max-new-tokens 768 --base-only

Generation is greedy (do_sample=False) so the numbers are reproducible run-to-run.
"""

import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

app = modal.App("excali-qlora-eval")

# Eval needs the validator, the converter, the schema, the held-out set — and the `dot`
# binary the converter shells out to.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("graphviz")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "peft==0.13.2",
        "bitsandbytes==0.44.1",
        "accelerate==1.1.1",
    )
    .env({"HF_HOME": "/cache/hf"})
    .add_local_file(str(REPO / "validation.jsonl"), "/root/validation.jsonl")
    .add_local_file(str(REPO / "validate_dsl.py"), "/root/validate_dsl.py")
    .add_local_file(str(REPO / "convert_dsl_to_excalidraw.py"), "/root/convert_dsl_to_excalidraw.py")
    .add_local_file(str(REPO / "dsl_schema.json"), "/root/dsl_schema.json")
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
def evaluate(
    run_name: str = "qwen7b-qlora-v1",
    max_new_tokens: int = 768,
    base_only: bool = False,
):
    import json
    import sys

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    sys.path.insert(0, "/root")
    from validate_dsl import load_schema, validate_dsl_text  # noqa: E402
    from convert_dsl_to_excalidraw import Converter  # noqa: E402

    schema = load_schema()

    # --- held-out prompts -------------------------------------------------
    items = []  # (messages_without_assistant, reference_dsl_obj_or_None)
    with open("/root/validation.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)["messages"]
            prompt = [m for m in msgs if m["role"] != "assistant"]
            ref = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            try:
                ref_obj = json.loads(ref) if ref else None
            except json.JSONDecodeError:
                ref_obj = None
            items.append((prompt, ref_obj))
    print(f"held-out prompts: {len(items)}")

    # --- tokenizer + base model ------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        attn_implementation="sdpa",
    )
    model.eval()

    def generate(prompt_msgs):
        ids = tokenizer.apply_chat_template(
            prompt_msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    def score(label):
        n = len(items)
        tally = {"json": 0, "valid": 0, "convert": 0, "type": 0}
        recall_sum = 0.0
        recall_n = 0
        failures = []
        for i, (prompt, ref) in enumerate(items):
            text = generate(prompt)
            # 1. JSON parse
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                failures.append((i, "json", text[:120]))
                continue
            tally["json"] += 1
            # 2. DSL validity (schema + semantic)
            res = validate_dsl_text(text, schema)
            if res.ok:
                tally["valid"] += 1
                # 3. converter
                try:
                    Converter(obj).build()
                    tally["convert"] += 1
                except Exception as e:  # noqa: BLE001
                    failures.append((i, "convert", str(e)[:120]))
            else:
                codes = ",".join(c for c, _ in res.errors)
                failures.append((i, "valid", codes))
            # 4. type match + node recall vs reference
            if ref:
                if obj.get("diagram") == ref.get("diagram"):
                    tally["type"] += 1
                ref_labels = {(node.get("label") or "").strip().lower()
                              for node in ref.get("nodes", [])}
                ref_labels.discard("")
                if ref_labels:
                    gen_labels = {(node.get("label") or "").strip().lower()
                                  for node in obj.get("nodes", [])}
                    recall_sum += len(ref_labels & gen_labels) / len(ref_labels)
                    recall_n += 1

        def pct(k):
            return f"{tally[k] / n * 100:5.1f}%  ({tally[k]}/{n})"

        print(f"\n----- {label} -----")
        print(f"  json_rate    : {pct('json')}")
        print(f"  valid_rate   : {pct('valid')}")
        print(f"  convert_rate : {pct('convert')}")
        print(f"  type_match   : {pct('type')}")
        if recall_n:
            print(f"  node_recall  : {recall_sum / recall_n * 100:5.1f}%  (avg over {recall_n})")
        if failures:
            print(f"  sample failures (up to 8):")
            for idx, kind, detail in failures[:8]:
                print(f"    #{idx} [{kind}] {detail}")
        return {k: tally[k] / n for k in tally}

    base_metrics = score("BASE  (Qwen2.5-Coder-7B-Instruct)")

    ft_metrics = None
    if not base_only:
        from peft import PeftModel

        adapter_dir = f"/out/{run_name}/final-adapter"
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()
        ft_metrics = score(f"FINE-TUNED  ({run_name})")

    # --- delta summary ----------------------------------------------------
    if ft_metrics:
        print("\n===== DELTA (fine-tuned - base) =====")
        for k in ("json", "valid", "convert", "type"):
            d = (ft_metrics[k] - base_metrics[k]) * 100
            print(f"  {k:>8}: {d:+5.1f} points")

    return {"base": base_metrics, "fine_tuned": ft_metrics}


@app.local_entrypoint()
def main(run_name: str = "qwen7b-qlora-v1", max_new_tokens: int = 768, base_only: bool = False):
    res = evaluate.remote(
        run_name=run_name, max_new_tokens=max_new_tokens, base_only=base_only
    )
    print("\ndone:", res)
