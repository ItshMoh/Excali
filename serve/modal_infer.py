"""
Inference endpoint for the fine-tuned Excalidraw-DSL model, on Modal.

This serves ONE thing over HTTP: `prompt -> DSL string`. It deliberately does NOT run the
validator or converter — those are cheap, deterministic CPU work and live on the Gradio Space
(`serve/app.py`). Keeping the GPU job minimal is what lets it scale to zero and keeps your
credits from draining while idle.

Lifecycle
---------
- The class loads base Qwen2.5-Coder-7B (4-bit) + your LoRA adapter ONCE per container
  (`@modal.enter`), reusing the same `excali-hf-cache` / `excali-out` Volumes as training.
- With no traffic the container shuts down after `SCALEDOWN_WINDOW` seconds -> $0 while idle.
- The first request after idle pays a cold start (~40-70s) to reload the 7B from cache.

Deploy (gives you a stable public URL)
--------------------------------------
  modal deploy serve/modal_infer.py
  # -> prints:  https://<workspace>--excali-serve-model-generate.modal.run
  # put that URL in the Space as the MODEL_ENDPOINT env var.

Quick local test
----------------
  modal serve serve/modal_infer.py        # ephemeral URL, live-reloads on save
  curl -X POST <url> -H 'content-type: application/json' \
       -d '{"prompt":"architecture for a url shortener"}'
"""

import modal

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
RUN_NAME = "qwen7b-qlora-v1"          # which trained adapter under excali-out to serve

# Must match the system prompt the model was fine-tuned on (the prompt->DSL task).
SYSTEM_PROMPT = (
    "You generate compact diagram DSL (JSON) for Excalidraw. "
    "Return only valid DSL JSON, no explanation."
)

SCALEDOWN_WINDOW = 300                # seconds a warm container lingers after last request

# Set True to lock the endpoint behind Modal proxy auth (recommended once it works). The Space
# then must send Modal-Key / Modal-Secret headers. Left False for first-run simplicity — a
# public URL means anyone who finds it can spend your GPU credits, so flip this before sharing.
REQUIRE_PROXY_AUTH = True 

app = modal.App("excali-serve")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "peft==0.13.2",
        "bitsandbytes==0.44.1",
        "accelerate==1.1.1",
        "fastapi[standard]",          # required for @modal.fastapi_endpoint
    )
    .env({"HF_HOME": "/cache/hf"})
)

cache_vol = modal.Volume.from_name("excali-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("excali-out", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


@app.cls(
    image=image,
    gpu="A100-40GB",
    volumes={"/cache": cache_vol, "/out": out_vol},
    secrets=[hf_secret],
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=600,
)
class Model:
    @modal.enter()
    def load(self):
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base, f"/out/{RUN_NAME}/final-adapter")
        model.eval()
        self.model = model
        print(f"loaded {MODEL_ID} + adapter {RUN_NAME}")

    def _generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        ids = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        do_sample = temperature is not None and temperature > 0
        with self.torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(
            out[0][ids.shape[1]:], skip_special_tokens=True
        ).strip()

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=REQUIRE_PROXY_AUTH)
    def generate(self, data: dict):
        """POST {"prompt": str, "max_new_tokens"?: int, "temperature"?: float}
        -> {"dsl": str}. `dsl` is the raw model output; validation/conversion happen Space-side."""
        prompt = (data or {}).get("prompt", "").strip()
        if not prompt:
            return {"error": "empty prompt"}
        dsl = self._generate(
            prompt,
            int((data or {}).get("max_new_tokens", 768)),
            float((data or {}).get("temperature", 0.0)),
        )
        return {"dsl": dsl}
