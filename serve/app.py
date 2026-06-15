"""
Gradio front-end for the Excalidraw-DSL pipeline (the HF Space).

Flow per request:
  prompt --> Modal GPU endpoint --> DSL string
         --> strip code fences  --> validate_dsl (one retry if invalid)
         --> Converter           --> Excalidraw scene JSON
         --> rendered live in an embedded (read-only) Excalidraw canvas
         --> downloadable .excalidraw file

The GPU work (generation) is remote on Modal; everything here is free CPU work. The validator
and converter are the SAME modules used in training/eval — imported, not reimplemented.

Files this Space needs (upload alongside this one):
  app.py  requirements.txt  packages.txt
  convert_dsl_to_excalidraw.py  validate_dsl.py  dsl_schema.json

Space env vars (Settings -> Variables and secrets):
  MODEL_ENDPOINT        the deployed Modal URL (…-model-generate.modal.run)   [required]
  MODEL_ENDPOINT_KEY    Modal-Key   — only if you enabled proxy auth          [optional]
  MODEL_ENDPOINT_SECRET Modal-Secret— only if you enabled proxy auth          [optional]
"""

import html
import json
import os
import re
import sys
import tempfile
import pathlib

import requests
import gradio as gr

# Import the canonical validator + converter. They sit next to this file on the Space; one dir
# up in the local repo. Add both so it works in either layout.
ROOT = pathlib.Path(__file__).resolve().parent
for p in (ROOT, ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from validate_dsl import load_schema, validate_dsl_text  # noqa: E402
from convert_dsl_to_excalidraw import Converter  # noqa: E402

SCHEMA = load_schema()
ENDPOINT = os.environ.get("MODEL_ENDPOINT", "").strip()
ENDPOINT_KEY = os.environ.get("MODEL_ENDPOINT_KEY", "").strip()
ENDPOINT_SECRET = os.environ.get("MODEL_ENDPOINT_SECRET", "").strip()

EXCALIDRAW_VERSION = "0.17.6"  # UMD build pinned for the embedded read-only canvas


# --------------------------------------------------------------------------- model call
def call_model(prompt: str, temperature: float = 0.0) -> str:
    if not ENDPOINT:
        raise RuntimeError(
            "MODEL_ENDPOINT is not set. Deploy serve/modal_infer.py and put its URL in the "
            "Space's environment variables."
        )
    headers = {"content-type": "application/json"}
    if ENDPOINT_KEY and ENDPOINT_SECRET:  # only when proxy auth is enabled
        headers["Modal-Key"] = ENDPOINT_KEY
        headers["Modal-Secret"] = ENDPOINT_SECRET
    r = requests.post(
        ENDPOINT,
        headers=headers,
        json={"prompt": prompt, "temperature": temperature},
        timeout=180,  # generous: covers a cold start of the scaled-to-zero container
    )
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise RuntimeError(f"model endpoint error: {body['error']}")
    return body["dsl"]


# --------------------------------------------------------------------------- DSL cleanup
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def strip_fences(text: str) -> str:
    """Models occasionally wrap output in ```json fences despite the system prompt."""
    text = _FENCE.sub("", text.strip())
    # Keep only the outermost JSON object if there is trailing prose.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return text.strip()


def get_valid_dsl(prompt: str):
    """Returns (dsl_obj, dsl_text, status_md). Retries once with sampling if the first
    (greedy) attempt fails validation."""
    attempts = [("greedy", 0.0), ("retry (sampled)", 0.4)]
    last_err = ""
    for label, temp in attempts:
        raw = call_model(prompt, temperature=temp)
        text = strip_fences(raw)
        res = validate_dsl_text(text, SCHEMA)
        if res.ok:
            warn = (f" · {len(res.warnings)} warning(s)" if res.warnings else "")
            return json.loads(text), text, f"✅ valid DSL ({label}){warn}"
        last_err = ", ".join(c for c, _ in res.errors) or "unparseable"
    # Both attempts failed — surface the raw text so the user can see what happened.
    return None, text, f"❌ DSL failed validation: {last_err}"


# --------------------------------------------------------------------------- render
def render_scene_html(scene: dict) -> str:
    """An <iframe> hosting a read-only Excalidraw canvas initialised with `scene`."""
    scene_json = json.dumps(scene)
    doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>html,body,#root{{margin:0;padding:0;height:100%;width:100%}}</style>
<script>window.EXCALIDRAW_ASSET_PATH="https://unpkg.com/@excalidraw/excalidraw@{EXCALIDRAW_VERSION}/dist/";</script>
<script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<script crossorigin src="https://unpkg.com/@excalidraw/excalidraw@{EXCALIDRAW_VERSION}/dist/excalidraw.production.min.js"></script>
</head>
<body>
<div id="root"></div>
<script>
  const scene = {scene_json};
  const App = () => React.createElement(ExcalidrawLib.Excalidraw, {{
    initialData: {{
      elements: scene.elements || [],
      appState: Object.assign({{}}, scene.appState, {{viewModeEnabled: true}}),
      files: scene.files || {{}},
      scrollToContent: true,
    }},
    viewModeEnabled: true,
    UIOptions: {{canvasActions: {{export: false, saveToActiveFile: false}}}},
  }});
  ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(App));
</script>
</body>
</html>"""
    srcdoc = html.escape(doc, quote=True)
    return (
        f'<iframe title="excalidraw preview" '
        f'style="width:100%;height:600px;border:1px solid #ddd;border-radius:8px" '
        f'srcdoc="{srcdoc}"></iframe>'
    )


def write_excalidraw_file(scene: dict) -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".excalidraw", delete=False, encoding="utf-8"
    )
    json.dump(scene, tmp, ensure_ascii=False, indent=2)
    tmp.close()
    return tmp.name


# --------------------------------------------------------------------------- main handler
EMPTY_PREVIEW = '<div style="height:600px;display:flex;align-items:center;justify-content:center;color:#999;border:1px dashed #ccc;border-radius:8px">Your diagram preview will appear here</div>'


def generate(prompt: str):
    prompt = (prompt or "").strip()
    if not prompt:
        return EMPTY_PREVIEW, "", None, "Enter a prompt to start."
    try:
        obj, dsl_text, status = get_valid_dsl(prompt)
    except Exception as e:  # noqa: BLE001 — show the user what broke
        return EMPTY_PREVIEW, "", None, f"❌ {type(e).__name__}: {e}"

    if obj is None:
        return EMPTY_PREVIEW, dsl_text, None, status

    try:
        scene = Converter(obj).build()
    except Exception as e:  # noqa: BLE001
        return EMPTY_PREVIEW, dsl_text, None, f"{status}, but converter failed: {e}"

    html_preview = render_scene_html(scene)
    file_path = write_excalidraw_file(scene)
    return html_preview, dsl_text, file_path, f"{status} · {len(scene.get('elements', []))} elements"


# --------------------------------------------------------------------------- UI
EXAMPLES = [
    "Left-to-right architecture for a food delivery app: customer app, API gateway, order service, payment service, Redis cache, Postgres, delivery worker.",
    "Flowchart for a user login flow with validation and a retry on failure.",
    "ER diagram for a blog: users, posts, comments, tags.",
    "Sequence diagram for a checkout: client, API, payment gateway, database.",
    "Data pipeline: ingest events from Kafka, transform in Spark, load into a warehouse, dashboard on top.",
]

with gr.Blocks(title="Excalidraw DSL Generator") as demo:
    gr.Markdown(
        "# 🖍️ Excalidraw Diagram Generator\n"
        "Describe a diagram → fine-tuned model emits compact DSL → deterministic converter "
        "produces a real `.excalidraw` file. Preview is live and editable after download."
    )
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Describe your diagram",
                placeholder="e.g. architecture for a URL shortener with cache and worker",
                lines=3,
            )
            go = gr.Button("Generate", variant="primary")
            status = gr.Markdown("")
            gr.Examples(examples=EXAMPLES, inputs=prompt)
        with gr.Column(scale=3):
            preview = gr.HTML(EMPTY_PREVIEW)
            download = gr.DownloadButton("⬇️ Download .excalidraw", interactive=True)
            with gr.Accordion("Generated DSL (JSON)", open=False):
                dsl_view = gr.Code(language="json", label="DSL")

    go.click(generate, inputs=prompt, outputs=[preview, dsl_view, download, status])
    prompt.submit(generate, inputs=prompt, outputs=[preview, dsl_view, download, status])

if __name__ == "__main__":
    demo.launch()
