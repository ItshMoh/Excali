---
title: Excalidraw Generator
emoji: 🖍️
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Describe a diagram in plain English, get a real editable .excalidraw file.
---

# 🖍️ Excalidraw Diagram Generator

**Upgrade your presentations and be promoted.**

Okay — we can't *literally* promise the promotion. But we can promise this: the slide that
actually gets you noticed is never the one with seven bullet points. It's the one with a clean
diagram that makes a hard idea look obvious in three seconds.

The problem is that *making* that diagram is a chore. You open a drawing tool, you drag
rectangles, you fight with arrows that won't stay attached, you nudge boxes by one pixel for
twenty minutes, and the meeting starts in ten. So you give up and paste another bullet list.

**This tool removes the chore.** You type what you want in plain English:

> *"Left-to-right architecture for a food delivery app: customer app, API gateway, order
> service, payment service, Redis cache, Postgres, delivery worker."*

…and you get back a real, **fully editable** `.excalidraw` file — laid out, wired up, and
ready to drop into your deck. Don't like where a box sits? Drag it. The arrows follow.

---

## When this is the right tool

- **The night-before-the-demo deck** — turn your architecture into a diagram while you still
  have time to sleep.
- **Explaining a system to a new teammate** — a flowchart they can *edit* beats a screenshot
  they can only squint at.
- **Design docs & RFCs** — sequence diagrams and ER diagrams without leaving your train of
  thought to go fight a canvas.
- **Whiteboarding sessions** — generate a first draft in seconds, then refine it live in
  Excalidraw like you drew it yourself.

It draws **8 kinds of diagram**: system architecture, flowchart, ER diagram, sequence
diagram, data pipeline, timeline, mind map, and mobile wireframe.

---

> NOTE: First request may take ~40–70s: the GPU sleeps when nobody's using it (so it costs nothing
> when idle) and takes a moment to wake up. After that it's quick. 
## Why it's different from "just ask ChatGPT"

A general LLM, asked for an Excalidraw file, has to hand-write hundreds of lines of strict
JSON — element IDs, x/y coordinates, arrow bindings, version nonces — and one slip produces a
file that won't open. That's a lot of ways to fail at something a computer should never get
wrong.

So we split the job in two:

```
your prompt
   │
   ▼
  fine-tuned model      →   writes a tiny, semantic diagram plan (a compact DSL)
   │                          "these boxes, these connections, this kind of node"
   ▼
  validator             →   checks the plan; retries once if it's off
   │
   ▼
  deterministic engine  →   computes every coordinate, arrow, and color — perfectly
   │
   ▼
  .excalidraw file      →   live preview + one-click download
```

**The model only decides *meaning*** (what connects to what). **Code handles everything a
computer can get exactly right** (layout, geometry, file format). That's why a small,
specialized model can be *more reliable* here than a giant general one — it never has to be
correct about format, only about ideas. A ~12-line plan becomes a ~340-line valid Excalidraw
file, every time.

Layout itself is done by Graphviz for the graph-shaped diagrams and purpose-built layouts for
the rest, so boxes don't overlap and arrows route cleanly.

---

## 🚀 How to use it

1. **Describe your diagram** in the box — be specific about the pieces and how they relate.
2. Hit **Generate**.
3. **Preview** it live in the embedded canvas.
4. **Download `.excalidraw`** and open it at [excalidraw.com](https://excalidraw.com) or the
   desktop app — it's yours to edit, restyle, and present.



---

## 🔗 Links

- **Live Space:** `https://huggingface.co/spaces/build-small-hackathon/excali-draw` 
- **Source code:** `https://github.com/ItshMoh/Excali` 
- **Demo Video:** `https://drive.google.com/drive/folders/1HzYcNPMHJa-1C_aB8V1GE4H-JtIdHG7X?usp=sharing`
---

##  Under the hood

| Piece | What it does |
|---|---|
| **Base model** | Qwen2.5-Coder-7B-Instruct, fine-tuned with QLoRA on a curated diagram-DSL dataset |
| **DSL** | A compact, semantic JSON the model emits instead of raw Excalidraw — small, easy to validate, hard to break |
| **Validator** | Structural + semantic checks (unique IDs, edges resolve, valid shapes per diagram type); one retry on failure |
| **Converter** | Deterministic DSL → Excalidraw: Graphviz layout for graphs, hand-built layouts for the rest; bidirectionally-bound arrows so they stay attached when you drag |
| **Serving** | Generation runs on a scale-to-zero Modal GPU; validation, conversion, and preview run on the free Space CPU |

The reliability bet is deliberate: the win isn't "prettier than GPT on every drawing," it's
**valid, editable files on the first try, cheaply, and fast.**

---

<sub>Built as a focused fine-tuning project. Diagrams are generated — give the output a
glance before you put it in front of the board. 😉</sub>
