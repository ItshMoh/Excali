# Diagram DSL — Specification & Rules (v1)

This is the contract between the **model** (produces DSL) and the **converter** (turns DSL into Excalidraw JSON). Freeze this before generating training data.

The formal structural contract is `dsl.schema.json`. This document covers:
1. The intent and shape (summary).
2. **Semantic rules that JSON Schema cannot express** — enforced by `validate_dsl.py`.
3. Per-diagram-type requirements.
4. The role → shape/color mapping (converter responsibility, fixed here for reference).

---

## 1. Shape summary

```json
{
  "diagram": "<one of 8 types>",
  "title": "string",
  "direction": "LR | RL | TB | BT | radial (optional)",
  "nodes": [ { "id", "label", "role", "group?", "color?", "meta?" } ],
  "edges": [ { "from", "to", "label?", "style?", "meta?" } ],
  "groups": [ { "id", "label" } ],
  "meta": {}
}
```

Principle: **only semantic decisions live here.** No `x`, `y`, `width`, `height`, hex colors, Excalidraw ids, `seed`, `versionNonce`, `roughness`, etc. Those are computed. See the litmus test: *if 10 careful people would all write the same value, it is deterministic → it belongs to the converter, not the DSL.*

---

## 2. Semantic validation rules (NOT checkable by JSON Schema)

`validate_dsl.py` MUST enforce all of these. Each is a binary accept/reject — these double as the dataset-generation filter and the inference-time validation gate.

### Universal (all diagram types)
- **U1 — Unique node ids.** No two `nodes[].id` are equal.
- **U2 — Unique group ids.** No two `groups[].id` are equal.
- **U3 — Edge endpoints resolve.** Every `edge.from` and `edge.to` equals some `node.id`.
- **U4 — Group membership resolves.** Every `node.group` (if present) equals some `group.id`.
- **U5 — No self-loops** unless the diagram type explicitly allows it (none do in v1) → reject `from == to`.
- **U6 — No duplicate edges.** The pair `(from, to)` (plus `label` for multigraphs) is unique. Exception: sequence_diagram may repeat a pair (multiple messages between the same participants).
- **U7 — No isolated nodes** for connected types (architecture, flowchart, pipeline, mind_map): every node appears in at least one edge. Allowed-isolated types: timeline, mobile_wireframe.
- **U8 — Color override sanity.** If `node.color` is set it must be in the enum (schema-checked) — but flag in review if >30% of nodes override color (smells like the model is doing the converter's job).
- **U9 — Label hygiene.** Labels trimmed, non-empty, no newlines (converter handles wrapping).

### flowchart
- **F1 — Exactly one `start` node** (role == "start").
- **F2 — At least one `end` node.**
- **F3 — Decision branches labeled.** Every outgoing edge from a `decision` node has a `label` (typically "yes"/"no"). A decision node should have ≥ 2 outgoing edges.
- **F4 — Reachability.** Every node is reachable from the `start` node by following edges.

### data_pipeline
- **P1 — Acyclic.** The edge set forms a DAG (no cycles). Reject on cycle detection.
- **P2 — Has ≥1 `source` and ≥1 `sink`.**
- **P3 — Sources have no incoming edges; sinks have no outgoing edges.**

### sequence_diagram
- **S1 — Nodes are participants.** All nodes have role `participant` (or `actor`).
- **S2 — Messages reference participants.** (covered by U3, but explicit here.)
- **S3 — Order is significant.** `edges` array order == top-to-bottom message order. The converter must NOT reorder.
- **S4 — `return` messages** (meta.kind == "return") should point back toward an earlier sender (warn, not hard-reject).

### er_diagram
- **E1 — Nodes are entities** (role `entity`).
- **E2 — Each entity should declare `meta.fields`** with ≥1 field; at most one field per entity has `key == "pk"`.
- **E3 — Relationships carry `meta.cardinality`** (schema-enforced via allOf, restated here).

### mind_map
- **M1 — Exactly one `root`** (role == "root").
- **M2 — Tree shape.** Treated as undirected, the graph is connected and acyclic (a tree). Every non-root node has exactly one parent.
- **M3 — Root reaches all nodes.**

### timeline
- **T1 — Every node has `meta.date`.** (schema-enforced.)
- **T2 — Dates parse/sort.** Converter orders milestones by `date`; validator confirms dates are sortable (lexical ISO or a known short form).

### mobile_wireframe
- **W1 — Every node has `meta.kind`.** (schema-enforced.)
- **W2 — Screens are groups.** Each node should belong to a `group` representing a screen; each group renders as one phone frame.
- **W3 — Within a screen, node array order == vertical stacking order** (top to bottom).

### Size / sanity guards
- **G1 — Node count** within `[1, 40]` (schema), but warn if a single diagram exceeds ~25 (layout quality degrades).
- **G2 — Edge count** within `[0, 80]` (schema).
- **G3 — Graph density** — warn if `edges > nodes * 3` (likely nonsense).

---

## 3. Direction defaults (applied by converter when `direction` omitted)

| diagram | default direction |
|---|---|
| system_architecture | LR |
| flowchart | TB |
| data_pipeline | LR |
| sequence_diagram | LR (participants across top) |
| er_diagram | LR (free layout) |
| timeline | LR |
| mind_map | radial |
| mobile_wireframe | TB (stack within screens) |

---

## 4. Role → shape & color (converter reference, fixed here)

The model emits `role`; the converter looks up shape + color. Colors are Excalidraw's standard palette. This table is the single source of truth for styling.

| role | shape | fill | stroke |
|---|---|---|---|
| actor | ellipse | gray `#e9ecef` | `#343a40` |
| client | rectangle | blue `#a5d8ff` | `#1971c2` |
| service | rectangle (rounded) | green `#b2f2bb` | `#2f9e44` |
| gateway | diamond | violet `#d0bfff` | `#7048e8` |
| database | rectangle (cylinder approx) | blue `#a5d8ff` | `#1971c2` |
| cache | rectangle (rounded) | orange `#ffd8a8` | `#e8590c` |
| queue | rectangle | orange `#ffec99` | `#f08c00` |
| worker | rectangle (rounded) | green `#b2f2bb` | `#2f9e44` |
| storage | rectangle | gray `#e9ecef` | `#495057` |
| external | rectangle (dashed) | gray `#f1f3f5` | `#868e96` |
| monitoring | rectangle | yellow `#ffec99` | `#f08c00` |
| auth | rectangle | violet `#d0bfff` | `#7048e8` |
| start | ellipse | green `#b2f2bb` | `#2f9e44` |
| end | ellipse | orange `#ffec99` | `#e8590c` |
| process | rectangle (rounded) | blue `#a5d8ff` | `#1971c2` |
| decision | diamond | yellow `#ffec99` | `#f08c00` |
| io | rectangle | gray `#e9ecef` | `#495057` |
| source | rectangle (rounded) | green `#b2f2bb` | `#2f9e44` |
| transform | rectangle (rounded) | blue `#a5d8ff` | `#1971c2` |
| sink | rectangle (rounded) | violet `#d0bfff` | `#7048e8` |
| entity | rectangle (table) | white `#ffffff` | `#1e1e1e` |
| participant | rectangle | blue `#a5d8ff` | `#1971c2` |
| milestone | ellipse | blue `#a5d8ff` | `#1971c2` |
| root | rectangle (rounded) | violet `#d0bfff` | `#7048e8` |
| branch | rectangle (rounded) | blue `#a5d8ff` | `#1971c2` |
| leaf | rectangle (rounded) | green `#b2f2bb` | `#2f9e44` |
| screen | rectangle (frame) | transparent | `#343a40` |
| ui_element | rectangle | gray `#e9ecef` | `#495057` |
| component | rectangle (rounded) | gray `#e9ecef` | `#495057` |
| generic | rectangle | gray `#e9ecef` | `#495057` |

If `node.color` override is present, it replaces the fill (converter picks a matching stroke shade). Edge `style` maps: solid → `solid`, dashed → `dashed`, dotted → `dotted`; all edges get `endArrowhead: "arrow"`.

---

## 5. Worked example (flowchart)

DSL the model produces:

```json
{
  "diagram": "flowchart",
  "title": "My Workflow",
  "direction": "TB",
  "nodes": [
    { "id": "start",   "label": "Start",        "role": "start"   },
    { "id": "proc",    "label": "Process Data",  "role": "process" },
    { "id": "check",   "label": "Valid?",        "role": "decision" },
    { "id": "done",    "label": "Done",          "role": "end"     },
    { "id": "fix",     "label": "Fix Errors",    "role": "process" }
  ],
  "edges": [
    { "from": "start", "to": "proc" },
    { "from": "proc",  "to": "check" },
    { "from": "check", "to": "done", "label": "yes" },
    { "from": "check", "to": "fix",  "label": "no" },
    { "from": "fix",   "to": "proc" }
  ]
}
```

Passes: U1–U9, F1 (one start), F2 (one end), F3 (decision `check` has labeled yes/no branches), F4 (all reachable from start). The converter computes every position, size, color, shape, arrow path, and Excalidraw field from this.
