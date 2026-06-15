#!/usr/bin/env python3
"""
validate_dsl.py

The Phase 2 accept/reject gate for Diagram DSL.

Two layers, both enforced here (zero pip dependencies — pure stdlib):

  1. STRUCTURAL  — a small JSON-Schema validator that reads `dsl_schema.json` and checks
     types, enums, required keys, additionalProperties, lengths, patterns, and the
     diagram-specific allOf/if-then conditionals. The schema file stays the single source
     of truth; this validator just interprets it.

  2. SEMANTIC    — the rules JSON Schema cannot express (DSL_SPEC.md section 2): unique ids,
     edge endpoints resolve, no isolated nodes, flowchart-has-one-start, pipeline-is-acyclic,
     mind_map-is-a-tree, etc. These are the rules that actually keep the dataset clean.

A DSL is ACCEPTED iff it produces zero errors. Warnings never reject — they flag rows worth
a human glance (e.g. very dense graphs, heavy color overrides).

Usage:
    # validate a single DSL file (or '-' for stdin)
    python3 validate_dsl.py diagram.dsl.json
    cat diagram.dsl.json | python3 validate_dsl.py -

    # validate every assistant DSL in a training file (the dataset gate)
    python3 validate_dsl.py --jsonl train.jsonl
    python3 validate_dsl.py --jsonl train.jsonl --max-show 20 --warnings

Exit code is 0 when everything passed, 1 otherwise — usable in CI / generation loops.

As a library:
    from validate_dsl import validate_dsl, validate_dsl_text, Result
    res = validate_dsl(dsl_dict)
    if res.ok: ...
"""

import argparse
import json
import os
import re
import sys

SCHEMA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsl_schema.json")

# Diagram types whose every node must touch at least one edge (DSL_SPEC U7).
CONNECTED_TYPES = {"system_architecture", "flowchart", "data_pipeline", "mind_map"}


# --------------------------------------------------------------------------------------
# Result accumulator
# --------------------------------------------------------------------------------------

class Result:
    """Collects validation findings. `ok` is True iff there are no errors."""

    def __init__(self):
        self.errors = []    # list of (code, message)
        self.warnings = []  # list of (code, message)

    def error(self, code, message):
        self.errors.append((code, message))

    def warn(self, code, message):
        self.warnings.append((code, message))

    @property
    def ok(self):
        return not self.errors

    def extend(self, other):
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def __str__(self):
        lines = []
        for code, msg in self.errors:
            lines.append(f"  ERROR [{code}] {msg}")
        for code, msg in self.warnings:
            lines.append(f"  WARN  [{code}] {msg}")
        if not lines:
            lines.append("  OK")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Layer 1: structural validation (a minimal JSON-Schema interpreter for our schema)
# --------------------------------------------------------------------------------------

_JSON_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def _type_matches(value, type_name):
    # bool is a subclass of int in Python; keep number/integer from swallowing booleans.
    if type_name in ("number", "integer") and isinstance(value, bool):
        return False
    if type_name == "boolean":
        return isinstance(value, bool)
    return isinstance(value, _JSON_TYPES[type_name])


class SchemaValidator:
    """Interprets the subset of JSON Schema 2020-12 that dsl_schema.json actually uses:
    $ref/$defs, type, enum, const, required, properties, additionalProperties, items,
    minItems/maxItems, minLength/maxLength, pattern, allOf, and if/then/else."""

    def __init__(self, schema):
        self.root = schema

    def validate(self, instance):
        errors = []
        self._check(instance, self.root, "$", errors)
        return errors

    # -- internals --------------------------------------------------------------------

    def _resolve(self, schema):
        # Follow a $ref chain ("#/$defs/node"). Sibling keywords in this schema are only
        # `description` (non-validating), so replacing the node with its target is safe.
        seen = 0
        while isinstance(schema, dict) and "$ref" in schema:
            ref = schema["$ref"]
            parts = [p for p in ref.lstrip("#/").split("/") if p]
            target = self.root
            for p in parts:
                target = target[p]
            schema = target
            seen += 1
            if seen > 50:
                break
        return schema

    def _matches(self, instance, schema):
        """True iff `instance` validates against `schema` (used for if/then)."""
        tmp = []
        self._check(instance, schema, "$", tmp)
        return not tmp

    def _check(self, instance, schema, path, errors):
        schema = self._resolve(schema)
        if not isinstance(schema, dict):
            return

        # allOf
        for sub in schema.get("allOf", []):
            self._check(instance, sub, path, errors)

        # if / then / else
        if "if" in schema:
            if self._matches(instance, schema["if"]):
                if "then" in schema:
                    self._check(instance, schema["then"], path, errors)
            elif "else" in schema:
                self._check(instance, schema["else"], path, errors)

        # type
        if "type" in schema:
            t = schema["type"]
            types = t if isinstance(t, list) else [t]
            if not any(_type_matches(instance, tn) for tn in types):
                errors.append(("schema", f"{path}: expected type {t}, got {_pytype(instance)}"))
                return  # deeper keyword checks would just produce noise

        # enum / const
        if "enum" in schema and instance not in schema["enum"]:
            errors.append(("schema", f"{path}: {instance!r} is not one of {schema['enum']}"))
        if "const" in schema and instance != schema["const"]:
            errors.append(("schema", f"{path}: expected const {schema['const']!r}"))

        if isinstance(instance, str):
            self._check_string(instance, schema, path, errors)
        elif isinstance(instance, list):
            self._check_array(instance, schema, path, errors)
        elif isinstance(instance, dict):
            self._check_object(instance, schema, path, errors)

    def _check_string(self, instance, schema, path, errors):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(("schema", f"{path}: shorter than minLength {schema['minLength']}"))
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(("schema", f"{path}: longer than maxLength {schema['maxLength']}"))
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append(("schema", f"{path}: {instance!r} does not match pattern {schema['pattern']}"))

    def _check_array(self, instance, schema, path, errors):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(("schema", f"{path}: fewer than minItems {schema['minItems']}"))
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(("schema", f"{path}: more than maxItems {schema['maxItems']}"))
        items = schema.get("items")
        if items is not None:
            for i, el in enumerate(instance):
                self._check(el, items, f"{path}[{i}]", errors)

    def _check_object(self, instance, schema, path, errors):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in instance:
                errors.append(("schema", f"{path}: missing required property '{req}'"))
        additional = schema.get("additionalProperties", True)
        for key, val in instance.items():
            if key in props:
                self._check(val, props[key], f"{path}.{key}", errors)
            elif additional is False:
                errors.append(("schema", f"{path}: unexpected property '{key}' (additionalProperties=false)"))
            elif isinstance(additional, dict):
                self._check(val, additional, f"{path}.{key}", errors)


def _pytype(v):
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, dict):
        return "object"
    if isinstance(v, list):
        return "array"
    if isinstance(v, str):
        return "string"
    if isinstance(v, (int, float)):
        return "number"
    if v is None:
        return "null"
    return type(v).__name__


# --------------------------------------------------------------------------------------
# Layer 2: semantic validation (DSL_SPEC.md section 2)
# --------------------------------------------------------------------------------------

def _semantic(dsl, res):
    diagram = dsl["diagram"]
    nodes = dsl["nodes"]
    edges = dsl.get("edges", []) or []
    groups = dsl.get("groups", []) or []

    node_ids = [n["id"] for n in nodes]
    id_set = set(node_ids)
    by_id = {n["id"]: n for n in nodes}
    group_ids = [g["id"] for g in groups]

    _universal(dsl, diagram, nodes, edges, groups, node_ids, id_set, group_ids, res)

    # Edges that actually resolve — later graph checks must not crash on dangling refs.
    valid_edges = [e for e in edges if e["from"] in id_set and e["to"] in id_set]

    dispatch = {
        "flowchart": _flowchart,
        "data_pipeline": _data_pipeline,
        "sequence_diagram": _sequence,
        "er_diagram": _er,
        "mind_map": _mind_map,
        "timeline": _timeline,
        "mobile_wireframe": _wireframe,
    }
    fn = dispatch.get(diagram)
    if fn:
        fn(nodes, valid_edges, edges, by_id, id_set, groups, res)

    _size_guards(nodes, edges, res)


def _roles(nodes, role):
    return [n for n in nodes if n.get("role") == role]


def _universal(dsl, diagram, nodes, edges, groups, node_ids, id_set, group_ids, res):
    # U1 — unique node ids
    for dup in _dups(node_ids):
        res.error("U1", f"duplicate node id '{dup}'")
    # U2 — unique group ids
    for dup in _dups(group_ids):
        res.error("U2", f"duplicate group id '{dup}'")
    group_set = set(group_ids)

    # U3 — edge endpoints resolve
    for i, e in enumerate(edges):
        if e["from"] not in id_set:
            res.error("U3", f"edges[{i}].from '{e['from']}' is not a node id")
        if e["to"] not in id_set:
            res.error("U3", f"edges[{i}].to '{e['to']}' is not a node id")

    # U4 — group membership resolves
    for n in nodes:
        g = n.get("group")
        if g is not None and g not in group_set:
            res.error("U4", f"node '{n['id']}' references unknown group '{g}'")

    # U5 — no self loops
    for i, e in enumerate(edges):
        if e["from"] == e["to"]:
            res.error("U5", f"edges[{i}] is a self-loop on '{e['from']}'")

    # U6 — no duplicate edges (sequence_diagram may legitimately repeat a pair)
    if diagram != "sequence_diagram":
        seen = set()
        for i, e in enumerate(edges):
            key = (e["from"], e["to"], e.get("label", ""))
            if key in seen:
                res.error("U6", f"edges[{i}] duplicates edge {e['from']}->{e['to']} (label {e.get('label','')!r})")
            seen.add(key)

    # U7 — no isolated nodes for connected types
    if diagram in CONNECTED_TYPES:
        touched = set()
        for e in edges:
            touched.add(e["from"])
            touched.add(e["to"])
        for n in nodes:
            if n["id"] not in touched:
                res.error("U7", f"node '{n['id']}' is isolated (no edge) in a connected diagram")

    # U8 — color override sanity (warn only)
    overrides = sum(1 for n in nodes if n.get("color"))
    if nodes and overrides / len(nodes) > 0.30:
        res.warn("U8", f"{overrides}/{len(nodes)} nodes override color (>30%) — styling should come from role")

    # U9 — label hygiene
    _label_hygiene("title", dsl.get("title", ""), res)
    for n in nodes:
        _label_hygiene(f"node '{n['id']}' label", n.get("label", ""), res)
    for g in groups:
        _label_hygiene(f"group '{g['id']}' label", g.get("label", ""), res)
    for i, e in enumerate(edges):
        if "label" in e:
            _label_hygiene(f"edges[{i}] label", e["label"], res)


def _label_hygiene(where, text, res):
    if "\n" in text or "\r" in text:
        res.error("U9", f"{where} contains a newline (converter handles wrapping)")
    elif text != text.strip():
        res.warn("U9", f"{where} has leading/trailing whitespace")


def _flowchart(nodes, edges, raw_edges, by_id, id_set, groups, res):
    starts = _roles(nodes, "start")
    # F1 — exactly one start
    if len(starts) != 1:
        res.error("F1", f"flowchart needs exactly one 'start' node, found {len(starts)}")
    # F2 — at least one end
    if not _roles(nodes, "end"):
        res.error("F2", "flowchart needs at least one 'end' node")

    out = _out_adj(edges)
    # F3 — decision branches labeled, >= 2 outgoing
    for n in nodes:
        if n.get("role") == "decision":
            outs = [e for e in edges if e["from"] == n["id"]]
            if len(outs) < 2:
                res.error("F3", f"decision node '{n['id']}' has {len(outs)} outgoing edges (need >= 2)")
            for e in outs:
                if not e.get("label", "").strip():
                    res.error("F3", f"decision node '{n['id']}' has an unlabeled branch to '{e['to']}'")

    # F4 — reachability from the (first) start
    if starts:
        reachable = _bfs(starts[0]["id"], out)
        for n in nodes:
            if n["id"] not in reachable:
                res.error("F4", f"node '{n['id']}' is not reachable from start '{starts[0]['id']}'")


def _data_pipeline(nodes, edges, raw_edges, by_id, id_set, groups, res):
    out = _out_adj(edges)
    indeg = _indegree(nodes, edges)
    outdeg = _outdegree(nodes, edges)

    # P1 — acyclic
    cycle = _find_cycle(nodes, out)
    if cycle:
        res.error("P1", f"data_pipeline must be acyclic; cycle through {' -> '.join(cycle)}")

    # P2 — has >=1 source and >=1 sink (by role)
    sources = _roles(nodes, "source")
    sinks = _roles(nodes, "sink")
    if not sources:
        res.error("P2", "data_pipeline needs at least one 'source' node")
    if not sinks:
        res.error("P2", "data_pipeline needs at least one 'sink' node")

    # P3 — sources have no incoming, sinks have no outgoing
    for n in sources:
        if indeg.get(n["id"], 0) > 0:
            res.error("P3", f"source '{n['id']}' has incoming edges")
    for n in sinks:
        if outdeg.get(n["id"], 0) > 0:
            res.error("P3", f"sink '{n['id']}' has outgoing edges")


def _sequence(nodes, edges, raw_edges, by_id, id_set, groups, res):
    # S1 — all nodes are participants (or actors)
    for n in nodes:
        if n.get("role") not in ("participant", "actor"):
            res.error("S1", f"sequence_diagram node '{n['id']}' has role '{n.get('role')}' (expected participant/actor)")

    # S4 — return messages should point back toward an earlier sender (warn only).
    prior_pairs = set()
    for i, e in enumerate(raw_edges):
        kind = (e.get("meta") or {}).get("kind")
        if kind == "return":
            if (e["to"], e["from"]) not in prior_pairs:
                res.warn("S4", f"edges[{i}] is a 'return' to '{e['to']}' with no preceding message from it")
        prior_pairs.add((e["from"], e["to"]))


def _er(nodes, edges, raw_edges, by_id, id_set, groups, res):
    # E1 — nodes are entities
    for n in nodes:
        if n.get("role") != "entity":
            res.error("E1", f"er_diagram node '{n['id']}' has role '{n.get('role')}' (expected entity)")
    # E2 — each entity declares >=1 field; at most one pk
    for n in nodes:
        fields = (n.get("meta") or {}).get("fields")
        if not fields:
            res.error("E2", f"entity '{n['id']}' declares no fields")
            continue
        pks = [f for f in fields if f.get("key") == "pk"]
        if len(pks) > 1:
            res.error("E2", f"entity '{n['id']}' has {len(pks)} primary keys (max 1)")
    # E3 — relationships carry cardinality (also schema-enforced; restated)
    for i, e in enumerate(raw_edges):
        if not (e.get("meta") or {}).get("cardinality"):
            res.error("E3", f"edges[{i}] ({e['from']}->{e['to']}) is missing meta.cardinality")


def _mind_map(nodes, edges, raw_edges, by_id, id_set, groups, res):
    roots = _roles(nodes, "root")
    # M1 — exactly one root
    if len(roots) != 1:
        res.error("M1", f"mind_map needs exactly one 'root' node, found {len(roots)}")

    n = len(nodes)
    # M2 — tree shape: undirected connected & acyclic, every non-root has one parent.
    if len(edges) != n - 1:
        res.error("M2", f"mind_map is not a tree: {n} nodes need exactly {n-1} edges, found {len(edges)}")
    undirected = _undirected_adj(edges)
    if roots:
        seen = _bfs(roots[0]["id"], undirected)
        if len(seen) != n:
            res.error("M2", f"mind_map is not connected: {len(seen)}/{n} nodes reachable from root (undirected)")
        # directed parent check: root in-degree 0, others exactly 1
        indeg = _indegree(nodes, edges)
        for nd in nodes:
            d = indeg.get(nd["id"], 0)
            if nd["id"] == roots[0]["id"]:
                if d != 0:
                    res.error("M2", f"root '{nd['id']}' has {d} incoming edges (expected 0)")
            elif d != 1:
                res.error("M2", f"node '{nd['id']}' has {d} parents (expected exactly 1)")

        # M3 — root reaches all (directed)
        out = _out_adj(edges)
        reach = _bfs(roots[0]["id"], out)
        for nd in nodes:
            if nd["id"] not in reach:
                res.error("M3", f"node '{nd['id']}' is not reachable from root (directed)")


def _timeline(nodes, edges, raw_edges, by_id, id_set, groups, res):
    # T1 — every node has meta.date (schema-enforced; restated). T2 — sortable.
    for n in nodes:
        date = (n.get("meta") or {}).get("date")
        if not date:
            res.error("T1", f"timeline node '{n['id']}' is missing meta.date")
        elif _date_sort_key(date) is None:
            res.warn("T2", f"timeline node '{n['id']}' date {date!r} is not obviously sortable")


def _wireframe(nodes, edges, raw_edges, by_id, id_set, groups, res):
    # W1 — every node has meta.kind (schema-enforced; restated)
    for n in nodes:
        if not (n.get("meta") or {}).get("kind"):
            res.error("W1", f"wireframe node '{n['id']}' is missing meta.kind")
    # W2 — every node belongs to a screen group
    if not groups:
        res.error("W2", "mobile_wireframe has no groups (screens)")
    for n in nodes:
        if not n.get("group"):
            res.error("W2", f"wireframe node '{n['id']}' is not assigned to a screen group")


def _size_guards(nodes, edges, res):
    # G1 / G3 — schema already bounds counts; these are quality warnings.
    if len(nodes) > 25:
        res.warn("G1", f"{len(nodes)} nodes — layout quality tends to degrade past ~25")
    if nodes and len(edges) > len(nodes) * 3:
        res.warn("G3", f"{len(edges)} edges for {len(nodes)} nodes (>3x) — likely too dense")


# --------------------------------------------------------------------------------------
# Graph helpers
# --------------------------------------------------------------------------------------

def _dups(seq):
    seen, dups = set(), []
    for x in seq:
        if x in seen and x not in dups:
            dups.append(x)
        seen.add(x)
    return dups


def _out_adj(edges):
    adj = {}
    for e in edges:
        adj.setdefault(e["from"], []).append(e["to"])
    return adj


def _undirected_adj(edges):
    adj = {}
    for e in edges:
        adj.setdefault(e["from"], []).append(e["to"])
        adj.setdefault(e["to"], []).append(e["from"])
    return adj


def _indegree(nodes, edges):
    deg = {n["id"]: 0 for n in nodes}
    for e in edges:
        deg[e["to"]] = deg.get(e["to"], 0) + 1
    return deg


def _outdegree(nodes, edges):
    deg = {n["id"]: 0 for n in nodes}
    for e in edges:
        deg[e["from"]] = deg.get(e["from"], 0) + 1
    return deg


def _bfs(start, adj):
    seen, stack = set(), [start]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                stack.append(nxt)
    return seen


def _find_cycle(nodes, adj):
    """Return a node sequence describing a directed cycle, or None. DFS with colors."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n["id"]: WHITE for n in nodes}
    parent = {}

    def visit(u):
        color[u] = GRAY
        for v in adj.get(u, []):
            if v not in color:           # endpoint outside node set; skip defensively
                continue
            if color[v] == WHITE:
                parent[v] = u
                r = visit(v)
                if r:
                    return r
            elif color[v] == GRAY:       # back edge u->v closes a cycle
                path = [u]               # walk parents from u up to the ancestor v
                x = u
                while x != v and x in parent:
                    x = parent[x]
                    path.append(x)
                path.reverse()           # now v ... u
                return path + [v]         # close the loop: v ... u -> v
        color[u] = BLACK
        return None

    sys.setrecursionlimit(10000)
    for n in nodes:
        if color[n["id"]] == WHITE:
            r = visit(n["id"])
            if r:
                return r
    return None


_DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


def _date_sort_key(date):
    """Return a sortable key for common date forms, or None if not obviously sortable.
    Accepts ISO-ish (YYYY, YYYY-MM, YYYY-MM-DD) and 'Q<n> YYYY' style labels."""
    date = date.strip()
    if _DATE_RE.match(date):
        return date
    m = re.match(r"^Q([1-4])\s+(\d{4})$", date)
    if m:
        return f"{m.group(2)}-Q{m.group(1)}"
    return None


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

_SCHEMA_CACHE = None


def load_schema(path=SCHEMA_FILE):
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        with open(path, "r", encoding="utf-8") as f:
            _SCHEMA_CACHE = json.load(f)
    return _SCHEMA_CACHE


def validate_dsl(dsl, schema=None):
    """Validate a parsed DSL object. Returns a Result (res.ok == accepted)."""
    res = Result()
    schema = schema if schema is not None else load_schema()

    # Layer 1 — structural. If it fails, semantic checks would just crash on the same
    # malformed data, so we stop here and report the structural errors.
    structural_errors = SchemaValidator(schema).validate(dsl)
    if structural_errors:
        for code, msg in structural_errors:
            res.error(code, msg)
        return res

    # Layer 2 — semantic.
    _semantic(dsl, res)
    return res


def validate_dsl_text(text, schema=None):
    """Validate DSL given as a JSON string (e.g. a model's assistant output)."""
    res = Result()
    try:
        dsl = json.loads(text)
    except json.JSONDecodeError as e:
        res.error("json", f"not valid JSON: {e}")
        return res
    if not isinstance(dsl, dict):
        res.error("json", f"top-level DSL must be an object, got {_pytype(dsl)}")
        return res
    return validate_dsl(dsl, schema)


def _assistant_content(row):
    """Pull the assistant DSL string out of a chat-format training row."""
    msgs = row.get("messages")
    if not isinstance(msgs, list):
        return None
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content")
    return None


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _read(path):
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _run_single(path, show_warnings):
    res = validate_dsl_text(_read(path))
    status = "PASS" if res.ok else "FAIL"
    print(f"{status}: {path}")
    for code, msg in res.errors:
        print(f"  ERROR [{code}] {msg}")
    if show_warnings:
        for code, msg in res.warnings:
            print(f"  WARN  [{code}] {msg}")
    return 0 if res.ok else 1


def _run_jsonl(path, show_warnings, max_show):
    text = _read(path)
    total = passed = failed = warned = 0
    shown = 0
    error_codes = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            failed += 1
            error_codes["json"] = error_codes.get("json", 0) + 1
            if shown < max_show:
                print(f"FAIL line {lineno}: row is not valid JSON: {e}")
                shown += 1
            continue

        content = _assistant_content(row) if isinstance(row, dict) else None
        # Accept either a chat row or a bare DSL object on the line.
        res = validate_dsl_text(content) if content is not None else validate_dsl(row) \
            if isinstance(row, dict) else Result()
        if content is None and not isinstance(row, dict):
            res = Result()
            res.error("json", "line is neither a chat row nor a DSL object")

        if res.warnings:
            warned += 1
        for code, _ in res.errors:
            error_codes[code] = error_codes.get(code, 0) + 1

        if res.ok:
            passed += 1
            if show_warnings and res.warnings and shown < max_show:
                print(f"WARN line {lineno}:")
                for code, msg in res.warnings:
                    print(f"  WARN  [{code}] {msg}")
                shown += 1
        else:
            failed += 1
            if shown < max_show:
                print(f"FAIL line {lineno}:")
                for code, msg in res.errors:
                    print(f"  ERROR [{code}] {msg}")
                shown += 1

    print("-" * 60)
    print(f"total={total}  passed={passed}  failed={failed}  with_warnings={warned}")
    if error_codes:
        breakdown = "  ".join(f"{c}={n}" for c, n in sorted(error_codes.items(), key=lambda x: -x[1]))
        print(f"error breakdown: {breakdown}")
    return 0 if failed == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate Diagram DSL (structural + semantic gate).")
    ap.add_argument("input", help="DSL JSON file, JSONL dataset, or '-' for stdin")
    ap.add_argument("--jsonl", action="store_true",
                    help="treat input as a JSONL dataset; validate each row's assistant DSL")
    ap.add_argument("--warnings", action="store_true", help="also show warnings")
    ap.add_argument("--max-show", type=int, default=20,
                    help="max failing/warning rows to print in --jsonl mode (default 20)")
    args = ap.parse_args(argv)

    if args.jsonl:
        return _run_jsonl(args.input, args.warnings, args.max_show)
    return _run_single(args.input, args.warnings)


if __name__ == "__main__":
    sys.exit(main())
