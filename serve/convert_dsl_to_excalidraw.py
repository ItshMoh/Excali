#!/usr/bin/env python3
"""
convert_dsl_to_excalidraw.py

Deterministic converter: Diagram DSL (see dsl.schema.json / DSL_SPEC.md)  ->  .excalidraw JSON.

Design:
  - Graph family (system_architecture, flowchart, data_pipeline, er_diagram) is laid out by
    Graphviz `dot` (called as a subprocess with -Tjson). Only dependency: the `dot` binary.
  - sequence_diagram, timeline, mind_map, mobile_wireframe use hand-rolled layouts.
  - A shared emit layer builds valid Excalidraw elements (shapes, bound text, bound arrows),
    handles z-ordering, and self-validates the output (no dangling refs, overlap warnings).

The DSL carries ONLY semantic decisions. Everything geometric/stylistic is computed here.

Usage:
    python3 convert_dsl_to_excalidraw.py input.dsl.json -o output.excalidraw
    cat input.dsl.json | python3 convert_dsl_to_excalidraw.py - > output.excalidraw
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

FONT = 20
TITLE_FONT = 28
FONT_FAMILY = 1            # 1 = Excalidraw hand-drawn (Virgil)
LINE_HEIGHT = 1.25
CHAR_W = 0.60             # avg glyph width as a fraction of font size (approx for Virgil)
PAD_X = 22
PAD_Y = 16
NODE_MIN_W = 120
NODE_MAX_W = 320
NODE_MIN_H = 56
INK = "#1e1e1e"
TRANSPARENT = "transparent"
WHITE = "#ffffff"

# role -> (shape, fill, stroke, rounded, dashed)
# shape in {"rectangle","ellipse","diamond"}.  Mirrors the table in DSL_SPEC.md.
ROLE_STYLE = {
    "actor":       ("ellipse",   "#e9ecef", "#343a40", False, False),
    "client":      ("rectangle", "#a5d8ff", "#1971c2", False, False),
    "service":     ("rectangle", "#b2f2bb", "#2f9e44", True,  False),
    "gateway":     ("diamond",   "#d0bfff", "#7048e8", False, False),
    "database":    ("rectangle", "#a5d8ff", "#1971c2", False, False),
    "cache":       ("rectangle", "#ffd8a8", "#e8590c", True,  False),
    "queue":       ("rectangle", "#ffec99", "#f08c00", False, False),
    "worker":      ("rectangle", "#b2f2bb", "#2f9e44", True,  False),
    "storage":     ("rectangle", "#e9ecef", "#495057", False, False),
    "external":    ("rectangle", "#f1f3f5", "#868e96", False, True),
    "monitoring":  ("rectangle", "#ffec99", "#f08c00", False, False),
    "auth":        ("rectangle", "#d0bfff", "#7048e8", False, False),
    "start":       ("ellipse",   "#b2f2bb", "#2f9e44", False, False),
    "end":         ("ellipse",   "#ffec99", "#e8590c", False, False),
    "process":     ("rectangle", "#a5d8ff", "#1971c2", True,  False),
    "decision":    ("diamond",   "#ffec99", "#f08c00", False, False),
    "io":          ("rectangle", "#e9ecef", "#495057", False, False),
    "source":      ("rectangle", "#b2f2bb", "#2f9e44", True,  False),
    "transform":   ("rectangle", "#a5d8ff", "#1971c2", True,  False),
    "sink":        ("rectangle", "#d0bfff", "#7048e8", True,  False),
    "entity":      ("rectangle", WHITE,     "#1e1e1e", False, False),
    "participant": ("rectangle", "#a5d8ff", "#1971c2", False, False),
    "milestone":   ("ellipse",   "#a5d8ff", "#1971c2", False, False),
    "root":        ("rectangle", "#d0bfff", "#7048e8", True,  False),
    "branch":      ("rectangle", "#a5d8ff", "#1971c2", True,  False),
    "leaf":        ("rectangle", "#b2f2bb", "#2f9e44", True,  False),
    "screen":      ("rectangle", TRANSPARENT, "#343a40", True, False),
    "ui_element":  ("rectangle", "#e9ecef", "#495057", False, False),
    "component":   ("rectangle", "#e9ecef", "#495057", True,  False),
    "generic":     ("rectangle", "#e9ecef", "#495057", False, False),
}

COLOR_OVERRIDE = {
    "red":    ("#ffc9c9", "#e03131"),
    "orange": ("#ffd8a8", "#e8590c"),
    "yellow": ("#ffec99", "#f08c00"),
    "green":  ("#b2f2bb", "#2f9e44"),
    "blue":   ("#a5d8ff", "#1971c2"),
    "violet": ("#d0bfff", "#7048e8"),
    "gray":   ("#e9ecef", "#495057"),
    "black":  ("#e9ecef", "#1e1e1e"),
}

STROKE_STYLE = {"solid": "solid", "dashed": "dashed", "dotted": "dotted"}

DEFAULT_DIRECTION = {
    "system_architecture": "LR",
    "flowchart": "TB",
    "data_pipeline": "LR",
    "er_diagram": "LR",
    "sequence_diagram": "LR",
    "timeline": "LR",
    "mind_map": "radial",
    "mobile_wireframe": "TB",
}

GRAPH_FAMILY = {"system_architecture", "flowchart", "data_pipeline", "er_diagram"}

# --------------------------------------------------------------------------------------
# Text measurement / sizing
# --------------------------------------------------------------------------------------

def text_dims(text, font=FONT):
    lines = str(text).split("\n")
    w = max((len(l) for l in lines), default=1) * font * CHAR_W
    h = len(lines) * font * LINE_HEIGHT
    return max(w, font * CHAR_W), max(h, font * LINE_HEIGHT)


def node_size(label):
    tw, th = text_dims(label, FONT)
    w = min(max(tw + 2 * PAD_X, NODE_MIN_W), NODE_MAX_W)
    h = max(th + 2 * PAD_Y, NODE_MIN_H)
    return w, h


# --------------------------------------------------------------------------------------
# Converter
# --------------------------------------------------------------------------------------

class Converter:
    def __init__(self, dsl, seed=None):
        self.dsl = dsl
        self.diagram = dsl["diagram"]
        self.title = dsl["title"]
        self.direction = dsl.get("direction") or DEFAULT_DIRECTION[self.diagram]
        self.nodes = dsl.get("nodes", [])
        self.edges = dsl.get("edges", [])
        self.groups = dsl.get("groups", [])
        self.node_by_id = {n["id"]: n for n in self.nodes}

        if seed is None:
            h = hashlib.md5((self.diagram + "|" + self.title).encode()).hexdigest()
            seed = int(h[:8], 16)
        self._seed = seed
        self._rng_state = seed & 0x7FFFFFFF

        self.elements = []        # accumulates element dicts (with temporary "_z")
        self.box_elem = {}        # node_id -> shape element dict (to append boundElements)
        self.boxes = {}           # node_id -> (x, y, w, h)
        self._idc = 0

    # ---- deterministic helpers -------------------------------------------------------

    def _rand(self):
        # simple LCG so output is reproducible without Date.now()/random module state
        self._rng_state = (self._rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        return self._rng_state

    def _new_id(self, prefix):
        self._idc += 1
        return f"{prefix}_{self._idc}"

    def _base(self, typ, x, y, w, h, z):
        return {
            "id": None, "type": typ,
            "x": round(x, 2), "y": round(y, 2),
            "width": round(w, 2), "height": round(h, 2),
            "angle": 0,
            "strokeColor": INK, "backgroundColor": TRANSPARENT,
            "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
            "roughness": 1, "opacity": 100,
            "groupIds": [], "frameId": None, "roundness": None,
            "seed": self._rand(), "version": 1, "versionNonce": self._rand(),
            "isDeleted": False, "boundElements": [], "updated": 1,
            "link": None, "locked": False,
            "_z": z,
        }

    def _add(self, el):
        self.elements.append(el)
        return el

    # ---- element builders ------------------------------------------------------------

    def add_box(self, node_id, x, y, w, h, role, color_override=None, z=2):
        shape, fill, stroke, rounded, dashed = ROLE_STYLE.get(role, ROLE_STYLE["generic"])
        if color_override and color_override in COLOR_OVERRIDE:
            fill, stroke = COLOR_OVERRIDE[color_override]
        el = self._base(shape, x, y, w, h, z)
        el["id"] = self._new_id("sh")
        el["backgroundColor"] = fill
        el["strokeColor"] = stroke
        if rounded and shape == "rectangle":
            el["roundness"] = {"type": 3}
        if dashed:
            el["strokeStyle"] = "dashed"
        self._add(el)
        if node_id is not None:
            self.box_elem[node_id] = el
            self.boxes[node_id] = (x, y, w, h)
        return el

    def add_label(self, container_el, text, align="center", valign="middle",
                  font=FONT, color=INK, z=3, cx=None, cy=None):
        tw, th = text_dims(text, font)
        bx, by = container_el["x"], container_el["y"]
        bw, bh = container_el["width"], container_el["height"]
        if cx is None:
            if align == "left":
                tx = bx + PAD_X
            else:
                tx = bx + (bw - tw) / 2
        else:
            tx = cx - tw / 2
        if cy is None:
            if valign == "top":
                ty = by + PAD_Y
            else:
                ty = by + (bh - th) / 2
        else:
            ty = cy - th / 2
        el = self._base("text", tx, ty, tw, th, z)
        el["id"] = self._new_id("tx")
        el["strokeColor"] = color
        el.update({
            "fontSize": font, "fontFamily": FONT_FAMILY,
            "text": str(text), "originalText": str(text),
            "textAlign": align, "verticalAlign": valign,
            "containerId": container_el["id"], "lineHeight": LINE_HEIGHT,
            "autoResize": True,
        })
        container_el["boundElements"].append({"type": "text", "id": el["id"]})
        self._add(el)
        return el

    def add_free_text(self, text, cx, top, font=FONT, color=INK, z=4):
        tw, th = text_dims(text, font)
        el = self._base("text", cx - tw / 2, top, tw, th, z)
        el["id"] = self._new_id("tx")
        el["strokeColor"] = color
        el.update({
            "fontSize": font, "fontFamily": FONT_FAMILY,
            "text": str(text), "originalText": str(text),
            "textAlign": "center", "verticalAlign": "top",
            "containerId": None, "lineHeight": LINE_HEIGHT, "autoResize": True,
        })
        self._add(el)
        return el

    def _linear(self, typ, pts, style="solid", arrowhead=True,
                stroke=INK, src_id=None, dst_id=None, z=1):
        x0, y0 = pts[0]
        rel = [[round(px - x0, 2), round(py - y0, 2)] for px, py in pts]
        xs = [p[0] for p in rel]
        ys = [p[1] for p in rel]
        el = self._base(typ, x0, y0, max(xs) - min(xs), max(ys) - min(ys), z)
        el["id"] = self._new_id("ln" if typ == "line" else "ar")
        el["strokeColor"] = stroke
        el["strokeStyle"] = STROKE_STYLE.get(style, "solid")
        el["roundness"] = {"type": 2}
        el["points"] = rel
        el["lastCommittedPoint"] = None
        el["startArrowhead"] = None
        el["endArrowhead"] = "arrow" if (arrowhead and typ == "arrow") else None
        el["startBinding"] = None
        el["endBinding"] = None
        if src_id is not None and src_id in self.box_elem:
            el["startBinding"] = {"elementId": self.box_elem[src_id]["id"], "focus": 0, "gap": 4}
            self.box_elem[src_id]["boundElements"].append({"type": "arrow", "id": el["id"]})
        if dst_id is not None and dst_id in self.box_elem:
            el["endBinding"] = {"elementId": self.box_elem[dst_id]["id"], "focus": 0, "gap": 4}
            self.box_elem[dst_id]["boundElements"].append({"type": "arrow", "id": el["id"]})
        self._add(el)
        return el

    def add_arrow(self, pts, label=None, style="solid", src_id=None, dst_id=None,
                  arrowhead=True, stroke=INK):
        el = self._linear("arrow", pts, style, arrowhead, stroke, src_id, dst_id)
        if label:
            mid = pts[len(pts) // 2]
            self.add_label(el, label, cx=mid[0], cy=mid[1], font=16)
        return el

    def add_line(self, pts, style="solid", stroke=INK):
        return self._linear("line", pts, style, arrowhead=False, stroke=stroke)

    def add_frame(self, x, y, w, h, label=None, stroke="#868e96"):
        el = self._base("rectangle", x, y, w, h, z=0)
        el["id"] = self._new_id("fr")
        el["strokeColor"] = stroke
        el["strokeStyle"] = "dashed"
        el["roundness"] = {"type": 3}
        self._add(el)
        if label:
            self.add_free_text(label, x + 70, y + 8, font=16, color=stroke, z=1)
        return el

    # ---- geometry --------------------------------------------------------------------

    @staticmethod
    def _clip(box, toward):
        """Point on box border in the direction of `toward` (center of another box)."""
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        dx, dy = toward[0] - cx, toward[1] - cy
        if dx == 0 and dy == 0:
            return (cx, cy)
        hw, hh = w / 2, h / 2
        scale = 1.0 / max(abs(dx) / hw if hw else 1e9, abs(dy) / hh if hh else 1e9)
        return (cx + dx * scale, cy + dy * scale)

    def _center(self, node_id):
        x, y, w, h = self.boxes[node_id]
        return (x + w / 2, y + h / 2)

    # ----------------------------------------------------------------------------------
    # Graphviz-backed layout (graph family)
    # ----------------------------------------------------------------------------------

    def build_graph_family(self):
        OFFX, OFFY = 80, 120
        rankdir = self.direction if self.direction in ("LR", "RL", "TB", "BT") else "TB"

        # precompute node render sizes
        sizes = {}
        labels = {}
        for n in self.nodes:
            if self.diagram == "er_diagram":
                label = self._er_label(n)
            else:
                label = n["label"]
            labels[n["id"]] = label
            sizes[n["id"]] = node_size(label) if self.diagram != "er_diagram" \
                else self._er_size(label)

        # build dot source
        src = ["digraph G {",
               f'  graph [rankdir={rankdir}, nodesep=0.5, ranksep=0.9, splines=true];',
               '  node [shape=box, fixedsize=true];']
        group_members = {g["id"]: [] for g in self.groups}
        for n in self.nodes:
            if n.get("group") in group_members:
                group_members[n["group"]].append(n["id"])
        grouped = set()
        for g in self.groups:
            members = group_members.get(g["id"], [])
            if not members:
                continue
            src.append(f'  subgraph cluster_{g["id"]} {{')
            src.append(f'    label="{self._esc(g["label"])}"; style=dashed; color="#868e96";')
            for nid in members:
                w, h = sizes[nid]
                src.append(f'    "{nid}" [width={w/72:.3f}, height={h/72:.3f}, label=""];')
                grouped.add(nid)
            src.append("  }")
        for n in self.nodes:
            if n["id"] in grouped:
                continue
            w, h = sizes[n["id"]]
            src.append(f'  "{n["id"]}" [width={w/72:.3f}, height={h/72:.3f}, label=""];')
        for e in self.edges:
            src.append(f'  "{e["from"]}" -> "{e["to"]}";')
        src.append("}")
        dot_src = "\n".join(src)

        data = self._run_dot(dot_src)
        bb = [float(v) for v in data["bb"].split(",")]
        H = bb[3]

        def flip(x, y):
            return (x + OFFX, (H - y) + OFFY)

        # nodes + clusters
        gvid_name = {}
        for o in data.get("objects", []):
            name = o.get("name", "")
            gvid_name[o.get("_gvid")] = name
            if name.startswith("cluster_") and "bb" in o:
                x0, y0, x1, y1 = [float(v) for v in o["bb"].split(",")]
                fx0, fy1 = flip(x0, y0)
                fx1, fy0 = flip(x1, y1)
                glabel = o.get("label") or ""
                self.add_frame(min(fx0, fx1), min(fy0, fy1),
                               abs(fx1 - fx0), abs(fy1 - fy0), label=glabel)
            elif name in self.node_by_id and "pos" in o:
                px, py = [float(v) for v in o["pos"].split(",")]
                cx, cy = flip(px, py)
                w, h = sizes[name]
                node = self.node_by_id[name]
                box = self.add_box(name, cx - w / 2, cy - h / 2, w, h,
                                   node["role"], node.get("color"))
                if self.diagram == "er_diagram":
                    self.add_label(box, labels[name], align="left", valign="top")
                else:
                    self.add_label(box, labels[name])

        # edges (match output edges back to DSL edges for label/style)
        pending = {}
        for e in self.edges:
            pending.setdefault((e["from"], e["to"]), []).append(e)
        for oe in data.get("edges", []):
            frm = gvid_name.get(oe.get("tail"))
            to = gvid_name.get(oe.get("head"))
            attrs = {}
            q = pending.get((frm, to))
            if q:
                attrs = q.pop(0)
            pts = self._spline(oe.get("pos", ""), flip)
            if len(pts) < 2:
                pts = [self._center(frm), self._center(to)]
            label = attrs.get("label")
            if self.diagram == "er_diagram" and attrs.get("meta", {}).get("cardinality"):
                label = attrs["meta"]["cardinality"] + (f"  {label}" if label else "")
            self.add_arrow(pts, label=label, style=attrs.get("style", "solid"),
                           src_id=frm, dst_id=to)

    @staticmethod
    def _spline(pos, flip):
        start = end = None
        mids = []
        for tok in pos.split():
            if tok.startswith("e,"):
                x, y = tok[2:].split(",")
                end = flip(float(x), float(y))
            elif tok.startswith("s,"):
                x, y = tok[2:].split(",")
                start = flip(float(x), float(y))
            elif "," in tok:
                x, y = tok.split(",")
                mids.append(flip(float(x), float(y)))
        poly = []
        if start:
            poly.append(start)
        poly.extend(mids)
        if end:
            poly.append(end)
        return poly

    def _er_label(self, node):
        lines = [node["label"]]
        for f in node.get("meta", {}).get("fields", []):
            key = f.get("key", "none")
            tag = {"pk": "🔑 ", "fk": "↗ "}.get(key, "")
            t = f.get("type", "")
            lines.append(f"{tag}{f['name']}" + (f": {t}" if t else ""))
        return "\n".join(lines)

    @staticmethod
    def _er_size(label):
        tw, th = text_dims(label, FONT)
        return min(max(tw + 2 * PAD_X, 160), 360), th + 2 * PAD_Y

    @staticmethod
    def _esc(s):
        return str(s).replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _run_dot(src):
        try:
            p = subprocess.run(["dot", "-Tjson"], input=src.encode(),
                               capture_output=True, timeout=30)
        except FileNotFoundError:
            raise SystemExit("ERROR: graphviz `dot` not found. Install graphviz.")
        if p.returncode != 0:
            raise SystemExit("ERROR: dot failed:\n" + p.stderr.decode())
        return json.loads(p.stdout.decode())

    # ----------------------------------------------------------------------------------
    # Hand-rolled layouts
    # ----------------------------------------------------------------------------------

    def build_sequence(self):
        MX, TOP = 100, 100
        gap = max((node_size(n["label"])[0] for n in self.nodes), default=160) + 80
        part_x = {}
        box_h = 50
        for i, n in enumerate(self.nodes):
            w, _ = node_size(n["label"])
            cx = MX + i * gap
            part_x[n["id"]] = cx
            box = self.add_box(n["id"], cx - w / 2, TOP, w, box_h, n["role"], n.get("color"))
            self.add_label(box, n["label"])
        n_msgs = max(len(self.edges), 1)
        life_bottom = TOP + box_h + 60 + n_msgs * 55 + 40
        for nid, cx in part_x.items():
            self.add_line([(cx, TOP + box_h), (cx, life_bottom)], style="dashed", stroke="#adb5bd")
        y = TOP + box_h + 60
        for e in self.edges:
            x1, x2 = part_x[e["from"]], part_x[e["to"]]
            kind = e.get("meta", {}).get("kind", "sync")
            style = "dashed" if kind in ("async", "return") else "solid"
            self.add_arrow([(x1, y), (x2, y)], label=e.get("label"), style=style)
            y += 55

    def build_timeline(self):
        nodes = sorted(self.nodes, key=lambda n: str(n.get("meta", {}).get("date", "")))
        MX, AXIS = 120, 320
        gap = max((node_size(n["label"])[0] for n in nodes), default=160) + 60
        xs = [MX + i * gap for i in range(len(nodes))]
        if xs:
            self.add_arrow([(MX - 40, AXIS), (xs[-1] + 60, AXIS)], stroke="#495057")
        for i, n in enumerate(nodes):
            cx = xs[i]
            marker = self.add_box(n["id"], cx - 9, AXIS - 9, 18, 18, "milestone", n.get("color"))
            above = (i % 2 == 0)
            w, h = node_size(n["label"])
            by = AXIS - 70 - h if above else AXIS + 70
            box = self.add_box(None, cx - w / 2, by, w, h, "generic", n.get("color"))
            self.add_label(box, n["label"])
            date = n.get("meta", {}).get("date", "")
            if date:
                self.add_free_text(str(date), cx, AXIS + (-95 if not above else 75), font=14,
                                   color="#868e96")
            connector_y = (by + h, AXIS) if above else (by, AXIS)
            self.add_line([(cx, connector_y[0]), (cx, connector_y[1])], stroke="#adb5bd")

    def build_mind_map(self):
        roots = [n for n in self.nodes if n["role"] == "root"] or self.nodes[:1]
        root = roots[0]
        adj = {n["id"]: [] for n in self.nodes}
        for e in self.edges:
            adj[e["from"]].append(e["to"])
            adj[e["to"]].append(e["from"])
        CX, CY = 700, 450
        placed = {root["id"]: (CX, CY)}
        order = [(root["id"], 0.0, 2 * math.pi, 0)]
        visited = {root["id"]}
        edges_to_draw = []
        while order:
            nid, a0, a1, depth = order.pop(0)
            children = [c for c in adj[nid] if c not in visited]
            if not children:
                continue
            span = (a1 - a0) / len(children)
            r = (depth + 1) * 230
            for i, c in enumerate(children):
                visited.add(c)
                ang = a0 + span * (i + 0.5)
                cx, cy = CX + r * math.cos(ang), CY + r * math.sin(ang)
                placed[c] = (cx, cy)
                edges_to_draw.append((nid, c))
                order.append((c, a0 + span * i, a0 + span * (i + 1), depth + 1))
        for n in self.nodes:
            cx, cy = placed.get(n["id"], (CX, CY))
            w, h = node_size(n["label"])
            role = "root" if n["id"] == root["id"] else n["role"]
            box = self.add_box(n["id"], cx - w / 2, cy - h / 2, w, h, role, n.get("color"))
            self.add_label(box, n["label"])
        for frm, to in edges_to_draw:
            p1 = self._clip(self.boxes[frm], self._center(to))
            p2 = self._clip(self.boxes[to], self._center(frm))
            self.add_arrow([p1, p2], arrowhead=False, src_id=frm, dst_id=to, stroke="#868e96")

    def build_wireframe(self):
        screens = self.groups or [{"id": "__all__", "label": self.dsl.get("title", "Screen")}]
        members = {g["id"]: [] for g in screens}
        for n in self.nodes:
            g = n.get("group", "__all__")
            members.setdefault(g, []).append(n)
        FW, FH, MX, TOP, GAP = 300, 600, 100, 120, 60
        kind_h = {"header": 56, "nav": 48, "tab": 44, "footer": 56, "button": 50,
                  "input": 50, "text": 40, "image": 140, "list": 200, "card": 120}
        for si, sc in enumerate(screens):
            fx = MX + si * (FW + GAP)
            self.add_frame(fx, TOP, FW, FH, label=sc["label"], stroke="#343a40")
            y = TOP + 20
            for n in members.get(sc["id"], []):
                kind = n.get("meta", {}).get("kind", "text")
                h = kind_h.get(kind, 50)
                box = self.add_box(n["id"], fx + 16, y, FW - 32, h, "ui_element", n.get("color"))
                self.add_label(box, n["label"])
                y += h + 14

    # ----------------------------------------------------------------------------------
    # Finalize
    # ----------------------------------------------------------------------------------

    def _content_bbox(self):
        xs0, ys0, xs1, ys1 = [], [], [], []
        for el in self.elements:
            xs0.append(el["x"]); ys0.append(el["y"])
            xs1.append(el["x"] + el["width"]); ys1.append(el["y"] + el["height"])
        if not xs0:
            return (0, 0, 200, 200)
        return (min(xs0), min(ys0), max(xs1), max(ys1))

    def build(self):
        if self.diagram in GRAPH_FAMILY:
            self.build_graph_family()
        elif self.diagram == "sequence_diagram":
            self.build_sequence()
        elif self.diagram == "timeline":
            self.build_timeline()
        elif self.diagram == "mind_map":
            self.build_mind_map()
        elif self.diagram == "mobile_wireframe":
            self.build_wireframe()
        else:
            raise SystemExit(f"Unsupported diagram type: {self.diagram}")

        x0, y0, x1, _ = self._content_bbox()
        self.add_free_text(self.title, (x0 + x1) / 2, y0 - 60, font=TITLE_FONT)

        self._assign_z_and_index()
        self._self_validate()
        return {
            "type": "excalidraw",
            "version": 2,
            "source": "convert_dsl_to_excalidraw.py",
            "elements": [self._strip(e) for e in self.elements],
            "appState": {"viewBackgroundColor": WHITE, "gridSize": None},
            "files": {},
        }

    def _assign_z_and_index(self):
        # stable sort by z layer, then assign fractional index in that order
        order = sorted(range(len(self.elements)), key=lambda i: (self.elements[i]["_z"], i))
        self.elements = [self.elements[i] for i in order]
        for i, el in enumerate(self.elements):
            el["index"] = f"a{i:07d}"

    @staticmethod
    def _strip(el):
        el = dict(el)
        el.pop("_z", None)
        return el

    def _self_validate(self):
        ids = {el["id"] for el in self.elements}
        warns = []
        for el in self.elements:
            for be in el.get("boundElements", []):
                if be["id"] not in ids:
                    warns.append(f"dangling boundElement {be['id']} on {el['id']}")
            if el.get("containerId") and el["containerId"] not in ids:
                warns.append(f"dangling containerId {el['containerId']} on {el['id']}")
            for b in ("startBinding", "endBinding"):
                if el.get(b) and el[b]["elementId"] not in ids:
                    warns.append(f"dangling {b} on {el['id']}")
        # overlap warning between node boxes
        items = list(self.boxes.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i][1], items[j][1]
                ox = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
                oy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
                if ox > 5 and oy > 5:
                    warns.append(f"overlap: {items[i][0]} & {items[j][0]}")
        if warns:
            sys.stderr.write("[self-validate] " + str(len(warns)) + " warning(s):\n")
            for w in warns[:20]:
                sys.stderr.write("  - " + w + "\n")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert Diagram DSL JSON to .excalidraw")
    ap.add_argument("input", help="DSL json file, or '-' for stdin")
    ap.add_argument("-o", "--output", help="output .excalidraw (default: stdout)")
    ap.add_argument("--seed", type=int, default=None, help="override deterministic seed")
    args = ap.parse_args()

    raw = sys.stdin.read() if args.input == "-" else open(args.input, encoding="utf-8").read()
    dsl = json.loads(raw)

    out = Converter(dsl, seed=args.seed).build()
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        sys.stderr.write(f"wrote {args.output} ({len(out['elements'])} elements)\n")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
