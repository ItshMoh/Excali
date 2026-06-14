#!/usr/bin/env python3
"""
generate_dataset.py

Phase 2, step 3 — the loop that turns specs into a validated training dataset.

    spec  ->  LLM (OpenRouter / DeepSeek / HF)  ->  user_request + DSL
          ->  validate_dsl  (accept / repair-retry)
          ->  dedup  ->  route to train.jsonl or validation.jsonl

Design choices that matter:

  * Provider-agnostic, stdlib only. OpenRouter / DeepSeek-direct / HuggingFace are all
    OpenAI-compatible chat endpoints; we hit them with urllib (no `openai`/`requests`).
    Default provider is OpenRouter.

  * The model's job is deliberately narrow. The *spec* already fixes the structure (nodes,
    roles, edges), so the model only (a) writes a natural user request and (b) polishes
    labels. It must not add/remove nodes. This is what makes a small distilled dataset
    reliable — structural validity is owned by `generate_specs`, not the model.

  * Every assistant DSL is run through the Phase 1 gate (`validate_dsl`). On failure we send
    the validator's errors back for one repair pass; still-bad rows are skipped (or, with
    --on-fail fallback, replaced by the deterministic `spec_to_dsl` skeleton).

  * Contamination control: entire patterns/domains are held out to validation.jsonl — never
    random rows — so the eval set shares no structure with training.

  * Task coverage (DSL_SPEC / project.md): prompt->DSL (main, model-generated) plus
    deterministic edit-DSL and repair-DSL rows (structurally exact, no API needed).

  * --provider none runs the whole pipeline offline using spec_to_dsl, so you can smoke-test
    plumbing and even produce a no-API dataset before wiring a key.

Usage:
    # offline smoke test (no key needed)
    python3 generate_dataset.py --count 50 --provider none --self-check

    # real run via OpenRouter (needs OPENROUTER_API_KEY in env or .env)
    python3 generate_dataset.py --count 2000 --provider openrouter \\
        --model-flash deepseek/deepseek-chat --model-pro deepseek/deepseek-r1

    # from a pre-generated spec file
    python3 generate_specs.py --count 2000 --out specs.jsonl
    python3 generate_dataset.py --specs specs.jsonl --provider openrouter
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from validate_dsl import validate_dsl
from generate_specs import (
    PATTERNS, DOMAINS, PROMPT_STYLES,
    build_spec, spec_to_dsl, generate as generate_specs,
)


# --------------------------------------------------------------------------------------
# Fixed system prompts stored in each training row (the model learns to obey these).
# --------------------------------------------------------------------------------------

SYSTEM_DSL = ("You generate compact diagram DSL (JSON) for Excalidraw. "
              "Return only valid DSL JSON, no explanation.")
SYSTEM_EDIT = ("You edit Excalidraw diagram DSL (JSON). Given an existing DSL and an edit "
               "instruction, return the full updated DSL JSON. Return only JSON.")
SYSTEM_REPAIR = ("You repair broken Excalidraw diagram DSL (JSON). Given an invalid DSL, "
                 "return a corrected, valid DSL JSON. Return only JSON.")

# Compact schema reminder handed to the generator model.
SCHEMA_SUMMARY = """DSL shape (JSON):
{
  "diagram": "system_architecture|flowchart|er_diagram|sequence_diagram|data_pipeline|timeline|mind_map|mobile_wireframe",
  "title": "<short>",
  "direction": "LR|RL|TB|BT|radial (optional)",
  "nodes": [ {"id":"snake_case","label":"...","role":"<role enum>","group":"<gid?>","meta":{...?}} ],
  "edges": [ {"from":"id","to":"id","label":"?","style":"solid|dashed|dotted?","meta":{...?}} ],
  "groups": [ {"id":"snake_case","label":"..."} ]
}
Rules: snake_case ids, no duplicate ids, every edge endpoint must be a node id, no positions/
sizes/colors/seeds (the converter computes those), labels human-readable and single-line."""


# --------------------------------------------------------------------------------------
# Provider configuration (all OpenAI-compatible /chat/completions)
# --------------------------------------------------------------------------------------

PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "flash": "deepseek/deepseek-v4-flash",
        "pro": "deepseek/deepseek-v4-pro",
        "extra_headers": {"HTTP-Referer": "https://localhost/excali-ft",
                          "X-Title": "excali-ft dataset gen"},
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "key_env": "DEEPSEEK_API_KEY",
        "flash": "deepseek-chat",
        "pro": "deepseek-reasoner",
        "extra_headers": {},
    },
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        "key_env": "HF_TOKEN",
        "flash": "deepseek-ai/DeepSeek-V3",
        "pro": "deepseek-ai/DeepSeek-R1",
        "extra_headers": {},
    },
}


def load_env(path):
    """Parse a .env file into a dict. Tolerates blank/comment/malformed (no '=') lines."""
    env = {}
    if not path or not os.path.exists(path):
        return env
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


# --------------------------------------------------------------------------------------
# OpenAI-compatible client (stdlib urllib)
# --------------------------------------------------------------------------------------

class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, provider, api_key, temperature=0.7, max_tokens=1500,
                 timeout=90, max_retries=4, json_mode=True):
        cfg = PROVIDERS[provider]
        self.base_url = cfg["base_url"].rstrip("/")
        self.extra_headers = cfg["extra_headers"]
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.json_mode = json_mode
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        self._usage_lock = threading.Lock()  # complete() runs on worker threads

    def complete(self, model, system, user):
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        headers.update(self.extra_headers)
        url = f"{self.base_url}/chat/completions"

        backoff = 2.0
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                u = body.get("usage", {})
                with self._usage_lock:
                    self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                    self.usage["completion_tokens"] += u.get("completion_tokens", 0)
                    self.usage["calls"] += 1
                return content
            except urllib.error.HTTPError as e:
                code = e.code
                detail = e.read().decode("utf-8", "replace")[:300]
                if code in (401, 403):
                    raise LLMError(f"auth failed ({code}); check your API key. {detail}")
                if code in (400, 404, 422):
                    raise LLMError(f"request rejected ({code}); check model id. {detail}")
                if code == 429 or 500 <= code < 600:
                    if attempt == self.max_retries - 1:
                        raise LLMError(f"giving up after {code}: {detail}")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise LLMError(f"HTTP {code}: {detail}")
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.max_retries - 1:
                    raise LLMError(f"network error: {e}")
                time.sleep(backoff)
                backoff *= 2
        raise LLMError("exhausted retries")


def extract_json_object(text):
    """Pull the first complete JSON object out of a model response (tolerates code fences
    and trailing prose). Returns a dict or raises ValueError."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s.strip("`")
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # balanced-brace scan, string-aware
    start = s.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(s[start:i + 1])
    raise ValueError("unbalanced JSON object")


# --------------------------------------------------------------------------------------
# Generators: spec -> (user_request, dsl)
# --------------------------------------------------------------------------------------

def _template_user_request(spec):
    """Plain, deterministic user request — the offline fallback / no-API path."""
    labels = [c["label"] for c in spec["components"]]
    phrase = spec["diagram_type"].replace("_", " ")
    shown = ", ".join(labels[:8]) + (", ..." if len(labels) > 8 else "")
    return (f"{spec['prompt_style']} {phrase} diagram for a {spec['domain']} "
            f"with: {shown}.").strip()


def _gen_prompt(spec):
    """The instruction we send to the generator model."""
    target = {"diagram": spec["diagram_type"],
              "title": f"<short title for a {spec['domain']} diagram>",
              "direction": spec["direction"],
              "nodes": spec["components"], "edges": spec["edges"], "groups": spec["groups"]}
    return f"""Produce ONE training example for a model that outputs Excalidraw diagram DSL.

You are given a diagram spec. Keep its structure EXACTLY: use the same node ids, roles, and
edges; do not add or remove nodes or edges. You MAY refine labels to suit the domain.

Return ONLY a JSON object with two keys:
  "user_request": a natural, specific request a person would type (use the phrasing style
                  "{spec['prompt_style']}"; mention the domain "{spec['domain']}"; do NOT
                  mention ids, roles, or the word DSL).
  "dsl": the diagram DSL object built from the spec — it MUST include a short "title".

{SCHEMA_SUMMARY}

Spec:
{json.dumps(target, indent=2)}
"""


class ModelGenerator:
    def __init__(self, client, model_flash, model_pro, pro_for, retries, on_fail,
                 log=None):
        self.client = client
        self.model_flash = model_flash
        self.model_pro = model_pro
        self.pro_for = pro_for          # "high" -> use pro on high-complexity specs
        self.retries = retries
        self.on_fail = on_fail          # "skip" | "fallback"
        self.log = log or (lambda m: None)

    def _model_for(self, spec):
        if self.pro_for == "high" and spec.get("complexity") == "high":
            return self.model_pro
        return self.model_flash

    def generate(self, spec):
        model = self._model_for(spec)
        tag = f"{spec['diagram_type']}/{spec['pattern']}"
        system, user = "You output only JSON.", _gen_prompt(spec)
        last_errors = None
        for attempt in range(self.retries + 1):
            if attempt == 0:
                content = self.client.complete(model, system, user)
            else:
                repair = (f"{user}\n\nYour previous answer was invalid. Validator errors:\n"
                          + "\n".join(f"- [{c}] {m}" for c, m in last_errors)
                          + "\n\nReturn a corrected JSON object with the same two keys.")
                content = self.client.complete(model, system, repair)
            try:
                obj = extract_json_object(content)
                user_request = str(obj["user_request"]).strip()
                dsl = obj["dsl"]
            except (ValueError, KeyError, TypeError):
                last_errors = [("parse", "response was not the expected JSON object")]
                dsl = None
            else:
                res = validate_dsl(dsl)
                if res.ok and user_request:
                    return user_request, dsl, "model"
                last_errors = res.errors or [("empty", "missing user_request")]

            codes = ",".join(sorted({c for c, _ in last_errors}))
            if attempt < self.retries:
                self.log(f"    repair {attempt + 1}/{self.retries} {tag}: invalid [{codes}]")
            else:
                self.log(f"    gave up {tag} after {self.retries + 1} attempts: invalid [{codes}]")

        if self.on_fail == "fallback":
            return _template_user_request(spec), spec_to_dsl(spec), "fallback"
        return None


class OfflineGenerator:
    """No-API path: deterministic DSL skeleton + templated request. Always valid."""
    def generate(self, spec):
        return _template_user_request(spec), spec_to_dsl(spec), "offline"


# --------------------------------------------------------------------------------------
# Deterministic edit / repair example builders (no API, structurally exact)
# --------------------------------------------------------------------------------------

def _close_deps(pattern, chosen):
    chosen = set(chosen)
    changed = True
    while changed:
        changed = False
        for sid in list(chosen):
            for dep in pattern["slots"][sid].get("requires", []):
                if dep not in chosen:
                    chosen.add(dep)
                    changed = True
    return chosen


def _materialize(pattern_name, domain, selected, direction):
    """Build a DSL directly from a pattern + explicit slot selection (mirrors spec_to_dsl)."""
    pattern = PATTERNS[pattern_name]
    comps = []
    for sid, slot in pattern["slots"].items():
        if sid not in selected:
            continue
        c = {"id": sid, "label": domain["labels"].get(sid, slot["label"]), "role": slot["role"]}
        if "group" in slot:
            c["group"] = slot["group"]
        if "meta" in slot:
            c["meta"] = copy.deepcopy(slot["meta"])
        comps.append(c)
    edges = []
    for e in pattern["edges"]:
        if e["from"] in selected and e["to"] in selected:
            edges.append(copy.deepcopy(e))
    used_groups = {c["group"] for c in comps if "group" in c}
    groups = [{"id": gid, "label": g["label"]}
              for gid, g in pattern["groups"].items() if gid in used_groups]
    title = f"{domain['name'].title()} {pattern_name.replace('_', ' ').title()}"[:80].strip()
    dsl = {"diagram": pattern["diagram"], "title": title, "direction": direction, "nodes": comps}
    if edges:
        dsl["edges"] = edges
    if groups:
        dsl["groups"] = groups
    return dsl


def make_edit_example(rng):
    """before-DSL + 'add X' instruction -> after-DSL. Returns a row dict or None."""
    candidates = [n for n, p in PATTERNS.items()
                  if any(s.get("optional") for s in p["slots"].values())]
    rng.shuffle(candidates)
    for pname in candidates:
        pattern = PATTERNS[pname]
        domain = rng.choice(DOMAINS)
        direction = pattern["direction"]
        required = {sid for sid, s in pattern["slots"].items() if not s.get("optional")}
        optional = [sid for sid, s in pattern["slots"].items() if s.get("optional")]
        # pick one optional to be "added"; base = everything else closeable minus it
        add = rng.choice(optional)
        if pattern["slots"][add].get("requires"):
            # adding a dependent node only makes sense once its deps exist; keep deps in base
            pass
        base_sel = _close_deps(pattern, required | {o for o in optional
                                                    if o != add and rng.random() < 0.4})
        if add in base_sel:
            continue
        aug_sel = _close_deps(pattern, base_sel | {add})
        if aug_sel == base_sel:
            continue
        before = _materialize(pname, domain, base_sel, direction)
        after = _materialize(pname, domain, aug_sel, direction)
        if not (validate_dsl(before).ok and validate_dsl(after).ok):
            continue
        added = pattern["slots"][add]
        label = domain["labels"].get(add, added["label"])
        # phrase a connection hint from the new node's incident edges
        hint = ""
        for e in pattern["edges"]:
            if e["to"] == add and e["from"] in base_sel:
                src = next(c["label"] for c in before["nodes"] if c["id"] == e["from"])
                hint = f" Connect it from the {src}."
                break
            if e["from"] == add and e["to"] in base_sel:
                dst = next(c["label"] for c in before["nodes"] if c["id"] == e["to"])
                hint = f" Connect it to the {dst}."
                break
        instruction = f"Add a {label} ({added['role']}) to the diagram.{hint}"
        user = json.dumps(before) + "\n\nEdit: " + instruction
        return {"messages": [
            {"role": "system", "content": SYSTEM_EDIT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(after)},
        ]}, pname, domain["name"], after
    return None


_CORRUPTIONS = ("dup_id", "dangling_edge", "drop_role", "bad_enum", "self_loop", "drop_title")


def make_repair_example(rng, valid_dsl):
    """valid DSL -> deliberately broken DSL + 'fix it' -> the valid DSL. Row dict or None."""
    for kind in rng.sample(_CORRUPTIONS, len(_CORRUPTIONS)):
        broken = copy.deepcopy(valid_dsl)
        nodes = broken["nodes"]
        if kind == "dup_id" and len(nodes) >= 2:
            nodes[1]["id"] = nodes[0]["id"]
        elif kind == "dangling_edge":
            broken.setdefault("edges", []).append({"from": nodes[0]["id"], "to": "ghost_node"})
        elif kind == "drop_role":
            nodes[0].pop("role", None)
        elif kind == "bad_enum":
            nodes[0]["role"] = "widget"
        elif kind == "self_loop":
            broken.setdefault("edges", []).append({"from": nodes[0]["id"], "to": nodes[0]["id"]})
        elif kind == "drop_title":
            broken.pop("title", None)
        else:
            continue
        if validate_dsl(broken).ok:
            continue  # corruption didn't actually break it; try another
        user = json.dumps(broken) + "\n\nThe above DSL is invalid. Return the corrected DSL JSON."
        return {"messages": [
            {"role": "system", "content": SYSTEM_REPAIR},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(valid_dsl)},
        ]}
    return None


# --------------------------------------------------------------------------------------
# Dedup + routing
# --------------------------------------------------------------------------------------

class Dedup:
    def __init__(self):
        self.requests = set()
        self.dsl_hashes = set()

    def is_dup(self, user_request, dsl):
        h = hashlib.md5(json.dumps(dsl, sort_keys=True).encode()).hexdigest()
        key = user_request.strip().lower()
        if key in self.requests or h in self.dsl_hashes:
            return True
        self.requests.add(key)
        self.dsl_hashes.add(h)
        return False


def _is_holdout(spec_pattern, spec_domain, holdout_patterns, holdout_domains):
    return spec_pattern in holdout_patterns or spec_domain in holdout_domains


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------

def _make_row(user_request, dsl):
    return {"messages": [
        {"role": "system", "content": SYSTEM_DSL},
        {"role": "user", "content": user_request},
        {"role": "assistant", "content": json.dumps(dsl)},
    ]}


def run(args, log=lambda m: print(m, file=sys.stderr)):
    rng = random.Random(args.seed)

    # 1. specs
    if args.specs:
        specs = [json.loads(l) for l in open(args.specs, encoding="utf-8") if l.strip()]
    else:
        specs = generate_specs(args.count, args.seed, log=log)
    if args.limit:
        specs = specs[:args.limit]

    # 2. generator
    if args.provider == "none":
        gen = OfflineGenerator()
        log("provider=none -> offline deterministic generation (no API calls)")
    else:
        cfg = PROVIDERS[args.provider]
        env = load_env(args.env_file)
        api_key = os.environ.get(cfg["key_env"]) or env.get(cfg["key_env"])
        if not api_key:
            raise SystemExit(f"missing {cfg['key_env']} (set it in env or {args.env_file}); "
                             f"or use --provider none for an offline run")
        client = LLMClient(args.provider, api_key, temperature=args.temperature,
                           max_tokens=args.max_tokens)
        gen = ModelGenerator(client,
                             args.model_flash or cfg["flash"],
                             args.model_pro or cfg["pro"],
                             args.pro_for, args.retries, args.on_fail, log=log)
        log(f"provider={args.provider} flash={args.model_flash or cfg['flash']} "
            f"pro={args.model_pro or cfg['pro']}")

    holdout_patterns = set(p for p in (args.holdout_patterns or "").split(",") if p)
    holdout_domains = set(d for d in (args.holdout_domains or "").split(",") if d)

    dedup = Dedup()
    train = open(args.train_out, "w", encoding="utf-8")
    val = open(args.val_out, "w", encoding="utf-8")
    stats = {"train": 0, "val": 0, "dup": 0, "failed": 0, "fallback": 0,
             "edit": 0, "repair": 0}
    valid_dsls = []  # pool for repair examples

    def write(row, to_val):
        f = val if to_val else train
        f.write(json.dumps(row) + "\n")
        f.flush()  # land each row on disk immediately so Ctrl-C never loses progress
        stats["val" if to_val else "train"] += 1

    total_specs = len(specs)
    workers = 1 if args.provider == "none" else max(1, args.concurrency)
    log(f"generating prompt->DSL rows for {total_specs} specs "
        f"(concurrency={workers})...")

    def _tok():
        if args.provider == "none":
            return ""
        u = gen.client.usage
        return f" | {u['calls']} calls, {u['prompt_tokens'] + u['completion_tokens']} tok"

    def _safe_generate(spec):
        """Worker body — network call only. Never touches shared state; returns a result
        tuple so the main thread does all dedup/write/stats lock-free."""
        try:
            return ("ok", gen.generate(spec))
        except LLMError as e:
            return ("error", str(e))

    def _consume(spec, kind, payload, n):
        tag = f"{spec['diagram_type']}/{spec['pattern']}"
        if kind == "error":
            log(f"  [{n}/{total_specs}] {tag} -> LLM ERROR: {payload}")
            stats["failed"] += 1
            return
        out = payload
        if out is None:
            log(f"  [{n}/{total_specs}] {tag} -> FAILED (invalid after retries){_tok()}")
            stats["failed"] += 1
            return
        user_request, dsl, source = out
        if source == "fallback":
            stats["fallback"] += 1
        if dedup.is_dup(user_request, dsl):
            stats["dup"] += 1
            log(f"  [{n}/{total_specs}] {tag} -> dup, skipped{_tok()}")
            return
        to_val = _is_holdout(spec["pattern"], spec["domain"],
                             holdout_patterns, holdout_domains)
        write(_make_row(user_request, dsl), to_val)
        valid_dsls.append((dsl, spec["pattern"], spec["domain"]))
        log(f"  [{n}/{total_specs}] {tag} -> {'val' if to_val else 'train'} "
            f"({source}){_tok()}")

    try:
        # 3. main prompt->DSL rows — fan out the API calls, consume on the main thread.
        if workers == 1:
            for n, spec in enumerate(specs, 1):
                kind, payload = _safe_generate(spec)
                _consume(spec, kind, payload, n)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_safe_generate, spec): spec for spec in specs}
                for n, fut in enumerate(as_completed(futures), 1):
                    spec = futures[fut]
                    kind, payload = fut.result()
                    _consume(spec, kind, payload, n)

        # 4. edit rows (deterministic). These dedup on their own input (the before-DSL +
        #    instruction), NOT the shared prompt->DSL hash pool — an edit target may
        #    legitimately equal a DSL that already appears as a prompt->DSL row.
        aux_seen = set()

        def _aux_dup(row):
            h = hashlib.md5(row["messages"][1]["content"].encode()).hexdigest()
            if h in aux_seen:
                return True
            aux_seen.add(h)
            return False

        n_edit = int(len(specs) * args.edit_frac)
        for _ in range(max(n_edit * 6, 30)):
            if stats["edit"] >= n_edit:
                break
            made = make_edit_example(rng)
            if not made:
                continue
            row, pname, dname, after = made
            if _aux_dup(row):
                continue
            write(row, _is_holdout(pname, dname, holdout_patterns, holdout_domains))
            stats["edit"] += 1

        # 5. repair rows (deterministic, from already-valid DSLs)
        n_repair = int(len(specs) * args.repair_frac)
        rng.shuffle(valid_dsls)
        for dsl, pname, dname in valid_dsls:
            if stats["repair"] >= n_repair:
                break
            row = make_repair_example(rng, dsl)
            if not row or _aux_dup(row):
                continue
            write(row, _is_holdout(pname, dname, holdout_patterns, holdout_domains))
            stats["repair"] += 1
    finally:
        train.close()
        val.close()

    # 6. summary
    total = stats["train"] + stats["val"]
    log("-" * 60)
    log(f"wrote {total} rows  (train={stats['train']}  validation={stats['val']})")
    log(f"  prompt->DSL: {total - stats['edit'] - stats['repair']}  "
        f"edit: {stats['edit']}  repair: {stats['repair']}")
    log(f"  skipped: duplicates={stats['dup']}  failed={stats['failed']}  "
        f"fallback_used={stats['fallback']}")
    if args.provider != "none":
        u = gen.client.usage
        log(f"  api: calls={u['calls']}  prompt_tokens={u['prompt_tokens']}  "
            f"completion_tokens={u['completion_tokens']}")
    log(f"  train -> {args.train_out}   validation -> {args.val_out}")

    if args.self_check:
        bad = 0
        for path in (args.train_out, args.val_out):
            for line in open(path, encoding="utf-8"):
                row = json.loads(line)
                content = row["messages"][-1]["content"]
                if not validate_dsl(json.loads(content)).ok:
                    bad += 1
        log(f"  self-check: {total - bad}/{total} assistant DSLs valid"
            + ("" if bad == 0 else f"  ({bad} INVALID)"))

    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate the prompt->DSL training dataset (Phase 2).")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--specs", help="JSONL of specs from generate_specs.py")
    src.add_argument("--count", type=int, default=500, help="generate this many specs on the fly")
    ap.add_argument("--limit", type=int, default=0, help="cap number of specs processed")

    ap.add_argument("--provider", choices=list(PROVIDERS) + ["none"], default="openrouter")
    ap.add_argument("--model-flash", default=None, help="override bulk model id")
    ap.add_argument("--model-pro", default=None, help="override complex/repair model id")
    ap.add_argument("--pro-for", choices=["high", "none"], default="high",
                    help="use the pro model on high-complexity specs (default) or never")
    ap.add_argument("--concurrency", type=int, default=5,
                    help="parallel API calls for prompt->DSL rows (forced to 1 for --provider none)")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--retries", type=int, default=2, help="repair retries per spec on invalid DSL")
    ap.add_argument("--on-fail", choices=["skip", "fallback"], default="skip",
                    help="when a spec never yields valid DSL: skip it, or use the offline skeleton")

    ap.add_argument("--edit-frac", type=float, default=0.1)
    ap.add_argument("--repair-frac", type=float, default=0.1)
    ap.add_argument("--holdout-domains", default="log analytics platform,ticket booking system",
                    help="comma-separated domains reserved entirely for validation.jsonl")
    ap.add_argument("--holdout-patterns", default="three_tier,blog_core",
                    help="comma-separated patterns reserved entirely for validation.jsonl")

    ap.add_argument("--train-out", default="train.jsonl")
    ap.add_argument("--val-out", default="validation.jsonl")
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-check", action="store_true",
                    help="re-validate every written assistant DSL after the run")
    args = ap.parse_args(argv)

    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
