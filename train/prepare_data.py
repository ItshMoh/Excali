"""
Local pre-flight sanity gate for the fine-tune dataset. Run this BEFORE spending GPU on
`modal_train.py`. It is read-only — it never modifies the JSONL.

Checks
------
1. Structure: every row is chat-format (system, user, assistant) with one fixed system prompt.
2. DSL validity: re-run every assistant payload through validate_dsl.py (errors block; warnings
   are reported). This is the same accept/reject gate used during generation, applied as a guard.
3. Token lengths: histogram + max, so you can confirm `--max-seq-len` (default 2048) won't
   truncate any example. Uses the real Qwen tokenizer if `transformers` is importable; otherwise
   falls back to a chars/4 estimate (clearly labeled).
4. Leakage: no user prompt or title shared between train.jsonl and validation.jsonl (the split is
   supposed to hold out entire patterns/domains — this catches accidental contamination).
5. Balance: diagram-type distribution per split.

Exit code is non-zero if any BLOCKING issue is found (bad structure, invalid DSL, leakage, or
truncation at the chosen max-seq-len), so it can gate a script/CI before training.

Usage
-----
  python3 train/prepare_data.py
  python3 train/prepare_data.py --train train.jsonl --val validation.jsonl --max-seq-len 2048
"""

import argparse
import collections
import json
import os
import sys

# Import the existing validator (repo root is the parent of this file's dir).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validate_dsl import load_schema, validate_dsl_text  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
EXPECTED_ROLES = ["system", "user", "assistant"]


# --------------------------------------------------------------------------- loading
def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append((i, json.loads(line)))
            except json.JSONDecodeError as e:
                rows.append((i, {"__bad_json__": str(e)}))
    return rows


# --------------------------------------------------------------------------- tokenizer
def get_tokenizer():
    """Return (encode_fn, label). Real Qwen tokenizer if available, else an estimate."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)

        def encode(messages):
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return len(tok(text, add_special_tokens=False)["input_ids"]), text

        return encode, "real (Qwen tokenizer)"
    except Exception as e:  # noqa: BLE001 — any failure -> estimate, don't crash
        print(f"  (transformers unavailable: {type(e).__name__}; using chars/4 estimate)")

        def encode(messages):
            text = "\n".join(m.get("content", "") for m in messages)
            return (len(text) + 3) // 4, text

        return encode, "estimate (chars/4)"


# --------------------------------------------------------------------------- checks
def check_structure(rows, expect_fixed_system):
    """Returns (errors, system_prompts_seen)."""
    errors = []
    systems = collections.Counter()
    for ln, row in rows:
        if "__bad_json__" in row:
            errors.append(f"line {ln}: invalid JSON ({row['__bad_json__']})")
            continue
        msgs = row.get("messages")
        if not isinstance(msgs, list):
            errors.append(f"line {ln}: missing 'messages' list")
            continue
        roles = [m.get("role") for m in msgs]
        if roles != EXPECTED_ROLES:
            errors.append(f"line {ln}: roles {roles} != {EXPECTED_ROLES}")
            continue
        if any(not (m.get("content") or "").strip() for m in msgs):
            errors.append(f"line {ln}: an empty message content")
        systems[msgs[0]["content"]] += 1
    return errors, systems


def assistant_dsl(row):
    for m in reversed(row.get("messages", [])):
        if m.get("role") == "assistant":
            return m.get("content")
    return None


def check_dsl(rows, schema):
    """Returns (error_lines, warn_count)."""
    errors, warns = [], 0
    for ln, row in rows:
        if "__bad_json__" in row:
            continue
        dsl = assistant_dsl(row)
        if dsl is None:
            errors.append(f"line {ln}: no assistant content")
            continue
        res = validate_dsl_text(dsl, schema)
        warns += len(res.warnings)
        if not res.ok:
            codes = ", ".join(c for c, _ in res.errors)
            errors.append(f"line {ln}: DSL invalid [{codes}]")
    return errors, warns


def token_stats(rows, encode, max_seq_len):
    lengths = []
    over = []
    for ln, row in rows:
        if "__bad_json__" in row or not isinstance(row.get("messages"), list):
            continue
        n, _ = encode(row["messages"])
        lengths.append(n)
        if n > max_seq_len:
            over.append((ln, n))
    return lengths, over


def histogram(lengths, buckets=(128, 256, 512, 768, 1024, 1536, 2048)):
    counts = collections.OrderedDict((b, 0) for b in buckets)
    counts["over"] = 0
    for n in lengths:
        placed = False
        for b in buckets:
            if n <= b:
                counts[b] += 1
                placed = True
                break
        if not placed:
            counts["over"] += 1
    return counts


def diagram_balance(rows):
    c = collections.Counter()
    for _, row in rows:
        dsl = assistant_dsl(row) if isinstance(row.get("messages"), list) else None
        if not dsl:
            continue
        try:
            c[json.loads(dsl).get("diagram", "?")] += 1
        except json.JSONDecodeError:
            c["?"] += 1
    return c


def leakage(train_rows, val_rows):
    """Returns (prompt_overlap, title_overlap) as sorted lists."""
    def field(rows, which):
        out = set()
        for _, row in rows:
            if not isinstance(row.get("messages"), list):
                continue
            if which == "prompt":
                for m in row["messages"]:
                    if m.get("role") == "user":
                        out.add((m.get("content") or "").strip())
            else:  # title
                dsl = assistant_dsl(row)
                if dsl:
                    try:
                        out.add(json.loads(dsl).get("title", "").strip())
                    except json.JSONDecodeError:
                        pass
        out.discard("")
        return out

    p = field(train_rows, "prompt") & field(val_rows, "prompt")
    t = field(train_rows, "title") & field(val_rows, "title")
    return sorted(p), sorted(t)


# --------------------------------------------------------------------------- report
def section(title):
    print(f"\n{'=' * 4} {title} {'=' * (60 - len(title))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--val", default="validation.jsonl")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    args = ap.parse_args()

    schema = load_schema()
    encode, tok_label = get_tokenizer()

    train_rows = load_rows(args.train)
    val_rows = load_rows(args.val)
    print(f"loaded train={len(train_rows)}  validation={len(val_rows)}")
    print(f"token counting: {tok_label}")

    blocking = 0

    for name, rows in (("TRAIN", train_rows), ("VALIDATION", val_rows)):
        section(f"{name}: structure")
        errs, systems = check_structure(rows, expect_fixed_system=True)
        if errs:
            blocking += len(errs)
            for e in errs[:20]:
                print("  ERR", e)
            if len(errs) > 20:
                print(f"  ... +{len(errs) - 20} more")
        else:
            print("  OK — all rows are [system, user, assistant] with non-empty content")
        if len(systems) > 1:
            print(f"  NOTE: {len(systems)} distinct system prompts (expected 1):")
            for s, c in systems.most_common():
                print(f"    {c:>4}x  {s[:70]!r}")

        section(f"{name}: DSL validity")
        derrs, warns = check_dsl(rows, schema)
        if derrs:
            blocking += len(derrs)
            for e in derrs[:20]:
                print("  ERR", e)
            if len(derrs) > 20:
                print(f"  ... +{len(derrs) - 20} more")
        else:
            print(f"  OK — every assistant DSL passes validate_dsl ({warns} warnings total)")

        section(f"{name}: token lengths ({tok_label})")
        lengths, over = token_stats(rows, encode, args.max_seq_len)
        if lengths:
            lengths_sorted = sorted(lengths)
            p50 = lengths_sorted[len(lengths_sorted) // 2]
            p95 = lengths_sorted[int(len(lengths_sorted) * 0.95)]
            print(f"  min={min(lengths)}  p50={p50}  p95={p95}  max={max(lengths)}")
            hist = histogram(lengths)
            for b, c in hist.items():
                bar = "#" * (c * 40 // max(1, len(lengths)))
                label = f"<={b}" if b != "over" else ">2048"
                print(f"    {label:>7}: {c:>4}  {bar}")
        if over:
            blocking += len(over)
            print(f"  ERR {len(over)} rows exceed max-seq-len={args.max_seq_len} "
                  f"(would be TRUNCATED): e.g. {over[:5]}")
        else:
            print(f"  OK — no row exceeds max-seq-len={args.max_seq_len}")

        section(f"{name}: diagram balance")
        for d, c in diagram_balance(rows).most_common():
            print(f"    {c:>4}  {d}")

    section("LEAKAGE: train vs validation")
    p_over, t_over = leakage(train_rows, val_rows)
    if p_over:
        blocking += len(p_over)
        print(f"  ERR {len(p_over)} user prompts appear in BOTH splits:")
        for s in p_over[:10]:
            print(f"    {s[:80]!r}")
    else:
        print("  OK — no shared user prompts")
    if t_over:
        # Titles can legitimately repeat across domains; report as a warning, not blocking.
        print(f"  WARN {len(t_over)} diagram titles appear in both splits (often benign):")
        for s in t_over[:10]:
            print(f"    {s[:80]!r}")
    else:
        print("  OK — no shared titles")

    section("RESULT")
    if blocking:
        print(f"  FAIL — {blocking} blocking issue(s). Fix before training.")
        sys.exit(1)
    print("  PASS — dataset is ready for modal_train.py")
    sys.exit(0)


if __name__ == "__main__":
    main()
