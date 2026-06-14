#!/usr/bin/env python3
"""
generate_specs.py

Phase 2, step 2 — the controlled, diverse spec generator.

A *spec* is a structured description of one diagram: which pattern, which domain, which
components and edges, what direction/complexity/prompt-style. It is the input that
`generate_dataset.py` hands to DeepSeek, which then writes a natural user prompt + the DSL.

The whole point of this file is DIVERSITY WITHOUT NONSENSE. Diversity is *sampled*, never
hand-written, from four independent axes:

    pattern  ×  domain  ×  optional-slot subset  ×  (direction, prompt phrasing)

…and nonsense is prevented two ways:
  1. Components/edges come from curated per-type *patterns* (a real web app, a real ETL DAG),
     never blind permutation. Edges are included iff both endpoints are selected, so a spec
     can never carry a dangling reference.
  2. Every candidate spec is turned into a deterministic DSL skeleton (`spec_to_dsl`) and run
     through the Phase 1 gate (`validate_dsl`). Anything that doesn't yield valid DSL is
     discarded. The generator is therefore self-correcting — a bad pattern shows up as a
     drop, not as poison in the dataset.

Nothing is ever silently dropped: duplicates, per-combo caps, and validation failures are all
counted and printed in the run summary.

Usage:
    python3 generate_specs.py --count 500 --out specs.jsonl --seed 7
    python3 generate_specs.py --count 2000 --out specs.jsonl --types flowchart,er_diagram
    python3 generate_specs.py --count 200 --out specs.jsonl --emit-dsl dsl_skeletons.jsonl
    python3 generate_specs.py --count 50  --self-check        # extra-loud validation

`spec_to_dsl(spec)` is also a usable no-API fallback: it produces a valid (if plainly-labeled)
DSL directly from a spec, so the pipeline can run end-to-end before DeepSeek is wired in.
"""

import argparse
import json
import math
import random
import sys

from validate_dsl import validate_dsl, Result


# --------------------------------------------------------------------------------------
# Domains — thematic flavor. labels are keyed by SLOT name (the common architecture slots);
# anything not listed falls back to the pattern's own default label. `subject` flavors titles.
# --------------------------------------------------------------------------------------

DOMAINS = [
    {"name": "food delivery app", "subject": "orders",
     "labels": {"user": "Customer", "frontend": "Customer App", "api": "Order API",
                "database": "Postgres DB", "cache": "Redis Cache", "queue": "Order Queue",
                "worker": "Delivery Worker", "auth": "Auth Service", "gateway": "API Gateway"}},
    {"name": "ride hailing service", "subject": "trips",
     "labels": {"user": "Rider", "frontend": "Mobile App", "api": "Trip API",
                "database": "Trips DB", "cache": "Geo Cache", "queue": "Dispatch Queue",
                "worker": "Matching Worker", "auth": "Auth Service", "gateway": "API Gateway"}},
    {"name": "online learning platform", "subject": "courses",
     "labels": {"user": "Student", "frontend": "Web App", "api": "Course API",
                "database": "Postgres DB", "cache": "Redis Cache", "queue": "Job Queue",
                "worker": "Certificate Worker", "auth": "Auth Service", "gateway": "API Gateway"}},
    {"name": "video streaming service", "subject": "videos",
     "labels": {"user": "Viewer", "frontend": "Web Player", "api": "Streaming API",
                "database": "Catalog DB", "cache": "CDN Cache", "queue": "Encode Queue",
                "worker": "Transcoder", "auth": "Auth Service", "gateway": "Edge Gateway"}},
    {"name": "banking app", "subject": "accounts",
     "labels": {"user": "Account Holder", "frontend": "Banking App", "api": "Banking API",
                "database": "Ledger DB", "cache": "Balance Cache", "queue": "Payments Queue",
                "worker": "Settlement Worker", "auth": "Identity Service", "gateway": "API Gateway"}},
    {"name": "IoT monitoring platform", "subject": "device readings",
     "labels": {"user": "Operator", "frontend": "Dashboard", "api": "Ingest API",
                "database": "Timeseries DB", "cache": "Hot Cache", "queue": "Telemetry Queue",
                "worker": "Aggregator", "auth": "Auth Service", "gateway": "Device Gateway"}},
    {"name": "hospital appointment system", "subject": "appointments",
     "labels": {"user": "Patient", "frontend": "Patient Portal", "api": "Scheduling API",
                "database": "Records DB", "cache": "Slot Cache", "queue": "Reminder Queue",
                "worker": "Reminder Worker", "auth": "Auth Service", "gateway": "API Gateway"}},
    {"name": "URL shortener", "subject": "links",
     "labels": {"user": "User", "frontend": "Web UI", "api": "Shortener API",
                "database": "Links DB", "cache": "Redirect Cache", "queue": "Analytics Queue",
                "worker": "Stats Worker", "auth": "Auth Service", "gateway": "API Gateway"}},
    {"name": "inventory management system", "subject": "stock",
     "labels": {"user": "Manager", "frontend": "Admin Console", "api": "Inventory API",
                "database": "Inventory DB", "cache": "Stock Cache", "queue": "Restock Queue",
                "worker": "Restock Worker", "auth": "Auth Service", "gateway": "API Gateway"}},
    {"name": "log analytics platform", "subject": "logs",
     "labels": {"user": "Engineer", "frontend": "Console", "api": "Query API",
                "database": "Index DB", "cache": "Query Cache", "queue": "Ingest Queue",
                "worker": "Indexer", "auth": "Auth Service", "gateway": "API Gateway"}},
    {"name": "social media app", "subject": "posts",
     "labels": {"user": "Member", "frontend": "Mobile App", "api": "Feed API",
                "database": "Posts DB", "cache": "Feed Cache", "queue": "Fanout Queue",
                "worker": "Fanout Worker", "auth": "Auth Service", "gateway": "API Gateway"}},
    {"name": "ticket booking system", "subject": "bookings",
     "labels": {"user": "Customer", "frontend": "Booking App", "api": "Booking API",
                "database": "Seats DB", "cache": "Seat Cache", "queue": "Booking Queue",
                "worker": "Confirmation Worker", "auth": "Auth Service", "gateway": "API Gateway"}},
]


PROMPT_STYLES = [
    "Create a", "Draw a", "I need an Excalidraw diagram for", "Design a simple",
    "Show the architecture of", "Map out the flow for", "Sketch a", "Generate a diagram of",
]


# --------------------------------------------------------------------------------------
# Patterns. Each pattern declares slots (some optional, some with `requires`), edges (kept
# iff both endpoints are selected), and optional group membership. Edge meta carries
# per-type data (er cardinality, sequence kind). Node meta carries er fields / timeline date
# / wireframe kind.
#
# slot:  id -> {role, label, optional?, requires?[ids], group?, meta?}
# edge:  {from, to, label?, style?, meta?}
# group: id -> {label}
# --------------------------------------------------------------------------------------

def _f(name, type=None, key=None):
    d = {"name": name}
    if type:
        d["type"] = type
    if key:
        d["key"] = key
    return d


PATTERNS = {
    # ---------------- system_architecture ----------------
    "basic_web_app": {
        "diagram": "system_architecture", "direction": "LR",
        "slots": {
            "user": {"role": "actor", "label": "User"},
            "frontend": {"role": "client", "label": "Frontend"},
            "api": {"role": "service", "label": "API Server"},
            "database": {"role": "database", "label": "Database", "group": "data_layer"},
            "cache": {"role": "cache", "label": "Cache", "optional": True, "group": "data_layer"},
            "queue": {"role": "queue", "label": "Job Queue", "optional": True},
            "worker": {"role": "worker", "label": "Worker", "optional": True, "requires": ["queue"]},
            "auth": {"role": "auth", "label": "Auth Service", "optional": True},
            "monitoring": {"role": "monitoring", "label": "Monitoring", "optional": True},
        },
        "edges": [
            {"from": "user", "to": "frontend", "label": "uses"},
            {"from": "frontend", "to": "api", "label": "HTTPS"},
            {"from": "api", "to": "database", "label": "read/write"},
            {"from": "api", "to": "cache", "label": "cache lookup", "style": "dashed"},
            {"from": "api", "to": "queue", "label": "enqueue job"},
            {"from": "queue", "to": "worker", "label": "consume job"},
            {"from": "api", "to": "auth", "label": "verify token"},
            {"from": "api", "to": "monitoring", "label": "metrics", "style": "dashed"},
        ],
        "groups": {"data_layer": {"label": "Data Layer"}},
    },
    "microservices": {
        "diagram": "system_architecture", "direction": "LR",
        "slots": {
            "user": {"role": "actor", "label": "User"},
            "gateway": {"role": "gateway", "label": "API Gateway"},
            "svc_orders": {"role": "service", "label": "Order Service", "group": "services"},
            "svc_payments": {"role": "service", "label": "Payment Service", "group": "services"},
            "database": {"role": "database", "label": "Orders DB"},
            "cache": {"role": "cache", "label": "Cache", "optional": True},
            "queue": {"role": "queue", "label": "Event Queue", "optional": True},
            "worker": {"role": "worker", "label": "Worker", "optional": True, "requires": ["queue"]},
            "monitoring": {"role": "monitoring", "label": "Monitoring", "optional": True},
        },
        "edges": [
            {"from": "user", "to": "gateway", "label": "request"},
            {"from": "gateway", "to": "svc_orders", "label": "route"},
            {"from": "gateway", "to": "svc_payments", "label": "route"},
            {"from": "svc_orders", "to": "database", "label": "store"},
            {"from": "svc_orders", "to": "cache", "label": "cache", "style": "dashed"},
            {"from": "svc_orders", "to": "queue", "label": "publish"},
            {"from": "queue", "to": "worker", "label": "consume"},
            {"from": "gateway", "to": "monitoring", "label": "metrics", "style": "dashed"},
        ],
        "groups": {"services": {"label": "Services"}},
    },
    "three_tier": {
        "diagram": "system_architecture", "direction": "LR",
        "slots": {
            "user": {"role": "actor", "label": "User"},
            "frontend": {"role": "client", "label": "Web Tier"},
            "api": {"role": "service", "label": "App Tier"},
            "database": {"role": "database", "label": "DB Tier"},
            "cache": {"role": "cache", "label": "Cache", "optional": True},
            "monitoring": {"role": "monitoring", "label": "Monitoring", "optional": True},
        },
        "edges": [
            {"from": "user", "to": "frontend", "label": "uses"},
            {"from": "frontend", "to": "api", "label": "calls"},
            {"from": "api", "to": "database", "label": "read/write"},
            {"from": "api", "to": "cache", "label": "cache", "style": "dashed"},
            {"from": "api", "to": "monitoring", "label": "metrics", "style": "dashed"},
        ],
        "groups": {},
    },
    "event_driven": {
        "diagram": "system_architecture", "direction": "LR",
        "slots": {
            "producer": {"role": "service", "label": "Producer"},
            "broker": {"role": "queue", "label": "Event Broker"},
            "consumer_a": {"role": "worker", "label": "Consumer A"},
            "consumer_b": {"role": "worker", "label": "Consumer B", "optional": True},
            "database": {"role": "database", "label": "Database", "optional": True},
            "monitoring": {"role": "monitoring", "label": "Monitoring", "optional": True},
        },
        "edges": [
            {"from": "producer", "to": "broker", "label": "publish"},
            {"from": "broker", "to": "consumer_a", "label": "subscribe"},
            {"from": "broker", "to": "consumer_b", "label": "subscribe"},
            {"from": "consumer_a", "to": "database", "label": "write"},
            {"from": "consumer_a", "to": "monitoring", "label": "metrics", "style": "dashed"},
        ],
        "groups": {},
    },

    # ---------------- flowchart ----------------
    "approval_flow": {
        "diagram": "flowchart", "direction": "TB",
        "slots": {
            "start": {"role": "start", "label": "Start"},
            "submit": {"role": "process", "label": "Submit Request"},
            "review": {"role": "decision", "label": "Manager Approves?"},
            "approved": {"role": "process", "label": "Mark Approved"},
            "rejected": {"role": "process", "label": "Notify Rejection"},
            "notify": {"role": "process", "label": "Send Email", "optional": True},
            "done": {"role": "end", "label": "Done"},
        },
        "edges": [
            {"from": "start", "to": "submit"},
            {"from": "submit", "to": "review"},
            {"from": "review", "to": "approved", "label": "yes"},
            {"from": "review", "to": "rejected", "label": "no"},
            {"from": "approved", "to": "done"},
            {"from": "rejected", "to": "done"},
            {"from": "approved", "to": "notify"},
            {"from": "notify", "to": "done"},
        ],
        "groups": {},
    },
    "retry_loop": {
        "diagram": "flowchart", "direction": "TB",
        "slots": {
            "start": {"role": "start", "label": "Start"},
            "fetch": {"role": "process", "label": "Fetch Data"},
            "validate": {"role": "decision", "label": "Valid?"},
            "process": {"role": "process", "label": "Process"},
            "fix": {"role": "process", "label": "Fix Errors"},
            "log": {"role": "process", "label": "Log Error", "optional": True},
            "done": {"role": "end", "label": "Done"},
        },
        "edges": [
            {"from": "start", "to": "fetch"},
            {"from": "fetch", "to": "validate"},
            {"from": "validate", "to": "process", "label": "yes"},
            {"from": "validate", "to": "fix", "label": "no"},
            {"from": "fix", "to": "fetch"},
            {"from": "process", "to": "done"},
            {"from": "fix", "to": "log"},
            {"from": "log", "to": "fetch"},
        ],
        "groups": {},
    },
    "checkout_flow": {
        "diagram": "flowchart", "direction": "TB",
        "slots": {
            "start": {"role": "start", "label": "Start"},
            "cart": {"role": "process", "label": "View Cart"},
            "login": {"role": "decision", "label": "Logged in?"},
            "do_login": {"role": "process", "label": "Log In"},
            "payment": {"role": "process", "label": "Enter Payment"},
            "valid": {"role": "decision", "label": "Payment OK?"},
            "confirm": {"role": "process", "label": "Confirm Order"},
            "fail": {"role": "process", "label": "Show Error"},
            "done": {"role": "end", "label": "Order Placed"},
        },
        "edges": [
            {"from": "start", "to": "cart"},
            {"from": "cart", "to": "login"},
            {"from": "login", "to": "payment", "label": "yes"},
            {"from": "login", "to": "do_login", "label": "no"},
            {"from": "do_login", "to": "payment"},
            {"from": "payment", "to": "valid"},
            {"from": "valid", "to": "confirm", "label": "yes"},
            {"from": "valid", "to": "fail", "label": "no"},
            {"from": "confirm", "to": "done"},
            {"from": "fail", "to": "payment"},
        ],
        "groups": {},
    },

    # ---------------- data_pipeline ----------------
    "etl_batch": {
        "diagram": "data_pipeline", "direction": "LR",
        "slots": {
            "src": {"role": "source", "label": "Raw Events"},
            "extract": {"role": "transform", "label": "Extract"},
            "clean": {"role": "transform", "label": "Clean & Validate"},
            "load": {"role": "transform", "label": "Load"},
            "warehouse": {"role": "sink", "label": "Warehouse"},
            "dq": {"role": "transform", "label": "Quality Check", "optional": True},
            "archive": {"role": "sink", "label": "Cold Storage", "optional": True},
        },
        "edges": [
            {"from": "src", "to": "extract"},
            {"from": "extract", "to": "clean"},
            {"from": "clean", "to": "load"},
            {"from": "load", "to": "warehouse"},
            {"from": "clean", "to": "dq"},
            {"from": "dq", "to": "load"},
            {"from": "load", "to": "archive"},
        ],
        "groups": {},
    },
    "streaming_pipeline": {
        "diagram": "data_pipeline", "direction": "LR",
        "slots": {
            "src": {"role": "source", "label": "Event Stream"},
            "ingest": {"role": "transform", "label": "Ingest"},
            "enrich": {"role": "transform", "label": "Enrich"},
            "aggregate": {"role": "transform", "label": "Aggregate"},
            "dashboard": {"role": "sink", "label": "Dashboard"},
            "store": {"role": "sink", "label": "Data Lake", "optional": True},
            "alert": {"role": "sink", "label": "Alerting", "optional": True},
        },
        "edges": [
            {"from": "src", "to": "ingest"},
            {"from": "ingest", "to": "enrich"},
            {"from": "enrich", "to": "aggregate"},
            {"from": "aggregate", "to": "dashboard"},
            {"from": "enrich", "to": "store"},
            {"from": "aggregate", "to": "alert"},
        ],
        "groups": {},
    },

    # ---------------- er_diagram ----------------
    "ecommerce_core": {
        "diagram": "er_diagram", "direction": "LR",
        "slots": {
            "customer": {"role": "entity", "label": "Customer",
                         "meta": {"fields": [_f("id", "int", "pk"), _f("name", "text"), _f("email", "text")]}},
            "order": {"role": "entity", "label": "Order",
                      "meta": {"fields": [_f("id", "int", "pk"), _f("customer_id", "int", "fk"),
                                          _f("total", "decimal"), _f("created_at", "timestamp")]}},
            "product": {"role": "entity", "label": "Product",
                        "meta": {"fields": [_f("id", "int", "pk"), _f("name", "text"), _f("price", "decimal")]}},
            "order_item": {"role": "entity", "label": "Order Item",
                           "meta": {"fields": [_f("id", "int", "pk"), _f("order_id", "int", "fk"),
                                               _f("product_id", "int", "fk"), _f("qty", "int")]}},
            "category": {"role": "entity", "label": "Category", "optional": True,
                         "meta": {"fields": [_f("id", "int", "pk"), _f("name", "text")]}},
        },
        "edges": [
            {"from": "customer", "to": "order", "label": "places", "meta": {"cardinality": "1:N"}},
            {"from": "order", "to": "order_item", "label": "contains", "meta": {"cardinality": "1:N"}},
            {"from": "product", "to": "order_item", "label": "in", "meta": {"cardinality": "1:N"}},
            {"from": "category", "to": "product", "label": "groups", "meta": {"cardinality": "1:N"}},
        ],
        "groups": {},
    },
    "blog_core": {
        "diagram": "er_diagram", "direction": "LR",
        "slots": {
            "user": {"role": "entity", "label": "User",
                     "meta": {"fields": [_f("id", "int", "pk"), _f("username", "text"), _f("email", "text")]}},
            "post": {"role": "entity", "label": "Post",
                     "meta": {"fields": [_f("id", "int", "pk"), _f("user_id", "int", "fk"),
                                         _f("title", "text"), _f("body", "text")]}},
            "comment": {"role": "entity", "label": "Comment",
                        "meta": {"fields": [_f("id", "int", "pk"), _f("post_id", "int", "fk"),
                                            _f("user_id", "int", "fk"), _f("body", "text")]}},
            "tag": {"role": "entity", "label": "Tag", "optional": True,
                    "meta": {"fields": [_f("id", "int", "pk"), _f("name", "text")]}},
            "post_tag": {"role": "entity", "label": "Post Tag", "optional": True, "requires": ["tag"],
                         "meta": {"fields": [_f("post_id", "int", "fk"), _f("tag_id", "int", "fk")]}},
        },
        "edges": [
            {"from": "user", "to": "post", "label": "writes", "meta": {"cardinality": "1:N"}},
            {"from": "post", "to": "comment", "label": "has", "meta": {"cardinality": "1:N"}},
            {"from": "user", "to": "comment", "label": "writes", "meta": {"cardinality": "1:N"}},
            {"from": "post", "to": "post_tag", "label": "tagged", "meta": {"cardinality": "1:N"}},
            {"from": "tag", "to": "post_tag", "label": "labels", "meta": {"cardinality": "1:N"}},
        ],
        "groups": {},
    },

    # ---------------- sequence_diagram (edge order = message order) ----------------
    "auth_sequence": {
        "diagram": "sequence_diagram", "direction": "LR",
        "slots": {
            "user": {"role": "actor", "label": "User"},
            "app": {"role": "participant", "label": "App"},
            "auth": {"role": "participant", "label": "Auth Service"},
            "db": {"role": "participant", "label": "User DB"},
            "cache": {"role": "participant", "label": "Session Cache", "optional": True},
        },
        "edges": [
            {"from": "user", "to": "app", "label": "enter credentials", "meta": {"kind": "sync"}},
            {"from": "app", "to": "auth", "label": "POST /login", "meta": {"kind": "sync"}},
            {"from": "auth", "to": "db", "label": "find user", "meta": {"kind": "sync"}},
            {"from": "db", "to": "auth", "label": "user row", "meta": {"kind": "return"}},
            {"from": "auth", "to": "cache", "label": "store session", "meta": {"kind": "sync"}},
            {"from": "cache", "to": "auth", "label": "ok", "meta": {"kind": "return"}},
            {"from": "auth", "to": "app", "label": "JWT token", "meta": {"kind": "return"}},
            {"from": "app", "to": "user", "label": "logged in", "meta": {"kind": "return"}},
        ],
        "groups": {},
    },
    "payment_sequence": {
        "diagram": "sequence_diagram", "direction": "LR",
        "slots": {
            "user": {"role": "actor", "label": "User"},
            "app": {"role": "participant", "label": "Checkout"},
            "payment": {"role": "participant", "label": "Payment Gateway"},
            "bank": {"role": "participant", "label": "Bank"},
            "ledger": {"role": "participant", "label": "Ledger Service", "optional": True},
        },
        "edges": [
            {"from": "user", "to": "app", "label": "place order", "meta": {"kind": "sync"}},
            {"from": "app", "to": "payment", "label": "charge card", "meta": {"kind": "sync"}},
            {"from": "payment", "to": "bank", "label": "authorize", "meta": {"kind": "sync"}},
            {"from": "bank", "to": "payment", "label": "approved", "meta": {"kind": "return"}},
            {"from": "app", "to": "ledger", "label": "record txn", "meta": {"kind": "async"}},
            {"from": "payment", "to": "app", "label": "payment ok", "meta": {"kind": "return"}},
            {"from": "app", "to": "user", "label": "order confirmed", "meta": {"kind": "return"}},
        ],
        "groups": {},
    },

    # ---------------- timeline (each node has a date; no edges) ----------------
    "product_launch": {
        "diagram": "timeline", "direction": "LR",
        "slots": {
            "kickoff": {"role": "milestone", "label": "Kickoff", "meta": {"date": "2024-01"}},
            "design": {"role": "milestone", "label": "Design Complete", "meta": {"date": "2024-02"}},
            "beta": {"role": "milestone", "label": "Beta Release", "meta": {"date": "2024-04"}},
            "launch": {"role": "milestone", "label": "Public Launch", "meta": {"date": "2024-06"}},
            "retro": {"role": "milestone", "label": "Retrospective", "optional": True, "meta": {"date": "2024-07"}},
        },
        "edges": [],
        "groups": {},
    },
    "project_roadmap": {
        "diagram": "timeline", "direction": "LR",
        "slots": {
            "q1": {"role": "milestone", "label": "Planning", "meta": {"date": "Q1 2024"}},
            "q2": {"role": "milestone", "label": "Development", "meta": {"date": "Q2 2024"}},
            "q3": {"role": "milestone", "label": "Testing", "meta": {"date": "Q3 2024"}},
            "q4": {"role": "milestone", "label": "Release", "meta": {"date": "Q4 2024"}},
        },
        "edges": [],
        "groups": {},
    },

    # ---------------- mind_map (one root, tree) ----------------
    "topic_breakdown": {
        "diagram": "mind_map", "direction": "radial",
        "slots": {
            "root": {"role": "root", "label": "Topic"},
            "b1": {"role": "branch", "label": "Subtopic A"},
            "b2": {"role": "branch", "label": "Subtopic B"},
            "b3": {"role": "branch", "label": "Subtopic C", "optional": True},
            "a1": {"role": "leaf", "label": "Detail A1"},
            "a2": {"role": "leaf", "label": "Detail A2"},
            "b2a": {"role": "leaf", "label": "Detail B1", "optional": True},
        },
        "edges": [
            {"from": "root", "to": "b1"},
            {"from": "root", "to": "b2"},
            {"from": "root", "to": "b3"},
            {"from": "b1", "to": "a1"},
            {"from": "b1", "to": "a2"},
            {"from": "b2", "to": "b2a"},
        ],
        "groups": {},
    },

    # ---------------- mobile_wireframe (each node has kind + screen group) ----------------
    "login_signup": {
        "diagram": "mobile_wireframe", "direction": "TB",
        "slots": {
            "l_header": {"role": "ui_element", "label": "Welcome", "group": "login_screen", "meta": {"kind": "header"}},
            "l_email": {"role": "ui_element", "label": "Email", "group": "login_screen", "meta": {"kind": "input"}},
            "l_pass": {"role": "ui_element", "label": "Password", "group": "login_screen", "meta": {"kind": "input"}},
            "l_signin": {"role": "ui_element", "label": "Sign In", "group": "login_screen", "meta": {"kind": "button"}},
            "l_footer": {"role": "ui_element", "label": "Forgot password?", "group": "login_screen",
                         "optional": True, "meta": {"kind": "footer"}},
            "s_header": {"role": "ui_element", "label": "Create Account", "group": "signup_screen",
                         "optional": True, "meta": {"kind": "header"}},
            "s_email": {"role": "ui_element", "label": "Email", "group": "signup_screen",
                        "optional": True, "requires": ["s_header"], "meta": {"kind": "input"}},
            "s_submit": {"role": "ui_element", "label": "Sign Up", "group": "signup_screen",
                         "optional": True, "requires": ["s_header"], "meta": {"kind": "button"}},
        },
        "edges": [],
        "groups": {"login_screen": {"label": "Login Screen"}, "signup_screen": {"label": "Sign Up Screen"}},
    },
    "feed_app": {
        "diagram": "mobile_wireframe", "direction": "TB",
        "slots": {
            "f_header": {"role": "ui_element", "label": "Home", "group": "feed_screen", "meta": {"kind": "header"}},
            "f_search": {"role": "ui_element", "label": "Search", "group": "feed_screen", "meta": {"kind": "input"}},
            "f_list": {"role": "ui_element", "label": "Posts", "group": "feed_screen", "meta": {"kind": "list"}},
            "f_nav": {"role": "ui_element", "label": "Bottom Nav", "group": "feed_screen", "meta": {"kind": "nav"}},
            "p_header": {"role": "ui_element", "label": "Profile", "group": "profile_screen",
                         "optional": True, "meta": {"kind": "header"}},
            "p_card": {"role": "ui_element", "label": "User Info", "group": "profile_screen",
                       "optional": True, "requires": ["p_header"], "meta": {"kind": "card"}},
            "p_nav": {"role": "ui_element", "label": "Bottom Nav", "group": "profile_screen",
                      "optional": True, "requires": ["p_header"], "meta": {"kind": "nav"}},
        },
        "edges": [],
        "groups": {"feed_screen": {"label": "Feed Screen"}, "profile_screen": {"label": "Profile Screen"}},
    },
}

# Directions we're allowed to sample per diagram type (first is the pattern default-ish).
DIRECTION_CHOICES = {
    "system_architecture": ["LR", "TB"],
    "flowchart": ["TB", "LR"],
    "data_pipeline": ["LR", "TB"],
    "er_diagram": ["LR", "TB"],
    "sequence_diagram": ["LR"],
    "timeline": ["LR"],
    "mind_map": ["radial"],
    "mobile_wireframe": ["TB"],
}


# --------------------------------------------------------------------------------------
# Spec construction
# --------------------------------------------------------------------------------------

def _complexity_optional_count(rng, n_opt):
    """How many optional slots to try to add, by complexity tier."""
    if n_opt == 0:
        return "low", 0
    tier = rng.choice(["low", "medium", "high"])
    if tier == "low":
        hi = min(1, n_opt)
        return tier, rng.randint(0, hi)
    if tier == "medium":
        return tier, rng.randint(1, max(1, math.ceil(n_opt / 2)))
    return tier, rng.randint(max(1, math.ceil(n_opt / 2)), n_opt)


def _select_slots(rng, pattern):
    """Return the set of selected slot ids: all required + a sampled, dependency-closed
    subset of optionals."""
    slots = pattern["slots"]
    required = [sid for sid, s in slots.items() if not s.get("optional")]
    optional = [sid for sid, s in slots.items() if s.get("optional")]

    complexity, k = _complexity_optional_count(rng, len(optional))
    chosen = set(rng.sample(optional, k)) if k else set()

    # dependency closure: pulling in an optional pulls in what it `requires`.
    changed = True
    while changed:
        changed = False
        for sid in list(chosen):
            for dep in slots[sid].get("requires", []):
                if dep not in chosen and dep not in required:
                    chosen.add(dep)
                    changed = True

    selected = set(required) | chosen
    return selected, complexity


def _label_for(slot_id, slot, domain):
    return domain["labels"].get(slot_id, slot["label"])


def build_spec(rng, pattern_name):
    """Build one spec dict from a pattern + a random domain + sampled optionals/direction/style."""
    pattern = PATTERNS[pattern_name]
    domain = rng.choice(DOMAINS)
    selected, complexity = _select_slots(rng, pattern)

    components = []
    for sid, slot in pattern["slots"].items():
        if sid not in selected:
            continue
        comp = {"id": sid, "label": _label_for(sid, slot, domain), "role": slot["role"]}
        if "group" in slot:
            comp["group"] = slot["group"]
        if "meta" in slot:
            comp["meta"] = json.loads(json.dumps(slot["meta"]))  # deep copy
        components.append(comp)

    edges = []
    for e in pattern["edges"]:
        if e["from"] in selected and e["to"] in selected:
            edge = {"from": e["from"], "to": e["to"]}
            if "label" in e:
                edge["label"] = e["label"]
            if "style" in e:
                edge["style"] = e["style"]
            if "meta" in e:
                edge["meta"] = json.loads(json.dumps(e["meta"]))
            edges.append(edge)

    # groups: keep only those that have at least one selected member.
    used_groups = {c["group"] for c in components if "group" in c}
    groups = [{"id": gid, "label": g["label"]}
              for gid, g in pattern["groups"].items() if gid in used_groups]

    direction = rng.choice(DIRECTION_CHOICES[pattern["diagram"]])

    spec = {
        "task_type": "prompt_to_dsl",
        "diagram_type": pattern["diagram"],
        "pattern": pattern_name,
        "domain": domain["name"],
        "subject": domain["subject"],
        "direction": direction,
        "complexity": complexity,
        "prompt_style": rng.choice(PROMPT_STYLES),
        "components": components,
        "edges": edges,
        "groups": groups,
    }
    return spec


def _pretty(name):
    return name.replace("_", " ").title()


def spec_to_dsl(spec):
    """Deterministically turn a spec into a valid DSL skeleton.

    Used (a) to validate that a spec is coherent, and (b) as a no-API fallback that produces
    real (plainly-labeled) training DSL without DeepSeek. DeepSeek's job on top of this is to
    write the natural user prompt and add labeling polish — not to invent structure."""
    title = f"{spec['domain'].title()} {_pretty(spec['pattern'])}"
    title = title[:80].strip()

    dsl = {"diagram": spec["diagram_type"], "title": title}
    if spec.get("direction"):
        dsl["direction"] = spec["direction"]

    dsl["nodes"] = [json.loads(json.dumps(c)) for c in spec["components"]]
    if spec["edges"]:
        dsl["edges"] = [json.loads(json.dumps(e)) for e in spec["edges"]]
    if spec["groups"]:
        dsl["groups"] = [json.loads(json.dumps(g)) for g in spec["groups"]]
    return dsl


def validate_spec(spec):
    """Light pre-DeepSeek contract check, plus the authoritative check: the spec must yield a
    DSL that passes the Phase 1 gate. Returns a Result."""
    res = Result()
    if spec.get("diagram_type") not in {p["diagram"] for p in PATTERNS.values()}:
        res.error("spec", f"unknown diagram_type {spec.get('diagram_type')!r}")
    if not spec.get("components"):
        res.error("spec", "spec has no components")
    ids = [c["id"] for c in spec.get("components", [])]
    if len(ids) != len(set(ids)):
        res.error("spec", "duplicate component ids")
    idset = set(ids)
    for e in spec.get("edges", []):
        if e["from"] not in idset or e["to"] not in idset:
            res.error("spec", f"edge {e['from']}->{e['to']} references a missing component")
    if res.errors:
        return res
    # Authoritative: does it convert to valid DSL?
    res.extend(validate_dsl(spec_to_dsl(spec)))
    return res


# --------------------------------------------------------------------------------------
# Diversity / dedup
# --------------------------------------------------------------------------------------

def _signature(spec):
    """Exact-duplicate key: same structure + domain + direction."""
    comp = tuple(sorted((c["id"], c["role"]) for c in spec["components"]))
    edge = tuple(sorted((e["from"], e["to"], e.get("label", "")) for e in spec["edges"]))
    return (spec["diagram_type"], spec["pattern"], spec["domain"], spec["direction"], comp, edge)


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------

def generate(count, seed, types=None, max_per_combo=0, self_check=False, log=print):
    rng = random.Random(seed)

    pattern_names = [n for n, p in PATTERNS.items()
                     if types is None or p["diagram"] in types]
    if not pattern_names:
        raise SystemExit(f"no patterns match types={types}")

    n_combos = len(pattern_names) * len(DOMAINS)
    if max_per_combo <= 0:
        max_per_combo = max(3, math.ceil(count / n_combos) + 2)

    specs = []
    seen_sigs = set()
    combo_counts = {}
    capped_combos = set()
    skipped = {"duplicate": 0, "combo_cap": 0, "invalid": 0}
    invalid_codes = {}

    max_attempts = count * 60
    attempts = 0
    while len(specs) < count and attempts < max_attempts:
        attempts += 1
        spec = build_spec(rng, rng.choice(pattern_names))

        sig = _signature(spec)
        if sig in seen_sigs:
            skipped["duplicate"] += 1
            continue

        combo = (spec["pattern"], spec["domain"])
        if combo_counts.get(combo, 0) >= max_per_combo:
            skipped["combo_cap"] += 1
            capped_combos.add(combo)
            continue

        res = validate_spec(spec)
        if not res.ok:
            skipped["invalid"] += 1
            for code, _ in res.errors:
                invalid_codes[code] = invalid_codes.get(code, 0) + 1
            continue

        spec["id"] = f"{spec['pattern']}-{len(specs):06d}"
        seen_sigs.add(sig)
        combo_counts[combo] = combo_counts.get(combo, 0) + 1
        specs.append(spec)

    # ---- summary (no silent truncation) ----
    log("-" * 60)
    log(f"requested={count}  generated={len(specs)}  attempts={attempts}")
    log(f"skipped: duplicate={skipped['duplicate']}  combo_cap={skipped['combo_cap']}  "
        f"invalid={skipped['invalid']}")
    if invalid_codes:
        log("  invalid breakdown: " + "  ".join(f"{c}={n}" for c, n in invalid_codes.items()))
    if capped_combos:
        log(f"  {len(capped_combos)} (pattern,domain) combos hit the cap of {max_per_combo} "
            f"and were limited")
    by_type = {}
    for s in specs:
        by_type[s["diagram_type"]] = by_type.get(s["diagram_type"], 0) + 1
    log("  by diagram type: " + "  ".join(f"{t}={n}" for t, n in sorted(by_type.items())))
    if len(specs) < count:
        log(f"  NOTE: produced fewer than requested — raise --max-per-combo, add patterns/"
            f"domains, or lower --count. Capacity ~= {n_combos} combos x {max_per_combo} cap "
            f"= {n_combos * max_per_combo}.")

    if self_check:
        bad = 0
        for s in specs:
            if not validate_dsl(spec_to_dsl(s)).ok:
                bad += 1
        log(f"  self-check: {len(specs) - bad}/{len(specs)} specs convert to valid DSL"
            + ("" if bad == 0 else f"  ({bad} FAILED)"))

    return specs


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate diverse, validated diagram specs for Phase 2.")
    ap.add_argument("--count", type=int, default=500, help="number of specs to generate")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible runs")
    ap.add_argument("--out", default="specs.jsonl", help="output JSONL path ('-' for stdout)")
    ap.add_argument("--types", default=None,
                    help="comma-separated diagram types to restrict to (default: all 8)")
    ap.add_argument("--max-per-combo", type=int, default=0,
                    help="cap specs per (pattern,domain); 0 = auto from --count")
    ap.add_argument("--emit-dsl", default=None,
                    help="also write deterministic DSL skeletons here (no-API fallback dataset)")
    ap.add_argument("--self-check", action="store_true",
                    help="re-validate every emitted spec's DSL and report")
    args = ap.parse_args(argv)

    types = None
    if args.types:
        valid_types = {p["diagram"] for p in PATTERNS.values()}
        types = set(t.strip() for t in args.types.split(","))
        unknown = types - valid_types
        if unknown:
            ap.error(f"unknown diagram types: {sorted(unknown)}; valid: {sorted(valid_types)}")

    # Diagnostics go to stderr so stdout stays clean when --out is '-'.
    log = lambda m: print(m, file=sys.stderr)
    specs = generate(args.count, args.seed, types=types,
                     max_per_combo=args.max_per_combo, self_check=args.self_check, log=log)

    out = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")
    try:
        for s in specs:
            out.write(json.dumps(s) + "\n")
    finally:
        if out is not sys.stdout:
            out.close()
            print(f"wrote {len(specs)} specs -> {args.out}")

    if args.emit_dsl:
        with open(args.emit_dsl, "w", encoding="utf-8") as f:
            for s in specs:
                f.write(json.dumps(spec_to_dsl(s)) + "\n")
        print(f"wrote {len(specs)} DSL skeletons -> {args.emit_dsl}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
