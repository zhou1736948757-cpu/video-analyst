#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-written stdlib JSON-schema (draft-07 subset) validator + schema registry.

Reuses the approach proven in ZHOU work/p6-01 validate_schemas.py (stdlib only,
no pydantic, no jsonschema package). Supports type/enum/pattern/minimum/maximum/
exclusiveMinimum/required/properties/additionalProperties/minItems/items/$ref.
"""

import json
import os
import re

from .common import SCHEMA_DIR, load_json

# ------------------------------ core validator (draft-07 subset) -------------


def _type_ok(val, t):
    if t == "string":
        return isinstance(val, str)
    if t == "number":
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if t == "integer":
        return isinstance(val, int) and not isinstance(val, bool)
    if t == "boolean":
        return isinstance(val, bool)
    if t == "object":
        return isinstance(val, dict)
    if t == "array":
        return isinstance(val, list)
    if t == "null":
        return val is None
    return True


def validate(instance, schema, root=None, path="$"):
    """Return list of error strings (empty == valid). Supports local #/definitions refs."""
    root = root if root is not None else schema
    errs = []
    if isinstance(schema, bool):
        return [] if schema else [f"{path}: schema=false"]
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/definitions/"):
            return [f"{path}: unsupported $ref {ref}"]
        target = root
        for seg in ref.split("/")[1:]:
            target = target.get(seg) if isinstance(target, dict) else None
        return validate(instance, target, root, path)
    if "type" in schema:
        ts = schema["type"]
        ts = [ts] if isinstance(ts, str) else ts
        if not any(_type_ok(instance, t) for t in ts):
            errs.append(f"{path}: type mismatch (want {ts}, got {type(instance).__name__})")
    if "enum" in schema and instance not in schema["enum"]:
        errs.append(f"{path}: {instance!r} not in enum {schema['enum']}")
    if "pattern" in schema and isinstance(instance, str):
        if re.search(schema["pattern"], instance) is None:
            errs.append(f"{path}: {instance!r} !~ {schema['pattern']}")
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errs.append(f"{path}: string shorter than minLength {schema['minLength']}")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errs.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errs.append(f"{path}: {instance} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errs.append(f"{path}: {instance} <= exclusiveMinimum {schema['exclusiveMinimum']}")
    if isinstance(instance, dict):
        props = schema.get("properties", {})
        if "required" in schema:
            for k in schema["required"]:
                if k not in instance:
                    errs.append(f"{path}: missing required {k!r}")
        for k, v in instance.items():
            if k in props:
                errs += validate(v, props[k], root, f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected property {k!r}")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errs.append(f"{path}: len {len(instance)} < minItems {schema['minItems']}")
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(instance):
                errs += validate(item, schema["items"], root, f"{path}[{i}]")
    return errs


# ------------------------------ registry -------------------------------------

SCHEMAS = {
    "coarse": {
        "coarse-batch-analysis.schema.json": "analysis/coarse/batches/*.json",
        "video-structure.schema.json": "analysis/coarse/video_structure.json",
        "refinement-plan.schema.json": "analysis/coarse/refinement_plan.json",
    },
    "fine": {
        "fine-batch-analysis.schema.json": "analysis/fine/batches/*.json",
        "temporal-events.schema.json": "analysis/fine/temporal_events.json",
        "fine-video-analysis.schema.json": "analysis/fine/fine_video_analysis.json",
        "refinement-plan.schema.json": "analysis/fine/second_refinement_plan.json",
    },
    "terminal": {
        "fine-batch-analysis.schema.json": "analysis/terminal_fine/*.json",
    },
}

_CACHE = {}


def load_schema(filename):
    if filename not in _CACHE:
        path = os.path.join(SCHEMA_DIR, filename)
        schema = load_json(path)
        if schema is None:
            raise RuntimeError(f"schema file missing/invalid: {path}")
        _CACHE[filename] = schema
    return _CACHE[filename]


def validate_dict(instance, filename):
    """Validates a python dict against a registry schema file. Returns (ok, errors)."""
    schema = load_schema(filename)
    errs = validate(instance, schema, schema)
    return (not errs), errs


def validate_file(path, filename):
    """Validates a JSON file on disk against a schema file. Returns (ok, errors)."""
    data = load_json(path)
    if data is None:
        return False, [f"{path}: not readable JSON"]
    ok, errs = validate_dict(data, filename)
    prefixed = [f"{os.path.relpath(path)}: {e}" for e in errs]
    return ok, prefixed


def all_schema_names():
    return sorted({name for stage in SCHEMAS.values() for name in stage})