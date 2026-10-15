"""Design files (.vfil): a FilterSpec, plus window settings, as readable JSON.

    {"format": "vektorfil-design", "version": 1, "program": "1.1.0",
     "spec": {"ftype": "Lowpass", "shape": "Chebyshev", ...},
     "view": {...}}                       # free-form window state (plot range, markers ...)

Infinite mask frequencies are stored as null.  Unknown keys are ignored when reading
and missing ones take their defaults, so files stay readable across versions; old
circuit names are translated by design() as usual.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from vektorfil import __version__
from vektorfil.filters.design import FilterSpec

FORMAT = "vektorfil-design"
SUFFIX = ".vfil"


def spec_to_dict(spec: FilterSpec) -> dict:
    """FilterSpec -> plain JSON-safe dict."""
    d = dataclasses.asdict(spec)
    d["mask"] = [[a, None if math.isinf(b) else b, db] for a, b, db in spec.mask]
    return d


def spec_from_dict(d: dict) -> FilterSpec:
    """Plain dict -> FilterSpec (unknown keys ignored, missing keys default)."""
    names = {f.name for f in dataclasses.fields(FilterSpec)}
    kw = {k: v for k, v in d.items() if k in names}
    if "mask" in kw:
        kw["mask"] = tuple((float(a), math.inf if b is None else float(b), float(db)) for a, b, db in kw["mask"])
    return FilterSpec(**kw)


def save(path, spec: FilterSpec, view: dict | None = None) -> None:
    """Write a design file."""
    doc = {"format": FORMAT, "version": 1, "program": __version__,
           "spec": spec_to_dict(spec), "view": view or {}}
    Path(path).write_text(json.dumps(doc, indent=2), encoding="utf-8")


def load(path) -> tuple[FilterSpec, dict]:
    """Read a design file: (spec, view)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("format") != FORMAT:
        raise ValueError("not a VektorFIL design file")
    return spec_from_dict(doc.get("spec", {})), doc.get("view", {})
