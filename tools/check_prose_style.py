#!/usr/bin/env python3
"""Flag writing in the notebooks and docs that does not match the register of the paper.

The manuscript is written plainly. Notebook markdown and module docstrings should read the same way,
so that a reader moving between them does not change gear. This script does not judge quality. It
finds a small set of constructions that recur in machine-written prose and are absent from the
paper, and prints them for a human to decide on.

What it looks for:

    dash            a dash used as punctuation. The paper uses commas and full stops instead.
                    Compound en-dashes are correct and are not flagged, so "mouse-human",
                    "Gromov-Wasserstein" and "sensorimotor-association" pass as written.
    colon           a colon used to introduce a flourish rather than a list or a definition.
    intensifier     exactly, precisely, crucially, remarkably, importantly, notably, simply
    editorial       the point is, the key insight, it turns out, worth noting, in other words
    hedge_stack     two hedges in one clause, for example "may potentially" or "could possibly"
    firstperson     we are, let us, let's, in this notebook we will
    superlative     powerful, elegant, seamless, robustly, dramatically, significantly better

Usage::

    cd otter && python3 tools/check_prose_style.py                 # notebooks and docs
    cd otter && python3 tools/check_prose_style.py notebooks/fig1_coupling.ipynb
    cd otter && python3 tools/check_prose_style.py --quiet         # exit code only

Exit code is 1 if anything was flagged, so it can gate a commit. A flag is not automatically a
defect. "significantly" is correct when it means a p-value.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                       # .../otter

DEFAULT_TARGETS = ["notebooks", "docs", "README.md", "src/otter/repro.py", "DATA.md"]

INTENSIFIERS = ("exactly", "precisely", "crucially", "remarkably", "importantly",
                "notably", "simply", "essentially", "fundamentally")
EDITORIAL = ("the point is", "the key insight", "it turns out", "worth noting",
             "in other words", "that is the point", "the real ", "note that this is")
SUPERLATIVE = ("powerful", "elegant", "seamless", "seamlessly", "robustly", "dramatically",
               "vastly", "trivially", "beautifully", "cleanly handles")
FIRST_PERSON = ("let us ", "let's ", "in this notebook we will", "we are going to",
                "now we will", "next we will")

RULES: list[tuple[str, re.Pattern[str]]] = [
    # An em-dash is punctuation wherever it appears. An en-dash is only punctuation when it stands
    # alone between spaces; joining two words it is correct compound usage, as the manuscript uses it.
    ("dash",        re.compile(r"—|(?<=\s)–(?=\s)|(?<=\s)–|–(?=\s)")),
    ("intensifier", re.compile(r"\b(" + "|".join(INTENSIFIERS) + r")\b", re.I)),
    ("editorial",   re.compile("|".join(re.escape(p) for p in EDITORIAL), re.I)),
    ("superlative", re.compile(r"\b(" + "|".join(SUPERLATIVE) + r")\b", re.I)),
    ("firstperson", re.compile("|".join(re.escape(p) for p in FIRST_PERSON), re.I)),
    ("hedge_stack", re.compile(r"\b(may|might|could|can)\s+(potentially|possibly|conceivably|"
                               r"perhaps|arguably)\b", re.I)),
    # A colon after a full clause, followed by prose rather than a list or a number. Skips
    # "Usage::", "Parameters", dict/annotation colons, URLs and times.
    ("colon",       re.compile(r"[a-z]{4,}\s+[a-z][a-z ,']{8,}: [a-z]")),
]

# Lines where a flag is expected and should not be reported.
EXEMPT = re.compile(r"^\s*(#\s*noqa|>>>|\.\.\.|\||\+|https?://)|::\s*$|significantly (higher|lower|"
                    r"different|greater|correlated)|p\s*[=<>]")


def markdown_of(path: Path) -> list[tuple[int, str]]:
    """(line number, text) for the prose in a file. Notebooks contribute markdown cells and
    docstrings; .py files contribute comments and docstrings; .md files contribute everything."""
    if path.suffix == ".ipynb":
        try:
            nb = json.loads(path.read_text())
        except Exception:                                        # noqa: BLE001
            return []
        out, n = [], 0
        for cell in nb.get("cells", []):
            src = cell.get("source", [])
            if cell.get("cell_type") == "markdown":
                for line in src:
                    n += 1
                    out.append((n, line.rstrip("\n")))
            else:
                for line in src:
                    n += 1
                    stripped = line.strip()
                    if stripped.startswith("#") or '"""' in stripped:
                        out.append((n, line.rstrip("\n")))
        return out
    if path.suffix == ".py":
        out, in_doc = [], False
        for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if line.count('"""') == 1:
                in_doc = not in_doc
                out.append((i, line))
            elif in_doc or line.strip().startswith("#"):
                out.append((i, line))
        return out
    return [(i, l) for i, l in enumerate(path.read_text(errors="ignore").splitlines(), 1)]


def collect(targets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for t in targets:
        p = (ROOT / t) if not Path(t).is_absolute() else Path(t)
        if p.is_dir():
            paths += [q for q in sorted(p.rglob("*"))
                      if q.suffix in {".ipynb", ".md", ".py"} and ".ipynb_checkpoints" not in str(q)]
        elif p.exists():
            paths.append(p)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", default=None)
    ap.add_argument("--quiet", action="store_true", help="exit code only")
    args = ap.parse_args()

    total = 0
    for path in collect(args.targets or DEFAULT_TARGETS):
        hits = []
        for lineno, text in markdown_of(path):
            if not text.strip() or EXEMPT.search(text):
                continue
            for label, rx in RULES:
                m = rx.search(text)
                if m:
                    hits.append((lineno, label, m.group(0)[:28], text.strip()[:88]))
        if hits:
            total += len(hits)
            if not args.quiet:
                try:
                    shown = path.relative_to(ROOT)
                except ValueError:
                    shown = path
                print(f"\n{shown}  ({len(hits)})")
                for lineno, label, found, text in hits:
                    print(f"  {lineno:5d}  {label:11s} {found!r:32s} {text}")

    if not args.quiet:
        print(f"\n{total} flagged construction(s). A flag is a prompt to reread the line, "
              f"not proof of a problem.")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
