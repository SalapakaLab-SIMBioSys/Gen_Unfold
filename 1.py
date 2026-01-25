# python
"""
Scan Gen_Unfold/ for python imports and generate a requirements candidate file.

Usage (Windows, from project root):
  python tools/gen_requirements_from_imports.py

Outputs:
  Gen_Unfold/requirements.generated.txt
"""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Set, Tuple

try:
    # Python 3.8+
    import importlib.metadata as importlib_metadata
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore


PROJECT_DIR = Path(__file__).resolve().parents[1]
TARGET_DIR = PROJECT_DIR / "Gen_Unfold"
OUTPUT_FILE = TARGET_DIR / "requirements.generated.txt"


# A conservative stdlib list is hard; keep a practical denylist and also drop relative imports.
STDLIB_TOPLEVEL = {
    "os",
    "sys",
    "re",
    "math",
    "json",
    "time",
    "datetime",
    "pathlib",
    "typing",
    "collections",
    "itertools",
    "functools",
    "dataclasses",
    "statistics",
    "random",
    "logging",
    "subprocess",
    "shlex",
    "hashlib",
    "pickle",
    "csv",
    "sqlite3",
    "threading",
    "multiprocessing",
    "concurrent",
    "asyncio",
    "unittest",
    "doctest",
    "inspect",
    "enum",
    "copy",
    "pprint",
    "traceback",
    "warnings",
    "argparse",
    "glob",
    "fnmatch",
    "tempfile",
    "io",
    "struct",
    "gzip",
    "bz2",
    "lzma",
    "base64",
    "uuid",
    "platform",
    "queue",
    "http",
    "urllib",
    "xml",
    "email",
}


# Common module -> PyPI distribution overrides
MODULE_TO_DIST = {
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "tensorboardX": "tensorboardX",
}


@dataclass(frozen=True)
class ImportHit:
    file: Path
    module: str


def iter_python_files(root: Path) -> Iterator[Path]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".py") and not name.startswith("."):
                yield Path(dirpath) / name


def top_level_name(dotted: str) -> str:
    return dotted.split(".", 1)[0].strip()


def extract_imports_from_file(path: Path) -> Set[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return set()

    mods: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name:
                    mods.add(top_level_name(name))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import
            if node.module:
                mods.add(top_level_name(node.module))
    return mods


def is_probably_third_party(mod: str) -> bool:
    if not mod or mod in STDLIB_TOPLEVEL:
        return False
    # Heuristic: project-internal packages often start with Gen_Unfold or live in repo; keep simple.
    if mod in {"Gen_Unfold"}:
        return False
    return True


def dist_name_for_module(mod: str) -> Optional[str]:
    if mod in MODULE_TO_DIST:
        return MODULE_TO_DIST[mod]

    # Try to map module -> installed distribution by scanning distributions' top-level names.
    # This is best-effort and depends on the current venv.
    try:
        for dist in importlib_metadata.distributions():
            name = dist.metadata.get("Name")
            if not name:
                continue
            top_level = dist.read_text("top_level.txt") or ""
            tops = {line.strip() for line in top_level.splitlines() if line.strip()}
            if mod in tops:
                return name
    except Exception:
        return None

    return None


def version_for_dist(dist_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(dist_name)
    except Exception:
        return None


def main() -> int:
    if not TARGET_DIR.exists():
        print(f"[ERROR] Target dir not found: {TARGET_DIR}")
        return 2

    all_mods: Set[str] = set()
    for py in iter_python_files(TARGET_DIR):
        all_mods |= extract_imports_from_file(py)

    third_party = sorted({m for m in all_mods if is_probably_third_party(m)})

    lines: list[str] = []
    unresolved: list[str] = []

    for mod in third_party:
        dist = dist_name_for_module(mod)
        if not dist:
            unresolved.append(mod)
            continue
        ver = version_for_dist(dist)
        if ver:
            lines.append(f"{dist}>={ver}")
        else:
            lines.append(dist)

    lines = sorted(set(lines), key=str.lower)

    header = [
        "# Auto-generated from imports under Gen_Unfold/ (best-effort).",
        "# Review manually before replacing requirement.txt.",
        "",
    ]

    footer = []
    if unresolved:
        footer += [
            "",
            "# Unresolved modules (could be local, stdlib, or not installed in current venv):",
        ]
        footer += [f"# - {m}" for m in sorted(set(unresolved))]

    OUTPUT_FILE.write_text("\n".join(header + lines + footer) + "\n", encoding="utf-8")
    print(f"[OK] Wrote: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
