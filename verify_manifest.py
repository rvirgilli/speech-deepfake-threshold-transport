"""Verify every file bound by the A2 anonymous-package manifest."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    failures: list[str] = []
    for relative, expected in sorted(manifest.items()):
        rel = PurePosixPath(relative)
        if rel.is_absolute() or ".." in rel.parts:
            failures.append(f"unsafe manifest path: {relative}")
            continue
        path = ROOT.joinpath(*rel.parts)
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        if path.stat().st_size != expected["bytes"]:
            failures.append(f"size mismatch: {relative}")
        if sha256(path) != expected["sha256"]:
            failures.append(f"SHA-256 mismatch: {relative}")

    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "manifest.json"
        and path.relative_to(ROOT).parts[0] != ".git"
    }
    unbound = sorted(actual - set(manifest))
    if unbound:
        failures.extend(f"unbound file: {path}" for path in unbound)

    if failures:
        print(f"FAILED ({len(failures)}):")
        print("\n".join(f"  - {failure}" for failure in failures))
        sys.exit(1)
    print(f"OK — {len(manifest)} files match size and SHA-256; no unbound files")


if __name__ == "__main__":
    main()
