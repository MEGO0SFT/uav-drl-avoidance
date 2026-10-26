#!/usr/bin/env python3
"""Pre-commit / CI guard: keep the repository free of oversized artefacts.

GitHub rejects any single file above 100 MB outright and warns above 50 MB, and
a repository that accumulates binary blobs becomes unpleasant to clone. This
script enforces the policy described in ``MODELS.md``: no model weights, no
archives, and nothing above a configurable size.

Usage
-----
    python scripts/repo_hygiene.py                 # check staged files
    python scripts/repo_hygiene.py --all           # check everything tracked
    python scripts/repo_hygiene.py --max-mb 25     # stricter threshold

Exit codes
----------
0  clean
1  violations found
2  git not available / not a repository
"""
from __future__ import annotations

import argparse
import subprocess
import sys

# Patterns that must never be tracked, regardless of size.
FORBIDDEN_SUFFIXES = (
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".pt", ".pth", ".ckpt", ".onnx", ".engine", ".tflite",
    ".mp4", ".avi", ".mov",
)

# Directories that must never be tracked (model / cache trees).
FORBIDDEN_PARTS = (
    "/models/", "/checkpoints/", "/__pycache__/", "/.venv/", "/venv/",
    "/wandb/", "/runs/",
)

# Intentional exceptions — files that look forbidden but are not.
ALLOWED_EXACT = {
    "requirements-legacy.txt",
}


def git(*args: str) -> str:
    try:
        out = subprocess.run(("git", *args), capture_output=True, text=True, check=True)
    except FileNotFoundError:
        print("error: git is not installed", file=sys.stderr)
        raise SystemExit(2)
    except subprocess.CalledProcessError as exc:
        print(f"error: git {' '.join(args)} failed: {exc.stderr.strip()}", file=sys.stderr)
        raise SystemExit(2)
    return out.stdout


def list_files(staged_only: bool) -> list[str]:
    if staged_only:
        return [p for p in git("diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines() if p]
    return [p for p in git("ls-files").splitlines() if p]


def size_of(path: str) -> int:
    try:
        import os

        return os.path.getsize(path)
    except OSError:
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Repository hygiene guard")
    ap.add_argument("--all", action="store_true", help="check all tracked files, not just staged")
    ap.add_argument("--max-mb", type=float, default=50.0, help="per-file size limit in MB (default 50)")
    args = ap.parse_args(argv)

    files = list_files(staged_only=not args.all)
    if not files:
        print("nothing to check")
        return 0

    limit = int(args.max_mb * 1024 * 1024)
    problems: list[str] = []
    total = 0

    for path in files:
        normalised = path.replace("\\", "/")
        if normalised in ALLOWED_EXACT or normalised.rsplit("/", 1)[-1] in ALLOWED_EXACT:
            continue

        if normalised.lower().endswith(FORBIDDEN_SUFFIXES):
            problems.append(f"forbidden file type: {path}")
            continue

        if any(part in f"/{normalised}" for part in FORBIDDEN_PARTS):
            problems.append(f"forbidden directory: {path}")
            continue

        size = size_of(path)
        total += size
        if size > limit:
            problems.append(f"too large ({size / 1e6:.1f} MB > {args.max_mb:.0f} MB): {path}")

    print(f"checked {len(files)} file(s), {total / 1e6:.1f} MB total")
    if total > 1_000 * 1024 * 1024:
        problems.append(f"repository payload {total / 1e9:.2f} GB exceeds the 1 GB guideline")

    if problems:
        print(f"\n{len(problems)} violation(s):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nModel weights and archives belong in a GitHub Release, not in git.\n"
            "See MODELS.md. Update .gitignore, then `git rm --cached <path>`.",
            file=sys.stderr,
        )
        return 1

    print("clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
