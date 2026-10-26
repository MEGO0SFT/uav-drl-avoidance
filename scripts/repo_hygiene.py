#!/usr/bin/env python3
"""Pre-commit / CI guard: keep the repository free of oversized artefacts.

GitHub rejects any single file above 100 MB outright and warns above 50 MB, and
a repository that accumulates binary blobs becomes unpleasant to clone. This
script enforces the policy described in ``MODELS.md``: no model weights, no
archives, and nothing above a configurable size.

``--docs`` additionally validates the bilingual layout: every ``X.md`` must have
an ``X.zh-CN.md`` twin, the two must carry a language-switch line pointing at
each other, and every relative link must resolve on disk.

Usage
-----
    python scripts/repo_hygiene.py                 # check staged files
    python scripts/repo_hygiene.py --all           # check everything tracked
    python scripts/repo_hygiene.py --max-mb 25     # stricter threshold
    python scripts/repo_hygiene.py --all --docs    # also validate the bilingual docs

Exit codes
---------
0  clean
1  violations found
2  git not available / not a repository
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from urllib.parse import unquote

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

# --- bilingual documentation checks ---------------------------------------

DOC_IGNORE_DIRS = {".git", "__pycache__", "node_modules", ".idea"}

# [text](target) with an optional <...> wrap and an optional title.
RELATIVE_LINK = re.compile(r"\[[^\]]*\]\(\s*<?([^)>\s]+)>?(?:\s+\"[^\"]*\")?\s*\)")
SWITCH_TO_ZH = re.compile(r"\[\*{0,2}简体中文\*{0,2}\]\(([^)]+)\)")
SWITCH_TO_EN = re.compile(r"\[\*{0,2}English\*{0,2}\]\(([^)]+)\)")


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
        return os.path.getsize(path)
    except OSError:
        return 0


def markdown_files(root: str = ".") -> list[str]:
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in DOC_IGNORE_DIRS)
        for name in sorted(filenames):
            if name.endswith(".md"):
                rel = os.path.relpath(os.path.join(dirpath, name), root)
                found.append(rel.replace(os.sep, "/"))
    return found


def resolve(base_dir: str, target: str) -> str:
    return os.path.normpath(os.path.join(base_dir, unquote(target))).replace(os.sep, "/")


def check_docs(root: str = ".") -> list[str]:
    """Validate the English / Chinese document pairs and the relative links."""
    problems: list[str] = []
    docs = markdown_files(root)
    if root != ".":
        problems_hint = True  # noqa: F841  (kept for readability of the branch below)
    for rel in docs:
        path = rel if root == "." else f"{root.rstrip('/')}/{rel}"
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except UnicodeDecodeError:
            problems.append(f"not valid UTF-8: {path}")
            continue

        if rel.endswith(".zh-CN.md"):
            twin = rel[: -len(".zh-CN.md")] + ".md"
            switch = SWITCH_TO_EN
        else:
            twin = rel[:-3] + ".zh-CN.md"
            switch = SWITCH_TO_ZH

        if not os.path.exists(twin if root == "." else f"{root.rstrip('/')}/{twin}"):
            problems.append(f"missing Chinese twin of {rel}: expected {twin}")
        else:
            match = switch.search(text)
            if match is None:
                problems.append(f"no language-switch line in {rel} (should link to {twin})")
            else:
                got = resolve(os.path.dirname(rel), match.group(1))
                if got != twin:
                    problems.append(f"language switch in {rel} points at {got}, expected {twin}")

        for match in RELATIVE_LINK.finditer(text):
            target = match.group(1).split("#", 1)[0]
            if not target or re.match(r"^(https?:|mailto:|#|tel:)", match.group(1)):
                continue
            resolved = resolve(os.path.dirname(rel), target)
            if not os.path.exists(resolved if root == "." else f"{root.rstrip('/')}/{resolved}"):
                problems.append(f"broken link in {rel}: {target} -> {resolved}")

    print(f"docs: {len(docs)} markdown file(s), {sum(1 for d in docs if d.endswith('.zh-CN.md'))} Chinese")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Repository hygiene guard")
    ap.add_argument("--all", action="store_true", help="check all tracked files, not just staged")
    ap.add_argument("--max-mb", type=float, default=50.0, help="per-file size limit in MB (default 50)")
    ap.add_argument("--docs", action="store_true", help="also validate the bilingual document pairs")
    ap.add_argument("--max-docs", action="store_true", help="alias kept for symmetry with --max-mb")
    args = ap.parse_args(argv)

    files = list_files(staged_only=not args.all)
    limit = int(args.max_mb * 1024 * 1024)
    problems: list[str] = []
    total = 0

    if not files:
        print("nothing to check")
    else:
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

    if args.docs:
        problems.extend(check_docs())

    if problems:
        print(f"\n{len(problems)} violation(s):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nModel weights and archives belong in a GitHub Release, not in git.\n"
            "See MODELS.md. Update .gitignore, then `git rm --cached <path>`.\n"
            "For document problems: every X.md needs an X.zh-CN.md twin, each with a\n"
            "language-switch line under the title, and every relative link must resolve.",
            file=sys.stderr,
        )
        return 1

    print("clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
