#!/usr/bin/env python3
"""
pybundler: bundle project files into a single LLM-friendly markdown file.

This tool collects files matching glob patterns under a root directory and
emits a single markdown document that includes optional human-provided
descriptions or generated short descriptions for each file/section. The
result is intended for copy/paste into LLM/web-UI uploads.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


class PyBundler:
    """Bundle files for LLM ingestion into a single markdown file.

    Contract:
    - Inputs: root directory, list of glob patterns, optional description map.
    - Output: Path to generated markdown containing sections per file.
    - Error modes: missing root or no files are handled gracefully; exceptions
      during reading a file are reported and skipped.
    """

    def __init__(self, root: Path, patterns: list[str] | None = None):
        self.root = Path(root)
        self.patterns = patterns or ["**/*.*"]
        self.output_lines: list[str] = []

    def add_header(self, title: str, level: int = 2) -> None:
        if level <= 1:
            self.output_lines.append(f"{'#' * level} {title}\n\n")
        else:
            self.output_lines.append(f"\n{'#' * level} {title}\n\n")

    def add_text(self, text: str) -> None:
        self.output_lines.append(text + "\n")

    def collect_files(self) -> list[Path]:
        files: list[Path] = []
        if not self.root.exists():
            return files
        for pat in self.patterns:
            files.extend([p for p in self.root.rglob(pat) if p.is_file()])
        # Deduplicate and sort by path
        unique = sorted({p.resolve(): p for p in files}.values(), key=lambda p: str(p))
        return unique

    def bundle(
        self,
        output: Path,
    ) -> Path:
        """Create the bundle and write to `output`.

        - description_map: mapping of relative path -> description.
        """
        self.add_header("Project Bundle for LLM", level=1)
        self.add_text(
            f"*Generated: {datetime.now().isoformat(sep=' ', timespec='seconds')}*"
        )

        files = self.collect_files()

        if not files:
            self.add_text("No files found. Nothing to bundle.")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("".join(self.output_lines), encoding="utf-8")
            return output

        for f in files:
            try:
                rel = f.relative_to(self.root)
            except Exception:
                rel = f.name

            rel_str = str(rel)
            # Attempt to read file
            try:
                content = f.read_text(encoding="utf-8")
            except Exception as exc:
                self.add_header(f"File: `{rel_str}`", level=2)
                self.add_text(f"*Error reading file: {exc}*")
                continue

            # Choose fenced block language by suffix
            suffix = f.suffix.lower().lstrip(".")
            fence_lang = suffix if suffix else "text"
            self.add_text("")
            self.add_text(f"```{fence_lang}")
            self.add_text(content.rstrip())
            self.add_text("```")

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(self.output_lines), encoding="utf-8")
        return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bundle project files into a single markdown for LLM ingestion."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory to bundle (default: current dir)",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default="**/*.*",
        help="Comma-separated glob patterns to include (default: '**/*.*')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="BUNDLE.md",
        help="Output markdown filename",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]

    bundler = PyBundler(root, patterns=patterns)
    out_path = Path(args.output)

    print(f"Bundling files from {root} using patterns: {patterns}")

    output = bundler.bundle(out_path)

    print(f"Created bundle: {output} ({output.stat().st_size / 1024:.2f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
