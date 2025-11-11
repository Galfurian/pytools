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

    This class collects files matching specified glob patterns under a root
    directory and generates a single markdown document containing all files
    with their contents in fenced code blocks.

    Attributes:
        root (Path):
            The root directory to bundle files from.
        patterns (list[str] | None):
            List of glob patterns to match files against. Defaults to ["**/*.*"].
        output_lines (list[str]):
            Internal list to accumulate markdown output lines.
    """

    def __init__(self, root: Path, patterns: list[str] | None = None):
        """Initialize the PyBundler with root directory and patterns.

        Args:
            root (Path):
                The root directory to search for files.
            patterns (list[str] | None):
                List of glob patterns to match files. If None, defaults to ["**/*.*"].
        """
        self.root = Path(root)
        self.patterns = patterns or ["**/*.*"]
        self.output_lines: list[str] = []

    def add_header(self, title: str, level: int = 2) -> None:
        """Add a markdown header to the output.

        Args:
            title (str):
                The header text.
            level (int):
                The header level (1-6). Defaults to 2.
        """
        if level <= 1:
            self.output_lines.append(f"{'#' * level} {title}\n\n")
        else:
            self.output_lines.append(f"\n{'#' * level} {title}\n\n")

    def add_text(self, text: str) -> None:
        """Add plain text to the output.

        Args:
            text (str):
                The text to add.
        """
        self.output_lines.append(text + "\n")

    def collect_files(self) -> list[Path]:
        """Collect all files matching the patterns under the root directory.

        Returns:
            list[Path]:
                List of matching file paths, deduplicated and sorted.
        """
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
        """Create the bundle and write to the specified output file.

        Collects all matching files, adds a header with generation timestamp,
        and writes each file's content in a fenced code block to the output
        markdown file.

        Args:
            output (Path):
                The path where the markdown bundle will be written.

        Returns:
            Path:
                The path to the created output file.
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
    """Parse command-line arguments.

    Args:
        argv (list[str] | None):
            List of command-line arguments. If None, uses sys.argv.

    Returns:
        argparse.Namespace:
            Parsed arguments.
    """
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
    """Main entry point for the pybundler command-line tool.

    Args:
        argv (list[str] | None):
            Command-line arguments. If None, uses sys.argv.

    Returns:
        int:
            Exit code (0 for success).
    """
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
