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
        max_file_size (int | None):
            Maximum file size in bytes to include. Files larger than this will be skipped.
            If None, no size limit is applied.
        include_hidden (bool):
            Whether to include hidden files and directories (starting with '.').
            Defaults to False.
        warn_size (int):
            File size threshold in bytes for warnings about potential LLM context window issues.
            Defaults to 50KB.
        generate_toc (bool):
            Whether to generate a table of contents at the top of the bundle.
            Defaults to False.
        toc_descriptions (dict[str, str] | None):
            Optional descriptions for files/folders in TOC, loaded from .bundler.toc file.
        output_lines (list[str]):
            Internal list to accumulate markdown output lines.
    """

    def __init__(
        self,
        root: Path,
        patterns: list[str] | None = None,
        max_file_size: int | None = None,
        include_hidden: bool = False,
        warn_size: int = 50 * 1024,  # 50KB default
        generate_toc: bool = False,
    ):
        """Initialize the PyBundler with root directory and patterns.

        Args:
            root (Path):
                The root directory to search for files.
            patterns (list[str] | None):
                List of glob patterns to match files. If None, defaults to ["**/*.*"].
            max_file_size (int | None):
                Maximum file size in bytes to include. If None, no size limit is applied.
            include_hidden (bool):
                Whether to include hidden files and directories. Defaults to False.
            warn_size (int):
                File size threshold in bytes for warnings. Defaults to 50KB.
            generate_toc (bool):
                Whether to generate a table of contents. Defaults to False.
        """
        self.root = Path(root)
        self.patterns = patterns or ["**/*.*"]
        self.max_file_size = max_file_size
        self.include_hidden = include_hidden
        self.warn_size = warn_size
        self.generate_toc = generate_toc
        self.toc_descriptions = self._load_toc_descriptions()
        self.output_lines: list[str] = []

    def _is_hidden_path(self, path: Path) -> bool:
        """Check if a path contains hidden files or directories.

        A path is considered hidden if it or any of its parent directories
        start with a dot ('.').

        Args:
            path (Path):
                The path to check.

        Returns:
            bool:
                True if the path contains hidden components, False otherwise.
        """
        # Check the file/directory name itself
        if path.name.startswith("."):
            return True

        # Check all parent directories
        for parent in path.parents:
            if parent.name.startswith("."):
                return True

        return False

    def _load_toc_descriptions(self) -> dict[str, str]:
        """Load TOC descriptions from .bundler.toc file if it exists.

        The .bundler.toc file should contain lines in the format:
        filename_or_folder: Description of what this contains

        Returns:
            dict[str, str]:
                Mapping of filenames/folders to their descriptions.
        """
        toc_file = self.root / ".bundler.toc"
        descriptions = {}
        
        if toc_file.exists():
            try:
                with open(toc_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        if ':' not in line:
                            print(f"Warning: Invalid line {line_num} in .bundler.toc: {line}")
                            continue
                        
                        key, description = line.split(':', 1)
                        key = key.strip()
                        description = description.strip()
                        
                        if key and description:
                            descriptions[key.lower()] = description
                            
            except Exception as e:
                print(f"Warning: Could not read .bundler.toc file: {e}")
        
        return descriptions

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

        Filters out binary files and files exceeding size limits.

        Returns:
            list[Path]:
                List of matching file paths, deduplicated and sorted.
        """
        files: list[Path] = []
        if not self.root.exists():
            return files

        # Common binary file extensions to skip
        binary_extensions = {
            # Python bytecode
            ".pyc",
            ".pyo",
            ".pyd",
            # Images
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".ico",
            # Videos
            ".mp4",
            ".avi",
            ".mkv",
            ".mov",
            ".wmv",
            # Audio
            ".mp3",
            ".wav",
            ".flac",
            ".aac",
            # Archives
            ".zip",
            ".tar",
            ".gz",
            ".bz2",
            ".xz",
            ".7z",
            ".rar",
            # Documents
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            # Executables/Libraries
            ".exe",
            ".dll",
            ".so",
            ".dylib",
            # Databases
            ".db",
            ".sqlite",
            ".sqlite3",
        }

        for pat in self.patterns:
            for p in self.root.rglob(pat):
                if not p.is_file():
                    continue

                # Skip hidden files/directories unless explicitly included
                if not self.include_hidden and self._is_hidden_path(p):
                    continue

                # Skip binary files by extension
                if p.suffix.lower() in binary_extensions:
                    continue

                # Skip files that are too large (if limit is enabled)
                if self.max_file_size is not None:
                    try:
                        if p.stat().st_size > self.max_file_size:
                            continue
                    except OSError:
                        continue

                files.append(p)

        # Deduplicate and sort by path
        unique = sorted({p.resolve(): p for p in files}.values(), key=lambda p: str(p))
        return unique

    def collect_files_with_warnings(self) -> tuple[list[Path], list[str]]:
        """Collect all files matching the patterns and return warnings for large files.

        Returns:
            tuple[list[Path], list[str]]:
                Tuple of (files, warnings) where warnings contains messages for files
                exceeding the warn_size threshold.
        """
        files = self.collect_files()
        warnings = []

        for f in files:
            try:
                file_size = f.stat().st_size
                if file_size > self.warn_size:
                    warnings.append(
                        f"Warning: {f.relative_to(self.root)} is {file_size} bytes "
                        f"({file_size/1024:.1f} KB) - may exceed LLM context window"
                    )
            except OSError:
                pass

        return files, warnings

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

        # Generate table of contents if requested
        if self.generate_toc and files:
            self.add_header("Table of Contents", level=2)
            
            # Collect unique top-level entries with descriptions
            toc_entries = set()
            for f in files:
                try:
                    rel = f.relative_to(self.root)
                except Exception:
                    rel = f.name
                rel_str = str(rel)
                
                # For files in subdirectories, use the top-level folder
                if '/' in rel_str:
                    top_level = rel_str.split('/')[0]
                else:
                    top_level = rel_str
                
                toc_entries.add(top_level)
            
            # Generate TOC entries
            for entry in sorted(toc_entries):
                description = None
                if self.toc_descriptions:
                    description = self.toc_descriptions.get(entry.lower())
                
                if description:
                    self.add_text(f"- **{entry}** - {description}")
                else:
                    self.add_text(f"- {entry}")
            
            self.add_text("")

        for f in files:
            try:
                rel = f.relative_to(self.root)
            except Exception:
                rel = f.name

            rel_str = str(rel)
            # Get file size
            try:
                file_size = f.stat().st_size
            except Exception:
                file_size = 0

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
            self.add_header(f"File: `{rel_str}` (Size: {file_size} bytes)", level=2)
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
        "--max-size",
        type=int,
        default=None,
        help="Maximum file size in bytes to include (default: no limit)",
    )
    parser.add_argument(
        "--warn-size",
        type=int,
        default=50 * 1024,
        help="File size threshold in bytes for LLM context window warnings (default: 50KB)",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files and directories (starting with '.')",
    )
    parser.add_argument(
        "--toc",
        action="store_true",
        help="Generate a table of contents at the top of the bundle",
    )
    parser.add_argument(
        "--generate-toc-config",
        action="store_true",
        help="Generate a starter .bundler.toc file with all folders/files and empty descriptions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="BUNDLE.md",
        help="Output markdown filename",
    )
    return parser.parse_args(argv)


def generate_toc_config(root: Path, patterns: list[str]) -> int:
    """Generate a starter .bundler.toc configuration file.

    Args:
        root (Path):
            Root directory to scan for files.
        patterns (list[str]):
            Glob patterns to match files.

    Returns:
        int:
            Exit code (0 for success).
    """
    # Create a temporary bundler to collect files
    temp_bundler = PyBundler(root, patterns=patterns, include_hidden=False)
    files = temp_bundler.collect_files()
    
    if not files:
        print("No files found matching the patterns. Cannot generate TOC config.")
        return 1
    
    # Collect unique top-level entries
    toc_entries = set()
    for f in files:
        try:
            rel = f.relative_to(root)
        except Exception:
            rel = f.name
        rel_str = str(rel)
        
        # Use top-level folder for files in subdirectories
        if '/' in rel_str:
            top_level = rel_str.split('/')[0]
        else:
            top_level = rel_str
        
        toc_entries.add(top_level)
    
    # Generate the .bundler.toc file
    toc_file = root / ".bundler.toc"
    
    with open(toc_file, 'w', encoding='utf-8') as f:
        f.write("# Bundler Table of Contents Configuration\n")
        f.write("# Format: name: Description of what this folder/file contains\n")
        f.write("# Lines starting with # are comments and ignored\n")
        f.write("# \n")
        f.write("# This file helps generate descriptive table of contents when using --toc\n")
        f.write("# Edit the descriptions below to customize your TOC entries\n")
        f.write("\n")
        
        for entry in sorted(toc_entries):
            f.write(f"{entry}: \n")
    
    print(f"Generated starter .bundler.toc file with {len(toc_entries)} entries.")
    print("Edit the file to add descriptions for each folder/file.")
    print("Then run with --toc to generate a bundle with descriptions.")
    
    return 0


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

    # Handle TOC config generation
    if args.generate_toc_config:
        return generate_toc_config(root, patterns)

    bundler = PyBundler(
        root,
        patterns=patterns,
        max_file_size=args.max_size,
        include_hidden=args.include_hidden,
        warn_size=args.warn_size,
        generate_toc=args.toc,
    )
    out_path = Path(args.output)

    print(f"Bundling files from {root} using patterns: {patterns}")

    # Collect files and check for size warnings
    _, warnings = bundler.collect_files_with_warnings()

    # Print warnings for large files
    for warning in warnings:
        print(f"⚠️  {warning}")

    if warnings:
        print(
            f"\nNote: {len(warnings)} file(s) exceed the warning threshold of {args.warn_size} bytes."
        )
        print(
            "Consider using --max-size to filter out large files or adjust --warn-size.\n"
        )

    output = bundler.bundle(out_path)

    print(f"Created bundle: {output} ({output.stat().st_size / 1024:.2f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
