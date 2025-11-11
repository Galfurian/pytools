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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_CONFIG_FILENAME = ".bundler.config"


@dataclass
class Config:
    """Configuration for pybundler containing patterns and TOC descriptions."""

    patterns: list[str]
    toc_descriptions: dict[str, str]

    @classmethod
    def load_from_files(cls, root: Path, config_files: list[str]) -> Config:
        """Load configuration from config files.

        Args:
            root (Path): Root directory containing config files.
            config_files (list[str]): List of config filenames to check.

        Returns:
            Config: Loaded configuration.
        """
        patterns_list: list[str] = []
        descriptions: dict[str, str] = {}

        for config_filename in config_files:
            config_file = root / config_filename

            if not config_file.exists():
                continue

            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    current_section = None
                    current_patterns: list[str] = []

                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue

                        # Section headers
                        if line.endswith(":") and not ":" in line[:-1]:
                            current_section = line[:-1].lower()
                            continue

                        # Parse patterns - list format only
                        if current_section == "patterns":
                            if line.startswith("- "):
                                pattern = line[2:].strip()
                                if pattern:
                                    current_patterns.append(pattern)

                        # Parse key-value pairs for TOC
                        elif current_section == "toc":
                            if ":" in line:
                                key, description = line.split(":", 1)
                                key = key.strip()
                                description = description.strip()

                                if key and description:
                                    descriptions[key.lower()] = description

                    # Merge patterns from this file (later files override)
                    if current_patterns:
                        patterns_list = current_patterns

            except Exception as e:
                print(f"Warning: Could not read {config_filename} file: {e}")

        return cls(patterns=patterns_list, toc_descriptions=descriptions)

    def save_to_file(self, config_file: Path) -> None:
        """Save configuration to a file.

        Args:
            config_file (Path): Path to save the config file.
        """
        with open(config_file, "w", encoding="utf-8") as f:
            f.write("# Bundler Configuration File\n")
            f.write(
                "# This file defines default patterns and TOC descriptions for your project\n"
            )
            f.write("# \n")
            f.write("# Patterns section: list of glob patterns to include by default\n")
            f.write("patterns:\n")
            for pattern in self.patterns:
                f.write(f"  - {pattern}\n")
            f.write("# \n")
            f.write(
                "# TOC section: descriptions for folders/files in table of contents\n"
            )
            f.write("toc:\n")
            for entry in sorted(self.toc_descriptions.keys()):
                description = self.toc_descriptions[entry]
                f.write(f"  {entry}: {description}\n")


def _is_hidden_path(path: Path) -> bool:
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


def _collect_files(
    root: Path,
    patterns: list[str],
    include_hidden: bool = False,
    max_file_size: int | None = None,
) -> list[Path]:
    """Collect all files matching the patterns under the root directory.

    Filters out binary files and files exceeding size limits.

    Args:
        root (Path):
            The root directory to search for files.
        patterns (list[str]):
            List of glob patterns to match files.
        include_hidden (bool):
            Whether to include hidden files and directories. Defaults to False.
        max_file_size (int | None):
            Maximum file size in bytes to include. If None, no size limit is applied.

    Returns:
        list[Path]:
            List of matching file paths, deduplicated and sorted.
    """
    files: list[Path] = []
    if not root.exists():
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

    for pat in patterns:
        # Handle directory patterns that end with "/" - convert to "**" to match all files
        if pat.endswith("/") and "*" not in pat and "?" not in pat:
            # Convert "player/" to "player/**" to match all files in the directory
            actual_pat = pat + "**"
        else:
            actual_pat = pat

        for p in root.rglob(actual_pat):
            if not p.is_file():
                continue

            # Skip hidden files/directories unless explicitly included
            if not include_hidden and _is_hidden_path(p):
                continue

            # Skip binary files by extension
            if p.suffix.lower() in binary_extensions:
                continue

            # Skip files that are too large (if limit is enabled)
            if max_file_size is not None:
                try:
                    if p.stat().st_size > max_file_size:
                        continue
                except OSError:
                    continue

            files.append(p)

    # Deduplicate and sort by path
    unique = sorted({p.resolve(): p for p in files}.values(), key=lambda p: str(p))
    return unique


def _is_valid_pattern(root: Path, pattern: str) -> tuple[bool, list[Path]]:
    """Check if a pattern is valid and return any matching files.

    Args:
        root (Path):
            The root directory to check against.
        pattern (str):
            The pattern to validate.

    Returns:
        tuple[bool, list[Path]]:
            Tuple of (is_valid, files) where is_valid indicates if the pattern is valid
            and files contains any matching files found.
    """
    # Handle directory patterns that end with "/" - convert to "**" for file matching
    if pattern.endswith("/") and "*" not in pattern and "?" not in pattern:
        actual_pattern = pattern + "**"
    else:
        actual_pattern = pattern

    # Check if pattern matches any files
    files = _collect_files(
        root,
        [actual_pattern],
        include_hidden=False,
        max_file_size=None,
    )
    if files:
        return True, files

    # If no files found, check if it's a valid directory path (for patterns ending with /)
    if pattern.endswith("/"):
        path_obj = root / pattern.rstrip("/")
        if path_obj.exists() and path_obj.is_dir():
            return True, []
    elif not ("*" in pattern or "?" in pattern):
        # For non-glob patterns, check if the path exists at all
        path_obj = root / pattern
        if path_obj.exists():
            return True, []

    return False, []


def _generate_config(
    root: Path,
    patterns: list[str],
    config_filename: str = DEFAULT_CONFIG_FILENAME,
) -> int:
    """Generate a starter bundler configuration file.

    Args:
        root (Path):
            Root directory to scan for files.
        patterns (list[str]):
            Glob patterns to match files.
        config_filename (str):
            Name of the config file to generate.

    Returns:
        int:
            Exit code (0 for success).
    """
    # Validate patterns individually and collect valid ones
    valid_patterns = []
    all_files = set()

    for pattern in patterns:
        # Check if pattern is valid and get any matching files
        is_valid, files = _is_valid_pattern(root, pattern)
        if is_valid:
            all_files.update(files)
            valid_patterns.append(pattern)
        else:
            print(
                f"Warning: Pattern '{pattern}' does not match any files or valid paths. Skipping."
            )

    if not valid_patterns:
        print("No valid patterns found. Cannot generate config.")
        return 1

    # Collect TOC entries based on valid pattern types
    toc_descriptions = {}

    # Analyze each valid pattern to determine TOC entries
    for pattern in valid_patterns:
        # If pattern ends with /** or /*, it's a directory pattern - add directory name
        if pattern.endswith("/**") or pattern.endswith("/*"):
            dir_name = pattern.rstrip("/*")
            if "/" in dir_name:
                # For nested directories like "player/**", add "player"
                toc_descriptions[dir_name.split("/")[0]] = ""
            else:
                # For top-level directories like "journal/**", add "journal"
                toc_descriptions[dir_name] = ""
        elif "*" in pattern or "?" in pattern:
            # Glob pattern - collect what it actually matches and add top-level entries
            for f in all_files:
                try:
                    rel = f.relative_to(root)
                except Exception:
                    rel = f.name
                rel_str = str(rel)
                if "/" in rel_str:
                    toc_descriptions[rel_str.split("/")[0]] = ""
                else:
                    toc_descriptions[rel_str] = ""
        else:
            # Specific path - check if it's a file or directory
            path_obj = root / pattern
            if path_obj.exists():
                if path_obj.is_file():
                    # Specific file
                    toc_descriptions[pattern] = ""
                elif path_obj.is_dir():
                    # Directory path - add directory name without trailing slash
                    dir_name = pattern.rstrip("/")
                    if "/" in dir_name:
                        toc_descriptions[dir_name.split("/")[0]] = ""
                    else:
                        toc_descriptions[dir_name] = ""

    # Create config object and save it
    config = Config(patterns=valid_patterns, toc_descriptions=toc_descriptions)
    config_file = root / config_filename
    config.save_to_file(config_file)

    print(
        f"Generated starter {config_filename} file with {len(valid_patterns)} patterns and {len(toc_descriptions)} TOC entries."
    )
    print("Edit the file to customize patterns and add descriptions.")
    print("Then run with --toc to generate bundles with descriptions.")
    print("Or run without --patterns to use the configured defaults.")

    return 0


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
        config_files: list[str] | None = None,
    ):
        """Initialize the PyBundler with root directory and patterns.

        Args:
            root (Path):
                The root directory to search for files.
            patterns (list[str] | None):
                List of glob patterns to match files. If None, loads from config files
                or defaults to ["**/*.*"].
            max_file_size (int | None):
                Maximum file size in bytes to include. If None, no size limit is applied.
            include_hidden (bool):
                Whether to include hidden files and directories. Defaults to False.
            warn_size (int):
                File size threshold in bytes for warnings. Defaults to 50KB.
            generate_toc (bool):
                Whether to generate a table of contents. Defaults to False.
            config_files (list[str] | None):
                List of config file paths to load. If None, uses default .bundler.config.
        """
        self.root = Path(root)
        self.config_files = config_files or [DEFAULT_CONFIG_FILENAME]

        # Load config from files
        config = Config.load_from_files(self.root, self.config_files)

        # Use provided patterns or load from config
        if patterns is None:
            self.patterns = config.patterns if config.patterns else ["**/*.*"]
        else:
            self.patterns = patterns

        self.max_file_size = max_file_size
        self.include_hidden = include_hidden
        self.warn_size = warn_size
        self.generate_toc = generate_toc
        self.toc_descriptions = config.toc_descriptions
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

    def collect_files_with_warnings(self) -> tuple[list[Path], list[str]]:
        """Collect all files matching the patterns and return warnings for large files.

        Returns:
            tuple[list[Path], list[str]]:
                Tuple of (files, warnings) where warnings contains messages for files
                exceeding the warn_size threshold.
        """
        files = _collect_files(
            self.root, self.patterns, self.include_hidden, self.max_file_size
        )
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

    def _generate_toc(self, files: list[Path]) -> None:
        """Generate table of contents for the bundled files.

        Args:
            files (list[Path]):
                List of files to include in the TOC.
        """
        self.add_header("Table of Contents", level=2)

        # Collect unique entries - individual files with descriptions get their own entries
        toc_entries = set()
        individual_files = set()

        for f in files:
            try:
                rel = f.relative_to(self.root)
            except Exception:
                rel = f.name
            rel_str = str(rel)

            # Check if this specific file has a description
            if self.toc_descriptions and self.toc_descriptions.get(rel_str.lower()):
                individual_files.add(rel_str)
            else:
                # For files in subdirectories, use the top-level folder
                if "/" in rel_str:
                    top_level = rel_str.split("/")[0]
                else:
                    top_level = rel_str
                toc_entries.add(top_level)

        # Generate TOC entries - individual files first, then grouped folders
        all_entries = sorted(individual_files) + sorted(
            toc_entries - set(entry.split("/")[0] for entry in individual_files)
        )

        for entry in all_entries:
            description = None
            if self.toc_descriptions:
                description = self.toc_descriptions.get(entry.lower())

            if description:
                self.add_text(f"- **{entry}** - {description}")
            else:
                self.add_text(f"- {entry}")

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

        files = _collect_files(
            self.root, self.patterns, self.include_hidden, self.max_file_size
        )

        if not files:
            self.add_text("No files found. Nothing to bundle.")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("".join(self.output_lines), encoding="utf-8")
            return output

        # Generate table of contents if requested
        if self.generate_toc and files:
            self._generate_toc(files)

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
        "--generate-config",
        type=str,
        nargs="?",
        const=".bundler.config",
        help="Generate a starter bundler configuration file (default: .bundler.config)",
    )
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        help="Configuration file(s) to use (can be specified multiple times). If not specified, uses .bundler.config",
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

    # Parse patterns - use None if default was used (to allow config loading)
    patterns_arg = args.patterns
    if patterns_arg == "**/*.*":  # This is the default
        patterns = None  # Will load from config or use default
    else:
        patterns = [p.strip() for p in patterns_arg.split(",") if p.strip()]

    # Handle config generation
    if args.generate_config:
        # For config generation, we need actual patterns
        actual_patterns = patterns if patterns else ["**/*.*"]
        return _generate_config(root, actual_patterns, args.generate_config)

    bundler = PyBundler(
        root,
        patterns=patterns,
        max_file_size=args.max_size,
        include_hidden=args.include_hidden,
        warn_size=args.warn_size,
        generate_toc=args.toc,
        config_files=args.config,
    )
    out_path = Path(args.output)

    # Format patterns for display
    if len(bundler.patterns) <= 3:
        patterns_display = str(bundler.patterns)
    else:
        patterns_display = f"[{bundler.patterns[0]}, {bundler.patterns[1]}, {bundler.patterns[2]}, ...] ({len(bundler.patterns)} total)"

    print(f"Bundling files from {root} using patterns: {patterns_display}")

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

    # Get file statistics
    file_size_kb = output.stat().st_size / 1024

    # Count lines and words
    try:
        content = output.read_text(encoding="utf-8")
        line_count = len(content.splitlines())
        word_count = len(content.split())
        # Rough token estimate: ~4 characters per token
        token_estimate = len(content) // 4
    except Exception:
        line_count = 0
        word_count = 0
        token_estimate = 0

    print(
        f"Created bundle: {output} ({file_size_kb:.2f} KB, {line_count} lines, {word_count} words, ~{token_estimate} tokens)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
