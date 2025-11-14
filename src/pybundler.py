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
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import fnmatch
from typing import Optional


DEFAULT_CONFIG_FILENAME = ".bundler.config"

logger = logging.getLogger(__name__)


# -- ColorFormatter (matches your repo style) --
class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",  # CYAN
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[41m",  # Red bg
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        if color:
            message = f"{color}{message}{self.RESET}"
        return message


@dataclass
class Config:
    """Configuration for pybundler containing patterns and TOC descriptions."""

    patterns: list[str]
    excludes: list[str]
    toc: dict[str, str]


def load_config_from_file(path: Path) -> Config | None:
    """Load configuration from config files.

    Args:
        path (Path): Path to the config file.

    Returns:
        Config | None: Loaded configuration or None if file does not exist.
    """

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            current_section = None
            patterns: list[str] = []
            excludes: list[str] = []
            toc: dict[str, str] = {}

            for line in f:
                original_line = line
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Section headers - assume they are not indented
                if (
                    line.endswith(":")
                    and not original_line.startswith(" ")
                    and ":" not in line[:-1]
                ):
                    section = line[:-1].lower()
                    if section not in ["patterns", "excludes", "toc"]:
                        logger.warning(
                            "Unknown section '%s' in config file %s. Ignoring.",
                            section,
                            path,
                        )
                        current_section = None
                    else:
                        current_section = section
                    continue

                # Parse patterns - list format only
                if current_section == "patterns":
                    if not line.startswith("- "):
                        logger.error(
                            "Invalid line in 'patterns' section of %s: '%s'. Expected format: '- pattern'",
                            path,
                            line,
                        )
                        continue
                    pattern = line[2:].strip()
                    if not pattern:
                        logger.warning(
                            "Empty pattern in 'patterns' section of %s",
                            path,
                        )
                        continue
                    patterns.append(pattern)

                # Parse excludes patterns - list format only
                elif current_section == "excludes":
                    if not line.startswith("- "):
                        logger.error(
                            "Invalid line in 'excludes' section of %s: '%s'. Expected format: '- pattern'",
                            path,
                            line,
                        )
                        continue
                    pattern = line[2:].strip()
                    if not pattern:
                        logger.warning(
                            "Empty pattern in 'excludes' section of %s",
                            path,
                        )
                        continue
                    excludes.append(pattern)

                # Parse key-value pairs for TOC
                elif current_section == "toc":
                    if ":" not in line:
                        logger.error(
                            "Invalid line in 'toc' section of %s: '%s'. Expected format: 'key: description'",
                            path,
                            line,
                        )
                        continue
                    key, description = line.split(":", 1)
                    key = key.strip()
                    description = description.strip()
                    if not key:
                        logger.warning("Empty key in 'toc' section of %s", path)
                        continue
                    toc[key.lower()] = description
                else:
                    logger.warning(
                        "Line outside any section in %s: '%s'. Ignoring.",
                        path,
                        line,
                    )

    except Exception as e:
        logger.warning("Could not read %s file: %s", path, e)
        return None

    return Config(
        patterns=patterns,
        toc=toc,
        excludes=excludes,
    )


def save_config_to_file(config: Config, config_file: Path) -> None:
    """Save configuration to a file.

    Args:
        config (Config): The configuration to save.
        config_file (Path): Path to save the config file.
    """
    try:
        # Ensure the directory exists
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w", encoding="utf-8") as f:
            f.write("# Bundler Configuration File\n")
            f.write(
                "# This file defines default patterns and TOC descriptions for your project\n"
            )
            f.write("# \n")
            f.write("# Patterns section: list of glob patterns to include by default\n")
            f.write("patterns:\n")
            for pattern in config.patterns:
                f.write(f"  - {pattern}\n")
            f.write("# \n")
            f.write(
                "# TOC section: descriptions for folders/files in table of contents\n"
            )
            f.write("toc:\n")
            for entry in sorted(config.toc.keys()):
                description = config.toc[entry]
                f.write(f"  {entry}: {description}\n")
            f.write("# \n")
            f.write("# Exclude section: patterns to excludes from bundling\n")
            f.write("excludes:\n")
            for pattern in config.excludes:
                f.write(f"  - {pattern}\n")
    except Exception as e:
        raise RuntimeError(f"Failed to save config file {config_file}: {e}")


def _collect_files(
    root: Path,
    patterns: list[str],
    include_hidden: bool = False,
    max_file_size: Optional[int] = None,
    excludes: Optional[list[str]] = None,
) -> list[Path]:
    """
    Collect files under `root` matching `patterns`, excluding hidden files,
    oversized files, and those matching `excludes`.
    """
    if not root.exists():
        return []

    matched_files = set()

    logger.debug("Collecting files from root: %s", root)
    logger.debug("Using patterns: %s", patterns)
    logger.debug("Exclude patterns: %s", excludes)
    logger.debug("Include hidden: %s", include_hidden)
    logger.debug("Max file size: %s", max_file_size)

    logger.debug("Starting file collection...")
    for pattern in patterns:
        logger.debug("  Processing pattern: `%s`...", pattern)
        for path in root.glob(pattern):
            if not path.is_file():
                continue

            # Exclude hidden files or folders
            if not include_hidden and any(part.startswith(".") for part in path.parts):
                continue

            # Exclude oversized files
            if max_file_size is not None:
                try:
                    if path.stat().st_size > max_file_size:
                        continue
                except OSError:
                    continue

            # Exclude based on patterns
            if excludes:
                rel_path = str(path.relative_to(root))
                if any(
                    fnmatch.fnmatch(rel_path, pat)
                    or fnmatch.fnmatch(path.name, pat)
                    or pat in path.parts
                    for pat in excludes
                ):
                    continue

            matched_files.add(path.resolve())

            logger.debug("    - %s", path)

    return sorted(matched_files)


def _generate_toc_entries(
    valid_patterns: list[str], all_files: set[Path], root: Path
) -> dict[str, str]:
    """Generate TOC entries based on patterns and collected files.

    Args:
        valid_patterns: List of valid glob patterns.
        all_files: Set of all collected file paths.
        root: Root directory path.

    Returns:
        Dictionary mapping TOC entry names to descriptions (initially empty strings).
    """
    toc = {}

    for pattern in valid_patterns:
        # If pattern ends with /** or /*, it's a directory pattern - add directory name
        if pattern.endswith("/**") or pattern.endswith("/*"):
            dir_name = pattern.rstrip("/*")
            if "/" in dir_name:
                # For nested directory patterns, add the top-level directory name
                toc[dir_name.split("/")[0]] = ""
            else:
                # For top-level directory patterns, add the directory name
                toc[dir_name] = ""
        elif "*" in pattern or "?" in pattern:
            # Glob pattern - collect what it actually matches and add top-level entries
            for f in all_files:
                try:
                    rel = f.relative_to(root)
                except Exception:
                    rel = f.name
                rel_str = str(rel)
                if "/" in rel_str:
                    toc[rel_str.split("/")[0]] = ""
                else:
                    toc[rel_str] = ""
        else:
            # Specific path - check if it's a file or directory
            path_obj = root / pattern
            if path_obj.exists():
                if path_obj.is_file():
                    # Specific file
                    toc[pattern] = ""
                elif path_obj.is_dir():
                    # Directory path - add directory name without trailing slash
                    dir_name = pattern.rstrip("/")
                    if "/" in dir_name:
                        toc[dir_name.split("/")[0]] = ""
                    else:
                        toc[dir_name] = ""

    return toc


def _generate_config(
    root: Path,
    patterns: list[str],
    config_filename: str = DEFAULT_CONFIG_FILENAME,
    excludes: list[str] | None = None,
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
    # Collect files from all patterns
    valid_patterns = patterns
    all_files = set()

    for pattern in patterns:
        files = _collect_files(
            root,
            [pattern],
            include_hidden=False,
            max_file_size=None,
            excludes=excludes,
        )
        all_files.update(files)

    # Collect TOC entries based on valid pattern types
    toc = _generate_toc_entries(valid_patterns, all_files, root)

    # Create config object and save it
    config = Config(
        patterns=valid_patterns,
        toc=toc,
        excludes=excludes or [],
    )
    config_file = root / config_filename
    save_config_to_file(config, config_file)

    logger.info(
        "Generated starter %s file with %d patterns and %d TOC entries.",
        config_filename,
        len(valid_patterns),
        len(toc),
    )
    logger.info("Edit the file to customize patterns and add descriptions.")
    logger.info("Then run with --toc to generate bundles with descriptions.")
    logger.info("Or run without --patterns to use the configured defaults.")

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
        toc (dict[str, str] | None):
            Optional descriptions for files/folders in TOC, loaded from .bundler.toc file.
        output_lines (list[str]):
            Internal list to accumulate markdown output lines.
    """

    def __init__(
        self,
        root: Path,
        patterns: list[str] = [],
        max_file_size: int | None = None,
        include_hidden: bool = False,
        warn_size: int = 50 * 1024,  # 50KB default
        generate_toc: bool = False,
        config_file: str | None = None,
        excludes: list[str] = [],
    ):
        """Initialize the PyBundler with root directory and patterns.

        Args:
            root (Path):
                The root directory to search for files.
            patterns (list[str]):
                List of glob patterns to match files.
            max_file_size (int | None):
                Maximum file size in bytes to include. If None, no size limit is applied.
            include_hidden (bool):
                Whether to include hidden files and directories. Defaults to False.
            warn_size (int):
                File size threshold in bytes for warnings. Defaults to 50KB.
            generate_toc (bool):
                Whether to generate a table of contents. Defaults to False.
            config_file (list[str]):
                List of config file paths to load. Uses default .bundler.config if not specified.
            excludes (list[str]):
                List of glob patterns to excludes from bundling. Applied after inclusion patterns.
        """
        self.root = Path(root)

        # Load config from files
        config = load_config_from_file(self.root / config_file) if config_file else None
        if config is None:
            config = Config(patterns, excludes, {})

        self.patterns = config.patterns
        self.excludes = config.excludes
        self.max_file_size = max_file_size
        self.include_hidden = include_hidden
        self.warn_size = warn_size
        self.generate_toc = generate_toc
        self.toc = config.toc
        self.output_lines: list[str] = []

        # Internal cache of collected files
        self._files: list[Path] = []
        # Internal cache of files grouped by directory
        self._files_by_directory: dict[str | None, list[Path]] = {}

    def add_header(self, title: str, level: int = 2) -> None:
        """Add a markdown header to the output.

        Args:
            title (str):
                The header text.
            level (int):
                The header level (1-6). Defaults to 2.
        """
        if level > 1:
            self.add_text("")
        self.add_text(f"{'#' * level} {title}")
        self.add_text("")

    def add_text(self, text: str) -> None:
        """Add plain text to the output.

        Args:
            text (str):
                The text to add.
        """
        self.output_lines.append(text + "\n")

    def fix_text(self) -> None:
        """
        Takes care of performing different types of quality of life fixes to the
        output text such as removing excessive newlines.
        """
        # Look for two consecutive empty lines and remove one to avoid excessive
        # spacing.
        for i in range(len(self.output_lines) - 1, 0, -1):
            prev_empty = not self.output_lines[i - 1].strip()
            curr_empty = not self.output_lines[i].strip()
            if curr_empty and prev_empty:
                del self.output_lines[i]

    def get_text(self) -> str:
        """Get the accumulated markdown output as a single string.

        Returns:
            str:
                The complete markdown output.
        """
        self.fix_text()
        return "".join(self.output_lines)

    def collect_files_with_warnings(self) -> tuple[list[Path], list[str]]:
        """Collect all files matching the patterns and return warnings for large files.

        Returns:
            tuple[list[Path], list[str]]:
                Tuple of (files, warnings) where warnings contains messages for files
                exceeding the warn_size threshold.
        """
        # Ensure we have collected files first (callers may call this before bundle())
        if not self._files:
            self._files = _collect_files(
                self.root,
                self.patterns,
                self.include_hidden,
                self.max_file_size,
                self.excludes,
            )
            logger.info(
                "Collected %d files matching the specified patterns.", len(self._files)
            )

        warnings: list[str] = []
        for f in self._files:
            try:
                file_size = f.stat().st_size
                if self.warn_size is not None and file_size > self.warn_size:
                    warnings.append(
                        f"Warning: {f.relative_to(self.root)} is {file_size} bytes "
                        f"({file_size/1024:.1f} KB) - may exceed LLM context window"
                    )
            except OSError:
                # If stat fails, skip warning for that file
                logger.debug("Could not stat file %s for warnings", f)

        return self._files, warnings

    def _group_files_by_directory(self) -> None:
        """Group files by directory patterns from self.patterns."""
        # Identify directory patterns
        directory_patterns = set()
        for pattern in self.patterns:
            # Check if pattern represents a directory
            if "*" not in pattern and "?" not in pattern:
                path_obj = self.root / pattern
                if path_obj.exists() and path_obj.is_dir():
                    # Normalize to include trailing slash
                    dir_pattern = pattern.rstrip("/") + "/"
                    directory_patterns.add(dir_pattern)
            elif pattern.endswith("/") and "*" not in pattern and "?" not in pattern:
                directory_patterns.add(pattern)
            elif "**" in pattern:
                # Handle glob patterns that represent directory structures
                # e.g., "journal/**" -> "journal/"
                # e.g., "test_journal/journal/**" -> "test_journal/journal/"
                if pattern.endswith("/**"):
                    dir_pattern = pattern[:-3] + "/"  # Remove /** and add /
                    directory_patterns.add(dir_pattern)
                elif "/**" in pattern:
                    # Extract the directory part before /**
                    dir_part = pattern.split("/**")[0]
                    if dir_part:
                        dir_pattern = dir_part.rstrip("/") + "/"
                        directory_patterns.add(dir_pattern)

        # Group files by matching directory patterns
        self._files_by_directory.clear()
        # Add the ungrouped key.
        self._files_by_directory[None] = []

        for file_path in self._files:
            try:
                rel_path = file_path.relative_to(self.root)
            except Exception:
                rel_path = file_path.name

            rel_str = str(rel_path)

            # Check if this file belongs to any directory pattern.
            groupped = False
            for dir_pattern in directory_patterns:
                dir_path = dir_pattern.rstrip("/")
                if rel_str.startswith(dir_path + "/") or rel_str == dir_path:
                    if dir_pattern not in self._files_by_directory:
                        self._files_by_directory[dir_pattern] = []
                    self._files_by_directory[dir_pattern].append(file_path)
                    groupped = True
                    break
            if not groupped:
                self._files_by_directory[None].append(file_path)

    def _generate_toc(self) -> None:
        """Generate table of contents for the bundled files.

        Args:
            files (list[Path]):
                List of files to include in the TOC.
        """
        if not self._files:
            logger.error("No files collected for TOC generation.")
            return

        self.add_header("Table of Contents", level=2)

        # Collect unique entries - individual files with descriptions get their own entries
        toc_entries = set()
        individual_files = set()

        for f in self._files:
            try:
                rel = f.relative_to(self.root)
            except Exception:
                rel = f.name
            rel_str = str(rel)

            # Check if this specific file has a description
            if self.toc and self.toc.get(rel_str.lower()):
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
            if self.toc:
                description = self.toc.get(entry.lower())

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

        self._files = _collect_files(
            self.root,
            self.patterns,
            self.include_hidden,
            self.max_file_size,
            self.excludes,
        )

        if not self._files:
            self.add_text("No files found. Nothing to bundle.")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(self.get_text(), encoding="utf-8")
            return output

        # Generate table of contents if requested
        if self.generate_toc:
            self._generate_toc()

        # Group files by directory for hierarchical output
        self._group_files_by_directory()

        # Process grouped files (directory headers)
        for dir_pattern, dir_files in self._files_by_directory.items():
            if dir_pattern is None:
                continue  # Handle ungrouped files separately

            if not dir_files:
                continue

            # Calculate directory statistics
            total_size = 0
            for f in dir_files:
                try:
                    if f.exists():
                        total_size += f.stat().st_size
                except OSError:
                    pass
            dir_name = dir_pattern.rstrip("/")

            # Create directory header
            self.add_header(
                f"Folder: `{dir_name}/` (Files: {len(dir_files)}, Size: {total_size:,} bytes)",
                level=2,
            )

            # Process files in this directory
            for f in sorted(dir_files):
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
                except Exception:
                    continue

                # Choose fenced block language by suffix
                suffix = f.suffix.lower().lstrip(".")
                fence_lang = suffix if suffix else "text"
                self.add_header(f"File: `{rel_str}` (Size: {file_size} bytes)", level=3)
                self.add_text(f"```{fence_lang}")
                self.add_text(content.rstrip())
                self.add_text("```")

        # Process ungrouped files (individual files not under directory headers)
        ungrouped_files = self._files_by_directory.get(None, [])
        for f in ungrouped_files:
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
            except Exception:
                continue

            # Choose fenced block language by suffix
            suffix = f.suffix.lower().lstrip(".")
            fence_lang = suffix if suffix else "text"
            self.add_header(f"File: `{rel_str}` (Size: {file_size} bytes)", level=2)
            self.add_text(f"```{fence_lang}")
            self.add_text(content.rstrip())
            self.add_text("```")

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.get_text(), encoding="utf-8")
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
        "--excludes",
        type=str,
        default="",
        help="Comma-separated glob patterns to excludes (applied after inclusion patterns)",
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
        default=64 * 1024,
        help="File size threshold in bytes for LLM context window warnings (default: 64KB)",
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
        const=DEFAULT_CONFIG_FILENAME,
        help=f"Generate a starter bundler configuration file (default: {DEFAULT_CONFIG_FILENAME})",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_FILENAME,
        help=f"Configuration file(s) to use (can be specified multiple times).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="BUNDLE.md",
        help="Output markdown filename",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
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

    # Set up logging.
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # Get the root path.
    root = Path(args.root).resolve()

    if not root.exists():
        logger.error("Root directory '%s' does not exist.", root)
        return 1

    # Parse patterns - use None if default was used (to allow config loading)
    patterns: list[str] = [p.strip() for p in args.patterns.split(",") if p.strip()]
    excludes: list[str] = [p.strip() for p in args.excludes.split(",") if p.strip()]

    # Handle config generation
    if args.generate_config:
        return _generate_config(
            root=root,
            patterns=patterns,
            config_filename=args.generate_config,
            excludes=excludes,
        )

    bundler = PyBundler(
        root,
        patterns=patterns,
        max_file_size=args.max_size,
        include_hidden=args.include_hidden,
        warn_size=args.warn_size,
        generate_toc=args.toc,
        config_file=args.config,
        excludes=excludes,
    )
    out_path = Path(args.output)

    # Format patterns for display
    if len(bundler.patterns) <= 3:
        patterns_display = str(bundler.patterns)
    else:
        patterns_display = f"[{bundler.patterns[0]}, {bundler.patterns[1]}, {bundler.patterns[2]}, ...] ({len(bundler.patterns)} total)"

    logger.info("Bundling files from %s using patterns: %s", root, patterns_display)

    output = bundler.bundle(out_path)

    # Collect files and check for size warnings
    _, warnings = bundler.collect_files_with_warnings()

    # Log warnings for large files
    for warning in warnings:
        logger.warning("%s", warning)

    if warnings:
        logger.warning(
            "%d file(s) exceed the warning threshold of %d bytes.",
            len(warnings),
            args.warn_size,
        )
        logger.info(
            "Consider using --max-size to filter out large files or adjust --warn-size."
        )

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

    logger.info(
        "Created bundle: %s (%.2f KB, %d lines, %d words, ~%d tokens)",
        output,
        file_size_kb,
        line_count,
        word_count,
        token_estimate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
