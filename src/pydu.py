#!/usr/bin/env python3
"""
pydu - Improved Disk Usage Analyzer

A tree-based disk usage analyzer with coloring, size filtering, and various
output options. Similar to 'du' but with enhanced tree visualization and
filtering capabilities. Hidden files and directories are excluded by default.
"""

import argparse
import fnmatch
import logging
import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class TreeNode:
    """Represents a file or directory node in the disk usage tree.

    Attributes:
        name (str):
            The name of the file or directory.
        size (int):
            The total size in bytes (for directories, includes all contents).
        mtime (float):
            The modification time as a Unix timestamp.
        children (list[TreeNode] | None):
            List of child nodes (None for files).

    """

    name: str
    size: int
    mtime: float
    children: list["TreeNode"] | None = None


class ColorFormatter(logging.Formatter):
    """
    Custom logging formatter that adds ANSI color codes to log messages.

    This formatter applies color coding based on log levels for better
    readability in terminal output.
    """

    COLORS = {
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        if color:
            message = f"{color}{message}{self.RESET}"
        return message


def human_size_parts(size: float) -> tuple[str, str]:
    """
    Convert a size in bytes to human-readable format with appropriate unit.

    Args:
        size (int | float):
            Size in bytes.

    Returns:
        tuple[str, str]:
            Tuple of (formatted_size, unit) where formatted_size is a string
            with 2 decimal places and unit is one of B, KB, MB, GB, TB

    Examples:
        >>> human_size_parts(1024)
        ('1.00', 'KB')
        >>> human_size_parts(1536)
        ('1.50', 'KB')

    """
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size_f = float(size)
    while size_f >= 1024 and i < len(units) - 1:
        size_f /= 1024
        i += 1
    return f"{size_f:.2f}", units[i]


def parse_size(size_str: str) -> int:
    """
    Parse human-readable size string into bytes.

    Supports formats like '100MB', '1.5GB', '500KB', or plain numbers for bytes.

    Args:
        size_str (str):
            Size string to parse (e.g., '100MB', '1.5GB', '500KB', '1024')

    Returns:
        int:
            Size in bytes as integer

    Raises:
        ValueError: If the size string format is invalid

    Examples:
        >>> parse_size('1MB')
        1048576
        >>> parse_size('1.5GB')
        1610612736
        >>> parse_size('1024')
        1024

    """
    if not size_str:
        return 0

    size_str = size_str.strip().upper()

    # Handle plain numbers as bytes
    try:
        return int(float(size_str))
    except ValueError:
        pass

    # Parse with units - find the longest matching unit
    units = {
        "TB": 1024**4,
        "GB": 1024**3,
        "MB": 1024**2,
        "KB": 1024,
        "B": 1,
    }

    for unit, multiplier in sorted(
        units.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if size_str.endswith(unit):
            try:
                value_str = size_str[: -len(unit)].strip()
                value = float(value_str)
                return int(value * multiplier)
            except ValueError:
                continue

    raise ValueError(
        f"Invalid size format: '{size_str}'. "
        f"Use formats like '100MB', '1.5GB', '500KB', or plain numbers for bytes."
    )


def color_size(size: int, human: bool, use_color: bool) -> str:
    """
    Format a size with optional human-readable conversion and ANSI coloring.

    Args:
        size (int):
            Size in bytes
        human (bool):
            Whether to use human-readable format (e.g., "1.2 MB")
        use_color (bool):
            Whether to apply ANSI color codes

    Returns:
        str:
            Formatted size string, optionally colored

    Note:
        Color scheme: - B: gray - KB: green - MB: yellow - GB: red - TB: magenta
        - bytes (no unit): cyan

    """
    if not human:
        s = str(size)
        return colored(s, "36") if use_color else s
    size_str, unit = human_size_parts(size=size)
    s = f"{size_str} {unit}"
    if not use_color:
        return s
    unit_colors = {
        "B": "90",  # gray
        "KB": "32",  # green
        "MB": "33",  # yellow
        "GB": "31",  # red
        "TB": "35",  # magenta
    }
    col = unit_colors.get(unit, "37")  # white fallback
    return colored(s, col)


def colored(text: str, color_code: str) -> str:
    """
    Apply ANSI color codes to text.

    Args:
        text (str):
            Text to color
        color_code (str):
            ANSI color code (e.g., "31" for red, "32" for green)

    Returns:
        str:
            Text wrapped with ANSI color codes

    Note:
        Always resets color to default at the end

    """
    return f"\033[{color_code}m{text}\033[0m"


def size_bar(
    size: int,
    max_size: int,
    width: int = 10,
    use_color: bool = True,
) -> str:
    """
    Generate a visual size bar showing relative size.

    Args:
        size (int):
            Current item size in bytes
        max_size (int):
            Maximum size in the current level (for scaling)
        width (int):
            Width of the size bar in characters
        use_color (bool):
            Whether to use colored bars

    Returns:
        str:
            String containing the visual size bar and optional percentage

    """
    if max_size > 0:
        # Calculate filled portion
        ratio = max(0.0, min(size / max_size, 1.0))
        filled = min(int(ratio * width), width)
        # Create bar with filled and empty portions
        bar = "█" * filled + "░" * (width - filled)
        # Color the bar based on size ratio
        if use_color:
            if ratio >= 0.8:
                bar = colored(bar, "31")  # Red for large items
            elif ratio >= 0.6:
                bar = colored(bar, "33")  # Yellow for medium items
            elif ratio >= 0.3:
                bar = colored(bar, "32")  # Green for small items
            else:
                bar = colored(bar, "90")  # Gray for very small items
    else:
        bar = "░" * width
        ratio = 0.0

    return f"[{bar}{ratio:4.0%}]"


def build_tree(
    path: str,
    excludes: list[str] | None = None,
    includes: list[str] | None = None,
    show_all: bool = False,
    min_size: int | None = None,
    max_size: int | None = None,
) -> TreeNode:
    """
    Recursively build a tree structure representing directory contents with filtering options.

    Args:
        path (str):
            Directory path to analyze
        excludes (list[str] | None):
            List of fnmatch patterns to exclude
        includes (list[str] | None):
            List of fnmatch patterns to include (if specified, only matching items shown)
        show_all (bool):
            Whether to include files in output (not just directories)
        min_size (int | None):
            Minimum size threshold in bytes (None for no minimum)
        max_size (int | None):
            Maximum size threshold in bytes (None for no maximum)

    Returns:
        TreeNode:
            Root TreeNode representing the directory/file at the given path,
            containing its total size and child nodes
    Note:
        Hidden files and directories (starting with '.') are excluded by default.
        Directories are filtered based on their total size (including contents).
        Files are filtered based on their individual size. Symlinks are counted
        by their link size, not target size.

    """
    if excludes is None:
        excludes = []
    if includes is None:
        includes = []
    total_size = 0
    children: list[TreeNode] = []

    try:
        with os.scandir(path) as it:
            for entry in it:
                # Check inclusion patterns first (if specified, only include matching items)
                if includes and not any(
                    fnmatch.fnmatch(entry.name, patt) for patt in includes
                ):
                    continue

                # Skip hidden files and directories by default (unless explicitly included)
                if entry.name.startswith(".") and not includes:
                    continue

                # Check exclusion patterns
                skip = any(fnmatch.fnmatch(entry.name, patt) for patt in excludes)
                if skip:
                    continue

                lstat = entry.stat(follow_symlinks=False)

                if entry.is_symlink():
                    # Count symlink size but don't traverse
                    total_size += lstat.st_size
                    continue

                if entry.is_file():
                    file_size = lstat.st_size
                    file_mtime = lstat.st_mtime
                    total_size += file_size
                    # Filter files by size if specified and show_all is enabled
                    if (
                        show_all
                        and (min_size is None or file_size >= min_size)
                        and (max_size is None or file_size <= max_size)
                    ):
                        children.append(
                            TreeNode(
                                name=entry.name,
                                size=file_size,
                                mtime=file_mtime,
                                children=None,
                            )
                        )

                elif entry.is_dir():
                    # Recursively scan subdirectory
                    sub_node = build_tree(
                        path=entry.path,
                        excludes=excludes,
                        includes=None,
                        show_all=show_all,
                        min_size=min_size,
                        max_size=max_size,
                    )
                    total_size += sub_node.size
                    # Filter directories by total size if specified
                    if (min_size is None or sub_node.size >= min_size) and (
                        max_size is None or sub_node.size <= max_size
                    ):
                        children.append(sub_node)

    except PermissionError:
        logging.exception(f"Permission denied: {path}")
    except OSError as e:
        logging.exception(f"Error accessing {path}: {e}")

    # Get stats for the current path
    try:
        path_stat = os.stat(path, follow_symlinks=False)
        path_mtime = path_stat.st_mtime
        # For files, total_size is already the file size
        # For directories, total_size includes all children
    except OSError:
        path_mtime = 0.0

    return TreeNode(
        name=os.path.basename(path) or "/",
        size=total_size,
        mtime=path_mtime,
        children=children if children else None,
    )


def print_tree(
    name: str,
    size: int,
    children: list[TreeNode] | None,
    prefix: str = "",
    depth: int = 0,
    max_depth: int | None = None,
    sort_by: str = "name",
    reverse: bool = False,
    human: bool = False,
    color: bool = False,
    size_bars: bool = False,
    max_size_in_level: int = 0,
    parent_total_size: int = 0,
    is_last: bool = True,
) -> None:
    """
    Recursively print a directory tree with size information.

    Args:
        name (str):
            Name of the current item (file or directory)
        size (int):
            Size of the current item in bytes
        children (list[TreeNode] | None):
            List of child TreeNode objects (None for files)
        prefix (str):
            Current tree prefix string for indentation
        depth (int):
            Current depth in the tree
        max_depth (int | None):
            Maximum depth to display (None for unlimited)
        sort_by (str):
            Sort criteria ("name" or "size")
        reverse (bool):
            Whether to reverse the sort order
        human (bool):
            Whether to use human-readable size format
        color (bool):
            Whether to use ANSI color codes
        size_bars (bool):
            Whether to display visual size bars
        max_size_in_level (int):
            Maximum size among siblings at this level (for bar scaling)
        parent_total_size (int):
            Total size of parent directory (for percentage calculation)
        is_last (bool):
            Whether this is the last item in its parent's list

    """
    if max_depth is not None and depth > max_depth:
        return

    # Create tree branch symbol
    line_prefix = prefix + ("└── " if is_last else "├── ")

    # Format size with optional coloring
    s = color_size(size, human, color)

    # Add size bar if enabled
    bar_str = ""
    if size_bars and max_size_in_level > 0:
        pct_size = parent_total_size if parent_total_size > 0 else max_size_in_level
        bar = size_bar(
            size=size,
            max_size=pct_size,
            width=10,
            use_color=color,
        )
        bar_str = f"{bar} "

    # Print current item
    if children is None:
        # File
        print(f"{bar_str}{line_prefix}{name} {s}")
    else:
        # Directory
        print(f"{bar_str}{line_prefix}{name}/ {s}")

    # Stop recursion if no children or max depth reached
    if children is None or (max_depth is not None and depth >= max_depth):
        return

    # Prepare prefix for child items
    new_prefix = prefix + ("    " if is_last else "│   ")

    # Sort children based on criteria
    if children:
        if sort_by == "size":
            key_func = lambda node: node.size
        elif sort_by == "mtime":
            key_func = lambda node: node.mtime
        else:  # name
            key_func = lambda node: node.name
        children.sort(key=key_func, reverse=reverse)

    # Calculate max size for this level (for size bars)
    level_max_size = (
        max((node.size for node in children), default=0)
        if size_bars and children
        else 0
    )

    # Recursively print children
    if children:
        for i, node in enumerate(children):
            sub_last = i == len(children) - 1
            print_tree(
                name=node.name,
                size=node.size,
                children=node.children,
                prefix=new_prefix,
                depth=depth + 1,
                max_depth=max_depth,
                sort_by=sort_by,
                reverse=reverse,
                human=human,
                color=color,
                size_bars=size_bars,
                max_size_in_level=level_max_size,
                parent_total_size=size,
                is_last=sub_last,
            )


def main() -> None:
    """
    Main entry point for the mydu disk usage analyzer.

    Parses command-line arguments, scans the specified directory, and prints the
    results in tree format with optional filtering and formatting.
    """
    parser = argparse.ArgumentParser(
        description="Improved disk usage analyzer with tree-like output and coloring.",
        conflict_handler="resolve",
        add_help=False,  # Disable automatic -h for help so we can use -h for human-readable
    )
    parser.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to analyze (default: current directory)",
    )
    parser.add_argument(
        "-a", "--all", action="store_true", help="Show files as well as directories"
    )
    parser.add_argument(
        "-H",
        "--human-readable",
        action="store_true",
        help="Print sizes in human-readable format (e.g., 1.2 MB)",
    )
    parser.add_argument(
        "-d",
        "--max-depth",
        type=int,
        default=None,
        help="Limit the display to the specified depth",
    )
    parser.add_argument(
        "-s",
        "--summarize",
        action="store_true",
        help="Display only a total for the specified path",
    )
    parser.add_argument(
        "-S",
        "--sort-by",
        choices=["name", "size", "mtime"],
        default="name",
        help="Sort output by name, size, or modification time (default: name)",
    )
    parser.add_argument(
        "-r", "--reverse", action="store_true", help="Reverse the sort order"
    )
    parser.add_argument(
        "-x",
        "--exclude",
        action="append",
        default=[],
        help="Exclude files/directories matching the pattern (can be used multiple times)",
    )
    parser.add_argument(
        "-i",
        "--include",
        action="append",
        default=[],
        help="Include only files/directories matching the pattern (can be used multiple times)",
    )
    parser.add_argument(
        "-m",
        "--min-size",
        type=str,
        default=None,
        help="Only show items larger than this size (e.g., '100MB', '1.5GB', '500KB')",
    )
    parser.add_argument(
        "-M",
        "--max-size",
        type=str,
        default=None,
        help="Only show items smaller than this size (e.g., '100MB', '1.5GB', '500KB')",
    )
    parser.add_argument(
        "-b",
        "--size-bars",
        action="store_true",
        help="Display visual size bars showing relative sizes",
    )
    parser.add_argument(
        "-c",
        "--color",
        choices=["always", "never", "auto"],
        default="auto",
        help="Use colors in output (default: auto)",
    )

    args = parser.parse_args()

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    # Determine if colors should be used
    use_color = args.color == "always" or (args.color == "auto" and sys.stdout.isatty())

    # Parse size filters with error handling
    min_size: int | None = None
    max_size: int | None = None
    try:
        if args.min_size:
            min_size = parse_size(args.min_size)
        if args.max_size:
            max_size = parse_size(args.max_size)
    except ValueError as e:
        logging.exception(f"Error: {e}")
        sys.exit(1)

    # Get absolute path and scan directory
    abs_path = os.path.abspath(args.path)
    root_node = build_tree(
        abs_path,
        args.exclude,
        args.include,
        args.all,
        min_size,
        max_size,
    )

    # Output results
    if args.summarize:
        s = color_size(root_node.size, args.human_readable, use_color)
        print(f"{s} {args.path}")
    else:
        root_max_size = (
            max((node.size for node in root_node.children), default=root_node.size)
            if root_node.children
            else root_node.size
        )
        print_tree(
            root_node.name,
            root_node.size,
            root_node.children,
            depth=0,
            max_depth=args.max_depth,
            sort_by=args.sort_by,
            reverse=args.reverse,
            human=args.human_readable,
            color=use_color,
            size_bars=args.size_bars,
            max_size_in_level=root_max_size,
            parent_total_size=0,
            is_last=True,
        )


if __name__ == "__main__":
    main()
