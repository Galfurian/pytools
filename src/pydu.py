#!/usr/bin/env python3
"""
pydu - Improved Disk Usage Analyzer.

A tree-based disk usage analyzer with coloring, size filtering, and various
output options. Similar to 'du' but with enhanced tree visualization and
filtering capabilities. Hidden files and directories are excluded by default.
"""

import argparse
import csv
import fnmatch
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Enumeration of filesystem node types."""

    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"


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

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with colors.

        Args:
            record (logging.LogRecord):
                The log record to format.

        Returns:
            str:
                The formatted log message with ANSI color codes.

        """
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        if color:
            message = f"{color}{message}{self.RESET}"
        return message


@dataclass
class TreeNode:
    """Represents a file or directory node in the disk usage tree.

    Attributes:
        name (str):
            The name of the file or directory.
        depth (int):
            The depth of the node in the tree (root is 0).
        size (int):
            The total size in bytes (for directories, includes all contents).
        mtime (float):
            The modification time as a Unix timestamp.
        children (list[TreeNode]):
            List of child nodes (empty for files).
        parent (TreeNode | None):
            Parent node (None for root).
        node_type (NodeType):
            Type of node: FILE, DIRECTORY, or SYMLINK.

    """

    name: str
    depth: int
    size: int
    mtime: float
    children: list["TreeNode"] = field(default_factory=list)
    parent: "TreeNode | None" = None
    node_type: NodeType = NodeType.DIRECTORY

    def get_path(self) -> str:
        """
        Get the full path from root to this node.
        """
        if self.parent is None:
            return self.name
        return os.path.join(self.parent.get_path(), self.name)

    def get_max_size(self) -> int:
        """
        Get the maximum size among the children of this node.
        """
        return max((child.size for child in self.children), default=0)


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


def parse_time_duration(duration_str: str) -> float:
    """
    Parse human-readable time duration into seconds.

    Supports formats like '30d', '1w', '2M', '1y' for days, weeks, months, years.

    Args:
        duration_str (str):
            Duration string to parse (e.g., '30d', '1w', '2M', '1y')

    Returns:
        float:
            Duration in seconds

    Raises:
        ValueError: If the duration string format is invalid

    Examples:
        >>> parse_time_duration('30d')
        2592000.0
        >>> parse_time_duration('1w')
        604800.0

    """
    if not duration_str:
        return 0

    duration_str = duration_str.strip().upper()

    # Define time multipliers (approximate)
    multipliers = {
        "S": 1,  # seconds
        "M": 30 * 24 * 3600,  # months (30 days)
        "W": 7 * 24 * 3600,  # weeks
        "D": 24 * 3600,  # days
        "H": 3600,  # hours
        "Y": 365 * 24 * 3600,  # years (365 days)
    }

    for unit, multiplier in multipliers.items():
        if duration_str.endswith(unit):
            try:
                value_str = duration_str[: -len(unit)].strip()
                value = float(value_str)
                return value * multiplier
            except ValueError:
                continue

    raise ValueError(
        f"Invalid duration format: '{duration_str}'. "
        f"Use formats like '30d', '1w', '2M', '1y', '5h', '10s'."
    )


def colored(text: str, color_code: str, light: bool = False) -> str:
    """
    Apply ANSI color codes to text.

    Args:
        text (str):
            Text to color
        color_code (str):
            ANSI color code (e.g., "31" for red, "32" for green)
        light (bool):
            Whether to use light (bright) variant of the color

    Returns:
        str:
            Text wrapped with ANSI color codes

    Note:
        Always resets color to default at the end

    """
    if light:
        color_code = f"1;{color_code}"
    return f"\033[{color_code}m{text}\033[0m"


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


def color_name(name: str, node_type: NodeType, use_color: bool) -> str:
    """
    Color a node name based on its type.

    Args:
        name (str):
            The node name to color
        node_type (NodeType):
            Type of the node
        use_color (bool):
            Whether to apply colors

    Returns:
        str:
            Colored name if use_color is True, otherwise plain name

    Note:
        Color scheme: directories blue, files green, symlinks cyan

    """
    if not use_color:
        return name

    type_colors = {
        NodeType.DIRECTORY: "34",
        NodeType.SYMLINK: "33",
        NodeType.FILE: "37",
    }

    color_code = type_colors.get(node_type, "37")
    return colored(name, color_code, True)


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


def process_patterns(patterns):
    """
    Process comma-separated pattern strings into a list of stripped patterns.

    Args:
        patterns (list[str]):
            List of pattern strings, potentially comma-separated

    Returns:
        list[str]:
            List of stripped, non-empty patterns
    """
    processed_patterns = []
    for pattern in patterns:
        processed_patterns.extend(pattern.split(","))
    return [pattern.strip() for pattern in processed_patterns if pattern.strip()]


def build_tree(
    path: str,
    depth: int,
    parent: TreeNode | None = None,
    progress_callback: Callable[[], None] = lambda: None,
) -> TreeNode:
    """
    Recursively build a complete tree structure representing directory contents.

    Args:
        path (str):
            Directory path to analyze
        depth (int):
            Current depth in the tree (root is 0).
        parent (TreeNode | None):
            Parent node (None for root)
        progress_callback (Callable[[], None]):
            Callback function to report progress (called for each item scanned)

    Returns:
        TreeNode:
            Root TreeNode representing the directory/file at the given path,
            containing its total size and child nodes
    """
    # Create the TreeNode first so we can pass it as parent to recursive calls
    node = TreeNode(
        name=os.path.basename(path) or "/",
        depth=depth,
        size=0,
        mtime=0.0,
        children=[],
        parent=parent,
    )

    try:
        with os.scandir(path) as it:
            for entry in it:
                lstat = entry.stat(follow_symlinks=False)

                progress_callback()

                if entry.is_symlink():
                    # Count symlink size but don't traverse
                    node.size += lstat.st_size
                    node.children.append(
                        TreeNode(
                            name=entry.name,
                            depth=depth + 1,
                            size=lstat.st_size,
                            mtime=lstat.st_mtime,
                            children=[],
                            parent=node,
                            node_type=NodeType.SYMLINK,
                        )
                    )

                elif entry.is_file():
                    file_size = lstat.st_size
                    file_mtime = lstat.st_mtime
                    node.size += file_size
                    node.children.append(
                        TreeNode(
                            name=entry.name,
                            depth=depth + 1,
                            size=file_size,
                            mtime=file_mtime,
                            children=[],
                            parent=node,
                            node_type=NodeType.FILE,
                        )
                    )

                elif entry.is_dir():
                    sub_node = build_tree(
                        path=entry.path,
                        depth=depth + 1,
                        parent=node,
                        progress_callback=progress_callback,
                    )
                    node.size += sub_node.size
                    node.children.append(sub_node)
    except PermissionError:
        logger.exception(f"Permission denied: {path}")
    except OSError:
        logger.exception(f"Error accessing {path}")
    # Update node with final mtime.
    try:
        path_stat = os.stat(path, follow_symlinks=False)
        node.mtime = path_stat.st_mtime
        node.node_type = NodeType.DIRECTORY
    except OSError:
        node.mtime = 0.0

    return node


def node_matches(node: TreeNode, args: argparse.Namespace) -> bool:
    """
    Determine whether a node matches the filtering criteria provided in args.

    This checks include/exclude patterns, size and age filters. Directories
    are considered matching if they themselves match or if their children may
    match (handled by filter_tree). This function only evaluates the node's
    own properties.
    """
    # Include/exclude by name patterns
    name = node.name
    if args.exclude:
        for pat in args.exclude:
            if fnmatch.fnmatch(name, pat):
                return False

    if args.include and node.node_type != NodeType.DIRECTORY:
        matched = False
        for pat in args.include:
            if fnmatch.fnmatch(name, pat):
                matched = True
                break
        if not matched:
            return False

    # Hidden files/directories.
    if not args.show_hidden and name.startswith("."):
        return False

    # Size filters (only for files and symlinks)
    if node.node_type != NodeType.DIRECTORY:
        if getattr(args, "min_size", None) is not None:
            if node.size < args.min_size:
                return False
        if getattr(args, "max_size", None) is not None:
            if node.size > args.max_size:
                return False

    # Age filter (older-than): node.mtime is compared to current time
    if getattr(args, "older_than", None):
        cutoff = time.time() - args.older_than
        if node.mtime > cutoff:
            return False

    return True


def filter_tree(node: TreeNode, args: argparse.Namespace) -> TreeNode | None:
    """
    Recursively filter a TreeNode tree according to args.

    Returns the node if it matches the filters, otherwise returns None.

    For directories, children are filtered first, and the directory is kept
    only if it has children left after filtering.
    """
    # Files and symlinks: simple match
    if node.node_type in (NodeType.FILE, NodeType.SYMLINK):
        return node if node_matches(node, args) else None

    # For directories, filter children first
    new_children: list[TreeNode] = []
    for child in list(node.children):
        kept = filter_tree(child, args)
        if kept is not None:
            new_children.append(kept)

    node.children = new_children

    # Recompute directory size as sum of children sizes (since we may have
    # removed some children). Keep original mtime/name.
    node.size = sum((c.size for c in node.children), 0)

    # Keep directory only if it matches the filters and has children left
    if node_matches(node, args) and node.children:
        return node

    return None


def sort_key_size(node: TreeNode) -> int:
    """Sort key function for sorting by size."""
    return node.size


def sort_key_mtime(node: TreeNode) -> float:
    """Sort key function for sorting by modification time."""
    return node.mtime


def sort_key_name(node: TreeNode) -> str:
    """Sort key function for sorting by name."""
    return node.name


def print_tree(
    node: TreeNode,
    prefix: str = "",
    max_depth: int | None = None,
    sort_by: str = "name",
    reverse: bool = False,
    human: bool = False,
    use_color: bool = False,
    size_bars: bool = False,
    is_last: bool = True,
) -> None:
    """
    Recursively print a directory tree with size information.

    Args:
        node (TreeNode):
            TreeNode to print
        prefix (str):
            Current tree prefix string for indentation
        depth (int):
            Current depth in the tree
        max_depth (int | None):
            Maximum depth to display (None for unlimited)
        sort_by (str):
            Sort criteria ("name" or "size" or "mtime")
        reverse (bool):
            Whether to reverse the sort order
        human (bool):
            Whether to use human-readable size format
        use_color (bool):
            Whether to use ANSI color codes
        size_bars (bool):
            Whether to display visual size bars
        is_last (bool):
            Whether this is the last item in its parent's list

    """
    if max_depth is not None and node.depth > max_depth:
        return

    # Create tree branch symbol (only for non-root nodes), and also update prefix.
    if node.depth == 0:
        new_prefix = ""
        line_prefix = ""
    else:
        new_prefix = prefix + ("   " if is_last else "│  ")
        line_prefix = prefix + ("└─ " if is_last else "├─ ")

    # Format size with optional coloring
    node_size = color_size(node.size, human, use_color)

    # Get the parent total size if not provided.
    parent_size = node.parent.size if node.parent else node.size

    # Add size bar if enabled.
    bar_str = ""
    if size_bars:
        bar_str = f"{size_bar(node.size, parent_size, 10, use_color)} "

    # Print current item.
    colored_name = color_name(node.name, node.node_type, use_color)
    if node.node_type == NodeType.FILE:
        print(f"{bar_str}{line_prefix}{colored_name} {node_size}")
    elif node.node_type == NodeType.SYMLINK:
        print(f"{bar_str}{line_prefix}{colored_name} {node_size}")
    else:
        print(f"{bar_str}{line_prefix}{colored_name}/ {node_size}")

    # Stop recursion if no children or max depth reached
    if node.node_type == NodeType.FILE or (
        max_depth is not None and node.depth >= max_depth
    ):
        return

    # Sort children based on criteria
    if node.children:
        if sort_by == "size":
            node.children.sort(key=sort_key_size, reverse=reverse)
        elif sort_by == "mtime":
            node.children.sort(key=sort_key_mtime, reverse=reverse)
        else:  # name
            node.children.sort(key=sort_key_name, reverse=reverse)

    # Recursively print children
    if node.children:
        for i, child in enumerate(node.children):
            is_last_child = i == len(node.children) - 1
            print_tree(
                node=child,
                prefix=new_prefix,
                max_depth=max_depth,
                sort_by=sort_by,
                reverse=reverse,
                human=human,
                use_color=use_color,
                size_bars=size_bars,
                is_last=is_last_child,
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
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to analyze (default: current directory)",
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
        "--sort-by",
        choices=["name", "size", "mtime"],
        default="name",
        help="Sort output by name, size, or modification time (default: name)",
    )
    parser.add_argument(
        "-S",
        "--summarize",
        action="store_true",
        help="Display only a total for the specified path",
    )
    parser.add_argument(
        "-r",
        "--reverse",
        action="store_true",
        help="Reverse the sort order",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        action="append",
        default=[],
        help="Exclude files/directories matching the pattern (can be used multiple times or comma-separated)",
    )
    parser.add_argument(
        "-i",
        "--include",
        action="append",
        default=[],
        help="Include only files/directories matching the pattern (can be used multiple times or comma-separated)",
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
        "--older-than",
        type=str,
        default=None,
        help="Only show items older than specified duration (e.g., '30d', '1w', '2M')",
    )
    parser.add_argument(
        "-b",
        "--size-bars",
        action="store_true",
        help="Display visual size bars showing relative sizes",
    )
    parser.add_argument(
        "--show-hidden",
        default=False,
        action="store_true",
        help="Include hidden files and directories (default is to exclude them)",
    )
    parser.add_argument(
        "-c",
        "--color",
        default=True,
        action="store_true",
        help="Use colors in output",
    )

    args = parser.parse_args()

    # Process comma-separated exclude and include patterns.
    args.exclude = process_patterns(args.exclude)
    args.include = process_patterns(args.include)

    # Set up logging with color support.
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Determine if colors should be used
    args.color = args.color and sys.stdout.isatty()

    # Parse size filters with error handling.
    try:
        if args.min_size:
            args.min_size = parse_size(args.min_size)
        if args.max_size:
            args.max_size = parse_size(args.max_size)
        if args.older_than:
            args.older_than = parse_time_duration(args.older_than)
    except ValueError:
        logger.exception("Error parsing size or time arguments")
        sys.exit(1)

    # Get absolute path and scan directory
    abs_path = os.path.abspath(args.path)

    # Validate that the path exists
    if not os.path.exists(abs_path):
        logger.error(f"The path '{abs_path}' does not exist.")
        sys.exit(1)

    # Set up progress tracking.
    scanned_count = 0

    # Callback to update and print progress.
    def progress_callback() -> None:
        nonlocal scanned_count
        scanned_count += 1
        print(f"\rScanned {scanned_count} items...", end="", file=sys.stderr)

    # Build the directory tree.
    root_node = build_tree(
        path=abs_path,
        depth=0,
        parent=None,
        progress_callback=progress_callback,
    )
    # Clear progress line.
    print(file=sys.stderr)

    # Apply filtering to the built tree. build_tree stays agnostic of filters.
    filtered_root = filter_tree(root_node, args)
    if filtered_root is None:
        logger.info("No items match the given filters.")
        sys.exit(0)

    # Output results
    if args.summarize:
        total_size = color_size(root_node.size, args.human_readable, args.color)
        print(f"{total_size} {args.path}")
    else:
        print_tree(
            node=filtered_root,
            max_depth=args.max_depth,
            sort_by=args.sort_by,
            reverse=args.reverse,
            human=args.human_readable,
            use_color=args.color,
            size_bars=args.size_bars,
            is_last=True,
        )


if __name__ == "__main__":
    main()
