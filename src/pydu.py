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
from dataclasses import dataclass, asdict, field
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
    size: int
    mtime: float
    children: list["TreeNode"] = field(default_factory=list)
    parent: "TreeNode | None" = None
    node_type: NodeType = NodeType.DIRECTORY

    def get_path(self) -> str:
        """Get the full path from root to this node."""
        if self.parent is None:
            return self.name
        return os.path.join(self.parent.get_path(), self.name)


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


def build_tree(
    path: str,
    excludes: list[str] | None = None,
    includes: list[str] | None = None,
    show_all: bool = False,
    min_size: int | None = None,
    max_size: int | None = None,
    older_than: float | None = None,
    progress_callback: Callable[[], None] | None = None,
    parent: TreeNode | None = None,
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
        older_than (float | None):
            Only include items older than this many seconds (None for no age filter)
        progress_callback (Callable[[], None] | None):
            Callback function to call for progress updates (None for no progress)

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
    current_time = time.time()
    age_threshold = current_time - (older_than or 0)

    # Create the TreeNode first so we can pass it as parent to recursive calls
    node = TreeNode(
        name=os.path.basename(path) or "/",
        size=0,
        mtime=0.0,
        children=[],
        parent=parent,
    )

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
                    # Count symlink size but don't traverse.
                    node.size += lstat.st_size
                    # Filter symlinks if show_all is enabled
                    if show_all:
                        node.children.append(
                            TreeNode(
                                name=entry.name,
                                size=lstat.st_size,
                                mtime=lstat.st_mtime,
                                children=[],
                                parent=node,
                                node_type=NodeType.SYMLINK,
                            )
                        )
                    continue

                if entry.is_file():
                    file_size = lstat.st_size
                    file_mtime = lstat.st_mtime
                    # Check age filter
                    if older_than and file_mtime > age_threshold:
                        continue
                    node.size += file_size
                    if progress_callback:
                        progress_callback()
                    # Filter files by size if specified and show_all is enabled
                    if (
                        show_all
                        and (min_size is None or file_size >= min_size)
                        and (max_size is None or file_size <= max_size)
                    ):
                        node.children.append(
                            TreeNode(
                                name=entry.name,
                                size=file_size,
                                mtime=file_mtime,
                                children=[],
                                parent=node,
                                node_type=NodeType.FILE,
                            )
                        )

                elif entry.is_dir():
                    # Check age filter for directory
                    dir_mtime = lstat.st_mtime
                    if older_than and dir_mtime > age_threshold:
                        continue
                    if progress_callback:
                        progress_callback()
                    # Recursively scan subdirectory
                    sub_node = build_tree(
                        path=entry.path,
                        excludes=excludes,
                        includes=None,
                        show_all=show_all,
                        min_size=min_size,
                        max_size=max_size,
                        older_than=older_than,
                        progress_callback=progress_callback,
                        parent=node,  # Pass current node as parent
                    )
                    node.size += sub_node.size
                    # Filter directories by total size if specified
                    if (min_size is None or sub_node.size >= min_size) and (
                        max_size is None or sub_node.size <= max_size
                    ):
                        node.children.append(sub_node)

    except PermissionError:
        logger.exception(f"Permission denied: {path}")
    except OSError:
        logger.exception(f"Error accessing {path}")

    # Update node with final mtime
    try:
        path_stat = os.stat(path, follow_symlinks=False)
        node.mtime = path_stat.st_mtime
    except OSError:
        node.mtime = 0.0

    return node


def sort_key_size(node: TreeNode) -> int:
    """Sort key function for sorting by size."""
    return node.size


def sort_key_mtime(node: TreeNode) -> float:
    """Sort key function for sorting by modification time."""
    return node.mtime


def sort_key_name(node: TreeNode) -> str:
    """Sort key function for sorting by name."""
    return node.name


def compute_depth_totals(node: TreeNode, max_depth: int) -> dict[int, int]:
    """
    Compute total sizes grouped by directory depth.

    Args:
        node (TreeNode): Root node to analyze
        max_depth (int): Maximum depth to compute totals for

    Returns:
        dict[int, int]: Dictionary mapping depth to total size at that depth
    """
    totals = {}

    def traverse(current_node: TreeNode, depth: int):
        if depth > max_depth:
            return
        if depth not in totals:
            totals[depth] = 0
        totals[depth] += current_node.size

        if current_node.children:
            for child in current_node.children:
                traverse(child, depth + 1)

    traverse(node, 0)
    return totals


def compute_file_types(node: TreeNode) -> dict[str, int]:
    """
    Compute cumulative sizes by file extension.

    Args:
        node (TreeNode): Root node to analyze

    Returns:
        dict[str, int]: Dictionary mapping file extension to total size
    """
    types = {}

    def traverse(current_node: TreeNode):
        if current_node.node_type == NodeType.FILE:
            # It's a file
            _, ext = os.path.splitext(current_node.name)
            ext = ext.lower() if ext else "(no extension)"
            if ext not in types:
                types[ext] = 0
            types[ext] += current_node.size
        elif current_node.children:
            for child in current_node.children:
                traverse(child)

    traverse(node)
    return types


def collect_all_nodes(node: TreeNode, include_files: bool = True) -> list[TreeNode]:
    """
    Collect all nodes in the tree for top-N analysis.

    Args:
        node (TreeNode): Root node to analyze
        include_files (bool): Whether to include files in the collection

    Returns:
        list[TreeNode]: List of all nodes
    """
    nodes = []

    def traverse(current_node: TreeNode):
        if current_node.node_type == NodeType.FILE:
            if include_files:
                nodes.append(current_node)
        else:
            nodes.append(current_node)
            if current_node.children:
                for child in current_node.children:
                    traverse(child)

    traverse(node)
    return nodes


def export_to_json(node: TreeNode) -> str:
    """
    Export tree structure to JSON.

    Args:
        node (TreeNode): Root node to export

    Returns:
        str: JSON representation of the tree
    """
    return json.dumps(asdict(node), indent=2)


def export_to_csv(node: TreeNode, include_files: bool = True) -> str:
    """
    Export tree structure to CSV format.

    Args:
        node (TreeNode): Root node to export
        include_files (bool): Whether to include files in export

    Returns:
        str: CSV representation of the tree data
    """
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["path", "size", "mtime", "type"])

    def traverse(current_node: TreeNode, path: str = ""):
        current_path = (
            os.path.join(path, current_node.name) if path else current_node.name
        )
        node_type = current_node.node_type.value
        if current_node.node_type == NodeType.FILE:
            if include_files:
                writer.writerow(
                    [current_path, current_node.size, current_node.mtime, node_type]
                )
        else:
            writer.writerow(
                [current_path, current_node.size, current_node.mtime, node_type]
            )
            if current_node.children:
                for child in current_node.children:
                    traverse(child, current_path)

    traverse(node)
    return output.getvalue()


def print_tree(
    node: TreeNode,
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
        node (TreeNode):
            TreeNode to print
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

    # Create tree branch symbol (only for non-root nodes), and also update prefix.
    if depth == 0:
        new_prefix = ""
        line_prefix = ""
    else:
        new_prefix = prefix + ("   " if is_last else "│  ")
        line_prefix = prefix + ("└─ " if is_last else "├─ ")

    # Format size with optional coloring
    s = color_size(node.size, human, color)

    # Add size bar if enabled
    bar_str = ""
    if size_bars and max_size_in_level > 0:
        pct_size = parent_total_size if parent_total_size > 0 else max_size_in_level
        bar = size_bar(
            size=node.size,
            max_size=pct_size,
            width=10,
            use_color=color,
        )
        bar_str = f"{bar} "

    # Print current item
    colored_name = color_name(node.name, node.node_type, color)
    if node.node_type == NodeType.FILE:
        print(f"{bar_str}{line_prefix}{colored_name} {s}")
    elif node.node_type == NodeType.SYMLINK:
        print(f"{bar_str}{line_prefix}{colored_name} {s}")
    else:
        print(f"{bar_str}{line_prefix}{colored_name}/ {s}")

    # Stop recursion if no children or max depth reached
    if node.node_type == NodeType.FILE or (
        max_depth is not None and depth >= max_depth
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

    # Calculate max size for this level (for size bars)
    level_max_size = (
        max((child.size for child in node.children), default=0)
        if size_bars and node.children
        else 0
    )

    # Recursively print children
    if node.children:
        for i, child in enumerate(node.children):
            sub_last = i == len(node.children) - 1
            print_tree(
                node=child,
                prefix=new_prefix,
                depth=depth + 1,
                max_depth=max_depth,
                sort_by=sort_by,
                reverse=reverse,
                human=human,
                color=color,
                size_bars=size_bars,
                max_size_in_level=level_max_size,
                parent_total_size=node.size,
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
        "-a",
        "--all",
        action="store_true",
        help="Show files as well as directories",
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
    parser.add_argument(
        "-t",
        "--depth-totals",
        type=int,
        default=None,
        help="Show size totals grouped by directory level up to specified depth",
    )
    parser.add_argument(
        "--by-type",
        action="store_true",
        help="Show cumulative size by file extension",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show top N largest items (files and directories)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Export results as JSON",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export results as CSV",
    )
    parser.add_argument(
        "--older-than",
        type=str,
        default=None,
        help="Only show items older than specified duration (e.g., '30d', '1w', '2M')",
    )

    args = parser.parse_args()

    # Process comma-separated exclude and include patterns
    processed_excludes = []
    for exclude in args.exclude:
        processed_excludes.extend(exclude.split(","))
    args.exclude = [
        pattern.strip() for pattern in processed_excludes if pattern.strip()
    ]

    processed_includes = []
    for include in args.include:
        processed_includes.extend(include.split(","))
    args.include = [
        pattern.strip() for pattern in processed_includes if pattern.strip()
    ]

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Determine if colors should be used
    use_color = args.color == "always" or (args.color == "auto" and sys.stdout.isatty())

    # Parse size filters with error handling
    min_size: int | None = None
    max_size: int | None = None
    older_than: float | None = None
    try:
        if args.min_size:
            min_size = parse_size(args.min_size)
        if args.max_size:
            max_size = parse_size(args.max_size)
        if args.older_than:
            older_than = parse_time_duration(args.older_than)
    except ValueError:
        logger.exception("Error parsing size or time arguments")
        sys.exit(1)

    # Get absolute path and scan directory
    abs_path = os.path.abspath(args.path)

    # Validate that the path exists
    if not os.path.exists(abs_path):
        logger.error(f"The path '{abs_path}' does not exist.")
        sys.exit(1)

    # Set up progress tracking
    scanned_count = 0

    def progress_callback():
        nonlocal scanned_count
        scanned_count += 1
        print(f"\rScanned {scanned_count} items...", end="", file=sys.stderr)

    root_node = build_tree(
        abs_path,
        args.exclude,
        args.include,
        args.all,
        min_size,
        max_size,
        older_than,
        progress_callback,
    )

    # Clear progress line
    print(file=sys.stderr)

    # Output results
    if args.summarize:
        s = color_size(root_node.size, args.human_readable, use_color)
        print(f"{s} {args.path}")
    elif args.depth_totals is not None:
        # Show depth totals
        totals = compute_depth_totals(root_node, args.depth_totals)
        for depth in sorted(totals.keys()):
            s = color_size(totals[depth], args.human_readable, use_color)
            indent = "  " * depth
            print(f"{indent}Depth {depth}: {s}")
    elif args.by_type:
        # Show file type aggregation
        types = compute_file_types(root_node)
        for ext, size in sorted(types.items(), key=lambda x: x[1], reverse=True):
            s = color_size(size, args.human_readable, use_color)
            print(f"{ext}: {s}")
    elif args.top is not None:
        # Show top N largest items
        all_nodes = collect_all_nodes(root_node, include_files=args.all)
        sorted_nodes = sorted(all_nodes, key=lambda n: n.size, reverse=True)[: args.top]
        for i, node in enumerate(sorted_nodes, 1):
            s = color_size(node.size, args.human_readable, use_color)
            item_type = node.node_type.value
            print(f"{i}. {node.get_path()} ({item_type}): {s}")
    elif args.json:
        # Export as JSON
        print(export_to_json(root_node))
    elif args.csv:
        # Export as CSV
        print(export_to_csv(root_node, include_files=args.all))
    else:
        root_max_size = (
            max((node.size for node in root_node.children), default=root_node.size)
            if root_node.children
            else root_node.size
        )
        print_tree(
            node=root_node,
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
