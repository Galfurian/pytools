#!/usr/bin/env python3
"""
pystats: Project-wide file statistics for LLM and developer workflow.

Counts lines, words, characters, tokens, code/comment/blank lines.
Highlights files approaching LLM context windows, supports filtering and output formats.
"""

import argparse
import os
import sys
import pathlib
import re
import json
import csv
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# -- ColorFormatter (matches your repo style) --
class ColorFormatter(logging.Formatter):
    COLORS = {
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


# ANSI color codes for table output
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RESET = "\033[0m"

# -- Defaults / Constants --
DEFAULT_EXTS = [".py", ".md", ".txt"]
LLM_LIMITS = [4096, 8192, 16384]
COMMENT_SYNTAX_MAP = {
    ".py": "#",
    ".js": "//",
    ".c": "//",
    ".cpp": "//",
    ".java": "//",
    ".sh": "#",
    ".rb": "#",
}


# -- Utility Functions --
def is_text_file(filepath):
    """Heuristically determines if a file is text (tries reading, checks for null bytes).

    Args:
        filepath (str):
            Path to the file to check.

    Returns:
        bool:
            True if the file appears to be text, False otherwise.
    """
    try:
        with open(filepath, "rb") as f:
            data = f.read(1024)
            if b"\x00" in data:
                return False
        with open(filepath, "r", encoding="utf-8") as f:
            f.read(1024)
        return True
    except Exception:
        return False


def word_count(text):
    """Count the number of words in the text.

    Args:
        text (str):
            The text to count words in.

    Returns:
        int:
            The number of words.
    """
    return len(re.findall(r"\b\w+\b", text))


def token_estimate(text):
    """Estimate the number of tokens in the text.

    Uses a simple heuristic of splitting on sequences of non-whitespace.

    Args:
        text (str):
            The text to estimate tokens for.

    Returns:
        int:
            The estimated number of tokens.
    """
    # Simple: split on sequences of non-whitespace for GPT-like counting
    return len(re.findall(r"\S+", text))


def detect_comment_lines(text, ext):
    """Detect the number of comment lines in the text based on file extension.

    Args:
        text (str):
            The text content of the file.
        ext (str):
            The file extension (e.g., '.py', '.js').

    Returns:
        int:
            The number of comment lines.
    """
    comment_prefix = COMMENT_SYNTAX_MAP.get(ext, None)
    if not comment_prefix:
        return 0
    return sum(
        1 for line in text.splitlines() if line.strip().startswith(comment_prefix)
    )


def safe_open(filepath):
    """Safely open and read a file, trying multiple encodings.

    Args:
        filepath (str):
            Path to the file to read.

    Returns:
        str:
            The file content as a string, or empty string if failed.
    """
    for encoding in ("utf-8", "latin-1", "ascii"):
        try:
            with open(filepath, "r", encoding=encoding) as f:
                return f.read()
        except Exception:
            continue
    return ""


# -- Statistics Storage Class --
class FileStats:
    def __init__(self, path):
        """Initialize FileStats with a file path.

        Args:
            path (str):
                Path to the file to analyze.
        """
        self.path = path
        self.ext = os.path.splitext(path)[1]
        self.lines = 0
        self.words = 0
        self.chars = 0
        self.tokens = 0
        self.blank_lines = 0
        self.comment_lines = 0
        self.code_lines = 0
        self.llm_status = ""
        self.relpath = ""

    def analyze(self, llm_limit):
        """Analyze the file and populate statistics.

        Args:
            llm_limit (int):
                Token limit for LLM context window check.
        """
        try:
            text = safe_open(self.path)
            self.chars = len(text)
            self.words = word_count(text)
            self.tokens = token_estimate(text)
            lines = text.splitlines()
            self.lines = len(lines)
            self.blank_lines = sum(1 for l in lines if not l.strip())
            self.comment_lines = detect_comment_lines(text, self.ext)
            self.code_lines = self.lines - self.blank_lines - self.comment_lines
            # LLM token window status
            if self.tokens >= llm_limit:
                self.llm_status = "EXCEEDS"
            elif self.tokens >= int(0.9 * llm_limit):
                self.llm_status = "WARNING"
            else:
                self.llm_status = "OK"
        except Exception as exc:
            logger.warning(f"Error analyzing {self.path}: {exc}")
            self.llm_status = "ERROR"

    def comment_percent(self):
        """Calculate the percentage of lines that are comments.

        Returns:
            float:
                Percentage of comment lines, rounded to 1 decimal place.
        """
        try:
            return round(100 * self.comment_lines / max(self.lines, 1), 1)
        except ZeroDivisionError:
            return 0.0

    def to_dict(self):
        """Convert the FileStats to a dictionary.

        Returns:
            dict:
                Dictionary representation of the file statistics.
        """
        return {
            "file": self.relpath or self.path,
            "lines": self.lines,
            "words": self.words,
            "chars": self.chars,
            "tokens": self.tokens,
            "blank_lines": self.blank_lines,
            "comment_lines": self.comment_lines,
            "code_lines": self.code_lines,
            "comment_percent": self.comment_percent(),
            "llm_status": self.llm_status,
        }


# -- Directory Traversal & Analysis --
def walk_directory(root, exts, exclude_dirs, include_hidden, force_binary, relpaths):
    """Walk the directory tree and collect FileStats for matching files.

    Args:
        root (str):
            Root directory to walk.
        exts (list[str]):
            List of file extensions to include (empty for all).
        exclude_dirs (list[str]):
            Directory names to exclude.
        include_hidden (bool):
            Whether to include hidden files and directories.
        force_binary (bool):
            Whether to force analysis of binary files.
        relpaths (bool):
            Whether to use relative paths in FileStats.

    Returns:
        list[FileStats]:
            List of FileStats objects for the matching files.
    """
    stats_list = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden directories unless --include-hidden
        if not include_hidden:
            dirnames[:] = [
                d for d in dirnames if not d.startswith(".") and d not in exclude_dirs
            ]
        else:
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        for fname in filenames:
            if not include_hidden and fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1]
            if exts and ext not in exts:
                continue
            fullpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fullpath, root) if relpaths else fullpath
            if not force_binary and not is_text_file(fullpath):
                logger.info(f"Skipping binary or unreadable file: {rel}")
                continue
            fs = FileStats(fullpath)
            fs.relpath = rel
            stats_list.append(fs)
    return stats_list


# -- Output Formatting --
def print_table(stats_list, llm_limit, group_ext, sort_by, summary):
    """Print a formatted table of file statistics.

    Args:
        stats_list (list[FileStats]):
            List of FileStats to display.
        llm_limit (int):
            Token limit for LLM context window.
        group_ext (bool):
            Whether to group files by extension.
        sort_by (str):
            Attribute to sort by (e.g., 'lines', 'tokens').
        summary (bool):
            Whether to show summary warnings.
    """
    # Determine max file name length for column width
    file_len = max(len(fs.relpath) for fs in stats_list) if stats_list else 4
    hdr = (
        "File",
        "Lines",
        "Words",
        "Chars",
        "Tokens",
        "Comments",
        "Blank",
        "Code",
        "%Comm",
        "LLM",
    )
    print(
        f"{hdr[0]:<{file_len}} {hdr[1]:>6} {hdr[2]:>6} {hdr[3]:>7} {hdr[4]:>7} {hdr[5]:>7} {hdr[6]:>6} {hdr[7]:>6} {hdr[8]:>7} {hdr[9]:>8}"
    )
    print("-" * (file_len + 74))
    stats_group = defaultdict(list)
    if group_ext:
        for fs in stats_list:
            stats_group[fs.ext].append(fs)
        for ext, group in sorted(stats_group.items()):
            print(f"[.{ext.lstrip('.')}]")
            for fs in sorted(group, key=lambda s: getattr(s, sort_by), reverse=True):
                line = f"{fs.relpath:<{file_len}} {fs.lines:6d} {fs.words:6d} {fs.chars:7d} {fs.tokens:7d} {fs.comment_lines:7d} {fs.blank_lines:6d} {fs.code_lines:6d} {fs.comment_percent():6.1f}%   {fs.llm_status:>8}"
                if fs.llm_status == "EXCEEDS":
                    print(f"{RED}{line}{RESET}")
                elif fs.llm_status == "WARNING":
                    print(f"{YELLOW}{line}{RESET}")
                elif fs.llm_status == "OK":
                    print(f"{GREEN}{line}{RESET}")
                else:
                    print(line)
    else:
        for fs in sorted(stats_list, key=lambda s: getattr(s, sort_by), reverse=True):
            line = f"{fs.relpath:<{file_len}} {fs.lines:6d} {fs.words:6d} {fs.chars:7d} {fs.tokens:7d} {fs.comment_lines:7d} {fs.blank_lines:6d} {fs.code_lines:6d} {fs.comment_percent():6.1f}%   {fs.llm_status:>8}"
            if fs.llm_status == "EXCEEDS":
                print(f"{RED}{line}{RESET}")
            elif fs.llm_status == "WARNING":
                print(f"{YELLOW}{line}{RESET}")
            elif fs.llm_status == "OK":
                print(f"{GREEN}{line}{RESET}")
            else:
                print(line)
    print("-" * (file_len + 74))
    # Summary
    total = FileStats("TOTAL")
    total.relpath = "TOTAL"
    for fs in stats_list:
        total.lines += fs.lines
        total.words += fs.words
        total.chars += fs.chars
        total.tokens += fs.tokens
        total.comment_lines += fs.comment_lines
        total.blank_lines += fs.blank_lines
        total.code_lines += fs.code_lines
    line = f"{total.relpath:<{file_len}} {total.lines:6d} {total.words:6d} {total.chars:7d} {total.tokens:7d} {total.comment_lines:7d} {total.blank_lines:6d} {total.code_lines:6d} {total.comment_percent():6.1f}%   "
    print(line)
    # LLM warning summary
    if summary:
        num_warn = sum(
            1 for fs in stats_list if fs.llm_status in ("EXCEEDS", "WARNING")
        )
        if num_warn:
            logger.warning(
                f"{num_warn} files approach or exceed the LLM context window ({llm_limit} tokens)"
            )


def write_json(stats_list, outfile):
    """Write file statistics to a JSON file.

    Args:
        stats_list (list[FileStats]):
            List of FileStats to write.
        outfile (str):
            Path to the output JSON file.
    """
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(json.dumps([fs.to_dict() for fs in stats_list], indent=2))


def write_csv(stats_list, outfile):
    """Write file statistics to a CSV file.

    Args:
        stats_list (list[FileStats]):
            List of FileStats to write.
        outfile (str):
            Path to the output CSV file.
    """
    hdr = [
        "file",
        "lines",
        "words",
        "chars",
        "tokens",
        "comment_lines",
        "blank_lines",
        "code_lines",
        "comment_percent",
        "llm_status",
    ]
    with open(outfile, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for fs in stats_list:
            w.writerow(fs.to_dict())


# -- Args Parsing and Main --
def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace:
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="pystats: file statistics tool for LLM and dev workflows."
    )
    parser.add_argument(
        "path", nargs="?", default=".", help="Root directory or file to analyze"
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=None,
        help="Comma-separated file extensions e.g. .py,.md",
    )
    parser.add_argument(
        "--exclude-dir",
        type=str,
        default="",
        help="Dirs to exclude (comma-separated e.g., venv,.git)",
    )
    parser.add_argument(
        "--include-hidden", action="store_true", help="Include hidden files and folders"
    )
    parser.add_argument(
        "--force-binary", action="store_true", help="Attempt to read binary files"
    )
    parser.add_argument(
        "--llm-limit",
        type=int,
        default=4096,
        help="LLM context token limit for warnings",
    )
    parser.add_argument(
        "--group-ext", action="store_true", help="Group output by file extension"
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="tokens",
        choices=["lines", "words", "chars", "tokens"],
        help="Sort output by metric",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Show only summary and LLM warnings"
    )
    parser.add_argument(
        "--json", type=str, default=None, help="Write JSON output to file"
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="Write CSV output to file"
    )
    parser.add_argument("--relpath", action="store_true", help="Show relative paths")
    parser.add_argument(
        "--min-words",
        type=int,
        default=None,
        help="Only show files with at least this many words",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Only show files with at most this many words",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=None,
        help="Only show files with at least this many lines",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Only show files with at most this many lines",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress file skip and error messages"
    )
    return parser.parse_args()


def main():
    """Main entry point for the pystats command-line tool.

    Parses arguments, analyzes files, and outputs statistics.

    Args:
        None

    Returns:
        None
    """
    # Setup logging with color
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    args = parse_args()

    root = args.path
    exts = [e.strip() for e in args.ext.split(",")] if args.ext else DEFAULT_EXTS
    exclude_dirs = [d.strip() for d in args.exclude_dir.split(",") if d.strip()]
    include_hidden = args.include_hidden
    force_binary = args.force_binary
    llm_limit = args.llm_limit
    group_ext = args.group_ext
    sort_by = args.sort
    relpaths = args.relpath
    summary = args.summary

    if not os.path.exists(root):
        logger.error(f"Path '{root}' does not exist!")
        sys.exit(1)

    # If path is a file: analyze just it; else traverse tree.
    stats_list = []
    if os.path.isfile(root):
        if not force_binary and not is_text_file(root):
            if not args.quiet:
                logger.info(f"Skipping binary or unreadable file: {root}")
        else:
            fs = FileStats(root)
            fs.relpath = os.path.basename(root) if relpaths else root
            fs.analyze(llm_limit)
            stats_list.append(fs)
    else:
        stats_list = walk_directory(
            root, exts, exclude_dirs, include_hidden, force_binary, relpaths
        )
        for fs in stats_list:
            fs.analyze(llm_limit)

    # Filter by lines and words if requested
    def stat_filter(fs):
        """Filter function to check if a FileStats meets the criteria.

        Args:
            fs (FileStats):
                The FileStats object to check.

        Returns:
            bool:
                True if the file meets the filter criteria, False otherwise.
        """
        if args.min_words is not None and fs.words < args.min_words:
            return False
        if args.max_words is not None and fs.words > args.max_words:
            return False
        if args.min_lines is not None and fs.lines < args.min_lines:
            return False
        if args.max_lines is not None and fs.lines > args.max_lines:
            return False
        return True

    stats_list = [fs for fs in stats_list if stat_filter(fs)]

    # Output
    if args.json:
        write_json(stats_list, args.json)
        logger.info(f"JSON written to {args.json}")
    if args.csv:
        write_csv(stats_list, args.csv)
        logger.info(f"CSV written to {args.csv}")

    if not summary:
        print_table(stats_list, llm_limit, group_ext, sort_by, summary)
    else:
        # Only show summary totals and LLM warnings.
        total = FileStats("TOTAL")
        for fs in stats_list:
            total.lines += fs.lines
            total.words += fs.words
            total.chars += fs.chars
            total.tokens += fs.tokens
            total.comment_lines += fs.comment_lines
            total.blank_lines += fs.blank_lines
            total.code_lines += fs.code_lines
        print(
            f"TOTAL: {total.lines} lines, {total.words} words, {total.chars} chars, {total.tokens} tokens, {total.comment_lines} comment, {total.blank_lines} blank, {total.code_lines} code"
        )
        num_warn = sum(
            1 for fs in stats_list if fs.llm_status in ("EXCEEDS", "WARNING")
        )
        if num_warn:
            logger.warning(
                f"{num_warn} files approach or exceed the LLM context window ({llm_limit} tokens)"
            )
        else:
            logger.info("No files exceed the LLM context window.")


if __name__ == "__main__":
    main()
