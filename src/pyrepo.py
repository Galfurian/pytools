#!/usr/bin/env python3

"""
Check git repositories for uncommitted changes and unpushed branches.

This script scans a directory tree for Git repositories and reports any
uncommitted changes or unpushed branches, helping maintain repository stability.
"""

import argparse
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


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


def is_git_repo(path):
    """
    Check if the given path is a Git repository.

    Args:
        path (str):
            Path to check for .git directory.

    Returns:
        bool:
            True if the path contains a .git directory, False otherwise.

    """
    return os.path.isdir(os.path.join(path, ".git"))


def get_uncommitted_changes(repo_path):
    """
    Get uncommitted changes in the Git repository.

    Args:
        repo_path (str):
            Path to the Git repository.

    Returns:
        str or None:
            String of uncommitted changes if any, None otherwise.

    """
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    changes = result.stdout.strip()
    return changes if changes else None


def get_unpushed_branches(repo_path):
    """
    Get branches with unpushed commits or no upstream set.

    Args:
        repo_path (str):
            Path to the Git repository.

    Returns:
        list[str] or None:
            List of unpushed branches if any, None otherwise.

    """
    # Get branches and their upstreams
    result = subprocess.run(
        [
            "git",
            "for-each-ref",
            "--format=%(refname:short) %(upstream:short)",
            "refs/heads",
        ],
        check=False,
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    branches = result.stdout.strip().split("\n")
    unpushed = []
    for branch_line in branches:
        if branch_line.strip():
            parts = branch_line.split()
            branch = parts[0]
            upstream = parts[1] if len(parts) > 1 and parts[1].strip() else None
            if upstream:
                # Check if branch is ahead of upstream
                ahead_result = subprocess.run(
                    ["git", "rev-list", "--count", f"{upstream}..{branch}"],
                    check=False,
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
                if (
                    ahead_result.returncode == 0
                    and int(ahead_result.stdout.strip() or "0") > 0
                ):
                    unpushed.append(f"{branch} (ahead of {upstream})")
            else:
                # No upstream set
                unpushed.append(f"{branch} (no upstream)")
    return unpushed if unpushed else None


def scan_repos(root_path):
    """
    Scan the directory tree for unstable Git repositories.

    Args:
        root_path (str):
            Root directory to scan.

    Returns:
        list[dict]:
            List of dictionaries with repository info for unstable repos.

    """
    unstable_repos = []
    for dirpath, dirnames, _ in os.walk(root_path):
        if is_git_repo(dirpath):
            changes = get_uncommitted_changes(dirpath)
            unpushed = get_unpushed_branches(dirpath)
            if changes or unpushed:
                unstable_repos.append(
                    {"path": dirpath, "uncommitted": changes, "unpushed": unpushed}
                )
            # Skip recursing into .git directories
            dirnames[:] = [d for d in dirnames if d != ".git"]
    return unstable_repos


def main():
    """
    Main entry point for the repository checker script.

    Parses command-line arguments, scans repositories, and reports unstable ones
    with colored logging output.
    """
    parser = argparse.ArgumentParser(
        description="Check git repositories for uncommitted changes and unpushed branches."
    )
    parser.add_argument(
        "root_path", type=str, help="Root path to scan for git repositories"
    )
    args = parser.parse_args()

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    root = args.root_path
    repos = scan_repos(root)
    if repos:
        logger.info("Unstable repositories:")
        for repo in repos:
            fullpath = os.path.abspath(repo["path"])
            logger.info(f"\n{fullpath}:")
            if repo["uncommitted"]:
                logger.error("  Uncommitted changes:")
                for line in repo["uncommitted"].split("\n"):
                    logger.error(f"    {line}")
            if repo["unpushed"]:
                logger.warning("  Unpushed branches:")
                for branch in repo["unpushed"]:
                    logger.warning(f"    {branch}")
    else:
        logger.info("All repositories are stable.")


if __name__ == "__main__":
    main()
