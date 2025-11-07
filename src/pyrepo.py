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
import time

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
                # Check ahead and behind counts
                ahead_result = subprocess.run(
                    ["git", "rev-list", "--count", f"{upstream}..{branch}"],
                    check=False,
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
                behind_result = subprocess.run(
                    ["git", "rev-list", "--count", f"{branch}..{upstream}"],
                    check=False,
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
                ahead_count = (
                    int(ahead_result.stdout.strip() or "0")
                    if ahead_result.returncode == 0
                    else 0
                )
                behind_count = (
                    int(behind_result.stdout.strip() or "0")
                    if behind_result.returncode == 0
                    else 0
                )

                if ahead_count > 0 or behind_count > 0:
                    status_parts = []
                    if ahead_count > 0:
                        status_parts.append(f"ahead {ahead_count}")
                    if behind_count > 0:
                        status_parts.append(f"behind {behind_count}")
                    unpushed.append(f"{branch} ({', '.join(status_parts)})")
            else:
                # No upstream set
                unpushed.append(f"{branch} (no upstream)")
    return unpushed if unpushed else None


def get_untracked_files(repo_path):
    """
    Get untracked files in the Git repository.

    Args:
        repo_path (str):
            Path to the Git repository.

    Returns:
        list[str] or None:
            List of untracked files if any, None otherwise.

    """
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        check=False,
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    untracked = result.stdout.strip().split("\n") if result.stdout.strip() else []
    return untracked if untracked else None


def get_special_git_states(repo_path):
    """
    Check for special Git states like detached HEAD or rebase in progress.

    Args:
        repo_path (str):
            Path to the Git repository.

    Returns:
        list[str] or None:
            List of special states if any, None otherwise.

    """
    states = []

    # Check for detached HEAD
    head_file = os.path.join(repo_path, ".git", "HEAD")
    if os.path.exists(head_file):
        with open(head_file, "r") as f:
            head_content = f.read().strip()
            if not head_content.startswith("ref: refs/heads/"):
                states.append("detached HEAD")

    # Check for rebase in progress
    rebase_merge = os.path.join(repo_path, ".git", "rebase-merge")
    rebase_apply = os.path.join(repo_path, ".git", "rebase-apply")
    if os.path.exists(rebase_merge):
        states.append("rebase in progress")
    elif os.path.exists(rebase_apply):
        states.append("rebase in progress")

    return states if states else None


def get_repo_freshness(repo_path, stale_threshold_days=30):
    """
    Check if repository is stale (no commits in threshold days).

    Args:
        repo_path (str):
            Path to the Git repository.
        stale_threshold_days (int):
            Number of days after which a repo is considered stale.

    Returns:
        int or None:
            Days since last commit if stale, None otherwise.

    """
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ct"],
        check=False,
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        last_commit_time = int(result.stdout.strip())
        current_time = int(time.time())
        days_since_commit = (current_time - last_commit_time) // (24 * 3600)
        if days_since_commit > stale_threshold_days:
            return days_since_commit
    return None


def scan_repos(root_path):
    """
    Scan the directory tree for Git repositories and their status.

    Args:
        root_path (str):
            Root directory to scan.

    Returns:
        list[dict]:
            List of dictionaries with repository info for all repos.

    """
    repos = []
    for dirpath, dirnames, _ in os.walk(root_path):
        if is_git_repo(dirpath):
            changes = get_uncommitted_changes(dirpath)
            unpushed = get_unpushed_branches(dirpath)
            untracked = get_untracked_files(dirpath)
            special_states = get_special_git_states(dirpath)
            stale_days = get_repo_freshness(dirpath)
            repos.append(
                {
                    "path": dirpath,
                    "uncommitted": changes,
                    "unpushed": unpushed,
                    "untracked": untracked,
                    "special_states": special_states,
                    "stale_days": stale_days,
                }
            )
            # Skip recursing into .git directories
            dirnames[:] = [d for d in dirnames if d != ".git"]
    return repos


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
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Output summary table instead of detailed view",
    )
    args = parser.parse_args()

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("[%(levelname)-9s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    root = args.root_path
    repos = scan_repos(root)
    if repos:
        if args.summary:
            # Output summary table
            print("PATH                 STATUS           UNPUSHED  STALE(days)")
            print("-" * 60)
            for repo in repos:
                path = os.path.basename(repo["path"])[:20]  # Truncate long paths
                is_unstable = (
                    repo["uncommitted"]
                    or repo["unpushed"]
                    or repo["untracked"]
                    or repo["special_states"]
                )
                status = "unstable" if is_unstable else "clean"
                unpushed_count = len(repo["unpushed"]) if repo["unpushed"] else 0
                stale = repo["stale_days"] if repo["stale_days"] is not None else ""
                print(f"{path:<20} {status:<15} {unpushed_count:<8} {stale}")
        else:
            # Detailed output
            logger.info("Git repositories:")
            for repo in repos:
                fullpath = os.path.abspath(repo["path"])
                is_unstable = (
                    repo["uncommitted"]
                    or repo["unpushed"]
                    or repo["untracked"]
                    or repo["special_states"]
                )
                print("")
                if is_unstable:
                    logger.error(f"Path  : {fullpath}:")
                elif repo["stale_days"]:
                    logger.warning(f"Path  : {fullpath}:")
                else:
                    logger.info(f"Path  : {fullpath}:")
                if not is_unstable and repo["stale_days"] is None:
                    logger.info("Status: clean")
                else:
                    logger.warning("Status: unstable")

                if repo["uncommitted"]:
                    logger.error("  Uncommitted changes:")
                    for line in repo["uncommitted"].split("\n"):
                        logger.error(f"    {line}")
                if repo["unpushed"]:
                    logger.warning("  Unpushed branches:")
                    for branch in repo["unpushed"]:
                        logger.warning(f"    {branch}")
                if repo["untracked"]:
                    logger.info("  Untracked files:")
                    for file in repo["untracked"]:
                        logger.info(f"    {file}")
                if repo["special_states"]:
                    logger.warning("  Special states:")
                    for state in repo["special_states"]:
                        logger.warning(f"    {state}")
                if repo["stale_days"] is not None:
                    logger.warning(f"  {repo['stale_days']} days since last commit")
    else:
        if args.summary:
            print("No Git repositories found.")
        else:
            logger.info("No Git repositories found.")


if __name__ == "__main__":
    main()
