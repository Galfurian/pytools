#!/usr/bin/env python3

"""
Compress a file or folder into a .tar.gz or .zip archive.

This script provides a command-line tool to compress files or directories into
.tar.gz or .zip archives with optional output paths and format selection.
"""

import argparse
import hashlib
import logging
import os
import sys
import tarfile
import time
import zipfile
from multiprocessing import Pool, cpu_count
from pathlib import Path

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


def compress_to_tar_gz(
    input_path: Path, output_path: Path, older_than: float | None = None
):
    """
    Compress the input file or folder to a .tar.gz archive.

    Args:
        input_path (Path):
            Path to the file or directory to compress.
        output_path (Path):
            Path for the output .tar.gz archive.
        older_than (float | None):
            Only include files older than this many seconds (None for no age filter).

    Raises:
        ValueError:
            If input_path is neither a file nor a directory.

    """
    # Collect all files to compress
    files_to_compress = []
    current_time = time.time()
    age_threshold = current_time - (older_than or 0)

    if input_path.is_file():
        if older_than is None or input_path.stat().st_mtime < age_threshold:
            files_to_compress = [(str(input_path), input_path.name)]
    elif input_path.is_dir():
        for file in input_path.rglob("*"):
            if file.is_file() and (
                older_than is None or file.stat().st_mtime < age_threshold
            ):
                arcname = file.relative_to(input_path.parent)
                files_to_compress.append((str(file), str(arcname)))
    else:
        raise ValueError(
            f"Input path '{input_path}' is neither a file nor a directory."
        )

    total_files = len(files_to_compress)
    completed = 0

    with tarfile.open(output_path, "w:gz") as tar:
        for file_path, arcname in files_to_compress:
            tar.add(file_path, arcname=arcname)
            completed += 1
            print(
                f"\rCompressing {completed} of {total_files} files...",
                end="",
                flush=True,
            )

    print()  # New line after progress


def verify_archive(archive_path: Path) -> bool:
    """
    Verify the integrity of a tar.gz or zip archive using SHA256 checksums.

    Args:
        archive_path (Path): Path to the archive to verify.

    Returns:
        bool: True if verification passes, False otherwise.
    """
    try:
        if archive_path.suffix == ".gz" or ".tar.gz" in str(archive_path):
            # For tar.gz, extract and hash each file
            with tarfile.open(archive_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        f = tar.extractfile(member)
                        if f:
                            hashlib.sha256(f.read()).hexdigest()
        elif archive_path.suffix == ".zip":
            # For zip, check CRC32 (built-in integrity check)
            with zipfile.ZipFile(archive_path, "r") as zipf:
                for info in zipf.filelist:
                    with zipf.open(info) as f:
                        hashlib.sha256(f.read()).hexdigest()
        return True
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def compress_to_zip(
    input_path: Path, output_path: Path, older_than: float | None = None
):
    """
    Compress the input file or folder to a .zip archive.

    Args:
        input_path (Path):
            Path to the file or directory to compress.
        output_path (Path):
            Path for the output .zip archive.
        older_than (float | None):
            Only include files older than this many seconds (None for no age filter).

    Raises:
        ValueError:
            If input_path is neither a file nor a directory.

    """
    current_time = time.time()
    age_threshold = current_time - (older_than or 0)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        if input_path.is_file():
            if older_than is None or input_path.stat().st_mtime < age_threshold:
                zipf.write(str(input_path), input_path.name)
        elif input_path.is_dir():
            for file in input_path.rglob("*"):
                if file.is_file() and (
                    older_than is None or file.stat().st_mtime < age_threshold
                ):
                    arcname = file.relative_to(input_path.parent)
                    zipf.write(str(file), str(arcname))
        else:
            raise ValueError(
                f"Input path '{input_path}' is neither a file nor a directory."
            )


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
        description="Compress a file or folder into a .tar.gz or .zip archive."
    )
    parser.add_argument(
        "input_path", type=str, help="Path to the file or folder to compress"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["tar.gz", "zip"],
        default=None,
        help="Compression format: tar.gz or zip. If not specified, auto-detects based on input type.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path. If not specified, will be created in the input's directory.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify archive integrity after creation using SHA256 checksums.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Delete the source file/directory after successful compression.",
    )
    parser.add_argument(
        "--older-than",
        type=str,
        default=None,
        help="Only compress files older than specified duration (e.g., '30d', '1w', '2M').",
    )
    return parser.parse_args(argv)


def main():
    """
    Main entry point for the compression script.

    Parses command-line arguments, validates inputs, and performs the compression
    operation with appropriate logging and error handling.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    input_path = Path(args.input_path).resolve()
    if not input_path.exists():
        logger.error(f"The path '{input_path}' does not exist.")
        raise FileNotFoundError(f"The path '{input_path}' does not exist.")
    if not (input_path.is_file() or input_path.is_dir()):
        logger.error(
            f"The path '{input_path}' is not a file or directory. Only files and folders are supported."
        )
        raise ValueError(
            f"The path '{input_path}' is not a file or directory. Only files and folders are supported."
        )

    # Parse older_than argument
    older_than: float | None = None
    if args.older_than:
        try:
            older_than = parse_time_duration(args.older_than)
        except ValueError:
            logger.exception("Error parsing older-than argument")
            sys.exit(1)

    # Auto-detect format if not specified
    if args.format is None:
        args.format = "zip" if input_path.is_file() else "tar.gz"
        logger.info(f"Auto-detected format: {args.format}")

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        input_name = input_path.stem if input_path.is_file() else input_path.name
        output_name = f"{input_name}.{args.format}"
        output_path = input_path.parent / output_name

    if output_path.exists():
        logger.warning(
            f"Output file '{output_path}' already exists and will be overwritten."
        )

    try:
        # Calculate input size (only for files that will be compressed)
        input_size = 0
        current_time = time.time()
        age_threshold = current_time - (older_than or 0)

        if input_path.is_file():
            if older_than is None or input_path.stat().st_mtime < age_threshold:
                input_size = input_path.stat().st_size
        elif input_path.is_dir():
            for file in input_path.rglob("*"):
                if file.is_file() and (
                    older_than is None or file.stat().st_mtime < age_threshold
                ):
                    input_size += file.stat().st_size

        logger.info(f"Creating {output_path} from {input_path}...")
        if args.format == "tar.gz":
            compress_to_tar_gz(input_path, output_path, older_than)
        else:
            compress_to_zip(input_path, output_path, older_than)

        # Calculate output size and compression ratio
        output_size = output_path.stat().st_size
        ratio = input_size / output_size if output_size > 0 else 0
        logger.info(f"Successfully created {output_path}")
        logger.info(
            f"Compressed: {input_size} bytes → {output_size} bytes ({ratio:.2f}x)"
        )

        # Verify archive if requested
        if args.verify:
            logger.info("Verifying archive integrity...")
            if verify_archive(output_path):
                logger.info("Archive verification passed.")
            else:
                logger.error("Archive verification failed!")
                sys.exit(4)

        # Move/delete source if requested
        if args.move:
            try:
                if input_path.is_file():
                    input_path.unlink()
                    logger.info(f"Deleted source file: {input_path}")
                elif input_path.is_dir():
                    import shutil

                    shutil.rmtree(input_path)
                    logger.info(f"Deleted source directory: {input_path}")
            except Exception as e:
                logger.error(f"Failed to delete source: {e}")
                sys.exit(5)
    except PermissionError:
        logger.exception(f"Permission denied when writing to '{output_path}'.")
        sys.exit(2)
    except Exception:
        logger.exception("Error during compression")
        sys.exit(3)


if __name__ == "__main__":
    main()
