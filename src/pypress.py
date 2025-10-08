#!/usr/bin/env python3

"""
Compress a file or folder into a .tar.gz or .zip archive.

This script provides a command-line tool to compress files or directories into
.tar.gz or .zip archives with optional output paths and format selection.
"""

import argparse
import logging
import tarfile
import zipfile
from pathlib import Path


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


def compress_to_tar_gz(input_path: Path, output_path: Path):
    """
    Compress the input file or folder to a .tar.gz archive.

    Args:
        input_path (Path):
            Path to the file or directory to compress.
        output_path (Path):
            Path for the output .tar.gz archive.

    Raises:
        ValueError:
            If input_path is neither a file nor a directory.

    """
    with tarfile.open(output_path, "w:gz") as tar:
        if input_path.is_file():
            tar.add(str(input_path), arcname=input_path.name)
        elif input_path.is_dir():
            for file in input_path.rglob("*"):
                if file.is_file():
                    arcname = file.relative_to(input_path.parent)
                    tar.add(str(file), arcname=str(arcname))
        else:
            logging.error(
                f"Input path '{input_path}' is neither a file nor a directory."
            )
            raise ValueError(
                f"Input path '{input_path}' is neither a file nor a directory."
            )


def compress_to_zip(input_path: Path, output_path: Path):
    """
    Compress the input file or folder to a .zip archive.

    Args:
        input_path (Path):
            Path to the file or directory to compress.
        output_path (Path):
            Path for the output .zip archive.

    Raises:
        ValueError:
            If input_path is neither a file nor a directory.

    """
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        if input_path.is_file():
            zipf.write(str(input_path), input_path.name)
        elif input_path.is_dir():
            for file in input_path.rglob("*"):
                if file.is_file():
                    arcname = file.relative_to(input_path.parent)
                    zipf.write(str(file), str(arcname))
        else:
            logging.error(
                f"Input path '{input_path}' is neither a file nor a directory."
            )
            raise ValueError(
                f"Input path '{input_path}' is neither a file nor a directory."
            )


def main():
    """
    Main entry point for the compression script.

    Parses command-line arguments, validates inputs, and performs the compression
    operation with appropriate logging and error handling.
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
        default="tar.gz",
        help="Compression format: tar.gz or zip (default: tar.gz)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path. If not specified, will be created in the input's directory.",
    )
    args = parser.parse_args()

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    input_path = Path(args.input_path).resolve()
    if not input_path.exists():
        logging.error(f"The path '{input_path}' does not exist.")
        raise FileNotFoundError(f"The path '{input_path}' does not exist.")
    if not (input_path.is_file() or input_path.is_dir()):
        logging.error(
            f"The path '{input_path}' is not a file or directory. Only files and folders are supported."
        )
        raise ValueError(
            f"The path '{input_path}' is not a file or directory. Only files and folders are supported."
        )

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        input_name = input_path.stem if input_path.is_file() else input_path.name
        output_name = f"{input_name}.{args.format}"
        output_path = input_path.parent / output_name

    if output_path.exists():
        logging.warning(
            f"Output file '{output_path}' already exists and will be overwritten."
        )

    try:
        logging.info(f"Creating {output_path} from {input_path}...")
        if args.format == "tar.gz":
            compress_to_tar_gz(input_path, output_path)
        else:
            compress_to_zip(input_path, output_path)
        logging.info(f"Successfully created {output_path}")
    except PermissionError:
        logging.exception(f"Permission denied when writing to '{output_path}'.")
        exit(2)
    except Exception as e:
        logging.exception(f"Error during compression: {e}")
        exit(3)


if __name__ == "__main__":
    main()
