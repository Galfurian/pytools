#!/usr/bin/env python3

"""Install Python scripts from src/ to a bin directory."""

import argparse
import logging
import shutil
import subprocess
import sys
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


def check_shebang(file_path):
    """Check if the file has the correct shebang."""
    with open(file_path) as f:
        first_line = f.readline().strip()
    return first_line == "#!/usr/bin/env python3"


def check_syntax(file_path):
    """Check if the Python file has valid syntax."""
    try:
        subprocess.run(
            [sys.executable, "-m", "py_compile", str(file_path)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        return False
    else:
        return True


def check_command_exists(command):
    """Check if a command already exists in PATH."""
    return shutil.which(command) is not None


def install_all_scripts(args):
    """Install all scripts in src/ with safety checks."""
    src_dir = Path(__file__).parent / "src"
    if not src_dir.exists():
        logger.error(f"{src_dir} does not exist.")
        return False

    install_dir = args.install_dir
    if install_dir is None:
        install_dir = Path.home() / ".local" / "bin"
    install_dir = Path(install_dir)
    if not args.dry_run:
        install_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"Would create directory {install_dir} (if it doesn't exist)")

    success_count = 0
    total_count = 0

    for script_path in src_dir.glob("*.py"):
        total_count += 1
        script_name = script_path.stem

        # Skip __init__.py and other special files
        if script_name == "__init__":
            continue

        logger.info(f"Processing {script_name}...")

        if not check_shebang(script_path):
            logger.warning(
                f"Skipped: {script_path} does not have the correct shebang '#!/usr/bin/env python3'."
            )
            continue

        if not check_syntax(script_path):
            logger.warning(f"Skipped: {script_path} has syntax errors.")
            continue

        target_name = script_name
        target_path = install_dir / target_name

        if target_path.exists():
            logger.warning(f"Warning: {target_path} already exists.")
            if args.dry_run:
                logger.info(f"Would prompt to overwrite {target_name} (dry-run)")
            else:
                response = input("  Do you want to overwrite? (Y/n): ").strip().lower()
                if response == "n":
                    logger.info(f"Skipped: Installation cancelled for {script_name}.")
                    continue

        if args.dry_run:
            logger.info(f"Would install {target_name} to {target_path} (dry-run)")
        else:
            shutil.copy2(script_path, target_path)
            target_path.chmod(0o755)
            logger.info(f"Successfully installed {target_name} to {target_path}")
        success_count += 1

    if args.dry_run:
        logger.info(f"\nWould install {success_count} out of {total_count} scripts (dry-run).")
    else:
        logger.info(f"\nInstalled {success_count} out of {total_count} scripts.")
        logger.info(f"Make sure {install_dir} is in your PATH.")
    return success_count > 0


def main():
    """
    Main entry point for the installation script.

    Parses command-line arguments and installs scripts to the specified directory.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Install all Python scripts from src/ to a bin directory."
    )
    parser.add_argument(
        "--install-dir",
        help="Directory to install to (default: ~/.local/bin)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be installed without actually installing",
    )
    args = parser.parse_args()

    success = install_all_scripts(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
