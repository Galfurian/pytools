#!/usr/bin/env python3

"""Install Python scripts from src/ to a bin directory."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_shebang(file_path):
    """Check if the file has the correct shebang."""
    with open(file_path, "r") as f:
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
        return True
    except subprocess.CalledProcessError:
        return False


def check_command_exists(command):
    """Check if a command already exists in PATH."""
    return shutil.which(command) is not None


def install_all_scripts(install_dir=None):
    """Install all scripts in src/ with safety checks."""
    src_dir = Path(__file__).parent / "src"
    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist.")
        return False

    if install_dir is None:
        install_dir = Path.home() / ".local" / "bin"
    install_dir = Path(install_dir)
    install_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    total_count = 0

    for script_path in src_dir.glob("*.py"):
        total_count += 1
        script_name = script_path.stem

        print(f"Processing {script_name}...")

        if not check_shebang(script_path):
            print(
                f"  Skipped: {script_path} does not have the correct shebang '#!/usr/bin/env python3'."
            )
            continue

        if not check_syntax(script_path):
            print(f"  Skipped: {script_path} has syntax errors.")
            continue

        target_name = script_name
        target_path = install_dir / target_name

        if target_path.exists():
            print(f"  Warning: {target_path} already exists.")
            response = input("  Do you want to overwrite? (y/N): ").strip().lower()
            if response != "y":
                print(f"  Skipped: Installation cancelled for {script_name}.")
                continue

        shutil.copy2(script_path, target_path)
        target_path.chmod(0o755)

        print(f"  Successfully installed {target_name} to {target_path}")
        success_count += 1

    print(f"\nInstalled {success_count} out of {total_count} scripts.")
    print(f"Make sure {install_dir} is in your PATH.")
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Install all Python scripts from src/ to a bin directory."
    )
    parser.add_argument(
        "--install-dir",
        help="Directory to install to (default: ~/.local/bin)",
    )
    args = parser.parse_args()

    success = install_all_scripts(args.install_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
