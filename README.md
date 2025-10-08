# pytools

A collection of Python utility scripts for common tasks.

## Tools

- **pypress**: Compress a file or folder into a .tar.gz or .zip archive.
- **pyrepo**: Check git repositories for uncommitted changes and unpushed branches.
- **pydu**: Display disk usage of files and directories in a human-readable format.
- **install**: Install all scripts from `src/` to a bin directory for easy command-line access.

## Installation

Clone the repository and run the installer:

```bash
git clone https://github.com/Galfurian/pytools.git
cd pytools
python install.py
```

This installs all scripts to `~/.local/bin` (ensure it's in your PATH).

## Requirements

- Python 3.10+
- Git (for pyrepo)

## Development

To set up the development environment with linting, type checking, and testing tools:

```bash
git clone https://github.com/Galfurian/pytools.git
cd pytools
pip install -e ".[dev]"
```

This installs the project in editable mode along with development dependencies (Ruff for linting, MyPy for type checking, and pytest for testing).

### Linting and Type Checking

To lint the code with Ruff:

```bash
ruff check .
```

To check types with MyPy:

```bash
mypy src/
```

## License

See LICENSE.md
