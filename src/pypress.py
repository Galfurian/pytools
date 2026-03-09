#!/usr/bin/env python3
"""
pypack.py — stdlib-only pack/unpack with safe defaults.

Goals
- Standard library only (no external tools, no third-party deps).
- "Do-all" within stdlib limits:
  - Archives: .zip, .tar
  - Compressed tars: .tar.gz/.tgz, .tar.bz2/.tbz2/.tbz, .tar.xz/.txz (if Python has lzma)
  - Single-file compression: .gz, .bz2, .xz (if lzma available)
- Safe-by-default:
  - Refuse overwrite unless -f/--force is passed
  - Safe extraction (prevents path traversal / zip-slip / tar-slip)
- Auto-detect by extension first, then simple magic sniff for zip/gz/bz2/xz when needed.
- Optional niceties inspired by your script:
  - colored logging
  - --older-than filtering when packing directories
  - --verify after pack (reads members / checks ZIP CRC)
  - --move deletes sources after successful pack

Notes / Limits (stdlib reality)
- No support for: .7z .rar .zst
- "Magic" detection is limited; extension detection is primary.
- lzma may be unavailable on some minimal Python builds; .xz support will be disabled gracefully.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import logging
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Sequence

logger = logging.getLogger("pypack")

# ---------------------------
# Logging (inspired by yours)
# ---------------------------


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[41m",  # red bg
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}" if color else msg


def setup_logging(verbose: int) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
    logger.handlers[:] = [handler]
    logger.setLevel(logging.DEBUG if verbose >= 1 else logging.INFO)


# ---------------------------
# Duration parsing (yours)
# ---------------------------


def parse_time_duration(duration_str: str) -> float:
    """
    Parse human-readable duration into seconds.

    Supports: Ns, Nh, Nd, Nw, NM, Ny (case-insensitive)
    - M = 30 days (approx), y = 365 days (approx)
    """
    if not duration_str:
        return 0.0

    s = duration_str.strip().upper()
    multipliers = {
        "S": 1,
        "H": 3600,
        "D": 24 * 3600,
        "W": 7 * 24 * 3600,
        "M": 30 * 24 * 3600,  # month ~ 30 days
        "Y": 365 * 24 * 3600,  # year ~ 365 days
    }

    for unit, mul in multipliers.items():
        if s.endswith(unit):
            val = s[: -len(unit)].strip()
            try:
                return float(val) * mul
            except ValueError as e:
                raise ValueError(f"Invalid duration number: {val!r}") from e

    raise ValueError(
        f"Invalid duration format: {duration_str!r}. "
        "Use formats like '30d', '1w', '2M', '1y', '5h', '10s'."
    )


# ---------------------------
# Format detection
# ---------------------------

XZ_AVAILABLE = True
try:
    import lzma  # noqa: F401
except Exception:
    XZ_AVAILABLE = False

TAR_READ_OPENER = {
    "tar": lambda path: tarfile.open(path, "r:"),
    "targz": lambda path: tarfile.open(path, "r:gz"),
    "tarbz2": lambda path: tarfile.open(path, "r:bz2"),
    "tarxz": lambda path: tarfile.open(path, "r:xz"),
}

TAR_WRITE_OPENER = {
    "tar": lambda path: tarfile.open(path, "w:"),
    "targz": lambda path: tarfile.open(path, "w:gz"),
    "tarbz2": lambda path: tarfile.open(path, "w:bz2"),
    "tarxz": lambda path: tarfile.open(path, "w:xz"),
}

SINGLE_FILE_WRITE_OPENER = {
    "gz": gzip.open,
    "bz2": bz2.open,
    "xz": (None if not XZ_AVAILABLE else __import__("lzma").open),
}

SINGLE_FILE_READ_OPENER = {
    "gz": gzip.open,
    "bz2": bz2.open,
    "xz": (None if not XZ_AVAILABLE else __import__("lzma").open),
}


def _lower_name(p: Path) -> str:
    return p.name.lower()


def detect_kind(path: Path) -> str:
    """
    Detect archive/compression kind.
    Returns one of:
      tar, targz, tarbz2, tarxz,
      zip,
      gz, bz2, xz,
      unknown

    Priority:
      1) extension-based
      2) simple magic sniff (zip/gz/bz2/xz) when extension doesn't help
    """
    n = _lower_name(path)

    # tar variants first
    if n.endswith(".tar"):
        return "tar"
    if n.endswith((".tar.gz", ".tgz")):
        return "targz"
    if n.endswith((".tar.bz2", ".tbz2", ".tbz")):
        return "tarbz2"
    if n.endswith((".tar.xz", ".txz")):
        return "tarxz"

    # zip
    if n.endswith(".zip"):
        return "zip"

    # single-file compressors
    if n.endswith(".gz"):
        return "gz"
    if n.endswith(".bz2"):
        return "bz2"
    if n.endswith(".xz"):
        return "xz"

    # magic sniff fallback
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return "unknown"

    if (
        head.startswith(b"PK\x03\x04")
        or head.startswith(b"PK\x05\x06")
        or head.startswith(b"PK\x07\x08")
    ):
        return "zip"
    if head[:2] == b"\x1f\x8b":
        return "gz"
    if head[:3] == b"BZh":
        return "bz2"
    if head.startswith(b"\xfd7zXZ\x00"):
        return "xz"

    return "unknown"


def infer_pack_type_from_output(out: Path, explicit: Optional[str]) -> str:
    if explicit:
        return explicit.lower()

    n = out.name.lower()
    if n.endswith((".tar.gz", ".tgz")):
        return "tar.gz"
    if n.endswith((".tar.bz2", ".tbz2", ".tbz")):
        return "tar.bz2"
    if n.endswith((".tar.xz", ".txz")):
        return "tar.xz"
    if n.endswith(".tar"):
        return "tar"
    if n.endswith(".zip"):
        return "zip"
    if n.endswith(".gz"):
        return "gz"
    if n.endswith(".bz2"):
        return "bz2"
    if n.endswith(".xz"):
        return "xz"
    # default
    return "tar.gz"


# ---------------------------
# Safety helpers
# ---------------------------


def refuse_overwrite(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite: {path} (use -f/--force)")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_within_directory(base: Path, target: Path) -> bool:
    """
    True if target resolves within base (prevents path traversal).
    """
    try:
        base_r = base.resolve()
        target_r = target.resolve()
        target_r.relative_to(base_r)
        return True
    except Exception:
        return False


def safe_extract_tar(tar: tarfile.TarFile, dest: Path, force: bool) -> None:
    """
    Extract tar safely into dest:
      - block absolute paths
      - block .. traversal
      - refuse overwrite unless force
    """
    for member in tar.getmembers():
        # TarInfo.name is a posix-style path
        name = member.name

        # Disallow absolute paths and drive letters-ish
        if name.startswith("/") or name.startswith("\\") or ":" in Path(name).drive:
            raise ValueError(f"Unsafe tar member path (absolute): {name}")

        out_path = dest / name
        if not is_within_directory(dest, out_path):
            raise ValueError(f"Unsafe tar member path (traversal): {name}")

        # Refuse overwrite for files/links
        if member.isfile() or member.islnk() or member.issym():
            if out_path.exists() and not force:
                raise FileExistsError(
                    f"Refusing to overwrite extracted file: {out_path} (use -f)"
                )

    # If all members pass, extract
    tar.extractall(path=dest)


def safe_extract_zip(zf: zipfile.ZipFile, dest: Path, force: bool) -> None:
    """
    Extract zip safely into dest:
      - block absolute paths
      - block .. traversal
      - refuse overwrite unless force
    """
    for info in zf.infolist():
        name = info.filename

        if name.startswith("/") or name.startswith("\\"):
            raise ValueError(f"Unsafe zip entry path (absolute): {name}")

        out_path = dest / name
        if not is_within_directory(dest, out_path):
            raise ValueError(f"Unsafe zip entry path (traversal): {name}")

        # zip entries ending with / are directories
        if not name.endswith("/"):
            if out_path.exists() and not force:
                raise FileExistsError(
                    f"Refusing to overwrite extracted file: {out_path} (use -f)"
                )

    zf.extractall(path=dest)


# ---------------------------
# Packing helpers
# ---------------------------


def iter_files_for_pack(root: Path, older_than: Optional[float]) -> Iterable[Path]:
    """
    Yield files under root (or root itself if file) filtered by mtime age threshold.
    older_than = seconds; include only files with mtime < now - older_than
    """
    now = time.time()
    threshold = now - older_than if older_than is not None else None

    def include(p: Path) -> bool:
        if threshold is None:
            return True
        try:
            return p.stat().st_mtime < threshold
        except OSError:
            return False

    if root.is_file():
        if include(root):
            yield root
        return

    if root.is_dir():
        for p in root.rglob("*"):
            if p.is_file() and include(p):
                yield p
        return

    raise ValueError(f"Input path is neither file nor directory: {root}")


def compute_input_size(paths: Sequence[Path], older_than: Optional[float]) -> int:
    total = 0
    for p in paths:
        for f in iter_files_for_pack(p, older_than):
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def add_to_tar(
    tar: tarfile.TarFile, input_path: Path, older_than: Optional[float], base_mode: str
) -> int:
    """
    Add files from input_path into tar.
    base_mode controls how arcnames are created:
      - "parent": relative to input_path.parent (like your script)
      - "self":   relative to input_path (keeps directory contents without parent)
    Returns number of files added.
    """
    count = 0
    input_path = input_path.resolve()

    if input_path.is_file():
        arcname = input_path.name
        tar.add(str(input_path), arcname=arcname)
        return 1

    if input_path.is_dir():
        for f in iter_files_for_pack(input_path, older_than):
            if base_mode == "parent":
                arcname = f.relative_to(input_path.parent)
            else:
                arcname = f.relative_to(input_path)
                # include top folder name to avoid dumping into root
                arcname = Path(input_path.name) / arcname
            tar.add(str(f), arcname=str(arcname))
            count += 1
        return count

    raise ValueError(f"Input path is neither file nor directory: {input_path}")


def add_to_zip(
    zf: zipfile.ZipFile, input_path: Path, older_than: Optional[float], base_mode: str
) -> int:
    count = 0
    input_path = input_path.resolve()

    if input_path.is_file():
        zf.write(str(input_path), arcname=input_path.name)
        return 1

    if input_path.is_dir():
        for f in iter_files_for_pack(input_path, older_than):
            if base_mode == "parent":
                arcname = f.relative_to(input_path.parent)
            else:
                arcname = Path(input_path.name) / f.relative_to(input_path)
            zf.write(str(f), arcname=str(arcname))
            count += 1
        return count

    raise ValueError(f"Input path is neither file nor directory: {input_path}")


# ---------------------------
# Verify helpers (stdlib)
# ---------------------------


def verify_archive(path: Path) -> bool:
    """
    Best-effort verification:
      - zip: zipfile.testzip() (CRC check)
      - tar*: iterate members + attempt to read contents (spot corruption)
      - gz/bz2/xz: attempt full decompression stream
    """
    kind = detect_kind(path)
    try:
        if kind == "zip":
            with zipfile.ZipFile(path, "r") as zf:
                bad = zf.testzip()
                return bad is None

        if kind in ("tar", "targz", "tarbz2", "tarxz"):
            opener = TAR_READ_OPENER.get(kind)
            if not opener:
                raise ValueError(f"Unsupported format: {kind}")
            with opener(path) as tf:
                for m in tf.getmembers():
                    if m.isfile():
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        # read stream to ensure it’s decodable
                        while f.read(1024 * 1024):
                            pass
            return True

        if kind in ("gz", "bz2", "xz"):
            # full stream read
            opener = SINGLE_FILE_READ_OPENER.get(kind)
            if not opener:
                raise ValueError(f"Unsupported format: {kind}")
            if opener is None:
                raise RuntimeError("xz verify requires lzma support")
            with opener(path, "rb") as f:
                while f.read(1024 * 1024):
                    pass
            return True

        return False
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


# ---------------------------
# Unpack / Decompress
# ---------------------------


def _default_single_out_name(src: Path) -> str:
    n = src.name
    for suf in (".gz", ".bz2", ".xz"):
        if n.lower().endswith(suf):
            return n[: -len(suf)]
    return n + ".out"


def unpack_one(src: Path, outdir: Path, force: bool) -> None:
    kind = detect_kind(src)
    ensure_dir(outdir)

    if kind == "zip":
        with zipfile.ZipFile(src, "r") as zf:
            safe_extract_zip(zf, outdir, force)
        return

    if kind in ("tar", "targz", "tarbz2", "tarxz"):
        opener = TAR_READ_OPENER.get(kind)
        if not opener:
            raise ValueError(f"Unsupported format: {kind}")
        if kind == "tarxz" and not XZ_AVAILABLE:
            raise RuntimeError(
                "This Python build lacks lzma; cannot unpack .tar.xz/.txz"
            )
        with opener(src) as tf:
            safe_extract_tar(tf, outdir, force)
        return

    # single-file decompression
    if kind in ("gz", "bz2", "xz"):
        if kind == "xz" and not XZ_AVAILABLE:
            raise RuntimeError("This Python build lacks lzma; cannot unpack .xz")
        out_path = outdir / _default_single_out_name(src)
        refuse_overwrite(out_path, force)

        opener = SINGLE_FILE_READ_OPENER.get(kind)
        if not opener:
            raise ValueError(f"Unsupported format: {kind}")
        if opener is None:
            raise RuntimeError("xz unpack requires lzma support")

        with opener(src, "rb") as fin, open(out_path, "wb") as fout:
            shutil.copyfileobj(fin, fout, length=1024 * 1024)
        return

    raise ValueError(f"Unsupported/unknown format: {src} (kind={kind})")


# ---------------------------
# Pack / Compress
# ---------------------------


def pack(
    output: Path,
    inputs: Sequence[Path],
    pack_type: str,
    force: bool,
    older_than: Optional[float],
    base_mode: str,
    level: Optional[int],
) -> None:
    """
    pack_type in:
      tar, tar.gz, tar.bz2, tar.xz, zip, gz, bz2, xz
    """
    refuse_overwrite(output, force)

    pt = pack_type.lower()

    if pt in ("gz", "bz2", "xz"):
        if len(inputs) != 1:
            raise ValueError(f"{pt} requires exactly one input file")
        inp = inputs[0].resolve()
        if not inp.is_file():
            raise ValueError(f"{pt} requires a file input, got: {inp}")
        # older-than applies to single file: if not old enough, create empty? Better: error.
        if older_than is not None:
            now = time.time()
            threshold = now - older_than
            if inp.stat().st_mtime >= threshold:
                raise ValueError(
                    f"Input file is not older than {older_than} seconds: {inp}"
                )

        if pt == "xz" and not XZ_AVAILABLE:
            raise RuntimeError("This Python build lacks lzma; cannot create .xz")

        opener = SINGLE_FILE_WRITE_OPENER.get(pt)
        if not opener:
            raise ValueError(f"Unsupported format: {pt}")

        # compressionlevel supported by gzip/bz2/lzma in modern Python
        kwargs = {}
        if level is not None:
            kwargs["compresslevel"] = level

        with open(inp, "rb") as fin, opener(output, "wb", **kwargs) as fout:
            shutil.copyfileobj(fin, fout, length=1024 * 1024)
        return

    if pt in ("tar", "tar.gz", "tar.bz2", "tar.xz"):
        opener = TAR_WRITE_OPENER.get(pt)
        if not opener:
            raise ValueError(f"Unsupported format: {pt}")
        if pt == "tar.xz" and not XZ_AVAILABLE:
            raise RuntimeError(
                "This Python build lacks lzma; cannot create .tar.xz/.txz"
            )

        # tarfile has compresslevel for gzip/bz2 in newer Pythons; lzma uses preset
        # We'll pass if supported, else ignore gracefully.
        tf_kwargs = {}
        if level is not None:
            # gzip/bz2: compresslevel; xz: preset
            if pt in ("tar.gz", "tar.bz2"):
                tf_kwargs["compresslevel"] = level
            elif pt == "tar.xz":
                tf_kwargs["preset"] = level

        added = 0
        with opener(output, **tf_kwargs) as tf:
            for inp in inputs:
                added += add_to_tar(tf, inp, older_than, base_mode)
        if added == 0:
            logger.warning("No files matched filters; created an empty tar archive.")
        return

    if pt == "zip":
        # zipfile compressionlevel supported in newer versions; handle gracefully.
        z_kwargs = {}
        if level is not None:
            z_kwargs["compresslevel"] = level

        added = 0
        with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED, **z_kwargs
        ) as zf:
            for inp in inputs:
                added += add_to_zip(zf, inp, older_than, base_mode)
        if added == 0:
            logger.warning("No files matched filters; created an empty zip archive.")
        return

    raise ValueError(f"Unknown pack type: {pack_type}")


# ---------------------------
# CLI
# ---------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="pypack.py — stdlib-only safe pack/unpack (zip, tar.*, gz/bz2/xz)."
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (repeatable)",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser(
        "unpack",
        help="Unpack/decompress archives and compressed files.",
    )
    u.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing files on extract/write.",
    )
    u.add_argument(
        "-C",
        "--directory",
        default=".",
        help="Destination directory (default: current).",
    )
    u.add_argument(
        "--verify",
        action="store_true",
        help="Verify archive stream before unpacking.",
    )
    u.add_argument(
        "files",
        nargs="+",
        help="Archives/compressed files to unpack.",
    )

    k = sub.add_parser(
        "pack",
        help="Create archives or compressed files.",
    )
    k.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output if it exists.",
    )
    k.add_argument(
        "-t",
        "--type",
        dest="type_",
        choices=["tar", "tar.gz", "tar.bz2", "tar.xz", "zip", "gz", "bz2", "xz"],
        help="Output type (default: inferred from output name; fallback tar.gz).",
    )
    k.add_argument(
        "-l",
        "--level",
        type=int,
        default=None,
        help="Compression level (varies by format; e.g. gzip/bz2 1-9, zip 0-9, xz preset 0-9).",
    )
    k.add_argument(
        "--older-than",
        type=str,
        default=None,
        help="Only include files older than duration (e.g. '30d', '1w', '2M', '5h').",
    )
    k.add_argument(
        "--base",
        choices=["parent", "self"],
        default="self",
        help="Archive path layout for directories: "
        "'self' stores dir as a top folder (default), "
        "'parent' stores paths relative to input's parent (like your script).",
    )
    k.add_argument(
        "--verify", action="store_true", help="Verify archive after creation."
    )
    k.add_argument(
        "--move", action="store_true", help="Delete inputs after successful pack."
    )
    k.add_argument("output", help="Output archive/compressed file name.")
    k.add_argument("inputs", nargs="+", help="Input files and/or directories.")

    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    try:
        if args.cmd == "unpack":
            outdir = Path(args.directory).resolve()
            ensure_dir(outdir)

            for f in args.files:
                src = Path(f).resolve()
                if not src.exists():
                    raise FileNotFoundError(f"Not found: {src}")

                if args.verify:
                    logger.info(f"Verifying {src} ...")
                    if not verify_archive(src):
                        raise RuntimeError(f"Verification failed: {src}")

                logger.info(f"Unpacking {src} -> {outdir}")
                unpack_one(src, outdir, args.force)
            return 0

        # pack
        output = Path(args.output).resolve()
        inputs = [Path(x).resolve() for x in args.inputs]

        for inp in inputs:
            if not inp.exists():
                raise FileNotFoundError(f"Not found: {inp}")
            if not (inp.is_file() or inp.is_dir()):
                raise ValueError(f"Unsupported input (not file/dir): {inp}")

        older_than = parse_time_duration(args.older_than) if args.older_than else None
        pack_type = infer_pack_type_from_output(output, args.type_)

        if pack_type in ("tar.xz", "xz") and not XZ_AVAILABLE:
            raise RuntimeError(
                "This Python build lacks lzma; xz support is unavailable."
            )

        # compute and log sizes
        input_size = compute_input_size(inputs, older_than)
        logger.info(f"Creating {output} ({pack_type}) from {len(inputs)} input(s)...")

        pack(
            output=output,
            inputs=inputs,
            pack_type=pack_type,
            force=args.force,
            older_than=older_than,
            base_mode=args.base,
            level=args.level,
        )

        out_size = output.stat().st_size if output.exists() else 0
        ratio = (input_size / out_size) if out_size > 0 else 0.0
        logger.info(f"Created {output}")
        logger.info(
            f"Compressed: {input_size} bytes -> {out_size} bytes ({ratio:.2f}x)"
        )

        if args.verify:
            logger.info("Verifying archive integrity...")
            if verify_archive(output):
                logger.info("Verification passed.")
            else:
                raise RuntimeError("Verification failed.")

        if args.move:
            # Only remove after successful creation (+ verify if requested)
            for inp in inputs:
                try:
                    if inp.is_file():
                        inp.unlink()
                        logger.info(f"Deleted source file: {inp}")
                    elif inp.is_dir():
                        shutil.rmtree(inp)
                        logger.info(f"Deleted source directory: {inp}")
                except Exception as e:
                    raise RuntimeError(f"Failed to delete source {inp}: {e}") from e

        return 0

    except FileExistsError as e:
        logger.error(str(e))
        return 1
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 2
    except PermissionError as e:
        logger.error(f"Permission error: {e}")
        return 3
    except (tarfile.TarError, zipfile.BadZipFile) as e:
        logger.error(f"Archive error: {e}")
        return 4
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
