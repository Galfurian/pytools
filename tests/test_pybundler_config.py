import json
import textwrap
from pathlib import Path

from pybundler import (
    DEFAULT_CONFIG_FILENAME,
    Config,
    _generate_config,
    load_config_from_file,
    main,
    save_config_to_file,
    _collect_files,
)


def test_save_and_load_json_config(tmp_path: Path):
    cfg = Config(
        patterns=["src/**/*.py", "README.md"],
        excludes=[".venv/**"],
        toc={"src": "Source files", "README.md": "Project readme"},
        output="dist/EXAMPLE.md",
    )

    config_file = tmp_path / ".bundler.json"
    save_config_to_file(cfg, config_file)
    assert config_file.exists()

    loaded = load_config_from_file(config_file)
    assert loaded is not None
    assert loaded.patterns == cfg.patterns
    assert loaded.excludes == cfg.excludes
    # keys are normalized to lowercase on load
    assert loaded.toc == {"src": "Source files", "readme.md": "Project readme"}
    assert loaded.output == "dist/EXAMPLE.md"


def test_legacy_config_format_still_supported(tmp_path: Path):
    # Create legacy-style config file (old .bundler.config)
    content = textwrap.dedent(
        """
        patterns:
        - src/**/*.py
        - README.md

        toc:
        - README.md: Project README

        excludes:
        - .venv/**
        """
    )
    cfg_file = tmp_path / ".bundler.config"
    cfg_file.write_text(content, encoding="utf-8")

    loaded = load_config_from_file(cfg_file)
    assert loaded is not None
    assert "src/**/*.py" in loaded.patterns
    assert "README.md" in loaded.patterns
    assert loaded.excludes == [".venv/**"]
    assert loaded.toc == {"readme.md": "Project README"}


def test_generate_config_writes_json(tmp_path: Path):
    # create simple project layout
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")

    rc = _generate_config(root=tmp_path, patterns=["**/*.*"], config_filename=DEFAULT_CONFIG_FILENAME)
    assert rc == 0

    cfg_file = tmp_path / DEFAULT_CONFIG_FILENAME
    assert cfg_file.exists()

    loaded = load_config_from_file(cfg_file)
    assert loaded is not None
    assert isinstance(loaded.patterns, list)
    assert "**/*.*" in loaded.patterns
    assert isinstance(loaded.toc, dict)


def test_config_output_used_when_no_cli(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')", encoding="utf-8")

    cfg = {"patterns": ["**/*.*"], "output": "out/CONFIG.md"}
    (tmp_path / ".bundler.json").write_text(json.dumps(cfg), encoding="utf-8")

    rc = main(["--root", str(tmp_path), "--config", ".bundler.json"])
    assert rc == 0
    assert (tmp_path / "out" / "CONFIG.md").exists()


def test_config_root_used_when_no_cli(tmp_path: Path):
    # config specifies a different root directory
    (tmp_path / "project").mkdir()
    (tmp_path / "project" / "file.txt").write_text("hello", encoding="utf-8")

    cfg = {"patterns": ["**/*.*"], "root": "project", "output": "out/config_root.md"}
    (tmp_path / ".bundler.json").write_text(json.dumps(cfg), encoding="utf-8")

    # Call without --root; main should use config.root resolved relative to the config file
    rc = main(["--config", str(tmp_path / ".bundler.json")])
    assert rc == 0
    assert (tmp_path / "project" / "out" / "config_root.md").exists()


def test_cli_root_overrides_config(tmp_path: Path):
    (tmp_path / "projectA").mkdir()
    (tmp_path / "projectA" / "file.txt").write_text("a", encoding="utf-8")
    (tmp_path / "projectB").mkdir()
    (tmp_path / "projectB" / "file.txt").write_text("b", encoding="utf-8")

    cfg = {"patterns": ["**/*.*"], "root": "projectA", "output": "out/config_root.md"}
    (tmp_path / ".bundler.json").write_text(json.dumps(cfg), encoding="utf-8")

    # CLI root should override config.root; config.output still applies (resolved against CLI root)
    rc = main(["--root", str(tmp_path / "projectB"), "--config", str(tmp_path / ".bundler.json")])
    assert rc == 0
    assert (tmp_path / "projectB" / "out" / "config_root.md").exists()
    assert not (tmp_path / "projectB" / "BUNDLE.md").exists()


def test_collect_files_pattern_order(tmp_path: Path):
    """Files should appear in the same order the patterns were specified."""
    # create two files with predictable names
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"

    # explicit patterns, reversed order
    files = _collect_files(tmp_path, ["b.txt", "a.txt"])
    assert files == [b, a]

    # glob followed by explicit: glob returns sorted list, and duplicate is skipped
    files = _collect_files(tmp_path, ["*.txt", "b.txt"])
    assert files == [a, b]

    # patterns with modifiers still respect order
    files = _collect_files(tmp_path, ["*.txt[+1]", "*.txt[-1]"])
    # first pattern keeps first entry (a), second keeps last entry (b)
    assert files == [a, b]


def test_collect_files_lexicographic_order_without_modifier(tmp_path: Path):
    """Glob matches should be lexicographically sorted without modifiers."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_41 = data_dir / "file41.md"
    file_40 = data_dir / "file40.md"
    file_41.write_text("41", encoding="utf-8")
    file_40.write_text("40", encoding="utf-8")

    files = _collect_files(tmp_path, ["data/**/*4*.md"])

    assert files == [file_40, file_41]


def test_bundle_respects_pattern_order(tmp_path: Path):
    """Bundled output must follow the order of the patterns list."""
    # create several files in different locations
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "one.py").write_text("print('1')", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.sh").write_text("echo hi", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "file.md").write_text("data", encoding="utf-8")

    cfg = {
        "patterns": [
            "README.md",
            "src/**/*.py",
            "scripts/*.sh",
            "data/**/*.md",
        ],
        "output": "bundle.md",
    }
    (tmp_path / ".bundler.json").write_text(json.dumps(cfg), encoding="utf-8")

    rc = main(["--root", str(tmp_path), "--config", ".bundler.json"])
    assert rc == 0
    out = (tmp_path / "bundle.md").read_text(encoding="utf-8")

    # ensure appearance order
    idx = {name: out.find(name) for name in ["README.md", "src/", "scripts/", "data/"]}
    assert idx["README.md"] != -1
    assert idx["src/"] != -1
    assert idx["scripts/"] != -1
    assert idx["data/"] != -1

    assert idx["README.md"] < idx["src/"] < idx["scripts/"] < idx["data/"]

    # verify each relevant section appears only once
    assert out.count("README.md") == 1
    assert out.count("src/") >= 1 and out.count("src/") < 3  # header + maybe paths
    # ensure the scripts header only once
    assert out.count("scripts/") == 1
    assert out.count("data/") >= 1 and out.count("data/") < 3


def test_toc_order_matches_pattern(tmp_path: Path):
    """Table of contents entries should follow the order of patterns."""
    # set up files similar to previous test
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "one.py").write_text("print('1')", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.sh").write_text("echo hi", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "file.md").write_text("data", encoding="utf-8")

    cfg = {
        "patterns": [
            "README.md",
            "src/**/*.py",
            "scripts/*.sh",
            "data/**/*.md",
        ],
        "output": "bundle.md",
        "toc": {"README.md": "read me"},
    }
    (tmp_path / ".bundler.json").write_text(json.dumps(cfg), encoding="utf-8")

    rc = main(["--root", str(tmp_path), "--config", ".bundler.json", "--toc"])
    assert rc == 0
    out = (tmp_path / "bundle.md").read_text(encoding="utf-8")

    # search TOC section specifically
    toc_start = out.find("## Table of Contents")
    assert toc_start != -1
    toc_section = out[toc_start:]
    # entries should appear in pattern sequence
    assert toc_section.index("README.md") < toc_section.index("src/")
    assert toc_section.index("src/") < toc_section.index("scripts/")
    assert toc_section.index("scripts/") < toc_section.index("data/")


def test_cli_output_overrides_config(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')", encoding="utf-8")

    cfg = {"patterns": ["**/*.*"], "output": "out/config.md"}
    (tmp_path / ".bundler.json").write_text(json.dumps(cfg), encoding="utf-8")

    cli_out = tmp_path / "override.md"
    rc = main(["--root", str(tmp_path), "--config", ".bundler.json", "--output", str(cli_out)])
    assert rc == 0
    assert cli_out.exists()
    assert not (tmp_path / "out" / "config.md").exists()
