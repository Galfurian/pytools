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
