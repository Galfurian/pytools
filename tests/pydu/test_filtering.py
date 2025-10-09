"""Tests for pydu filtering functions."""

import argparse
import copy
import pytest
from src.pydu import node_matches, filter_tree, TreeNode, NodeType


@pytest.fixture
def sample_tree():
    """Create a sample tree for testing."""
    # Root directory
    root = TreeNode(
        name="root",
        depth=0,
        size=0,  # Will be calculated
        original_size=0,
        mtime=0,
        mode=0o755,
        uid=1000,
        gid=1000,
        node_type=NodeType.DIRECTORY,
    )

    # Files
    file_py = TreeNode(
        name="script.py",
        depth=1,
        size=1024,
        original_size=1024,
        mtime=0,
        mode=0o644,
        uid=1000,
        gid=1000,
        node_type=NodeType.FILE,
        parent=root,
    )

    file_txt = TreeNode(
        name="readme.txt",
        depth=1,
        size=512,
        original_size=512,
        mtime=0,
        mode=0o644,
        uid=1000,
        gid=1000,
        node_type=NodeType.FILE,
        parent=root,
    )

    file_hidden = TreeNode(
        name=".hidden",
        depth=1,
        size=256,
        original_size=256,
        mtime=0,
        mode=0o644,
        uid=1000,
        gid=1000,
        node_type=NodeType.FILE,
        parent=root,
    )

    # Subdirectory
    subdir = TreeNode(
        name="subdir",
        depth=1,
        size=0,  # Will be calculated
        original_size=0,
        mtime=0,
        mode=0o755,
        uid=1000,
        gid=1000,
        node_type=NodeType.DIRECTORY,
        parent=root,
    )

    # File in subdir
    sub_file = TreeNode(
        name="data.txt",
        depth=2,
        size=2048,
        original_size=2048,
        mtime=0,
        mode=0o644,
        uid=1000,
        gid=1000,
        node_type=NodeType.FILE,
        parent=subdir,
    )

    # Symlink
    symlink = TreeNode(
        name="link",
        depth=1,
        size=10,
        original_size=10,
        mtime=0,
        mode=0o777,
        uid=1000,
        gid=1000,
        node_type=NodeType.SYMLINK,
        parent=root,
    )

    # Build tree
    subdir.children = [sub_file]
    subdir.size = sub_file.size
    subdir.original_size = sub_file.size

    root.children = [file_py, file_txt, file_hidden, subdir, symlink]
    root.size = sum(child.size for child in root.children)
    root.original_size = root.size

    return copy.deepcopy(root)


class TestNodeMatches:
    """Test node_matches function."""

    def test_no_filters(self):
        """Test with no filters applied."""
        args = argparse.Namespace(
            exclude=[],
            include=[],
            show_hidden=False,
            only_dirs=False,
            min_size=None,
            max_size=None,
            older_than=None,
        )
        node = TreeNode(
            name="file.txt",
            depth=1,
            size=1000,
            original_size=1000,
            mtime=0,
            mode=0o644,
            uid=1000,
            gid=1000,
            node_type=NodeType.FILE,
        )
        assert node_matches(node, args) is True

    def test_exclude_pattern(self):
        """Test exclude pattern matching."""
        args = argparse.Namespace(
            exclude=["*.txt"],
            include=[],
            show_hidden=False,
            only_dirs=False,
            min_size=None,
            max_size=None,
            older_than=None,
        )
        node = TreeNode(
            name="file.txt",
            depth=1,
            size=1000,
            original_size=1000,
            mtime=0,
            mode=0o644,
            uid=1000,
            gid=1000,
            node_type=NodeType.FILE,
        )
        assert node_matches(node, args) is False

    def test_include_pattern(self):
        """Test include pattern matching for files."""
        args = argparse.Namespace(
            exclude=[],
            include=["*.py"],
            show_hidden=False,
            only_dirs=False,
            min_size=None,
            max_size=None,
            older_than=None,
        )
        file_node = TreeNode(
            name="script.py",
            depth=1,
            size=1000,
            original_size=1000,
            mtime=0,
            mode=0o644,
            uid=1000,
            gid=1000,
            node_type=NodeType.FILE,
        )
        txt_node = TreeNode(
            name="file.txt",
            depth=1,
            size=1000,
            original_size=1000,
            mtime=0,
            mode=0o644,
            uid=1000,
            gid=1000,
            node_type=NodeType.FILE,
        )
        assert node_matches(file_node, args) is True
        assert node_matches(txt_node, args) is False

    def test_hidden_files(self):
        """Test show_hidden filter."""
        args = argparse.Namespace(
            exclude=[],
            include=[],
            show_hidden=False,
            only_dirs=False,
            min_size=None,
            max_size=None,
            older_than=None,
        )
        hidden_node = TreeNode(
            name=".hidden",
            depth=1,
            size=1000,
            original_size=1000,
            mtime=0,
            mode=0o644,
            uid=1000,
            gid=1000,
            node_type=NodeType.FILE,
        )
        assert node_matches(hidden_node, args) is False

        args.show_hidden = True
        assert node_matches(hidden_node, args) is True

    def test_only_dirs(self):
        """Test only_dirs filter."""
        args = argparse.Namespace(
            exclude=[],
            include=[],
            show_hidden=False,
            only_dirs=True,
            min_size=None,
            max_size=None,
            older_than=None,
        )
        file_node = TreeNode(
            name="file.txt",
            depth=1,
            size=1000,
            original_size=1000,
            mtime=0,
            mode=0o644,
            uid=1000,
            gid=1000,
            node_type=NodeType.FILE,
        )
        dir_node = TreeNode(
            name="dir",
            depth=1,
            size=1000,
            original_size=1000,
            mtime=0,
            mode=0o755,
            uid=1000,
            gid=1000,
            node_type=NodeType.DIRECTORY,
        )
        assert node_matches(file_node, args) is False
        assert node_matches(dir_node, args) is True

    def test_size_filters(self):
        """Test min_size and max_size filters."""
        args = argparse.Namespace(
            exclude=[],
            include=[],
            show_hidden=False,
            only_dirs=False,
            min_size=500,
            max_size=2000,
            older_than=None,
        )
        small_node = TreeNode(
            name="small.txt",
            depth=1,
            size=100,
            original_size=100,
            mtime=0,
            mode=0o644,
            uid=1000,
            gid=1000,
            node_type=NodeType.FILE,
        )
        medium_node = TreeNode(
            name="medium.txt",
            depth=1,
            size=1000,
            original_size=1000,
            mtime=0,
            mode=0o644,
            uid=1000,
            gid=1000,
            node_type=NodeType.FILE,
        )
        large_node = TreeNode(
            name="large.txt",
            depth=1,
            size=3000,
            original_size=3000,
            mtime=0,
            mode=0o644,
            uid=1000,
            gid=1000,
            node_type=NodeType.FILE,
        )
        assert node_matches(small_node, args) is False  # too small
        assert node_matches(medium_node, args) is True
        assert node_matches(large_node, args) is False  # too large


class TestFilterTree:
    """Test filter_tree function."""

    def test_no_filters(self, sample_tree):
        """Test with no filters - should keep everything."""
        args = argparse.Namespace(
            exclude=[],
            include=[],
            show_hidden=True,  # Include hidden files
            only_dirs=False,
            min_size=None,
            max_size=None,
            older_than=None,
        )

        result = filter_tree(sample_tree, args)
        assert result is not None
        assert len(result.children) == 5  # All children kept

    def test_filter_out_hidden(self, sample_tree):
        """Test filtering out hidden files."""
        args = argparse.Namespace(
            exclude=[],
            include=[],
            show_hidden=False,
            only_dirs=False,
            min_size=None,
            max_size=None,
            older_than=None,
        )

        result = filter_tree(sample_tree, args)
        assert result is not None
        # Should have 4 children: script.py, readme.txt, subdir, link (.hidden filtered out)
        assert len(result.children) == 4
        names = [child.name for child in result.children]
        assert ".hidden" not in names

    def test_only_dirs(self, sample_tree):
        """Test only_dirs filter."""
        args = argparse.Namespace(
            exclude=[],
            include=[],
            show_hidden=False,
            only_dirs=True,
            min_size=None,
            max_size=None,
            older_than=None,
        )

        result = filter_tree(sample_tree, args)
        assert result is not None
        # Should have 1 child: subdir (only directory)
        assert len(result.children) == 1
        assert result.children[0].name == "subdir"

    def test_exclude_pattern(self, sample_tree):
        """Test exclude pattern."""
        args = argparse.Namespace(
            exclude=["*.txt"],
            include=[],
            show_hidden=False,  # Hidden files excluded
            only_dirs=False,
            min_size=None,
            max_size=None,
            older_than=None,
        )

        result = filter_tree(sample_tree, args)
        assert result is not None
        # Should exclude readme.txt, data.txt, .hidden
        names = [child.name for child in result.children]
        assert "readme.txt" not in names
        assert ".hidden" not in names
        # subdir is removed because its child data.txt is excluded
        assert "subdir" not in names
        # Only script.py and link remain
        assert names == ["script.py", "link"]

    def test_include_pattern(self, sample_tree):
        """Test include pattern."""
        args = argparse.Namespace(
            exclude=[],
            include=["*.py"],
            show_hidden=False,
            only_dirs=False,
            min_size=None,
            max_size=None,
            older_than=None,
        )

        result = filter_tree(sample_tree, args)
        assert result is not None
        # Should only include script.py
        names = [child.name for child in result.children]
        assert names == ["script.py"]

    def test_size_filters(self, sample_tree):
        """Test size filters."""
        args = argparse.Namespace(
            exclude=[],
            include=[],
            show_hidden=False,
            only_dirs=False,
            min_size=1000,  # >= 1KB
            max_size=1500,  # <= 1.5KB
            older_than=None,
        )

        result = filter_tree(sample_tree, args)
        assert result is not None
        # Should include script.py (1024), exclude readme.txt (512 too small), subdir (2048 too big), etc.
        names = [child.name for child in result.children]
        assert "script.py" in names
        assert "readme.txt" not in names
        assert "subdir" not in names