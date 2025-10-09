"""Tests for pydu filtering functions."""

import argparse
from src.pydu import node_matches, filter_tree, TreeNode, NodeType


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

    def test_no_filters(self):
        """Test with no filters - should keep everything."""
        args = argparse.Namespace(
            exclude=[],
            include=[],
            show_hidden=False,
            only_dirs=False,
            min_size=None,
            max_size=None,
            older_than=None,
        )
        root = TreeNode(
            name="root",
            depth=0,
            size=2000,
            original_size=2000,
            mtime=0,
            mode=0o755,
            uid=1000,
            gid=1000,
            node_type=NodeType.DIRECTORY,
        )
        child = TreeNode(
            name="file.txt",
            depth=1,
            size=1000,
            original_size=1000,
            mtime=0,
            mode=0o644,
            uid=1000,
            gid=1000,
            node_type=NodeType.FILE,
            parent=root,
        )
        root.children = [child]

        result = filter_tree(root, args)
        assert result is not None
        assert len(result.children) == 1

    def test_filter_out_file(self):
        """Test filtering out a file."""
        args = argparse.Namespace(
            exclude=["*.txt"],
            include=[],
            show_hidden=False,
            only_dirs=False,
            min_size=None,
            max_size=None,
            older_than=None,
        )
        root = TreeNode(
            name="root",
            depth=0,
            size=2000,
            original_size=2000,
            mtime=0,
            mode=0o755,
            uid=1000,
            gid=1000,
            node_type=NodeType.DIRECTORY,
        )
        child = TreeNode(
            name="file.txt",
            depth=1,
            size=1000,
            original_size=1000,
            mtime=0,
            mode=0o644,
            uid=1000,
            gid=1000,
            node_type=NodeType.FILE,
            parent=root,
        )
        root.children = [child]

        result = filter_tree(root, args)
        assert result is None  # root has no children left

    def test_keep_directory_with_matching_children(self):
        """Test keeping directory when it has matching children."""
        args = argparse.Namespace(
            exclude=[],
            include=["*.py"],
            show_hidden=False,
            only_dirs=False,
            min_size=None,
            max_size=None,
            older_than=None,
        )
        root = TreeNode(
            name="root",
            depth=0,
            size=2000,
            original_size=2000,
            mtime=0,
            mode=0o755,
            uid=1000,
            gid=1000,
            node_type=NodeType.DIRECTORY,
        )
        py_child = TreeNode(
            name="script.py",
            depth=1,
            size=1000,
            original_size=1000,
            mtime=0,
            mode=0o644,
            uid=1000,
            gid=1000,
            node_type=NodeType.FILE,
            parent=root,
        )
        txt_child = TreeNode(
            name="file.txt",
            depth=1,
            size=1000,
            original_size=1000,
            mtime=0,
            mode=0o644,
            uid=1000,
            gid=1000,
            node_type=NodeType.FILE,
            parent=root,
        )
        root.children = [py_child, txt_child]

        result = filter_tree(root, args)
        assert result is not None
        assert len(result.children) == 1
        assert result.children[0].name == "script.py"