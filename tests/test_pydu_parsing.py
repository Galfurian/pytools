"""Tests for pydu module."""

import pytest
from pydu import parse_size, parse_time_duration, process_patterns


class TestParseSize:
    """Test parse_size function."""

    def test_parse_bytes(self):
        assert parse_size("1024") == 1024

    def test_parse_kb(self):
        assert parse_size("1KB") == 1024

    def test_parse_mb(self):
        assert parse_size("1MB") == 1024**2

    def test_parse_gb(self):
        assert parse_size("1GB") == 1024**3

    def test_parse_tb(self):
        assert parse_size("1TB") == 1024**4

    def test_parse_with_spaces(self):
        assert parse_size(" 1 MB ") == 1024**2

    def test_parse_float(self):
        assert parse_size("1.5MB") == int(1.5 * 1024**2)

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            parse_size("invalid")


class TestParseTimeDuration:
    """Test parse_time_duration function."""

    def test_parse_seconds(self):
        assert parse_time_duration("60s") == 60

    def test_parse_minutes(self):
        assert parse_time_duration("1M") == 30 * 24 * 3600  # 30 days

    def test_parse_hours(self):
        assert parse_time_duration("1h") == 3600

    def test_parse_days(self):
        assert parse_time_duration("1d") == 24 * 3600

    def test_parse_weeks(self):
        assert parse_time_duration("1w") == 7 * 24 * 3600

    def test_parse_years(self):
        assert parse_time_duration("1y") == 365 * 24 * 3600

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            parse_time_duration("invalid")


class TestProcessPatterns:
    """Test process_patterns function."""

    def test_single_pattern(self):
        assert process_patterns(["*.py"]) == ["*.py"]

    def test_comma_separated(self):
        assert process_patterns(["*.py,*.txt"]) == ["*.py", "*.txt"]

    def test_multiple_args(self):
        assert process_patterns(["*.py", "*.txt"]) == ["*.py", "*.txt"]

    def test_empty_and_whitespace(self):
        assert process_patterns(["", " ", "*.py"]) == ["*.py"]
