from pathlib import Path
import pytest
import contextlib

from slapp.deploy import working_directory


def test_working_directory(tmp_path):
    with working_directory(tmp_path):
        assert Path.cwd() == tmp_path


def test_working_directory_fails_if_not_exists(tmp_path):
    with pytest.raises(FileNotFoundError):
        with working_directory(tmp_path / "abc123xyz"):
            pass
