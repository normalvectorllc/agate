"""
Tests for the utils module.
"""

import os
import sys
import platform
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from interview import utils


def test_check_python_version():
    """Test check_python_version function."""
    # Current Python version should be compatible
    assert utils.check_python_version() is True
    
    # Test with a future version
    future_version = (sys.version_info[0] + 1, 0)
    assert utils.check_python_version(future_version) is False


def test_get_python_version_info():
    """Test get_python_version_info function."""
    info = utils.get_python_version_info()
    
    assert isinstance(info, dict)
    assert 'version' in info
    assert 'implementation' in info
    assert 'compiler' in info
    assert 'build' in info
    assert 'executable' in info
    
    assert info['version'] == platform.python_version()
    assert info['implementation'] == platform.python_implementation()


def test_check_package_installed():
    """Test check_package_installed function."""
    # Test with a package that should be installed
    assert utils.check_package_installed('os') is True
    
    # Test with a package that should not be installed
    assert utils.check_package_installed('non_existent_package_123') is False


def test_get_package_version():
    """Test get_package_version function."""
    # Mock pkg_resources.get_distribution
    mock_dist = MagicMock()
    mock_dist.version = '1.2.3'
    
    with patch('pkg_resources.get_distribution', return_value=mock_dist):
        version = utils.get_package_version('some_package')
        assert version == '1.2.3'
    
    # Test with ImportError
    with patch('pkg_resources.get_distribution', side_effect=ImportError):
        version = utils.get_package_version('some_package')
        assert version is None


def test_check_required_packages():
    """Test check_required_packages function."""
    with patch('interview.utils.check_package_installed') as mock_check:
        # Set up the mock to return True for 'os' and False for 'non_existent_package'
        def side_effect(package):
            return package == 'os'
        
        mock_check.side_effect = side_effect
        
        result = utils.check_required_packages(['os', 'non_existent_package'])
        
        assert isinstance(result, dict)
        assert result['os'] is True
        assert result['non_existent_package'] is False


def test_get_os_info():
    """Test get_os_info function."""
    info = utils.get_os_info()
    
    assert isinstance(info, dict)
    assert 'system' in info
    assert 'release' in info
    assert 'version' in info
    assert 'machine' in info
    assert 'processor' in info
    
    assert info['system'] == platform.system()
    assert info['release'] == platform.release()


def test_is_notebook():
    """Test is_notebook function."""
    # In a standard Python environment, this should return False
    assert utils.is_notebook() is False


def test_format_time():
    """Test format_time function."""
    # Test with seconds
    assert utils.format_time(30) == "30.00 seconds"
    
    # Test with minutes
    assert utils.format_time(90) == "1.50 minutes"
    
    # Test with hours
    assert utils.format_time(3600) == "1.00 hours"
    assert utils.format_time(5400) == "1.50 hours"


def test_create_directory_if_not_exists(tmpdir):
    """Test create_directory_if_not_exists function."""
    # Test with a non-existent directory
    test_dir = tmpdir / "test_dir"
    utils.create_directory_if_not_exists(test_dir)
    assert os.path.exists(test_dir)
    
    # Test with an existing directory
    utils.create_directory_if_not_exists(test_dir)
    assert os.path.exists(test_dir)


def test_get_project_root():
    """Test get_project_root function."""
    root = utils.get_project_root()
    
    assert isinstance(root, Path)
    assert (root / "interview").exists()


def test_get_notebook_path():
    """Test get_notebook_path function."""
    path = utils.get_notebook_path()
    
    assert isinstance(path, Path)
    assert path.name == "notebooks"
    assert path.parent == utils.get_project_root()


def test_get_dataset_path():
    """Test get_dataset_path function."""
    path = utils.get_dataset_path()
    
    assert isinstance(path, Path)
    assert path.name == "datasets"
    assert path.parent == utils.get_project_root()


def test_print_section_header(capsys):
    """Test print_section_header function."""
    utils.print_section_header("Test Header")
    
    captured = capsys.readouterr()
    assert "=" * 80 in captured.out
    assert "Test Header" in captured.out


def test_print_step_header(capsys):
    """Test print_step_header function."""
    utils.print_step_header(1, "Test Step")
    
    captured = capsys.readouterr()
    assert "-" * 80 in captured.out
    assert "Step 1: Test Step" in captured.out


def test_print_code_hint(capsys):
    """Test print_code_hint function."""
    utils.print_code_hint("Test Hint")
    
    captured = capsys.readouterr()
    assert "Hint:" in captured.out
    assert "Test Hint" in captured.out


def test_print_success_message(capsys):
    """Test print_success_message function."""
    utils.print_success_message("Test Success")
    
    captured = capsys.readouterr()
    assert "✅" in captured.out
    assert "Test Success" in captured.out


def test_print_error_message(capsys):
    """Test print_error_message function."""
    utils.print_error_message("Test Error")
    
    captured = capsys.readouterr()
    assert "❌" in captured.out
    assert "Test Error" in captured.out


def test_print_warning_message(capsys):
    """Test print_warning_message function."""
    utils.print_warning_message("Test Warning")
    
    captured = capsys.readouterr()
    assert "⚠️" in captured.out
    assert "Test Warning" in captured.out


def test_print_info_message(capsys):
    """Test print_info_message function."""
    utils.print_info_message("Test Info")
    
    captured = capsys.readouterr()
    assert "ℹ️" in captured.out
    assert "Test Info" in captured.out