"""
General utility functions for the AI/ML coding interview assessment platform.
"""

import os
import sys
import platform
import subprocess
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any


def check_python_version(min_version: Tuple[int, int] = (3, 9)) -> bool:
    """
    Check if the current Python version meets the minimum requirement.
    
    Args:
        min_version (Tuple[int, int], optional): Minimum required version. Defaults to (3, 9).
    
    Returns:
        bool: True if the current version meets the requirement, False otherwise
    """
    current_version = sys.version_info[:2]
    return current_version >= min_version


def get_python_version_info() -> Dict[str, str]:
    """
    Get information about the current Python environment.
    
    Returns:
        Dict[str, str]: Dictionary with Python version information
    """
    return {
        'version': platform.python_version(),
        'implementation': platform.python_implementation(),
        'compiler': platform.python_compiler(),
        'build': platform.python_build()[0],
        'executable': sys.executable
    }


def check_package_installed(package_name: str) -> bool:
    """
    Check if a Python package is installed.
    
    Args:
        package_name (str): Name of the package to check
    
    Returns:
        bool: True if the package is installed, False otherwise
    """
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def get_installed_packages() -> List[str]:
    """
    Get a list of installed Python packages.
    
    Returns:
        List[str]: List of installed package names
    """
    try:
        import pkg_resources
        return [pkg.key for pkg in pkg_resources.working_set]
    except ImportError:
        return []


def get_package_version(package_name: str) -> Optional[str]:
    """
    Get the version of an installed package.
    
    Args:
        package_name (str): Name of the package
    
    Returns:
        Optional[str]: Version string if the package is installed, None otherwise
    """
    try:
        import pkg_resources
        return pkg_resources.get_distribution(package_name).version
    except (ImportError, pkg_resources.DistributionNotFound):
        return None


def check_required_packages(required_packages: List[str]) -> Dict[str, bool]:
    """
    Check if all required packages are installed.
    
    Args:
        required_packages (List[str]): List of required package names
    
    Returns:
        Dict[str, bool]: Dictionary mapping package names to installation status
    """
    return {pkg: check_package_installed(pkg) for pkg in required_packages}


def get_os_info() -> Dict[str, str]:
    """
    Get information about the operating system.
    
    Returns:
        Dict[str, str]: Dictionary with OS information
    """
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor()
    }


def is_notebook() -> bool:
    """
    Check if the code is running in a Jupyter notebook.
    
    Returns:
        bool: True if running in a notebook, False otherwise
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal IPython
        else:
            return False  # Other type
    except NameError:
        return False  # Standard Python interpreter


def get_memory_usage() -> Dict[str, float]:
    """
    Get memory usage information for the current process.
    
    Returns:
        Dict[str, float]: Dictionary with memory usage in MB
    """
    import psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        'rss': mem_info.rss / (1024 * 1024),  # Resident Set Size in MB
        'vms': mem_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
    }


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def create_directory_if_not_exists(directory_path: Union[str, Path]) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path (Union[str, Path]): Path to the directory
    """
    path = Path(directory_path)
    if not path.exists():
        path.mkdir(parents=True)


def get_project_root() -> Path:
    """
    Get the root directory of the project.
    
    Returns:
        Path: Path to the project root directory
    """
    return Path(__file__).parent.parent


def get_notebook_path() -> Path:
    """
    Get the path to the notebooks directory.
    
    Returns:
        Path: Path to the notebooks directory
    """
    return get_project_root() / "notebooks"


def get_dataset_path() -> Path:
    """
    Get the path to the datasets directory.
    
    Returns:
        Path: Path to the datasets directory
    """
    return get_project_root() / "datasets"


def print_section_header(title: str, width: int = 80, char: str = "=") -> None:
    """
    Print a section header with the given title.
    
    Args:
        title (str): Section title
        width (int, optional): Width of the header. Defaults to 80.
        char (str, optional): Character to use for the header. Defaults to "=".
    """
    print(char * width)
    print(title.center(width))
    print(char * width)


def print_step_header(step_number: int, title: str, width: int = 80) -> None:
    """
    Print a step header with the given step number and title.
    
    Args:
        step_number (int): Step number
        title (str): Step title
        width (int, optional): Width of the header. Defaults to 80.
    """
    header = f"Step {step_number}: {title}"
    print("\n" + "-" * width)
    print(header)
    print("-" * width + "\n")


def print_code_hint(hint: str) -> None:
    """
    Print a code hint with proper formatting.
    
    Args:
        hint (str): The hint to print
    """
    print("\nHint:")
    print("-----")
    print(hint)
    print("-----\n")


def print_success_message(message: str) -> None:
    """
    Print a success message with proper formatting.
    
    Args:
        message (str): The success message to print
    """
    print("\n✅ " + message + "\n")


def print_error_message(message: str) -> None:
    """
    Print an error message with proper formatting.
    
    Args:
        message (str): The error message to print
    """
    print("\n❌ " + message + "\n")


def print_warning_message(message: str) -> None:
    """
    Print a warning message with proper formatting.
    
    Args:
        message (str): The warning message to print
    """
    print("\n⚠️ " + message + "\n")


def print_info_message(message: str) -> None:
    """
    Print an info message with proper formatting.
    
    Args:
        message (str): The info message to print
    """
    print("\nℹ️ " + message + "\n")