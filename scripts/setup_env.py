#!/usr/bin/env python3
"""
Setup script for the AI/ML coding interview assessment platform.

This script:
1. Checks the Python version
2. Creates a virtual environment
3. Installs the package and its dependencies
4. Verifies the installation
"""

import sys
import subprocess
import os
import platform
from pathlib import Path


def check_for_pyenv():
    """
    Check if pyenv is installed and available.
    
    Returns:
        bool: True if pyenv is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["which", "pyenv"] if os.name != "nt" else ["where", "pyenv"],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def get_available_python_versions():
    """
    Get available Python versions using pyenv.
    
    Returns:
        list: List of available Python versions in the format (major, minor)
    """
    try:
        result = subprocess.run(
            ["pyenv", "versions", "--bare"],
            capture_output=True,
            text=True,
            check=True
        )
        
        versions = []
        for line in result.stdout.strip().split("\n"):
            if line:
                # Parse version string like "3.9.0" or "3.10.2"
                parts = line.split(".")
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    major = int(parts[0])
                    minor = int(parts[1])
                    if major == 3 and 9 <= minor <= 12:
                        versions.append((major, minor))
        
        return versions
    except Exception as e:
        print(f"Error getting Python versions from pyenv: {e}")
        return []


def check_python_version():
    """
    Check if the current Python version meets the requirements.
    
    Returns:
        bool: True if the current version meets the requirements, False otherwise
    """
    min_version = (3, 9)
    max_version = (3, 13)  # Exclusive
    
    # First check if pyenv is available and has a compatible version
    if check_for_pyenv():
        try:
            # Check if .python-version file exists (created by pyenv local)
            pyenv_version_file = Path(".python-version")
            if pyenv_version_file.exists():
                with open(pyenv_version_file, 'r') as f:
                    pyenv_version = f.read().strip()
                    print(f"Found pyenv local version: {pyenv_version}")
                    
                    # Parse the version string
                    parts = pyenv_version.split(".")
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        major = int(parts[0])
                        minor = int(parts[1])
                        if major == 3 and 9 <= minor <= 12:
                            print(f"Pyenv local version {major}.{minor} is compatible.")
                            return True
                        else:
                            print(f"Pyenv local version {major}.{minor} is not compatible (need 3.9-3.12).")
            
            # If no compatible local version, check available versions
            print("Checking for compatible Python versions in pyenv...")
            available_versions = get_available_python_versions()
            
            if available_versions:
                # Sort versions to get the highest available
                available_versions.sort(reverse=True)
                highest_version = available_versions[0]
                print(f"Found compatible Python version: {highest_version[0]}.{highest_version[1]}")
                print(f"Please run: pyenv local {highest_version[0]}.{highest_version[1]}")
                print("Then run this setup script again.")
            else:
                print("No compatible Python versions found in pyenv.")
                print("Please install a compatible version (3.9-3.12) using:")
                print("pyenv install 3.12.0")
                
            return False
        except Exception as e:
            print(f"Error checking pyenv version: {e}")
    
    # Fall back to checking the current Python version
    current_version = sys.version_info[:2]
    
    if current_version < min_version or current_version >= max_version:
        print(f"Error: Python 3.9 through 3.12 is required.")
        print(f"Current version: {current_version[0]}.{current_version[1]}")
        
        if not check_for_pyenv():
            print("pyenv is not available. Please install Python 3.9-3.12 manually.")
        
        return False
    
    print(f"Python version {current_version[0]}.{current_version[1]} is compatible.")
    return True


def get_best_python_executable():
    """
    Get the path to the best available Python executable.
    
    Returns:
        str: Path to the Python executable
    """
    # If pyenv is available, try to use the local version first
    if check_for_pyenv():
        try:
            # Check if .python-version file exists (created by pyenv local)
            pyenv_version_file = Path(".python-version")
            if pyenv_version_file.exists():
                with open(pyenv_version_file, 'r') as f:
                    pyenv_version = f.read().strip()
                    print(f"Found pyenv local version: {pyenv_version}")
                    
                    # Parse the version string
                    parts = pyenv_version.split(".")
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        major = int(parts[0])
                        minor = int(parts[1])
                        if major == 3 and 9 <= minor <= 12:
                            # Get the path to the Python executable for this version
                            result = subprocess.run(
                                ["pyenv", "which", f"python{major}.{minor}"],
                                capture_output=True,
                                text=True,
                                check=True
                            )
                            python_path = result.stdout.strip()
                            if os.path.exists(python_path):
                                print(f"Using Python {major}.{minor} from pyenv local: {python_path}")
                                return python_path
            
            # If no compatible local version, use the highest available version
            available_versions = get_available_python_versions()
            if available_versions:
                # Sort versions to get the highest available
                available_versions.sort(reverse=True)
                highest_version = available_versions[0]
                version_str = f"{highest_version[0]}.{highest_version[1]}"
                
                # Get the path to the Python executable for this version
                result = subprocess.run(
                    ["pyenv", "which", f"python{version_str}"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                python_path = result.stdout.strip()
                if os.path.exists(python_path):
                    print(f"Using Python {version_str} from pyenv: {python_path}")
                    return python_path
        except Exception as e:
            print(f"Error getting Python path from pyenv: {e}")
    
    # Fall back to the current Python executable
    print(f"Using current Python: {sys.executable}")
    return sys.executable


def create_venv():
    """
    Create a virtual environment.
    
    Returns:
        Path: Path to the virtual environment
    """
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print(f"Virtual environment already exists at {venv_path}")
        return venv_path
    
    print("Creating virtual environment...")
    try:
        python_executable = get_best_python_executable()
        subprocess.run([python_executable, "-m", "venv", str(venv_path)], check=True)
        print(f"Virtual environment created at {venv_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        sys.exit(1)
    
    return venv_path


def get_python_executable(venv_path):
    """
    Get the path to the Python executable in the virtual environment.
    
    Args:
        venv_path (Path): Path to the virtual environment
        
    Returns:
        Path: Path to the Python executable
    """
    if os.name == "nt":  # Windows
        return venv_path / "Scripts" / "python.exe"
    else:  # Unix/Linux/MacOS
        return venv_path / "bin" / "python"


def install_package(venv_path):
    """
    Install the package and its dependencies.
    
    Args:
        venv_path (Path): Path to the virtual environment
    """
    python_path = get_python_executable(venv_path)
    
    print("Installing package and dependencies...")
    try:
        # Upgrade pip
        subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install the package in development mode
        subprocess.run([str(python_path), "-m", "pip", "install", "-e", "."], check=True)
        
        print("Installation complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing package: {e}")
        sys.exit(1)


def verify_installation(venv_path):
    """
    Verify that the package and its dependencies are installed correctly.
    
    Args:
        venv_path (Path): Path to the virtual environment
        
    Returns:
        bool: True if the installation is verified, False otherwise
    """
    python_path = get_python_executable(venv_path)
    
    print("Verifying installation...")
    
    # Check if the package can be imported
    try:
        result = subprocess.run(
            [str(python_path), "-c", "import interview; print('Package imported successfully')"],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error importing package: {e}")
        print(f"Output: {e.output}")
        return False
    
    # Check if required dependencies are installed
    dependencies = [
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "jupyter",
        "ipykernel"
    ]
    
    for dep in dependencies:
        try:
            result = subprocess.run(
                [str(python_path), "-c", f"import {dep}; print('{dep} imported successfully')"],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout.strip())
        except subprocess.CalledProcessError:
            print(f"Error importing {dep}")
            return False
    
    print("All dependencies verified.")
    return True


def print_activation_instructions(venv_path):
    """
    Print instructions for activating the virtual environment.
    
    Args:
        venv_path (Path): Path to the virtual environment
    """
    print("\nSetup complete! You can activate the virtual environment with:")
    
    if os.name == "nt":  # Windows
        print(f"{venv_path}\\Scripts\\activate")
    else:  # Unix/Linux/MacOS
        print(f"source {venv_path}/bin/activate")
    
    print("\nAfter activation, you can run the assessment notebook with:")
    print("jupyter notebook notebooks/pokemon_assessment.ipynb")
    
    print("\nOr with VSCode:")
    print("code notebooks/pokemon_assessment.ipynb")


def main():
    """Main function."""
    print("=" * 80)
    print("AI/ML Coding Interview Assessment Platform Setup".center(80))
    print("=" * 80)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    venv_path = create_venv()
    
    # Install package
    install_package(venv_path)
    
    # Verify installation
    if verify_installation(venv_path):
        print_activation_instructions(venv_path)
    else:
        print("Installation verification failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()