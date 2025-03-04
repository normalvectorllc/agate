# GitHub Codespaces Configuration for AI/ML Interviews

This directory contains configuration files for setting up a GitHub Codespace optimized for conducting AI/ML technical interviews using this repository.

## Setup Process

The environment setup follows these steps:

1. The Dockerfile creates a base container with Python 3.12 and essential system dependencies
2. When the codespace is created, `scripts/setup_env.py` is executed automatically which:
   - Creates a Python virtual environment (`.venv`)
   - Installs the project package and its dependencies
   - Verifies the installation of core libraries
3. Additional ML libraries are installed after the virtual environment is created
4. VS Code settings are updated to use the Python interpreter from the virtual environment

## Setup Features

- **Python 3.12** environment with virtual environment (`.venv`)
- Pre-installed ML libraries:
  - Core packages from project dependencies: pandas, numpy, scikit-learn, tensorflow, torch, matplotlib, seaborn
  - Additional ML packages: plotly, xgboost, lightgbm, catboost, shap, eli5
- Jupyter notebook support with extensions
- Development tools for code quality (black, flake8, isort)
- Git and GitHub CLI tools

## Using This for Interviews

1. **Starting a Codespace**:
   - Click the "Code" button in your GitHub repository
   - Select the "Codespaces" tab
   - Click "Create codespace on main"

2. **Prepare the Interview**:
   - The codespace will automatically run `setup_env.py` to set up the environment
   - The virtual environment will be activated by default
   - Open the `notebooks/` directory to access or create interview notebooks
   - Use the Pokemon dataset from `datasets/pokemon/` for data analysis tasks

3. **During the Interview**:
   - Share your Codespace URL with the candidate (if pair programming)
   - Or use VS Code's Live Share extension for real-time collaboration
   - Guide candidates through the assessment using the repository resources

4. **Post-Interview**:
   - Save any candidate solutions within the Codespace
   - Commit and push changes if you want to preserve the work
   - Stop the Codespace when not in use to conserve resources

## Customization

If you need to customize the environment further:

- Edit `Dockerfile` to add system dependencies 
- Modify `devcontainer.json` to change VS Code settings or extensions
- Update the `postCreateCommand` to install additional packages

## Troubleshooting

If encountering issues with the environment:

1. Check the terminal logs for any errors during the execution of `setup_env.py`
2. Manually run the setup script: `python scripts/setup_env.py`
3. Verify that all dependencies were installed correctly with `. .venv/bin/activate && pip list`
4. Rebuild the container if necessary from VS Code's command palette with "Rebuild Container" 