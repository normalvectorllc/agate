# AI/ML Coding Interview Assessment Platform

A standardized environment for conducting AI/ML coding interviews using the Kaggle Pokemon dataset. This platform allows candidates to demonstrate their data analysis and model building skills in a live interview setting.

## Features

- **Standardized Assessment**: Structured prompts to evaluate candidates' skills in data exploration, statistical analysis, and model building
- **Cross-Platform Compatibility**: Works seamlessly across macOS, Linux, and Windows
- **Python Version Support**: Compatible with Python 3.9-3.12
- **Flexible Environment**: Supports both external Jupyter notebooks and VSCode's integrated notebook experience
- **Essential Libraries**: Includes pandas, numpy, scikit-learn, tensorflow, pytorch, matplotlib, seaborn, and other common data science packages

## Assessment Structure

The assessment follows a structured four-step process:

1. **Data Exploration**: Evaluate the candidate's familiarity with Python and data analysis libraries
2. **Distribution Analysis**: Assess statistical understanding and visualization skills
3. **Type Prediction Model**: Evaluate model building skills for classification
4. **Attack Prediction Model**: Assess model building skills for regression

## Getting Started

### Prerequisites

- Python 3.9 or higher

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/agate.git
   cd agate
   ```

2. Run the setup script to create a virtual environment and install dependencies:
   ```
   python scripts/setup_env.py
   ```

3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Unix/Linux/MacOS: `source .venv/bin/activate`

### Manual Installation

If you prefer to set up the environment manually:

1. Create a virtual environment:
   ```
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Unix/Linux/MacOS: `source .venv/bin/activate`

3. Install the package:
   ```
   pip install -e .
   ```

## Usage

1. Activate the virtual environment as described above.

2. Open the assessment notebook:
   ```
   jupyter notebook notebooks/pokemon_assessment.ipynb
   ```
   
   Or with VSCode:
   ```
   code notebooks/pokemon_assessment.ipynb
   ```

3. Follow the prompts in the notebook to guide the candidate through the assessment.

## Project Structure

```
agate/
├── datasets/              # Dataset files
│   └── pokemon/           # Pokemon dataset
├── interview/             # Main module package
│   ├── data.py            # Data loading and processing
│   ├── visualization.py   # Plotting and visualization
│   ├── models.py          # Model building utilities
│   └── utils.py           # General utilities
├── notebooks/             # Jupyter notebooks
│   ├── pokemon_assessment.ipynb  # Main assessment
│   └── solutions/         # Reference solutions
├── scripts/               # Utility scripts
│   └── setup_env.py       # Environment setup script
└── tests/                 # Test suite
```

## Data Conventions

The platform uses a consistent naming convention for dataset columns:

- All column names are lowercase (e.g., `attack` instead of `Attack`)
- Multi-word column names use underscores (e.g., `sp_attack` instead of `Sp. Atk`)
- Type columns are named `type1` and `type2`
- Stat columns include: `hp`, `attack`, `defense`, `sp_attack`, `sp_def`, `speed`
- Other columns include: `weight_kg`, `height_m`, `is_legendary`, `generation`

This convention makes the code more pythonic and easier to work with programmatically.

## Development

### Running Tests

```
pytest
```

### Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Pokemon dataset is sourced from Kaggle and is in the public domain (CC0)