# Stroke Prediction Analysis Project

## Overview
This project analyzes brain stroke data to identify risk factors and predict stroke occurrence using statistic methods and machine learning techniques. The analysis includes data cleaning, exploratory data analysis, visualization, and predictive modeling.

## Primary Source/Article
- Dataset from Kaggle: Brain Stroke Dataset
- URL: https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset

## Project Data
Raw data file: `brain_stroke.csv` (4,981 records with 11 features)

## Project Structure
```
stroke_py_project/
│
├── src/                        # Source code
│   ├── data_analysis.py       # Statistical analysis functions
│   ├── data_cleaning.py       # Data preprocessing functions
│   ├── data_visualization.py  # Visualization functions
│   └── model_tree.py         # Machine learning model implementation
│
├── tests/                     # Test files
│   ├── test_data_analysis.py
│   ├── test_data_cleaning.py
│   └── test_model_tree.py
│
├── graphs/                    # Generated visualizations
├── model_output/             # Model evaluation results
├── brain_stroke.csv          # Raw data file
├── main.py                   # Main execution script
├── my_project.code-workspace # VS Code workspace settings
├── pyproject.toml            # Project dependencies and settings
├── README.md                 # Project documentation
└── stoke_report.docx         # Scientific report
```

### Features Analyzed
- Demographic: age, gender, ever_married
- Medical: hypertension, heart_disease
- Lifestyle: bmi, smoking_status, avg_glucose_level
- Socioeconomic: work_type, Residence_type

## Installation and Setup

All commands should run under project root directory:

```bash
## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/SagiHai/stroke_py_project.git
cd stroke_py_project

# Install virtualenv
pip install virtualenv

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\.venv\Scripts\activate
# (venv) should appear as prefix

# Update pip
python.exe -m pip install --upgrade pip

# Install project packages
pip install -r requirements.txt

# Install development packages
pip install -e .[dev]
```

## Running the Analysis

1. Activate the virtual environment:
```bash
.\.venv\Scripts\activate
```

2. Run the main analysis:
```bash
python main.py
```

3. Run tests:
```bash
python -m pytest tests/
```

## Project Components

1. Data Cleaning
   - Missing value handling
   - Outlier removal

2. Analysis Pipeline
   - Exploratory Data Analysis
   - Statistical Testing
   - Feature Importance Analysis
   - Analyze Stoke by Gender

3. Visualization
   - Distribution Analysis
   - Risk Factor Correlations
   - Visualize Stoke by Gender
   
4. Prediction ML Model
   - Data type conversion #preprocessing
   - ML Model Development
   - Model Performance Metrics #Visualization - results
   - Classification report #Model results

## Results

Results are saved in:
- `graphs/`: Visualization outputs
- `model_output/`: Model performance metrics and classification report
- `stoke_report.docx`: Full scientific report

## Dependencies
- Python 3.8+
- Core packages:
  - pandas>=2.2.3
  - numpy>=2.2.1
  - scikit-learn>=1.6.1
  - matplotlib>=3.10.0
  - seaborn>=0.13.2

## Development
- Testing: pytest
- Formatting: black
- Linting: pylint
- Type checking: mypy