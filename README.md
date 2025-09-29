[![Test Pipeline](https://github.com/treychase/IDS_mini2/actions/workflows/ci.yml/badge.svg)](https://github.com/treychase/IDS_mini2/actions/workflows/ci.yml)

# Driveline Open Biomechanics Analysis

# Baseball Pitch Speed Prediction using Biomechanics Data

This repository contains a machine learning analysis that predicts baseball pitch velocity from peak biomechanics metrics. The project implements and compares three different regression models to determine the best approach for pitch speed prediction.

## Repository Structure

```
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore        
├── Makefile
├── driveline.py
├── test_driveline.py
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites
- Python 3.7 or higher
- pip3 package manager
- Your biomechanics data file named `hp_obp.csv`

### Installation & Basic Usage

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies:**
   ```bash
   make install
   ```

3. **Verify your data file exists:**
   ```bash
   make validate-data
   ```

4. **Run the complete analysis:**
   ```bash
   make run
   ```

5. **View all available commands:**
   ```bash
   make help
   ```

## Refactoring Changes

This section documents the refactoring improvements made to `driveline.py` to enhance code quality, readability, and maintainability.

### Extract Method Refactorings

Three new functions were extracted to reduce complexity and improve code organization:

#### 1. **`train_all_models()`** - Consolidated Model Training

**Before:**
```python
def run_complete_pipeline(filepath: str = 'hp_obp.csv') -> pd.DataFrame:
    # ... earlier code ...
    
    # Step 5: Train models
    print("\nStep 6: Training models...")
    lr_model = train_linear_regression(X_train_scaled, y_train)
    bayes_model = train_bayesian_ridge(X_train_scaled, y_train)
    rf_model = train_random_forest(X_train_scaled, y_train)
    
    # ... rest of pipeline ...
```

**After:**
```python
def train_all_models(train_features_scaled: np.ndarray,
                     train_targets: pd.Series) -> Dict[str, any]:
    """Train all regression models."""
    print("\nTraining models...")
    
    linear_regression_model = train_linear_regression(train_features_scaled, train_targets)
    bayesian_ridge_model = train_bayesian_ridge(train_features_scaled, train_targets)
    random_forest_model = train_random_forest(train_features_scaled, train_targets)
    
    return {
        'linear_regression': linear_regression_model,
        'bayesian_ridge': bayesian_ridge_model,
        'random_forest': random_forest_model
    }

# Usage in pipeline:
trained_models_dict = train_all_models(train_features_scaled, train_targets)
```

**Benefits:**
- Single responsibility: One function handles all model training
- Returns organized dictionary of models
- Easier to add new models in the future
- Can be tested independently

#### 2. **`evaluate_all_models()`** - Consolidated Model Evaluation

**Before:**
```python
def run_complete_pipeline(filepath: str = 'hp_obp.csv') -> pd.DataFrame:
    # ... earlier code ...
    
    # Step 6: Evaluate models
    print("\nStep 7: Evaluating models...")
    lr_results = evaluate_model(lr_model, X_test_scaled, y_test, "Linear Regression")
    bayes_results = evaluate_model(bayes_model, X_test_scaled, y_test, "Bayesian Ridge")
    rf_results = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    
    evaluation_results = [lr_results, bayes_results, rf_results]
    
    # ... rest of pipeline ...
```

**After:**
```python
def evaluate_all_models(trained_models_dict: Dict[str, any],
                       test_features_scaled: np.ndarray,
                       test_targets: pd.Series) -> List[Dict]:
    """Evaluate all trained models."""
    print("\nEvaluating models...")
    
    linear_regression_results = evaluate_model(
        trained_models_dict['linear_regression'],
        test_features_scaled,
        test_targets,
        "Linear Regression")
    
    bayesian_ridge_results = evaluate_model(
        trained_models_dict['bayesian_ridge'],
        test_features_scaled,
        test_targets,
        "Bayesian Ridge")
    
    random_forest_results = evaluate_model(
        trained_models_dict['random_forest'],
        test_features_scaled,
        test_targets,
        "Random Forest")
    
    return [linear_regression_results, bayesian_ridge_results, random_forest_results]

# Usage in pipeline:
evaluation_results_list = evaluate_all_models(trained_models_dict, test_features_scaled, test_targets)
```

**Benefits:**
- Encapsulates all evaluation logic
- Works seamlessly with `train_all_models()`
- Consistent naming and structure
- Easier to maintain and extend

#### 3. **`create_model_visualizations()`** - Consolidated Visualization Creation

**Before:**
```python
def run_complete_pipeline(filepath: str = 'hp_obp.csv') -> pd.DataFrame:
    # ... earlier code ...
    
    # Step 7: Create visualizations
    print("\nStep 8: Creating model comparison plots...")
    plot_actual_vs_predicted(y_test, lr_results['predictions'], "Linear Regression", 'blue')
    plot_actual_vs_predicted(y_test, bayes_results['predictions'], "Bayesian Ridge", 'purple')
    plot_actual_vs_predicted(y_test, rf_results['predictions'], "Random Forest", 'green')

    predictions_dict = {
        'Linear Regression': lr_results['predictions'],
        'Bayesian Ridge': bayes_results['predictions'],
        'Random Forest': rf_results['predictions']
    }
    plot_model_comparison(y_test, predictions_dict)
    
    # ... rest of pipeline ...
```

**After:**
```python
def create_model_visualizations(test_targets: pd.Series,
                                evaluation_results_list: List[Dict]) -> None:
    """Create all model comparison visualizations."""
    print("\nCreating model comparison plots...")
    
    # Individual model plots
    plot_actual_vs_predicted(
        test_targets,
        evaluation_results_list[0]['predictions'],
        "Linear Regression",
        'blue')
    
    plot_actual_vs_predicted(
        test_targets,
        evaluation_results_list[1]['predictions'],
        "Bayesian Ridge",
        'purple')
    
    plot_actual_vs_predicted(
        test_targets,
        evaluation_results_list[2]['predictions'],
        "Random Forest",
        'green')

    # Comparison plot
    model_predictions_dict = {
        'Linear Regression': evaluation_results_list[0]['predictions'],
        'Bayesian Ridge': evaluation_results_list[1]['predictions'],
        'Random Forest': evaluation_results_list[2]['predictions']
    }
    plot_model_comparison(test_targets, model_predictions_dict)

# Usage in pipeline:
create_model_visualizations(test_targets, evaluation_results_list)
```

**Benefits:**
- Separates visualization logic from pipeline flow
- All plotting code in one place
- Easier to modify visualization behavior
- Cleaner main pipeline function

### Variable Name Changes

Comprehensive renaming for clarity and consistency:

| Category | Before | After | Rationale |
|----------|--------|-------|-----------|
| **DataFrames** | `data` | `dataset` | More descriptive, standard ML terminology |
| **Models** | `lr_model` | `linear_regression_model` | Full name removes ambiguity |
| | `bayes_model` | `bayesian_ridge_model` | Explicit model type |
| | `rf_model` | `random_forest_model` | Clear and professional |
| **Features/Targets** | `X` | `feature_matrix` | Self-documenting |
| | `y` | `target_vector` | Explicit purpose |
| | `X_train` | `train_features` | Readable naming convention |
| | `y_train` | `train_targets` | Consistent with above |
| | `X_test` | `test_features` | Improved clarity |
| | `y_test` | `test_targets` | Matches pattern |
| | `X_train_scaled` | `train_features_scaled` | Consistent naming |
| | `X_test_scaled` | `test_features_scaled` | Consistent naming |
| **Parameters** | `factor` | `threshold_factor` | More specific |
| | `strategy` | `imputation_strategy` | Context-specific |
| | `n_estimators` | `number_of_trees` | Intuitive for non-ML experts |
| | `column` | `target_column` / `speed_column` | Context-aware naming |
| | `test_size` | `test_proportion` | More accurate description |
| | `random_state` | `random_seed` | Common terminology |
| **Utilities** | `scaler` | `feature_scaler` | Specifies what it scales |
| | `data_info` | `data_info_buffer` | Clarifies it's a StringIO buffer |
| | `color` | `scatter_color` / `plot_color` | Context-specific |
| | `model_name` | `model_display_name` | Clarifies it's for display |
| **Collections** | `filtered_cols` | `filtered_column_names` | Full descriptive name |
| | `float_cols` | `float_column_names` | Consistency |
| | `all_desired_cols` | `all_desired_columns` | No abbreviations |
| | `evaluation_results` | `evaluation_results_list` | Type indication |
| | `predictions_dict` | `model_predictions_dict` | More descriptive |

### Code Quality Improvements

**Before `run_complete_pipeline()`:**
- 80+ lines of code
- Multiple responsibilities mixed together
- Hard to understand at a glance
- Difficult to test individual components

**After `run_complete_pipeline()`:**
- ~50 lines of code
- Clear, step-by-step flow
- Each step delegates to focused functions
- Easy to modify or extend

**Example Comparison:**

```python
# BEFORE - Complex, inline logic
def run_complete_pipeline(filepath: str = 'hp_obp.csv') -> pd.DataFrame:
    # ... data loading and preprocessing ...
    
    # Lots of inline model training
    lr_model = train_linear_regression(X_train_scaled, y_train)
    bayes_model = train_bayesian_ridge(X_train_scaled, y_train)
    rf_model = train_random_forest(X_train_scaled, y_train)
    
    # Lots of inline evaluation
    lr_results = evaluate_model(lr_model, X_test_scaled, y_test, "Linear Regression")
    bayes_results = evaluate_model(bayes_model, X_test_scaled, y_test, "Bayesian Ridge")
    rf_results = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    
    # Lots of inline plotting
    plot_actual_vs_predicted(y_test, lr_results['predictions'], "Linear Regression", 'blue')
    plot_actual_vs_predicted(y_test, bayes_results['predictions'], "Bayesian Ridge", 'purple')
    plot_actual_vs_predicted(y_test, rf_results['predictions'], "Random Forest", 'green')
    # ... more plotting code ...

# AFTER - Clean, delegated logic
def run_complete_pipeline(filepath: str = 'hp_obp.csv') -> pd.DataFrame:
    # ... data loading and preprocessing ...
    
    # Clean delegation to focused functions
    trained_models_dict = train_all_models(train_features_scaled, train_targets)
    evaluation_results_list = evaluate_all_models(trained_models_dict, test_features_scaled, test_targets)
    create_model_visualizations(test_targets, evaluation_results_list)
```

### Impact Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Pipeline Function Length** | ~80 lines | ~50 lines | 37% reduction |
| **Functions with Single Responsibility** | Lower | Higher | Better modularity |
| **Variable Name Clarity** | Mixed | Consistent | Easier to understand |
| **Testability** | Harder | Easier | Better quality assurance |
| **Code Duplication** | Some | Minimal | DRY principle applied |
| **Maintainability** | Moderate | High | Easier future changes |

### Backward Compatibility

✅ **All changes are backward compatible:**
- Function signatures remain unchanged for public API functions
- Test suite continues to pass without modifications
- Pipeline produces identical results
- Only internal implementation and naming improved

## Detailed Setup Instructions

### 1. Environment Setup

**Option A: Using Make (Recommended)**
```bash
# Complete development environment setup
make setup

# Verify everything is working
make dev-check
```

**Option B: Manual Setup**
```bash
# Install Python dependencies
pip3 install --user --upgrade pip
pip3 install --user pandas matplotlib scikit-learn scipy numpy

# Verify installations
python3 -c "import pandas, matplotlib, sklearn, numpy, scipy; print('All dependencies installed successfully!')"
```

### 2. Data Preparation

Your CSV file must be named `hp_obp.csv` and include these required columns:
- `pitch_speed_mph`: Target variable (pitch velocity in mph)
- `athlete_uid`: Unique identifier for each athlete
- `test_date`: Date of biomechanics assessment
- Columns containing 'peak' in the name (biomechanics metrics)

**Example data structure:**
```csv
pitch_speed_mph,athlete_uid,test_date,peak_velocity_x,peak_velocity_y,peak_acceleration_z
85.5,athlete_001,2024-01-15,12.3,8.7,45.2
87.2,athlete_002,2024-01-15,13.1,9.2,47.8
```

**Data validation:**
```bash
# Check if your data file exists and is properly formatted
make validate-data

# Quick data inspection
python3 -c "import pandas as pd; data = pd.read_csv('hp_obp.csv'); print(f'Data shape: {data.shape}'); print(f'Columns: {list(data.columns)}')"
```

### 3. Running the Analysis

**Basic execution:**
```bash
# Run the complete pipeline
make run

# Or run with data validation first
make run-full
```

**Development workflow:**
```bash
# Complete development setup and testing
make dev-setup

# Run quality checks
make dev-check

# Format code (if making changes)
make format
```

### 4. Testing

**Run all tests:**
```bash
make test
```

**Detailed test output:**
```bash
make test-verbose
```

**Test with coverage reporting:**
```bash
make test-coverage
```

**Quick smoke test:**
```bash
make test-quick
```

### 5. Code Quality

**Lint the code:**
```bash
make lint
```

**Auto-format code:**
```bash
make format
```

**Complete CI pipeline:**
```bash
make ci
```

## Understanding the Output

### Console Output
The script will display:

1. **Data Loading**: Confirmation of successful data loading and shape
2. **Exploratory Data Analysis**: First 5 rows, data types, summary statistics
3. **Data Preprocessing**: Number of outliers removed, missing values handled
4. **Model Training**: Confirmation of each model training
5. **Model Evaluation**: MSE and R² scores for each model
6. **Final Summary**: Best performing model and recommendations

### Generated Visualizations
The analysis creates several plots:

1. **Pitch Speed Histogram**: Distribution of pitch velocities with mean line
2. **Individual Model Plots**: Actual vs. predicted scatter plots for each model
3. **Model Comparison Plot**: All three models overlaid for comparison

### Results Table
A comparison table showing:
- Model names
- Mean Squared Error (MSE)
- R² Score (coefficient of determination)

## Troubleshooting

### Common Issues

**1. Missing Data File**
```
Error: hp_obp.csv not found
Solution: Ensure your data file is named exactly 'hp_obp.csv' and is in the same directory
```

**2. Missing Dependencies**
```
Error: ModuleNotFoundError: No module named 'pandas'
Solution: Run 'make install' or 'pip3 install pandas matplotlib scikit-learn scipy'
```

**3. Insufficient Data**
```
Error: Not enough data for train/test split
Solution: Ensure your dataset has at least 10 rows of data
```

**4. No Peak Columns Found**
```
Error: No columns containing 'peak' found
Solution: Verify your biomechanics columns contain the word 'peak' in their names
```

**5. Memory Issues**
```
Error: Memory error during processing
Solution: Try reducing dataset size or closing other applications
```

### Getting Help

**Check dependencies:**
```bash
make check-deps
```

**Validate environment:**
```bash
make test-quick
```

**Check available commands:**
```bash
make help
make help-test
make help-dev
```

## Dataset Requirements

The analysis expects a CSV file named `hp_obp.csv` containing:
- `pitch_speed_mph`: Target variable (pitch velocity)
- `athlete_uid`: Unique athlete identifier
- `test_date`: Date of biomechanics assessment
- Multiple columns with 'peak' in the name (biomechanics metrics)

**Minimum requirements:**
- At least 10 rows of data
- At least 2 peak biomechanics columns
- No more than 50% missing values in any column
- Pitch speeds between 40-120 mph (outliers will be automatically removed)

## Analysis Overview

The `driveline.py` script follows a systematic machine learning workflow:

### Step 1: Exploratory Data Analysis (EDA)
- **Data inspection**: Displays first 5 rows, data types, and summary statistics
- **Quality assessment**: Checks for missing values and duplicate records
- **Data profiling**: Generates descriptive statistics for all variables

### Step 2: Data Preprocessing
- **Feature selection**: Filters dataset to include only:
  - Peak biomechanics metrics (columns containing 'peak')
  - Pitch speed (target variable)
  - Athlete identifiers and test dates
- **Outlier removal**: Uses Interquartile Range (IQR) method to remove pitch speed outliers
  - Lower bound: Q1 - 1.5 × IQR
  - Upper bound: Q3 + 1.5 × IQR
- **Missing value imputation**: Fills null values with column means
- **Feature scaling**: Standardizes features using StandardScaler

### Step 3: Machine Learning Implementation

Three regression models are implemented and compared:

#### Model 1: Linear Regression
- **Algorithm**: Classic ordinary least squares regression
- **Use case**: Baseline model for linear relationships
- **Performance**: MSE ≈ 29.1, R² ≈ 0.67

#### Model 2: Bayesian Ridge Regression
- **Algorithm**: Bayesian approach to linear regression with regularization
- **Use case**: Handles uncertainty and prevents overfitting
- **Performance**: MSE ≈ 29.1, R² ≈ 0.67

#### Model 3: Random Forest Regressor
- **Algorithm**: Ensemble method using multiple decision trees
- **Use case**: Captures non-linear relationships and feature interactions
- **Performance**: MSE ≈ 27.07, R² ≈ 0.69

## Key Findings

### Model Performance Comparison

| Model | Mean Squared Error | R² Score | 
|-------|-------------------|----------|
| Linear Regression | 29.1 | 0.67 |
| Bayesian Ridge | 29.1 | 0.67 |
| **Random Forest** | **27.07** | **0.69** |

### Insights

1. **Best Performer**: Random Forest achieved the lowest MSE and highest R² score
2. **Linear Models**: Both linear approaches performed nearly identically
3. **Predictive Power**: All models explain ~67-69% of pitch speed variance
4. **Recommendation**: Despite Random Forest's slight edge, **Linear Regression is recommended** for:
   - **Simplicity**: Easier to interpret and implement
   - **Cost efficiency**: Lower computational requirements
   - **Minimal performance difference**: Only ~2% improvement for added complexity

### Data Quality Observations
- Successfully removed outliers to improve model stability
- Peak biomechanics metrics show predictive value for pitch velocity
- Missing value imputation with means proved effective

## Visualizations

The script generates several plots:
- **Histogram**: Distribution of pitch speeds with mean indicator
- **Scatter plots**: Actual vs. predicted values for each model
- **Comparison plot**: All three models' predictions overlaid

## Technical Notes

- **Train/test split**: 80/20 random split with fixed seed (42)
- **Feature scaling**: Applied only after train/test split to prevent data leakage
- **Evaluation metrics**: Mean Squared Error and R² coefficient of determination
- **Cross-validation**: Could be added for more robust performance estimation

## Advanced Usage

### Custom Data File
```bash
# Edit driveline.py to change the filename
python3 -c "
from driveline import run_complete_pipeline
results = run_complete_pipeline('your_custom_file.csv')
print(results)
"
```

### Importing as Module
```python
from driveline import load_data, run_complete_pipeline

# Load and inspect data
data = load_data('hp_obp.csv')
print(data.head())

# Run analysis
results = run_complete_pipeline('hp_obp.csv')
```

### Running Specific Components
```python
from driveline import (
    load_data, perform_eda, filter_columns, 
    remove_outliers, prepare_features_target
)

# Step by step analysis
data = load_data('hp_obp.csv')
perform_eda(data)
filtered_data = filter_columns(data)
clean_data = remove_outliers(filtered_data, 'pitch_speed_mph')
X, y = prepare_features_target(clean_data)
```

## Dependencies

Core dependencies:
- `pandas>=1.3.0`: Data manipulation and analysis
- `numpy>=1.21.0`: Numerical computing
- `matplotlib>=3.3.0`: Data visualization
- `scikit-learn>=1.0.0`: Machine learning algorithms and preprocessing
- `scipy>=1.7.0`: Scientific computing

Development dependencies:
- `flake8>=4.0.0`: Code linting
- `coverage>=6.0.0`: Test coverage reporting

## Future Improvements

1. **Feature engineering**: Create interaction terms between biomechanics metrics
2. **Cross-validation**: Implement k-fold CV for more robust evaluation
3. **Hyperparameter tuning**: Grid search for Random Forest optimization
4. **Additional models**: Try XGBoost, Support Vector Regression
5. **Feature importance**: Analyze which biomechanics metrics matter most
6. **Time series analysis**: Incorporate temporal patterns if multiple measurements per athlete
7. **Web interface**: Create a simple web app for easy analysis
8. **Real-time prediction**: API endpoint for live pitch speed prediction

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Check code quality: `make lint`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.