[![Run Baseball Analysis](https://github.com/treychase/IDS_mini2/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/treychase/IDS_mini2/actions/workflows/ci.yml)

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
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites
- Python 3.x
- pip3

### Installation & Usage

1. **Install dependencies:**
   ```bash
   make install
   ```

2. **Run the analysis:**
   ```bash
   make run
   ```

3. **Lint the code:**
   ```bash
   make lint
   ```

4. **Clean cache files:**
   ```bash
   make clean
   ```

5. **View all available commands:**
   ```bash
   make help
   ```

## Dataset Requirements

The analysis expects a CSV file named `hp_obp.csv` containing:
- `pitch_speed_mph`: Target variable (pitch velocity)
- `athlete_uid`: Unique athlete identifier
- `test_date`: Date of biomechanics assessment
- Multiple columns with 'peak' in the name (biomechanics metrics)

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

## Dependencies

- `pandas`: Data manipulation and analysis
- `matplotlib`: Data visualization
- `scikit-learn`: Machine learning algorithms and preprocessing
- `flake8`: Code linting (development)

## Future Improvements

1. **Feature engineering**: Create interaction terms between biomechanics metrics
2. **Cross-validation**: Implement k-fold CV for more robust evaluation
3. **Hyperparameter tuning**: Grid search for Random Forest optimization
4. **Additional models**: Try XGBoost, Support Vector Regression
5. **Feature importance**: Analyze which biomechanics metrics matter most
6. **Time series analysis**: Incorporate temporal patterns if multiple measurements per athlete
