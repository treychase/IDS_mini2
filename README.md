[![Run Jupyter Notebook](https://github.com/treychase/IDS_mini2/actions/workflows/main.yml/badge.svg)](https://github.com/treychase/IDS_mini2/actions/workflows/main.yml)

# Driveline Open Biomechanics Analysis

## Overview

This notebook analyzes a dataset of pitching biomechanics and performance metrics to predict pitch velocity (`pitch_speed_mph`) from peak biomechanics metrics using several machine learning models. The workflow includes data loading, cleaning, exploratory analysis, preprocessing, model training, evaluation, and comparison.

---

## Steps

### 1. Load Data and Libraries

- Import necessary libraries: `pandas`, `matplotlib`, and scikit-learn modules.
- Load the dataset (`hp_obp.csv`) into a pandas DataFrame.

### 2. Exploratory Data Analysis (EDA)

- Display the first few rows, data info, and descriptive statistics.
- Check for missing values and duplicate rows to assess data quality.

### 3. Data Preprocessing

- **Feature Selection:** Select columns related to peak biomechanics, pitch speed, test date, and athlete ID.
- **Outlier Removal:** Remove outliers in `pitch_speed_mph` using the IQR (interquartile range) method.
- **Missing Value Imputation:** Fill missing values in numeric columns with the column mean.
- Display the cleaned data and its summary statistics.

### 4. Data Visualization

- Plot a histogram of `pitch_speed_mph` with a dashed red line indicating the mean pitch speed.

### 5. Machine Learning Models

#### Model 1: Classic Linear Regression

- Select numeric features (excluding `pitch_speed_mph`).
- Split data into training and test sets.
- Scale features using `StandardScaler`.
- Train a linear regression model and evaluate its performance (MSE and R²).
- Visualize actual vs. predicted pitch speeds.

#### Model 2: Bayesian Ridge Regression

- Train a Bayesian Ridge regression model on the same data.
- Evaluate and visualize predictions.

#### Model 3: Random Forest Regressor

- Train a Random Forest regressor.
- Evaluate and visualize predictions.

### 6. Model Comparison

- Create a DataFrame summarizing the Mean Squared Error (MSE) and R² score for all three models.
- Visualize all models' predictions against actual values in a single scatter plot.

---

## Findings

- **All three models—Linear Regression, Bayesian Ridge, and Random Forest—show similar performance in predicting pitch velocity.**
- **Random Forest** achieves the lowest mean squared error (27.07) and the highest R² score (0.69), indicating slightly better predictive accuracy and variance explanation.
- **Linear Regression** and **Bayesian Ridge** perform nearly identically, with MSEs around 29.1 and R² scores around 0.67.
- **Recommendation:** While Random Forest performs best, the difference is small. For cost efficiency and simplicity, classic linear regression is recommended.

---

## Usage

1. Place your data file (`hp_obp.csv`) in the working directory.
2. Run the notebook cells sequentially to reproduce the analysis and results.
3. Review the summary and model comparison to select the best approach for pitch
