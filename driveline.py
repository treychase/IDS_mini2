#!/usr/bin/env python3
"""
Baseball Pitch Analysis and Machine Learning Pipeline

This script performs exploratory data analysis and machine learning modeling
on baseball pitching data to predict pitch velocity based on biomechanical metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load baseball pitching data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Successfully loaded data with shape: {data.shape}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filepath}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"File is empty: {filepath}")


def perform_eda(data: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        data (pd.DataFrame): Input dataset
    """
    print("#### First 5 Rows of Data")
    print(data.head())
    
    print("\n#### Data Info")
    data_info = io.StringIO()
    data.info(buf=data_info)
    print(data_info.getvalue())
    
    print("#### Data Description")
    print(data.describe())
    
    print("#### Missing Values per Column")
    print(data.isnull().sum().to_frame("Null Count").T)
    
    print("#### Total Duplicate Rows")
    print(f"**{data.duplicated().sum()}**")


def filter_columns(data: pd.DataFrame, 
                  include_patterns: List[str] = None,
                  additional_cols: List[str] = None) -> pd.DataFrame:
    """
    Filter DataFrame columns based on patterns and additional specified columns.
    
    Args:
        data (pd.DataFrame): Input dataset
        include_patterns (List[str]): Patterns to match in column names (case-insensitive)
        additional_cols (List[str]): Additional columns to include
        
    Returns:
        pd.DataFrame: Filtered dataset
    """
    if include_patterns is None:
        include_patterns = ['peak']
    if additional_cols is None:
        additional_cols = ['pitch_speed_mph', 'test_date', 'athlete_uid']
    
    # Find columns matching patterns
    pattern_matched_cols = []
    for pattern in include_patterns:
        pattern_matched_cols.extend([col for col in data.columns if pattern.lower() in col.lower()])
    
    # Combine with additional columns, only keeping existing ones
    all_desired_cols = []
    all_desired_cols.extend(pattern_matched_cols)
    all_desired_cols.extend([col for col in additional_cols if col in data.columns])
    
    # Remove duplicates while preserving order
    filtered_cols = []
    for col in all_desired_cols:
        if col not in filtered_cols:
            filtered_cols.append(col)
    
    print(f"Selected {len(filtered_cols)} columns: {sorted(filtered_cols)}")
    return data[filtered_cols]


def remove_outliers(data: pd.DataFrame, 
                   column: str, 
                   method: str = 'iqr', 
                   factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from a specific column using the specified method.
    
    Args:
        data (pd.DataFrame): Input dataset
        column (str): Column name to remove outliers from
        method (str): Method to use ('iqr' or 'zscore')
        factor (float): Factor for outlier detection (1.5 for IQR, 3 for z-score)
        
    Returns:
        pd.DataFrame: Dataset with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    original_size = len(data)
    
    if method.lower() == 'iqr':
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        filtered_data = data[
            (data[column] >= lower_bound) & 
            (data[column] <= upper_bound)
        ]
        
    elif method.lower() == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(data[column].dropna()))
        filtered_data = data[z_scores < factor]
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")
    
    removed_count = original_size - len(filtered_data)
    print(f"Removed {removed_count} outliers from '{column}' using {method} method")
    
    return filtered_data


def handle_missing_values(data: pd.DataFrame, 
                         strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        data (pd.DataFrame): Input dataset
        strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
        
    Returns:
        pd.DataFrame: Dataset with missing values handled
    """
    missing_count = data.isnull().sum().sum()
    print(f"Found {missing_count} missing values")
    
    if missing_count == 0:
        return data
    
    if strategy == 'mean':
        filled_data = data.fillna(data.mean(numeric_only=True))
    elif strategy == 'median':
        filled_data = data.fillna(data.median(numeric_only=True))
    elif strategy == 'drop':
        filled_data = data.dropna()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"Handled missing values using '{strategy}' strategy")
    return filled_data


def prepare_features_target(data: pd.DataFrame, 
                           target_column: str = 'pitch_speed_mph') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target variable for machine learning.
    
    Args:
        data (pd.DataFrame): Input dataset
        target_column (str): Name of the target column
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Select only float columns except target
    float_cols = data.select_dtypes(include='float').columns.tolist()
    if target_column in float_cols:
        float_cols.remove(target_column)
    
    X = data[float_cols]
    y = data[target_column]
    
    print(f"Prepared {len(float_cols)} features and {len(y)} target samples")
    return X, y


def split_and_scale_data(X: pd.DataFrame, 
                        y: pd.Series, 
                        test_size: float = 0.2, 
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler]:
    """
    Split data into train/test sets and scale features.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of test data
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple: X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Split data: {len(X_train)} train, {len(X_test)} test samples")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_linear_regression(X_train: np.ndarray, 
                           y_train: pd.Series) -> LinearRegression:
    """
    Train a Linear Regression model.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (pd.Series): Training targets
        
    Returns:
        LinearRegression: Trained model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Trained Linear Regression model")
    return model


def train_bayesian_ridge(X_train: np.ndarray, 
                        y_train: pd.Series) -> BayesianRidge:
    """
    Train a Bayesian Ridge Regression model.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (pd.Series): Training targets
        
    Returns:
        BayesianRidge: Trained model
    """
    model = BayesianRidge()
    model.fit(X_train, y_train)
    print("Trained Bayesian Ridge model")
    return model


def train_random_forest(X_train: np.ndarray, 
                       y_train: pd.Series, 
                       n_estimators: int = 100, 
                       random_state: int = 42) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor model.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (pd.Series): Training targets
        n_estimators (int): Number of trees
        random_state (int): Random seed
        
    Returns:
        RandomForestRegressor: Trained model
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    print(f"Trained Random Forest model with {n_estimators} trees")
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: pd.Series, 
                  model_name: str) -> Dict[str, float]:
    """
    Evaluate a trained model and return metrics.
    
    Args:
        model: Trained scikit-learn model
        X_test (np.ndarray): Test features
        y_test (pd.Series): Test targets
        model_name (str): Name of the model for printing
        
    Returns:
        Dict[str, float]: Dictionary with MSE and R² scores
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} - MSE: {mse:.2f}, R²: {r2:.3f}")
    
    return {
        'model_name': model_name,
        'mse': mse,
        'r2': r2,
        'predictions': y_pred
    }


def plot_pitch_speed_histogram(data: pd.DataFrame, 
                              column: str = 'pitch_speed_mph',
                              title: str = "Histogram of Pitch Speed (mph)",
                              save_path: Optional[str] = None) -> None:
    """
    Plot histogram of pitch speed data.
    
    Args:
        data (pd.DataFrame): Dataset containing pitch speed
        column (str): Column name for pitch speed
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data[column], bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Pitch Speed (mph)")
    plt.ylabel("Frequency")
    plt.title(title)
    
    # Add mean line
    mean_speed = data[column].mean()
    plt.axvline(mean_speed, color='red', linestyle='dashed', linewidth=2, 
                label=f"Mean: {mean_speed:.2f}")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_actual_vs_predicted(y_test: pd.Series, 
                           y_pred: np.ndarray, 
                           model_name: str,
                           color: str = 'blue',
                           save_path: Optional[str] = None) -> None:
    """
    Plot actual vs predicted values for a model.
    
    Args:
        y_test (pd.Series): Actual test values
        y_pred (np.ndarray): Predicted values
        model_name (str): Name of the model
        color (str): Color for the scatter plot
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color=color)
    plt.xlabel("Actual Pitch Speed (mph)")
    plt.ylabel("Predicted Pitch Speed (mph)")
    plt.title(f"{model_name}: Actual vs. Predicted Pitch Speed")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='1:1 line')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_model_comparison(y_test: pd.Series, 
                         predictions_dict: Dict[str, np.ndarray],
                         colors_dict: Dict[str, str] = None,
                         markers_dict: Dict[str, str] = None,
                         save_path: Optional[str] = None) -> None:
    """
    Plot comparison of multiple models' predictions.
    
    Args:
        y_test (pd.Series): Actual test values
        predictions_dict (Dict[str, np.ndarray]): Dictionary of model predictions
        colors_dict (Dict[str, str]): Colors for each model
        markers_dict (Dict[str, str]): Markers for each model
        save_path (Optional[str]): Path to save the plot
    """
    if colors_dict is None:
        colors_dict = {
            'Linear Regression': 'blue',
            'Bayesian Ridge': 'purple', 
            'Random Forest': 'green'
        }
    
    if markers_dict is None:
        markers_dict = {
            'Linear Regression': 'o',
            'Bayesian Ridge': '^',
            'Random Forest': 's'
        }
    
    plt.figure(figsize=(10, 8))
    
    for model_name, y_pred in predictions_dict.items():
        color = colors_dict.get(model_name, 'black')
        marker = markers_dict.get(model_name, 'o')
        plt.scatter(y_test, y_pred, alpha=0.6, label=model_name, 
                   color=color, marker=marker)
    
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', label='1:1 Line')
    plt.xlabel("Actual Pitch Speed (mph)")
    plt.ylabel("Predicted Pitch Speed (mph)")
    plt.title("Actual vs. Predicted Pitch Speed: Model Comparison")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def create_results_dataframe(evaluation_results: List[Dict]) -> pd.DataFrame:
    """
    Create a comparison DataFrame from model evaluation results.
    
    Args:
        evaluation_results (List[Dict]): List of evaluation result dictionaries
        
    Returns:
        pd.DataFrame: Comparison results
    """
    results_df = pd.DataFrame({
        'Model': [result['model_name'] for result in evaluation_results],
        'Mean Squared Error': [result['mse'] for result in evaluation_results],
        'R^2 Score': [result['r2'] for result in evaluation_results]
    })
    
    return results_df


def run_complete_pipeline(filepath: str = 'hp_obp.csv') -> pd.DataFrame:
    """
    Run the complete machine learning pipeline.
    
    Args:
        filepath (str): Path to the data file
        
    Returns:
        pd.DataFrame: Results comparison DataFrame
    """
    print("=== Baseball Pitch Analysis Pipeline ===\n")
    
    # Step 1: Load and explore data
    print("Step 1: Loading data...")
    data = load_data(filepath)
    
    print("\nStep 2: Performing EDA...")
    perform_eda(data)
    
    # Step 2: Data preprocessing
    print("\nStep 3: Data preprocessing...")
    filtered_data = filter_columns(data)
    filtered_data = remove_outliers(filtered_data, 'pitch_speed_mph')
    filtered_data = handle_missing_values(filtered_data)
    
    print(f"Final preprocessed data shape: {filtered_data.shape}")
    print(filtered_data.describe())
    
    # Step 3: Plot histogram
    print("\nStep 4: Creating visualization...")
    plot_pitch_speed_histogram(filtered_data)
    
    # Step 4: Prepare data for ML
    print("\nStep 5: Preparing features and target...")
    X, y = prepare_features_target(filtered_data)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Step 5: Train models
    print("\nStep 6: Training models...")
    lr_model = train_linear_regression(X_train_scaled, y_train)
    bayes_model = train_bayesian_ridge(X_train_scaled, y_train)
    rf_model = train_random_forest(X_train_scaled, y_train)
    
    # Step 6: Evaluate models
    print("\nStep 7: Evaluating models...")
    lr_results = evaluate_model(lr_model, X_test_scaled, y_test, "Linear Regression")
    bayes_results = evaluate_model(bayes_model, X_test_scaled, y_test, "Bayesian Ridge")
    rf_results = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    
    evaluation_results = [lr_results, bayes_results, rf_results]
    
    # Step 7: Create visualizations
    print("\nStep 8: Creating model comparison plots...")
    plot_actual_vs_predicted(y_test, lr_results['predictions'], "Linear Regression", 'blue')
    plot_actual_vs_predicted(y_test, bayes_results['predictions'], "Bayesian Ridge", 'purple')
    plot_actual_vs_predicted(y_test, rf_results['predictions'], "Random Forest", 'green')
    
    # Model comparison plot
    predictions_dict = {
        'Linear Regression': lr_results['predictions'],
        'Bayesian Ridge': bayes_results['predictions'],
        'Random Forest': rf_results['predictions']
    }
    plot_model_comparison(y_test, predictions_dict)
    
    # Step 8: Create results summary
    print("\nStep 9: Creating results summary...")
    results_df = create_results_dataframe(evaluation_results)
    print("\n=== Model Comparison Results ===")
    print(results_df)
    
    # Print summary
    best_model_idx = results_df['R^2 Score'].idxmax()
    best_model = results_df.loc[best_model_idx, 'Model']
    best_r2 = results_df.loc[best_model_idx, 'R^2 Score']
    
    print(f"\n=== Summary ===")
    print(f"Best performing model: {best_model} (R² = {best_r2:.3f})")
    print("All models show similar performance, suggesting the relationships")
    print("in the data are primarily linear. For production use, consider")
    print("the simplest model (Linear Regression) for cost efficiency.")
    
    return results_df


def main():
    """Main function to run the analysis pipeline."""
    try:
        results = run_complete_pipeline()
        print("\nPipeline completed successfully!")
        return results
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()