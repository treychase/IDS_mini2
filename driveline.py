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
        dataset = pd.read_csv(filepath)
        print(f"Successfully loaded data with shape: {dataset.shape}")
        return dataset
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filepath}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"File is empty: {filepath}")


def perform_eda(dataset: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis on the dataset.

    Args:
        dataset (pd.DataFrame): Input dataset
    """
    print("#### First 5 Rows of Data")
    print(dataset.head())

    print("\n#### Data Info")
    data_info_buffer = io.StringIO()
    dataset.info(buf=data_info_buffer)
    print(data_info_buffer.getvalue())

    print("#### Data Description")
    print(dataset.describe())

    print("#### Missing Values per Column")
    print(dataset.isnull().sum().to_frame("Null Count").T)

    print("#### Total Duplicate Rows")
    print(f"**{dataset.duplicated().sum()}**")


def filter_columns(dataset: pd.DataFrame,
                   include_patterns: List[str] = None,
                   additional_columns: List[str] = None) -> pd.DataFrame:
    """
    Filter DataFrame columns based on patterns and additional specified columns.

    Args:
        dataset (pd.DataFrame): Input dataset
        include_patterns (List[str]): Patterns to match in column names (case-insensitive)
        additional_columns (List[str]): Additional columns to include

    Returns:
        pd.DataFrame: Filtered dataset
    """
    if include_patterns is None:
        include_patterns = ['peak']
    if additional_columns is None:
        additional_columns = ['pitch_speed_mph', 'test_date', 'athlete_uid']

    # Find columns matching patterns
    pattern_matched_columns = []
    for pattern in include_patterns:
        pattern_matched_columns.extend(
            [col for col in dataset.columns if pattern.lower() in col.lower()])

    # Combine with additional columns, only keeping existing ones
    all_desired_columns = []
    all_desired_columns.extend(pattern_matched_columns)
    all_desired_columns.extend(
        [col for col in additional_columns if col in dataset.columns])

    # Remove duplicates while preserving order
    filtered_column_names = []
    for col in all_desired_columns:
        if col not in filtered_column_names:
            filtered_column_names.append(col)

    print(
        f"Selected {
            len(filtered_column_names)} columns: {
            sorted(filtered_column_names)}")
    return dataset[filtered_column_names]


def remove_outliers(dataset: pd.DataFrame,
                    target_column: str,
                    method: str = 'iqr',
                    threshold_factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from a specific column using the specified method.

    Args:
        dataset (pd.DataFrame): Input dataset
        target_column (str): Column name to remove outliers from
        method (str): Method to use ('iqr' or 'zscore')
        threshold_factor (float): Factor for outlier detection (1.5 for IQR, 3 for z-score)

    Returns:
        pd.DataFrame: Dataset with outliers removed
    """
    if target_column not in dataset.columns:
        raise ValueError(f"Column '{target_column}' not found in data")

    original_row_count = len(dataset)

    if method.lower() == 'iqr':
        first_quartile = dataset[target_column].quantile(0.25)
        third_quartile = dataset[target_column].quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        lower_threshold = first_quartile - threshold_factor * interquartile_range
        upper_threshold = third_quartile + threshold_factor * interquartile_range

        cleaned_dataset = dataset[
            (dataset[target_column] >= lower_threshold) &
            (dataset[target_column] <= upper_threshold)
        ]

    elif method.lower() == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(dataset[target_column].dropna()))
        cleaned_dataset = dataset[z_scores < threshold_factor]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")

    outliers_removed_count = original_row_count - len(cleaned_dataset)
    print(
        f"Removed {outliers_removed_count} outliers from '{target_column}' using {method} method")

    return cleaned_dataset


def handle_missing_values(dataset: pd.DataFrame,
                          imputation_strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        dataset (pd.DataFrame): Input dataset
        imputation_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')

    Returns:
        pd.DataFrame: Dataset with missing values handled
    """
    total_missing_values = dataset.isnull().sum().sum()
    print(f"Found {total_missing_values} missing values")

    if total_missing_values == 0:
        return dataset

    if imputation_strategy == 'mean':
        imputed_dataset = dataset.fillna(dataset.mean(numeric_only=True))
    elif imputation_strategy == 'median':
        imputed_dataset = dataset.fillna(dataset.median(numeric_only=True))
    elif imputation_strategy == 'drop':
        imputed_dataset = dataset.dropna()
    else:
        raise ValueError(f"Unknown strategy: {imputation_strategy}")

    print(f"Handled missing values using '{imputation_strategy}' strategy")
    return imputed_dataset


def prepare_features_target(dataset: pd.DataFrame,
                            target_column_name: str = 'pitch_speed_mph') -> Tuple[pd.DataFrame,
                                                                                  pd.Series]:
    """
    Prepare features and target variable for machine learning.

    Args:
        dataset (pd.DataFrame): Input dataset
        target_column_name (str): Name of the target column

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)
    """
    if target_column_name not in dataset.columns:
        raise ValueError(
            f"Target column '{target_column_name}' not found in data")

    # Select only float columns except target
    float_column_names = dataset.select_dtypes(
        include='float').columns.tolist()
    if target_column_name in float_column_names:
        float_column_names.remove(target_column_name)

    feature_matrix = dataset[float_column_names]
    target_vector = dataset[target_column_name]

    print(
        f"Prepared {
            len(float_column_names)} features and {
            len(target_vector)} target samples")
    return feature_matrix, target_vector


def split_and_scale_data(feature_matrix: pd.DataFrame,
                         target_vector: pd.Series,
                         test_proportion: float = 0.2,
                         random_seed: int = 42) -> Tuple[np.ndarray,
                                                         np.ndarray,
                                                         pd.Series,
                                                         pd.Series,
                                                         StandardScaler]:
    """
    Split data into train/test sets and scale features.

    Args:
        feature_matrix (pd.DataFrame): Features
        target_vector (pd.Series): Target variable
        test_proportion (float): Proportion of test data
        random_seed (int): Random seed for reproducibility

    Returns:
        Tuple: train_features_scaled, test_features_scaled, train_targets, test_targets, feature_scaler
    """
    # Split data
    train_features, test_features, train_targets, test_targets = train_test_split(
        feature_matrix, target_vector, test_size=test_proportion, random_state=random_seed)

    # Scale features
    feature_scaler = StandardScaler()
    train_features_scaled = feature_scaler.fit_transform(train_features)
    test_features_scaled = feature_scaler.transform(test_features)

    print(
        f"Split data: {
            len(train_features)} train, {
            len(test_features)} test samples")
    return train_features_scaled, test_features_scaled, train_targets, test_targets, feature_scaler


def train_linear_regression(train_features: np.ndarray,
                            train_targets: pd.Series) -> LinearRegression:
    """
    Train a Linear Regression model.

    Args:
        train_features (np.ndarray): Training features
        train_targets (pd.Series): Training targets

    Returns:
        LinearRegression: Trained model
    """
    linear_model = LinearRegression()
    linear_model.fit(train_features, train_targets)
    print("Trained Linear Regression model")
    return linear_model


def train_bayesian_ridge(train_features: np.ndarray,
                         train_targets: pd.Series) -> BayesianRidge:
    """
    Train a Bayesian Ridge Regression model.

    Args:
        train_features (np.ndarray): Training features
        train_targets (pd.Series): Training targets

    Returns:
        BayesianRidge: Trained model
    """
    bayesian_model = BayesianRidge()
    bayesian_model.fit(train_features, train_targets)
    print("Trained Bayesian Ridge model")
    return bayesian_model


def train_random_forest(train_features: np.ndarray,
                        train_targets: pd.Series,
                        number_of_trees: int = 100,
                        random_seed: int = 42) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor model.

    Args:
        train_features (np.ndarray): Training features
        train_targets (pd.Series): Training targets
        number_of_trees (int): Number of trees
        random_seed (int): Random seed

    Returns:
        RandomForestRegressor: Trained model
    """
    forest_model = RandomForestRegressor(
        n_estimators=number_of_trees,
        random_state=random_seed)
    forest_model.fit(train_features, train_targets)
    print(f"Trained Random Forest model with {number_of_trees} trees")
    return forest_model


def evaluate_model(trained_model,
                   test_features: np.ndarray,
                   test_targets: pd.Series,
                   model_display_name: str) -> Dict[str,
                                                    float]:
    """
    Evaluate a trained model and return metrics.

    Args:
        trained_model: Trained scikit-learn model
        test_features (np.ndarray): Test features
        test_targets (pd.Series): Test targets
        model_display_name (str): Name of the model for printing

    Returns:
        Dict[str, float]: Dictionary with MSE and R² scores
    """
    predicted_values = trained_model.predict(test_features)
    mean_squared_error_value = mean_squared_error(
        test_targets, predicted_values)
    r_squared_score = r2_score(test_targets, predicted_values)

    print(
        f"{model_display_name} - MSE: {
            mean_squared_error_value:.2f}, R²: {
            r_squared_score:.3f}")

    return {
        'model_name': model_display_name,
        'mse': mean_squared_error_value,
        'r2': r_squared_score,
        'predictions': predicted_values
    }


def plot_pitch_speed_histogram(
        dataset: pd.DataFrame,
        speed_column: str = 'pitch_speed_mph',
        plot_title: str = "Histogram of Pitch Speed (mph)",
        save_path: Optional[str] = None) -> None:
    """
    Plot histogram of pitch speed data.

    Args:
        dataset (pd.DataFrame): Dataset containing pitch speed
        speed_column (str): Column name for pitch speed
        plot_title (str): Plot title
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.hist(
        dataset[speed_column],
        bins=30,
        color='skyblue',
        edgecolor='black')
    plt.xlabel("Pitch Speed (mph)")
    plt.ylabel("Frequency")
    plt.title(plot_title)

    # Add mean line
    average_speed = dataset[speed_column].mean()
    plt.axvline(average_speed, color='red', linestyle='dashed', linewidth=2,
                label=f"Mean: {average_speed:.2f}")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_actual_vs_predicted(actual_targets: pd.Series,
                             predicted_values: np.ndarray,
                             model_display_name: str,
                             scatter_color: str = 'blue',
                             save_path: Optional[str] = None) -> None:
    """
    Plot actual vs predicted values for a model.

    Args:
        actual_targets (pd.Series): Actual test values
        predicted_values (np.ndarray): Predicted values
        model_display_name (str): Name of the model
        scatter_color (str): Color for the scatter plot
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(
        actual_targets,
        predicted_values,
        alpha=0.7,
        color=scatter_color)
    plt.xlabel("Actual Pitch Speed (mph)")
    plt.ylabel("Predicted Pitch Speed (mph)")
    plt.title(f"{model_display_name}: Actual vs. Predicted Pitch Speed")
    plt.plot([actual_targets.min(), actual_targets.max()], [
             actual_targets.min(), actual_targets.max()], 'r--', label='1:1 line')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_model_comparison(actual_targets: pd.Series,
                          model_predictions_dict: Dict[str, np.ndarray],
                          color_mapping: Dict[str, str] = None,
                          marker_mapping: Dict[str, str] = None,
                          save_path: Optional[str] = None) -> None:
    """
    Plot comparison of multiple models' predictions.

    Args:
        actual_targets (pd.Series): Actual test values
        model_predictions_dict (Dict[str, np.ndarray]): Dictionary of model predictions
        color_mapping (Dict[str, str]): Colors for each model
        marker_mapping (Dict[str, str]): Markers for each model
        save_path (Optional[str]): Path to save the plot
    """
    if color_mapping is None:
        color_mapping = {
            'Linear Regression': 'blue',
            'Bayesian Ridge': 'purple',
            'Random Forest': 'green'
        }

    if marker_mapping is None:
        marker_mapping = {
            'Linear Regression': 'o',
            'Bayesian Ridge': '^',
            'Random Forest': 's'
        }

    plt.figure(figsize=(10, 8))

    for model_name, predicted_values in model_predictions_dict.items():
        plot_color = color_mapping.get(model_name, 'black')
        plot_marker = marker_mapping.get(model_name, 'o')
        plt.scatter(
            actual_targets,
            predicted_values,
            alpha=0.6,
            label=model_name,
            color=plot_color,
            marker=plot_marker)

    plt.plot([actual_targets.min(), actual_targets.max()], [
             actual_targets.min(), actual_targets.max()], 'r--', label='1:1 Line')
    plt.xlabel("Actual Pitch Speed (mph)")
    plt.ylabel("Predicted Pitch Speed (mph)")
    plt.title("Actual vs. Predicted Pitch Speed: Model Comparison")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def create_results_dataframe(
        evaluation_results_list: List[Dict]) -> pd.DataFrame:
    """
    Create a comparison DataFrame from model evaluation results.

    Args:
        evaluation_results_list (List[Dict]): List of evaluation result dictionaries

    Returns:
        pd.DataFrame: Comparison results
    """
    results_comparison = pd.DataFrame({
        'Model': [result['model_name'] for result in evaluation_results_list],
        'Mean Squared Error': [result['mse'] for result in evaluation_results_list],
        'R^2 Score': [result['r2'] for result in evaluation_results_list]
    })

    return results_comparison


def train_all_models(train_features_scaled: np.ndarray,
                     train_targets: pd.Series) -> Dict[str, any]:
    """
    Train all regression models.

    Args:
        train_features_scaled (np.ndarray): Scaled training features
        train_targets (pd.Series): Training targets

    Returns:
        Dict[str, any]: Dictionary containing all trained models
    """
    print("\nTraining models...")

    linear_regression_model = train_linear_regression(
        train_features_scaled, train_targets)
    bayesian_ridge_model = train_bayesian_ridge(
        train_features_scaled, train_targets)
    random_forest_model = train_random_forest(
        train_features_scaled, train_targets)

    return {
        'linear_regression': linear_regression_model,
        'bayesian_ridge': bayesian_ridge_model,
        'random_forest': random_forest_model
    }


def evaluate_all_models(trained_models_dict: Dict[str, any],
                        test_features_scaled: np.ndarray,
                        test_targets: pd.Series) -> List[Dict]:
    """
    Evaluate all trained models.

    Args:
        trained_models_dict (Dict[str, any]): Dictionary of trained models
        test_features_scaled (np.ndarray): Scaled test features
        test_targets (pd.Series): Test targets

    Returns:
        List[Dict]: List of evaluation results for all models
    """
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

    return [
        linear_regression_results,
        bayesian_ridge_results,
        random_forest_results]


def create_model_visualizations(test_targets: pd.Series,
                                evaluation_results_list: List[Dict]) -> None:
    """
    Create all model comparison visualizations.

    Args:
        test_targets (pd.Series): Test target values
        evaluation_results_list (List[Dict]): List of evaluation results
    """
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
    raw_dataset = load_data(filepath)

    print("\nStep 2: Performing EDA...")
    perform_eda(raw_dataset)

    # Step 2: Data preprocessing
    print("\nStep 3: Data preprocessing...")
    filtered_dataset = filter_columns(raw_dataset)
    cleaned_dataset = remove_outliers(filtered_dataset, 'pitch_speed_mph')
    preprocessed_dataset = handle_missing_values(cleaned_dataset)

    print(f"Final preprocessed data shape: {preprocessed_dataset.shape}")
    print(preprocessed_dataset.describe())

    # Step 3: Plot histogram
    print("\nStep 4: Creating visualization...")
    plot_pitch_speed_histogram(preprocessed_dataset)

    # Step 4: Prepare data for ML
    print("\nStep 5: Preparing features and target...")
    feature_matrix, target_vector = prepare_features_target(
        preprocessed_dataset)
    train_features_scaled, test_features_scaled, train_targets, test_targets, feature_scaler = split_and_scale_data(
        feature_matrix, target_vector)

    # Step 5: Train all models
    print("\nStep 6: Training models...")
    trained_models_dict = train_all_models(
        train_features_scaled, train_targets)

    # Step 6: Evaluate all models
    print("\nStep 7: Evaluating models...")
    evaluation_results_list = evaluate_all_models(
        trained_models_dict, test_features_scaled, test_targets)

    # Step 7: Create visualizations
    print("\nStep 8: Creating model comparison plots...")
    create_model_visualizations(test_targets, evaluation_results_list)

    # Step 8: Create results summary
    print("\nStep 9: Creating results summary...")
    results_comparison_dataframe = create_results_dataframe(
        evaluation_results_list)
    print("\n=== Model Comparison Results ===")
    print(results_comparison_dataframe)

    # Print summary
    best_model_index = results_comparison_dataframe['R^2 Score'].idxmax()
    best_performing_model = results_comparison_dataframe.loc[best_model_index, 'Model']
    best_r2_score = results_comparison_dataframe.loc[best_model_index, 'R^2 Score']

    print(f"\n=== Summary ===")
    print(
        f"Best performing model: {best_performing_model} (R² = {
            best_r2_score:.3f})")
    print("All models show similar performance, suggesting the relationships")
    print("in the data are primarily linear. For production use, consider")
    print("the simplest model (Linear Regression) for cost efficiency.")

    return results_comparison_dataframe


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
