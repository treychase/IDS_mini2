#!/usr/bin/env python3
"""
Unit tests for the refactored driveline.py module.

This test suite covers all the functions in the refactored baseball pitch analysis pipeline.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
from io import StringIO
import tempfile

# Add the current directory to sys.path to import driveline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the functions from the refactored driveline module
try:
    from driveline import (
        load_data, perform_eda, filter_columns, remove_outliers,
        handle_missing_values, prepare_features_target, split_and_scale_data,
        train_linear_regression, train_bayesian_ridge, train_random_forest,
        evaluate_model, plot_pitch_speed_histogram, plot_actual_vs_predicted,
        plot_model_comparison, create_results_dataframe, run_complete_pipeline,
        train_all_models, evaluate_all_models, create_model_visualizations
    )
except ImportError as e:
    print(f"Could not import driveline functions: {e}")
    print("Make sure the refactored driveline.py is in the same directory")


class TestDataLoading(unittest.TestCase):
    """Test data loading functionality"""

    def setUp(self):
        """Set up test data"""
        self.sample_csv_content = """pitch_speed_mph,peak_velocity_x,peak_velocity_y,test_date,athlete_uid
85.0,12.5,8.2,2024-01-01,athlete_1
87.5,13.2,8.9,2024-01-01,athlete_2
90.2,14.1,9.5,2024-01-01,athlete_3"""

    @patch('pandas.read_csv')
    def test_load_data_success(self, mock_read_csv):
        """Test successful data loading"""
        mock_df = pd.DataFrame({
            'pitch_speed_mph': [85.0, 87.5, 90.2],
            'peak_velocity_x': [12.5, 13.2, 14.1]
        })
        mock_read_csv.return_value = mock_df

        result = load_data('test.csv')

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        mock_read_csv.assert_called_once_with('test.csv')

    @patch('pandas.read_csv')
    def test_load_data_file_not_found(self, mock_read_csv):
        """Test FileNotFoundError handling"""
        mock_read_csv.side_effect = FileNotFoundError("File not found")

        with self.assertRaises(FileNotFoundError):
            load_data('nonexistent.csv')

    @patch('pandas.read_csv')
    def test_load_data_empty_file(self, mock_read_csv):
        """Test empty file handling"""
        mock_read_csv.side_effect = pd.errors.EmptyDataError("No data")

        with self.assertRaises(pd.errors.EmptyDataError):
            load_data('empty.csv')


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functions"""

    def setUp(self):
        """Set up test data for preprocessing tests"""
        self.sample_data = pd.DataFrame({
            'pitch_speed_mph': [85.0, 87.5, 90.2, 92.1, 88.7, 91.3, 89.4, 86.8, 93.2, 87.9],
            'peak_velocity_x': [12.5, 13.2, 14.1, 15.0, 13.8, 14.7, 13.9, 12.8, 15.5, 13.4],
            'peak_velocity_y': [8.2, 8.9, 9.5, 10.1, 9.2, 9.8, 9.1, 8.5, 10.3, 8.8],
            'peak_acceleration_z': [3.1, 3.4, 3.8, 4.0, 3.6, 3.9, 3.5, 3.2, 4.2, 3.3],
            'test_date': ['2024-01-01'] * 10,
            'athlete_uid': [f'athlete_{i}' for i in range(10)],
            'extra_column': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })

        self.data_with_outliers = self.sample_data.copy()
        self.data_with_outliers.loc[len(self.data_with_outliers)] = {
            'pitch_speed_mph': 150.0,  # Extreme outlier
            'peak_velocity_x': 20.0,
            'peak_velocity_y': 15.0,
            'peak_acceleration_z': 6.0,
            'test_date': '2024-01-01',
            'athlete_uid': 'outlier_athlete',
            'extra_column': 11
        }

        self.data_with_nulls = self.sample_data.copy()
        self.data_with_nulls.loc[2, 'peak_velocity_x'] = np.nan
        self.data_with_nulls.loc[5, 'peak_acceleration_z'] = np.nan

    def test_filter_columns_default(self):
        """Test column filtering with default parameters"""
        result = filter_columns(self.sample_data)

        expected_cols = {
            'peak_velocity_x',
            'peak_velocity_y',
            'peak_acceleration_z',
            'pitch_speed_mph',
            'test_date',
            'athlete_uid'}

        # Check that extra_column is NOT present
        self.assertNotIn('extra_column', result.columns)

        # Check that all expected columns are present
        for col in expected_cols:
            self.assertIn(col, result.columns)

    def test_filter_columns_custom(self):
        """Test column filtering with custom parameters"""
        result = filter_columns(
            self.sample_data,
            include_patterns=['velocity'],
            additional_columns=['athlete_uid']
        )

        expected_cols = {'peak_velocity_x', 'peak_velocity_y', 'athlete_uid'}
        self.assertEqual(set(result.columns), expected_cols)

    def test_remove_outliers_iqr(self):
        """Test outlier removal using IQR method"""
        result = remove_outliers(
            self.data_with_outliers,
            'pitch_speed_mph',
            method='iqr')

        # The extreme outlier (150.0) should be removed
        self.assertLess(len(result), len(self.data_with_outliers))
        self.assertNotIn(150.0, result['pitch_speed_mph'].values)

    def test_remove_outliers_invalid_column(self):
        """Test outlier removal with invalid column"""
        with self.assertRaises(ValueError):
            remove_outliers(self.sample_data, 'nonexistent_column')

    def test_remove_outliers_invalid_method(self):
        """Test outlier removal with invalid method"""
        with self.assertRaises(ValueError):
            remove_outliers(
                self.sample_data,
                'pitch_speed_mph',
                method='invalid')

    def test_handle_missing_values_mean(self):
        """Test handling missing values with mean strategy"""
        # FIXED: Changed parameter name from 'strategy' to 'imputation_strategy'
        result = handle_missing_values(self.data_with_nulls, imputation_strategy='mean')

        # Check that no nulls remain in numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        self.assertEqual(result[numeric_cols].isnull().sum().sum(), 0)

    def test_handle_missing_values_drop(self):
        """Test handling missing values with drop strategy"""
        # FIXED: Changed parameter name from 'strategy' to 'imputation_strategy'
        result = handle_missing_values(self.data_with_nulls, imputation_strategy='drop')

        # Should have fewer rows after dropping nulls
        self.assertLess(len(result), len(self.data_with_nulls))
        self.assertEqual(result.isnull().sum().sum(), 0)

    def test_handle_missing_values_invalid_strategy(self):
        """Test handling missing values with invalid strategy"""
        with self.assertRaises(ValueError):
            # FIXED: Changed parameter name from 'strategy' to 'imputation_strategy'
            handle_missing_values(self.data_with_nulls, imputation_strategy='invalid')

    def test_prepare_features_target(self):
        """Test feature and target preparation"""
        X, y = prepare_features_target(self.sample_data)

        # X should contain only float columns except target
        expected_feature_cols = {
            'peak_velocity_x',
            'peak_velocity_y',
            'peak_acceleration_z'}
        self.assertEqual(set(X.columns), expected_feature_cols)

        # y should be the target column
        self.assertEqual(y.name, 'pitch_speed_mph')
        self.assertEqual(len(y), len(self.sample_data))

    def test_prepare_features_target_missing_target(self):
        """Test feature preparation with missing target column"""
        data_no_target = self.sample_data.drop('pitch_speed_mph', axis=1)

        with self.assertRaises(ValueError):
            prepare_features_target(data_no_target)

    def test_split_and_scale_data(self):
        """Test data splitting and scaling"""
        X, y = prepare_features_target(self.sample_data)
        # FIXED: Changed variable names to match refactored code
        train_features_scaled, test_features_scaled, train_targets, test_targets, feature_scaler = split_and_scale_data(
            X, y)

        # Check shapes
        total_samples = len(X)
        self.assertEqual(
            len(train_features_scaled) +
            len(test_features_scaled),
            total_samples)
        self.assertEqual(len(train_targets) + len(test_targets), total_samples)

        # Check that test size is approximately 20%
        self.assertAlmostEqual(
            len(test_features_scaled) /
            total_samples,
            0.2,
            places=1)

        # Check that data is scaled (mean ≈ 0, std ≈ 1)
        self.assertAlmostEqual(np.mean(train_features_scaled), 0, places=1)
        self.assertAlmostEqual(np.std(train_features_scaled), 1, places=1)


class TestModelTraining(unittest.TestCase):
    """Test model training functions"""

    def setUp(self):
        """Set up test data for model training"""
        np.random.seed(42)
        self.X_train = np.random.randn(100, 5)
        self.y_train = pd.Series(np.random.randn(100) * 10 + 90)
        self.X_test = np.random.randn(20, 5)
        self.y_test = pd.Series(np.random.randn(20) * 10 + 90)

    def test_train_linear_regression(self):
        """Test linear regression training"""
        model = train_linear_regression(self.X_train, self.y_train)

        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

        # Test prediction
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

    def test_train_bayesian_ridge(self):
        """Test Bayesian Ridge training"""
        model = train_bayesian_ridge(self.X_train, self.y_train)

        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

        # Test prediction
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

    def test_train_random_forest(self):
        """Test Random Forest training"""
        # FIXED: Changed parameter name from 'n_estimators' to 'number_of_trees'
        model = train_random_forest(
            self.X_train, self.y_train, number_of_trees=10)

        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

        # Test prediction
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

    def test_evaluate_model(self):
        """Test model evaluation"""
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(self.X_train, self.y_train)

        result = evaluate_model(model, self.X_test, self.y_test, "Test Model")

        self.assertIn('model_name', result)
        self.assertIn('mse', result)
        self.assertIn('r2', result)
        self.assertIn('predictions', result)
        self.assertEqual(result['model_name'], "Test Model")
        self.assertGreaterEqual(result['mse'], 0)
        self.assertLessEqual(result['r2'], 1.0)

    def test_train_all_models(self):
        """Test training all models at once"""
        models_dict = train_all_models(self.X_train, self.y_train)

        # Check that all models are returned
        self.assertIn('linear_regression', models_dict)
        self.assertIn('bayesian_ridge', models_dict)
        self.assertIn('random_forest', models_dict)

        # Check that all models can make predictions
        for model_name, model in models_dict.items():
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))

    def test_evaluate_all_models(self):
        """Test evaluating all models at once"""
        # First train all models
        models_dict = train_all_models(self.X_train, self.y_train)

        # Then evaluate them
        results_list = evaluate_all_models(models_dict, self.X_test, self.y_test)

        # Check that we have results for all 3 models
        self.assertEqual(len(results_list), 3)

        # Check that each result has required keys
        for result in results_list:
            self.assertIn('model_name', result)
            self.assertIn('mse', result)
            self.assertIn('r2', result)
            self.assertIn('predictions', result)


class TestVisualization(unittest.TestCase):
    """Test visualization functions"""

    def setUp(self):
        """Set up test data for visualization tests"""
        self.sample_data = pd.DataFrame({
            'pitch_speed_mph': np.random.normal(90, 5, 100)
        })
        self.y_test = pd.Series(np.random.normal(90, 5, 20))
        self.y_pred = np.random.normal(90, 5, 20)

    @patch('matplotlib.pyplot.show')
    def test_plot_pitch_speed_histogram(self, mock_show):
        """Test histogram plotting"""
        plot_pitch_speed_histogram(self.sample_data)
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_actual_vs_predicted(self, mock_show):
        """Test actual vs predicted plotting"""
        plot_actual_vs_predicted(self.y_test, self.y_pred, "Test Model")
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_model_comparison(self, mock_show):
        """Test model comparison plotting"""
        predictions_dict = {
            'Model 1': self.y_pred,
            'Model 2': self.y_pred + np.random.normal(0, 1, len(self.y_pred))
        }
        plot_model_comparison(self.y_test, predictions_dict)
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_create_model_visualizations(self, mock_show):
        """Test creating all model visualizations"""
        evaluation_results = [
            {'model_name': 'Linear Regression', 'mse': 25.0, 'r2': 0.75, 'predictions': self.y_pred},
            {'model_name': 'Bayesian Ridge', 'mse': 26.0, 'r2': 0.74, 'predictions': self.y_pred},
            {'model_name': 'Random Forest', 'mse': 24.0, 'r2': 0.76, 'predictions': self.y_pred}
        ]

        create_model_visualizations(self.y_test, evaluation_results)

        # Should be called 4 times (3 individual plots + 1 comparison plot)
        self.assertEqual(mock_show.call_count, 4)

    def test_create_results_dataframe(self):
        """Test results DataFrame creation"""
        evaluation_results = [
            {'model_name': 'Model 1', 'mse': 25.0, 'r2': 0.75},
            {'model_name': 'Model 2', 'mse': 30.0, 'r2': 0.70}
        ]

        result_df = create_results_dataframe(evaluation_results)

        self.assertEqual(len(result_df), 2)
        self.assertIn('Model', result_df.columns)
        self.assertIn('Mean Squared Error', result_df.columns)
        self.assertIn('R^2 Score', result_df.columns)
        self.assertEqual(list(result_df['Model']), ['Model 1', 'Model 2'])


class TestPipelineIntegration(unittest.TestCase):
    """Test full pipeline integration"""

    def setUp(self):
        """Set up test data for pipeline tests"""
        # Create a temporary CSV file for testing
        self.test_data = pd.DataFrame({
            'pitch_speed_mph': np.random.normal(90, 5, 50),
            'peak_velocity_x': np.random.normal(13, 2, 50),
            'peak_velocity_y': np.random.normal(9, 1.5, 50),
            'peak_acceleration_x': np.random.normal(50, 10, 50),
            'peak_acceleration_y': np.random.normal(25, 5, 50),
            'test_date': ['2024-01-01'] * 50,
            'athlete_uid': [f'athlete_{i}' for i in range(50)]
        })

    @patch('matplotlib.pyplot.show')
    @patch('driveline.load_data')
    def test_run_complete_pipeline(self, mock_load_data, mock_show):
        """Test the complete pipeline execution"""
        mock_load_data.return_value = self.test_data

        # This should run without errors
        result = run_complete_pipeline('test.csv')

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)  # Should have 3 models
        self.assertIn('Model', result.columns)
        mock_load_data.assert_called_once_with('test.csv')


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def test_empty_dataframe_handling(self):
        """Test behavior with empty DataFrame"""
        empty_df = pd.DataFrame()
        result = filter_columns(empty_df)
        self.assertEqual(len(result.columns), 0)

    def test_single_row_dataframe(self):
        """Test behavior with single row DataFrame"""
        single_row_df = pd.DataFrame({
            'pitch_speed_mph': [90.0],
            'peak_velocity_x': [13.0]
        })

        # Should handle single row gracefully
        result = remove_outliers(single_row_df, 'pitch_speed_mph')
        self.assertEqual(len(result), 1)

    def test_all_null_column(self):
        """Test behavior with completely null columns"""
        data_with_null_col = pd.DataFrame({
            'pitch_speed_mph': [85, 87, 90],
            'all_null_col': [np.nan, np.nan, np.nan],
            'peak_velocity_x': [1, 2, 3]
        })

        # FIXED: Changed parameter name from 'strategy' to 'imputation_strategy'
        result = handle_missing_values(data_with_null_col, imputation_strategy='mean')
        # All null column should remain null (mean of all NaN is NaN)
        self.assertTrue(result['all_null_col'].isnull().all())

    def test_no_numeric_columns(self):
        """Test behavior with no numeric columns"""
        text_only_df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'category': ['A', 'B', 'C']
        })

        with self.assertRaises(ValueError):
            prepare_features_target(text_only_df)


if __name__ == '__main__':
    # Suppress excessive output during testing
    import warnings
    warnings.filterwarnings('ignore')

    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestDataLoading,
        TestDataPreprocessing,
        TestModelTraining,
        TestVisualization,
        TestPipelineIntegration,
        TestEdgeCases
    ]

    for test_class in test_classes:
        suite.addTest(unittest.makeSuite(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"TEST SUMMARY")
    print(f"{'=' * 50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, failure in result.failures:
            print(f"  {test}: {failure.splitlines()[-1]}")

    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, error in result.errors:
            print(f"  {test}: {error.splitlines()[-1]}")

    if not result.failures and not result.errors:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {len(result.failures + result.errors)} test(s) failed")