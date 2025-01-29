import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_analysis import (
   load_data,
   analyze_binary_column,
   analyze_categorical_column,
   analyze_numerical_column,
   analyze_modifiable_vs_nonmodifiable,
   calculate_category_analysis
)

class TestDataAnalysis(unittest.TestCase):
    """
    Unit tests for data analysis functions.
    - Positive Test Cases: Ensure expected outputs for valid inputs.
    - Negative Test Cases: Validate handling of invalid inputs or edge cases.
    - Edge Cases: Validate function behavior under special circumstances.
    """

    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame({
            'stroke': [0, 0, 1, 0, 1],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'age': [45, 32, 67, 28, 71],
            'avg_glucose_level': [110, 90, 140, 85, 160],
            'bmi': [25, 22, 28, 21, 30],
            'smoking_status': ['never', 'smokes', 'former', 'never', 'smokes'],
            'heart_disease': [0, 0, 1, 0, 1],
            'work_type': ['Private', 'Govt_job', 'Private', 'Self-employed', 'Private']
        })

    def test_load_data(self):
      """
      Test load_data function.
      - Positive Test Case: Valid file path with proper CSV data.
      - Error Test Case: File not found or empty file.
      """
      # Positive: Valid file path
      valid_file_path = 'test_data.csv'
      test_df = pd.DataFrame({
         'col1': [1, 2, 3],
         'col2': ['A', 'B', 'C']
      })
      test_df.to_csv(valid_file_path, index=False)

      loaded_df = load_data(valid_file_path)
      pd.testing.assert_frame_equal(loaded_df, test_df)
      os.remove(valid_file_path)

      # Error: File not found
      with self.assertRaises(FileNotFoundError):
         load_data('non_existent_file.csv')

      # Error: Empty file
      empty_file_path = 'empty_file.csv'
      pd.DataFrame().to_csv(empty_file_path, index=False)
      with self.assertRaises(pd.errors.EmptyDataError):
         load_data(empty_file_path)
      os.remove(empty_file_path)

    
    def test_analyze_binary_column(self):
        """
        Test analyze_binary_column function.
        - Positive Test Case: Expected chi-squared test results and distribution percentages.
        - Edge Test Case: Binary column with uneven distribution.
        - Error Test Case: Column not found.
        """
        result = analyze_binary_column('heart_disease', self.df)
        chi2, p_value, result_df = result

        # Positive: Check chi-squared results
        expected_chi2 = 3.125
        expected_p_value = 0.077

        self.assertAlmostEqual(chi2, expected_chi2, places=3)
        self.assertAlmostEqual(p_value, expected_p_value, places=3)

        # Positive: Check distribution percentages
        expected_percentages = {
            0: [80.0, 20.0],  # [no_stroke_percentage, stroke_percentage]
            1: [0.0, 100.0]
        }

        unique_values = result_df['heart_disease'].unique()  # Extract unique binary values dynamically
        for val in unique_values:
            row = result_df[result_df['heart_disease'] == val]

            # Validate the percentage of non-stroke cases
            self.assertAlmostEqual(
                row['no_stroke_percentage'].iloc[0], 
                expected_percentages[val][0], 
                places=1,
                msg=f"Mismatch in no_stroke_percentage for heart_disease={val}"
            )

            # Validate the percentage of stroke cases
            self.assertAlmostEqual(
                row['stroke_percentage'].iloc[0], 
                expected_percentages[val][1], 
                places=1,
                msg=f"Mismatch in stroke_percentage for heart_disease={val}"
            )

        # Error: Column not found
        with self.assertRaises(KeyError):
            analyze_binary_column('non_existent_column', self.df)

    def test_analyze_categorical_column(self):
        """
        Test analyze_categorical_column function.
        - Positive Test Case: Expected chi-squared test results and distribution percentages.
        - Negative Test Case: Empty DataFrame.
        - Error Test Case: Non-categorical column input.
        - Edge Test case: Columns with single unique value (no variance) - can't calculate valid chi-val.
        """
        result = analyze_categorical_column('work_type', self.df)
        chi2, p_value, rel_dist_df, cat_dist_df = result

        # Positive: Check chi-squared results
        expected_chi2 = 2.756
        expected_p_value = 0.431

        self.assertAlmostEqual(chi2, expected_chi2, places=3)
        self.assertAlmostEqual(p_value, expected_p_value, places=3)

        # Positive: Check distribution percentages
        expected_rel_dist = {
            'Private': [75.0, 25.0],
            'Govt_job': [100.0, 0.0],
            'Self-employed': [100.0, 0.0]
        }

        for work_type in expected_rel_dist:
            row = rel_dist_df[rel_dist_df['work_type'] == work_type]
            # Validate the percentage of non-stroke cases
            self.assertAlmostEqual(row['no_stroke_percentage'].iloc[0],
                                   expected_rel_dist[work_type][0], places=1)
            # Validate the percentage of stroke cases
            self.assertAlmostEqual(row['stroke_percentage'].iloc[0],
                                   expected_rel_dist[work_type][1], places=1)

        # Negative: Empty DataFrame
        empty_df = pd.DataFrame({'work_type': [], 'stroke': []})
        with self.assertRaises(ValueError):
            analyze_categorical_column('work_type', empty_df)

        # Error: Non-categorical input
        with self.assertRaises(TypeError):
            analyze_categorical_column('age', self.df)
         
         # Edge case: Single unique value (no variance) - can't calculate valid chi-val
        single_value_df = pd.DataFrame({
            'stroke': [0, 0],
            'test_col': ['A', 'A']
        })
        with self.assertRaises(ValueError):
            analyze_categorical_column('test_col', single_value_df)

    def test_analyze_numerical_column(self):
        """
        Test analyze_numerical_column function.
        - Positive Test Case: Correlation, t-test, and descriptive statistics.
        - Null Test Case: Column with missing values.
        - Boundary Test Case: Numerical column with minimum data points.
        """
        result = analyze_numerical_column('age', self.df)
        corr, p_value_corr, group_stats, stats_desc, t_stat, p_value_ttest = result

        # Positive: Check correlation and t-test results
        expected_corr = 0.845
        expected_p_value = 0.036
        expected_t_stat = -3.247

        self.assertAlmostEqual(corr, expected_corr, places=3)
        self.assertAlmostEqual(p_value_corr, expected_p_value, places=3)
        self.assertAlmostEqual(t_stat, expected_t_stat, places=3)

        # Positive: Check descriptive statistics
        expected_means = {0: 35.0, 1: 69.0}
        for stroke in [0, 1]:
            self.assertAlmostEqual(group_stats.loc[stroke, 'mean'], 
                                   expected_means[stroke], places=1)

        # Null: Missing values in column
        df_with_nan = self.df.copy()
        df_with_nan.loc[0, 'age'] = np.nan
        with self.assertRaises(ValueError):
            analyze_numerical_column('age', df_with_nan)

    def test_analyze_modifiable_vs_nonmodifiable(self):
        """
        Test analyze_modifiable_vs_nonmodifiable function.
        - Positive Test Case: Proper categorization of modifiable and non-modifiable factors.
        - Error Test Case: Empty correlation dictionary.
        """
        corr_dict = {
            'avg_glucose_level': 0.3,
            'bmi': 0.2,
            'smoking_status': 0.15,
            'work_type': 0.1,
            'age': 0.4,
            'gender': 0.1,
            'hypertension': 0.25,
            'heart_disease': 0.35
        }

        # Positive: Check calculated impacts
        result = analyze_modifiable_vs_nonmodifiable(corr_dict)
        mod_impact, nonmod_impact, ratio = result

        expected_mod = 0.75
        expected_nonmod = 1.1
        expected_ratio = 0.682

        self.assertAlmostEqual(mod_impact, expected_mod, places=3)
        self.assertAlmostEqual(nonmod_impact, expected_nonmod, places=3)
        self.assertAlmostEqual(ratio, expected_ratio, places=3)

        # Error: Empty dictionary
        with self.assertRaises(ZeroDivisionError):
            analyze_modifiable_vs_nonmodifiable({})

    def test_calculate_category_analysis(self):
        """
        Test calculate_category_analysis function.
        - Positive Test Case: Valid combinations of categories and expected rates.
        - Negative Test Case: Non-existent column input.
        - Null Test Case: DataFrame with missing category values.
        """
        result = calculate_category_analysis(self.df, 'gender', 'smoking_status')

        # Positive: Check stroke rates
        expected_rates = {
            ('Male', 'never'): 50.0,
            ('Male', 'smokes'): 0.0,
            ('Female', 'never'): 0.0,
            ('Female', 'smokes'): 100.0
        }

        for (gender, smoking), rate in expected_rates.items():
            actual_rate = result.loc[(gender, smoking), 'stroke_rate']
            self.assertAlmostEqual(actual_rate, rate, places=1)

        # Negative: Non-existent column
        with self.assertRaises(KeyError):
            calculate_category_analysis(self.df, 'non_existent', 'smoking_status')

        # Null: Missing category values
        df_with_nan = self.df.copy()
        df_with_nan.loc[0, 'gender'] = np.nan
        with self.assertRaises(ValueError):
            calculate_category_analysis(df_with_nan, 'gender', 'smoking_status')


if __name__ == '__main__':
    unittest.main()
