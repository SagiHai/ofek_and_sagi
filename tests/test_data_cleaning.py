import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


from data_cleaning import classify_column_type, create_age_groups, check_missing_values, handle_missing_values, remove_outliers

class TestDataCleaning(unittest.TestCase):

    def test_classify_column_type(self):
        """
        Test classify_column_type function.
        - Positive Test Case: Expected classification of numerical, binary, and categorical columns.
        - Negative Test Case: Non-existent column or invalid inputs.
        """
        df = pd.DataFrame({
            'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M'],
            'work_type': ['Private', 'Self-employed', 'Private', 'Govt_job', 'Private', 'Self-employed', 'Private', 'Govt_job', 'Private', 'Self-employed', 'Private'],
            'age': [10 , 20, 30, 40, 50, 60, 70, 80, 90, 11, 22] # Assumption: if more than 10 unique values, likely numerical
        })
        
        self.assertEqual(classify_column_type('gender', df), 'binary')
        self.assertEqual(classify_column_type('work_type', df), 'categorical')
        self.assertEqual(classify_column_type('age', df), 'numerical')
        
        # Negative Test Case: Non-existent column
        with self.assertRaises(KeyError):
            classify_column_type('non_existent_col', df)

    def test_create_age_groups(self):
        """
        Test create_age_groups function.
        - Positive Test Case: Expected grouping of ages into defined bins.
        - Boundary Test Case: Age values at bin edges (e.g., 9, 18, etc.).
        """
        df = pd.DataFrame({'age': [5, 9, 10, 18, 25, 45, 75, 80]})
        result = create_age_groups(df)
        expected_groups = ['0-9', '0-9', '10-18', '10-18', '19-30', '31-45', '61-75', '75+']
        self.assertListEqual(result['age_group'].tolist(), expected_groups)

    def test_check_missing_values(self):
        """
        Test check_missing_values function.
        - Positive Test Case: Detect missing values in DataFrame.
        - Negative Test Case: DataFrame without missing values.
        """
        df_with_missing = pd.DataFrame({
            'age': [25, np.nan, 35],
            'gender': ['M', np.nan, 'F']
        })
        self.assertTrue(check_missing_values(df_with_missing))
        
        df_without_missing = pd.DataFrame({
            'age': [25, 30, 35],
            'gender': ['M', 'F', 'F']
        })
        self.assertFalse(check_missing_values(df_without_missing))

    def test_handle_missing_values(self):
        """
        Test handle_missing_values function.
        - Positive Test Case: Proper filling of numerical and categorical columns.
        - Additional Test Case: Categorical column with multiple modes, and age groups with no values.
        """
        df = pd.DataFrame({
            'age': [25, np.nan, 35, 45, 25, np.nan, 35, 45 ,25, np.nan, 35, 45],
            'gender': ['M', 'F', np.nan, 'F','M', 'F', np.nan, 'F','M', 'F', np.nan, 'F'],
            'bmi': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, 11]
        })
        result = handle_missing_values(df)
        
        # Ensure age is filled with mean
        self.assertAlmostEqual(result['age'].iloc[1], 35)
        
        # Ensure gender is filled with mode
        self.assertEqual(result['gender'].iloc[2], 'F')

        # Test case with multiple modes in categorical column, and no values in an age group
        df_with_modes = pd.DataFrame({
            'age': [25, 26, 35, 36],
            'gender': [np.nan, np.nan, 'F','M']
        })
        result_with_modes = handle_missing_values(df_with_modes)
        self.assertIn(result_with_modes['gender'].iloc[2], ['M', 'F'])

    def test_remove_outliers(self):
        """
        Test remove_outliers function.
        - Positive Test Case: Detect and remove outliers based on z-scores.
        - Edge Test Case: Handle empty DataFrame.
        - Boundary Test Case: Include edge cases for numeric column ranges.
        """
        df = pd.DataFrame({
            'age': [25, 30, 35, 20, 22, 25, 32, 23, 20, 30, 25, 100],  # Column age has an outlier, and numeric (int\float and above 10 unique values)
            'bmi': [18.5, 24.0, 24.5, 30.0, 18.5, 24.0, 24.5, 30.0, 18.5, 24.0, 24.5, 30.0]  # Column bmi has no outliers
        })
        result = remove_outliers(df)
        self.assertNotIn(100, result['age'].values)

        # Edge Case: Empty DataFrame
        empty_df = pd.DataFrame()
        result_empty = remove_outliers(empty_df)
        pd.testing.assert_frame_equal(result_empty, empty_df)

if __name__ == '__main__':
    unittest.main()
