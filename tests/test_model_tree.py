import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model_tree import prepare_data_for_model

class TestModelTree(unittest.TestCase):

    def test_prepare_data_for_model(self):
        """
        Test prepare_data_for_model function.
        - Positive Test Case: Check if categorical variables are properly encoded and features/target are separated.
        - Additional Case: Ensure 'stroke' column is separated as target.
        """
        df = pd.DataFrame({
            'gender': ['M', 'F', 'M', 'F'],
            'age': [25, 30, 35, 40],
            'stroke': [0, 1, 0, 1]
        })
        x, y = prepare_data_for_model(df)
        
        # Check that 'stroke' column is the target
        self.assertTrue('stroke' not in x.columns)
        self.assertTrue('stroke' in df.columns)
        self.assertTrue(all(y == df['stroke']))
        
        # Check if categorical variables are encoded
        self.assertTrue(x['gender'].dtype in [np.int32, np.int64])

if __name__ == '__main__':
    unittest.main()
