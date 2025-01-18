import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pointbiserialr
from data_cleaning import * 
from data_visualization_delete import *
from data_analysis import *
 
# כאן נכניס דאטא קלינינג!!!

def main():
    # Load the dataset
    input_file = "brain_stroke.csv"  # Path to the dataset
    df = pd.read_csv(input_file)

    # Loop through all columns in the DataFrame
    for column in df.columns:
    
        # Classify the column type
        column_type = classify_column_type(column, df)

        # Analyze the column using the central function
        analysis_result = analyze_column(column, df, column_type)[2]

        # Call the corresponding visualization function
        if column_type == 'binary':
            plot_categorical_analysis(analysis_result, column)
        elif column_type == 'categorical':
            plot_categorical_analysis(analysis_result, column)
        else: # column_type == 'numerical'
            plot_numerical_analysis(analysis_result, column)


if __name__ == "__main__":
    main()