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
from data_visualization import *
from data_analysis import *


def main():
    # Load the dataset
    input_file = "brain_stroke.csv"  # Path to the dataset
    df = load_data(input_file)

    # Data cleaning - Missing values
    if check_missing_values(df):
        df = handle_missing_values(df)
    
    # Data cleaning - Outliers
    df = remove_outliers(df)

    # Calculate correlations and variabels distribution with target variable (Stroke)
    correlations = {}

    # Loop through all columns in the DataFrame
    for column in df.columns:
    
        # Classify the column type
        column_type = classify_column_type(column, df)

        # Analyze the column using the central function
        analysis_result = analyze_column(column, df, column_type)
        analysis_result_df = analysis_result[2]
        correlation = analysis_result[0]
        if column != 'stroke':
            correlations[column] = correlation

        # Call the corresponding visualization distributions function
        if column=='stroke':
            plot_stroke_proportion(analysis_result_df)
        elif column_type == 'binary':
            plot_categorical_analysis(analysis_result_df, column)
        elif column_type == 'categorical':
            plot_categorical_analysis(analysis_result_df, column)
        else: # column_type == 'numerical'
            plot_numerical_analysis(analysis_result_df, column)
    
    # Variable's Importance by Correlations plot
    sorted_correlations = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
    plot_variable_importance(sorted_correlations)

    # Modifiable vs. Nonmodifiable 
    modifiable_result = analyze_modifiable_vs_nonmodifiable(sorted_correlations)
    plot_modifiable_vs_nonmodifiable_combined(modifiable_result[0],modifiable_result[1],modifiable_result[2])

    # Analyze age, bmi and smoking status bygender
    gender_analysis_results = analyze_by_gender(df)
    visualize_stroke_data_by_gender(gender_analysis_results)


if __name__ == "__main__":
    main()