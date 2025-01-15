import pandas as pd
import numpy as np
from typing import List, Union

def load_data(file_path):
    """
    Load the stroke dataset from CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} was not found")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("The file is empty")

def check_missing_values(df):
    """
    Check for missing values in each column.
    תזכורת- לבדוק אם בכלל בא לנו את זה
    """
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    return missing_percentages[missing_percentages > 0]

def create_age_groups(df, age_column = 'age'):
    """
    Create age groups from continuous age data.
    """
    df_with_groups = df.copy()
    
    # Define age bins and labels
    age_bins = [0, 9, 18, 30, 45, 60, 75, float('inf')]
    age_labels = ['0-9', '10-18', '19-30', '31-45', '46-60', '61-75', '75+']
    
    # Create age groups
    df_with_groups['age_group'] = pd.cut(df_with_groups[age_column], 
                                        bins=age_bins, 
                                        labels=age_labels)
    return df_with_groups

def handle_missing_values(df):
   """
   Handle missing values based on age groups and column types.
   
   Args:
       df (pd.DataFrame): Input dataset
       
   Returns:
       pd.DataFrame: Clean dataset without missing values
   """
   # Create a copy of the dataframe
   df_clean = df.copy()
   
   # Step 1: Fill missing age values with mean
   df_clean['age'].fillna(df_clean['age'].mean(), inplace=True)
   
   # Step 2: Create age groups
   df_clean = create_age_groups(df_clean)
   
   # Step 3: Identify column types
   binary_cols = [] #'gender', 'ever_married', 'hypertension', 'heart_disease', 'stroke'
   numeric_cols = [] #'avg_glucose_level', 'bmi'
   categorical_cols = [] #'work_type', 'Residence_type', 'smoking_status'
   
   for col in df_clean.columns:
       if col in ['age', 'age_group']:  # Skip already handled columns
           continue
           
       # Check if column is binary (has only 2 unique values)
       if df_clean[col].nunique() == 2:
           binary_cols.append(col)
       # Check if column is numeric
       elif df_clean[col].dtype in ['float64', 'int64']:
           numeric_cols.append(col)
       # If not numeric and not binary, it's categorical
       else:
           categorical_cols.append(col)
   
   # Step 4: Fill missing values by age group and column type
   for group in df_clean['age_group'].unique():
       group_data = df_clean[df_clean['age_group'] == group]
       
       # Handle binary columns - fill with mode (most frequent value)
       for col in binary_cols:
           if df_clean[col].isna().any():
               mode_val = group_data[col].mode()[0]
               df_clean.loc[df_clean['age_group'] == group, col].fillna(mode_val, inplace=True)
       
       # Handle numeric columns - fill with mean
       for col in numeric_cols:
           if df_clean[col].isna().any():
               mean_val = group_data[col].mean()
               df_clean.loc[df_clean['age_group'] == group, col].fillna(mean_val, inplace=True)
       
       # Handle categorical columns - fill with mode
       for col in categorical_cols:
           if df_clean[col].isna().any():
               mode_val = group_data[col].mode()[0]
               df_clean.loc[df_clean['age_group'] == group, col].fillna(mode_val, inplace=True)
   
   df_clean = df_clean.drop('age_group', axis=1)
   return df_clean

def remove_outliers(df, n_std = 3):
    """
    Remove outliers from numeric columns using z-score method.
    """
    df_clean = df.copy()
    
    # Get numeric columns and filter out those with 2 or fewer unique values
    numeric_columns = df_clean.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = [col for col in numeric_columns if (df_clean[col].nunique() > 2)]
    
    for column in numeric_columns:
        # Calculate z-scores
        z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / df_clean[column].std())
        # Keep only rows where z-scores are below n_std
        df_clean = df_clean[z_scores < n_std]
    
    return df_clean