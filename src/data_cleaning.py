import pandas as pd
import numpy as np
from typing import List, Union

def classify_column_type (column_name, df):
    """
    Classifies a column as binary, categorical, or numerical
    
    Parameters:
    column_name (str): Name of the column
    df (pd.DataFrame): DataFrame containing the column
    
    Returns:
    str: Column type ('binary', 'categorical', or 'numerical')
    """
    unique_values = df[column_name].nunique()
    
    if pd.api.types.is_numeric_dtype(df[column_name]):
        if unique_values == 2:
            return 'binary'
        elif unique_values > 10:  # Assumption: if more than 10 unique values, likely numerical
            return 'numerical'
        else:
            return 'categorical'
    else:  # If column contains strings
        if unique_values == 2:
            return 'binary'
        else:
            return 'categorical'

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

def check_missing_values(df):
    """
    Check for missing values in each column.
    """
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        return False
    else:
        return True

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
   
   # Fill missing age values with mean
   df_clean['age'] = df_clean['age'].fillna(df_clean['age'].mean())

   
   # Create age groups
   df_clean = create_age_groups(df_clean)
   
   # Identify column types
   numeric_cols = [] #'avg_glucose_level', 'bmi'
   categorical_cols = [] #'work_type', 'Residence_type', 'smoking_status', #'gender', 'ever_married', 'hypertension', 'heart_disease', 'stroke'
   
   # Iteration only original coloumns
   cols = df_clean.columns.tolist()
   if 'age_group' in cols:
       cols.remove('age_group')

   for col in cols:
       column_type = classify_column_type(col, df)
       if col == 'age':# Skip already handled column
           continue
       elif column_type in ['binary','categorical']:
           categorical_cols.append(col)
       else:# column_type == 'numerical'
           numeric_cols.append(col)
   
   # Fill missing values by age group and column type
   for group in df_clean['age_group'].unique():
       group_data = df_clean[df_clean['age_group'] == group]
       # Handle binary and categorical columns - fill with mode (most frequent value)
       for col in categorical_cols:
           if df_clean[col].isna().any():
               mode_values = group_data[col].mode()
               if len(mode_values) == 0:
                   mode_val = df_clean[col].mode()[0] # General mode if there are no values in age group
               else:
                   mode_val = group_data[col].mode()[0]
               df_clean.loc[df_clean['age_group'] == group, col] = (df_clean.loc[df_clean['age_group'] == group, col].fillna(mode_val))
       
       # Handle numeric columns - fill with mean
       for col in numeric_cols:
           if df_clean[col].isna().any():
               mean_val = group_data[col].mean()
               if mean_val>0 or mean_val<=0:# Not Nan
                   pass
               else:
                   mean_val = df_clean[col].mean() # General mean if there are no values in age group
               df_clean.loc[df_clean['age_group'] == group, col] = df_clean.loc[df_clean['age_group'] == group, col].fillna(mean_val)
   
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