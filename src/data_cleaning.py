import pandas as pd
import numpy as np
from typing import List, Union

def load_data(file_path: str) -> pd.DataFrame:
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

def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Check for missing values in each column.
    """
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    return missing_percentages[missing_percentages > 0]

def create_age_groups(df: pd.DataFrame, age_column: str = 'age') -> pd.DataFrame:
    """
    Create age groups from continuous age data.
    """
    df_with_groups = df.copy()
    
    # Define age bins and labels
    age_bins = [0, 18, 30, 45, 60, 75, float('inf')]
    age_labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '75+']
    
    # Create age groups
    df_with_groups['age_group'] = pd.cut(df_with_groups[age_column], 
                                        bins=age_bins, 
                                        labels=age_labels, 
                                        right=False)
    return df_with_groups

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    """
    df_clean = df.copy()
    
    # Handle numeric columns - fill with median
    numeric_columns = df_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Handle categorical columns - fill with mode
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    return df_clean

def remove_outliers(df: pd.DataFrame, columns: List[str], n_std: float = 3) -> pd.DataFrame:
    """
    Remove outliers using z-score method.
    """
    df_clean = df.copy()
    
    for column in columns:
        if df_clean[column].dtype in ['float64', 'int64']:
            # Calculate z-scores
            z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / df_clean[column].std())
            # Keep only rows where all z-scores are below n_std
            df_clean = df_clean[z_scores < n_std]
    
    return df_clean

# קוד ראשי לבדיקה
if __name__ == "__main__":
    # טען את הנתונים
    df = load_data('brain_stroke.csv')
    
    # בדוק ערכים חסרים
    missing = check_missing_values(df)
    print("\nMissing values before cleaning:")
    print(missing)
    
    # טפל בערכים חסרים
    df_clean = handle_missing_values(df)
    
    # בדוק שוב ערכים חסרים
    missing_after = check_missing_values(df_clean)
    print("\nMissing values after cleaning:")
    print(missing_after if not missing_after.empty else "No missing values!")
    
    # הסר ערכים חריגים מעמודות מספריות
    numeric_columns = ['age', 'avg_glucose_level', 'bmi']
    df_clean = remove_outliers(df_clean, numeric_columns)
    print(f"\nShape after removing outliers: {df_clean.shape}")
    
    # צור קבוצות גיל
    df_clean = create_age_groups(df_clean)
    print("\nAge groups distribution:")
    print(df_clean['age_group'].value_counts())
    
    print("\nColumns in cleaned dataset:")
    print(df_clean.columns.tolist())