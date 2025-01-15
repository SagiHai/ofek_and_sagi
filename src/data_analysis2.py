import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def read_stroke_data():
    """
    Reads the stroke dataset from CSV file
    
    Returns:
    pd.DataFrame: Stroke data DataFrame
    """
    try:
        df = pd.read_csv('brain_stroke.csv')
        print(f"Read {len(df)} rows and {len(df.columns)} columns from file")
        return df
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

def classify_column_type(column_name, df):
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

def analyze_binary_column(column_name, df):
    """
    Analyzes a binary column in relation to the target variable (stroke)
    
    Parameters:
    column_name (str): Name of the column
    df (pd.DataFrame): DataFrame
    """
    if column_name == 'stroke':
        print(f"Distribution of target variable {column_name}:")
        print(df[column_name].value_counts(normalize=True) * 100)
        return
        
    # Calculate distribution for each value relative to stroke
    cross_tab = pd.crosstab(df[column_name], df['stroke'], normalize='index') * 100
    print(f"\nDistribution of {column_name} relative to stroke (percentage):")
    print(cross_tab)
    
    # Chi-square test for independence
    chi2, p_value = stats.chi2_contingency(pd.crosstab(df[column_name], df['stroke']))[:2]
    print(f"\nChi-square test:")
    print(f"Chi-square value: {chi2:.2f}")
    print(f"p-value: {p_value:.4f}")

def analyze_categorical_column(column_name, df):
    """
    Analyzes a categorical column in relation to the target variable
    
    Parameters:
    column_name (str): Name of the column
    df (pd.DataFrame): DataFrame
    """
    # Category distribution
    print(f"\nCategory distribution in {column_name}:")
    print(df[column_name].value_counts(normalize=True) * 100)
    
    # Distribution relative to stroke
    cross_tab = pd.crosstab(df[column_name], df['stroke'], normalize='index') * 100
    print(f"\nDistribution of {column_name} relative to stroke (percentage):")
    print(cross_tab)
    
    # Chi-square test
    chi2, p_value = stats.chi2_contingency(pd.crosstab(df[column_name], df['stroke']))[:2]
    print(f"\nChi-square test:")
    print(f"Chi-square value: {chi2:.2f}")
    print(f"p-value: {p_value:.4f}")

def analyze_numerical_column(column_name, df):
    """
    Analyzes a numerical column
    
    Parameters:
    column_name (str): Name of the column
    df (pd.DataFrame): DataFrame
    """
    # Basic descriptive statistics
    stats_desc = df[column_name].describe()
    print(f"\nDescriptive statistics for {column_name}:")
    print(stats_desc)
    
    # Statistics by stroke group
    print(f"\nStatistics by stroke groups:")
    print(df.groupby('stroke')[column_name].describe())
    
    # T-test for group differences
    stroke_yes = df[df['stroke'] == 1][column_name]
    stroke_no = df[df['stroke'] == 0][column_name]
    t_stat, p_value = stats.ttest_ind(stroke_yes, stroke_no)
    print(f"\nt-test for group comparison:")
    print(f"t-statistic: {t_stat:.2f}")
    print(f"p-value: {p_value:.4f}")

def analyze_column(column_name, df):
    """
    Main function that routes analysis based on column type
    
    Parameters:
    column_name (str): Name of the column
    df (pd.DataFrame): DataFrame
    """
    column_type = classify_column_type(column_name, df)
    print(f"\nAnalyzing column: {column_name}")
    print(f"Column type: {column_type}")
    
    if column_type == 'binary':
        analyze_binary_column(column_name, df)
    elif column_type == 'categorical':
        analyze_categorical_column(column_name, df)
    else:  # numerical
        analyze_numerical_column(column_name, df)

def analyze_variable_importance(df):
    """
    Analyzes the relative importance of each variable in relation to stroke
    
    Parameters:
    df (pd.DataFrame): DataFrame
    """
    # Create copy of data
    df_encoded = df.copy()
    
    # Convert categorical variables to numerical
    le = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    # Calculate correlations with target variable
    correlations = {}
    for col in df_encoded.columns:
        if col != 'stroke':
            correlation = np.abs(df_encoded[col].corr(df_encoded['stroke']))
            correlations[col] = correlation
    
    # Sort by importance
    sorted_correlations = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_correlations

def analyze_modifiable_vs_nonmodifiable(df):
    """
    Compares cumulative impact of modifiable vs non-modifiable variables
    
    Parameters:
    df (pd.DataFrame): DataFrame
    """
    modifiable_vars = ['avg_glucose_level', 'bmi', 'smoking_status', 'work_type']
    nonmodifiable_vars = ['age', 'gender', 'Residence_type']
    
    df_encoded = df.copy()
    le = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    # Calculate cumulative impact
    modifiable_impact = sum([np.abs(df_encoded[col].corr(df_encoded['stroke'])) 
                           for col in modifiable_vars if col in df_encoded.columns])
    nonmodifiable_impact = sum([np.abs(df_encoded[col].corr(df_encoded['stroke'])) 
                              for col in nonmodifiable_vars if col in df_encoded.columns])
    
    results = {
        'modifiable_impact': modifiable_impact,
        'nonmodifiable_impact': nonmodifiable_impact,
        'ratio': modifiable_impact / nonmodifiable_impact
    }
    
    return results

def analyze_age_gender_stroke(df):
    """
    Analyzes age impact on stroke between males and females
    
    Parameters:
    df (pd.DataFrame): DataFrame
    """
    results = df.groupby(['gender', pd.cut(df['age'], bins=5)])['stroke'].mean()
    return results

def analyze_heart_disease_gender_stroke(df):
    """
    Analyzes heart disease impact on stroke between males and females
    
    Parameters:
    df (pd.DataFrame): DataFrame
    """
    results = df.groupby(['gender', 'heart_disease'])['stroke'].mean()
    return results

def analyze_hypertension_gender_stroke(df):
    """
    Analyzes hypertension impact on stroke between males and females
    
    Parameters:
    df (pd.DataFrame): DataFrame
    """
    results = df.groupby(['gender', 'hypertension'])['stroke'].mean()
    return results

def analyze_numerical_correlations(df):
    """
    Analyzes correlations between numerical variables
    
    Parameters:
    df (pd.DataFrame): DataFrame
    
    Returns:
    pd.DataFrame: Correlation matrix
    """
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlations = df[numerical_cols].corr()
    return correlations

# Usage example
if __name__ == "__main__":
    # Read the data
    df = read_stroke_data()
    
    if df is not None:
        # Example of analyzing a specific column
        print("\nAnalyzing age column:")
        analyze_column('age', df)
        
        # Example of analyzing variable importance
        print("\nRelative importance of variables:")
        importance = analyze_variable_importance(df)
        for var, score in importance.items():
            print(f"{var}: {score:.3f}")
        
        # Example of analyzing modifiable vs non-modifiable variables
        print("\nComparing modifiable vs non-modifiable variables:")
        impact = analyze_modifiable_vs_nonmodifiable(df)
        print(f"Cumulative impact of modifiable variables: {impact['modifiable_impact']:.3f}")
        print(f"Cumulative impact of non-modifiable variables: {impact['nonmodifiable_impact']:.3f}")
        print(f"Ratio: {impact['ratio']:.3f}")