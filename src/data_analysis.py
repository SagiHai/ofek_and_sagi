import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pointbiserialr
from data_cleaning import create_age_groups, classify_column_type

def load_data(file_path):
    """
    Load the stroke dataset from CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} was not found")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("The file is empty")


def analyze_binary_column (column_name, df):
    """
    Analyzes a binary column in relation to the target variable (stroke).
    
    Parameters:
    column_name (str): Name of the column to analyze
    df (pd.DataFrame): DataFrame containing the data
    
    Returns:
    tuple: A tuple containing the analysis Chi-square value, p-value and DataFrame
    """
    if column_name == 'stroke':
        target_distribution = df[column_name].value_counts(normalize=True) * 100
        result_df = pd.DataFrame(target_distribution).reset_index().rename(
            columns={"index": column_name, column_name: "stroke"}
        )
        return None, None, result_df
    
    # Compute the cross-tabulation (distribution relative to the target variable)
    cross_tab = pd.crosstab(df[column_name], df['stroke'], normalize='index') * 100
    result_df = cross_tab.reset_index().rename_axis(None, axis=1)
    result_df.columns = [column_name, 'no_stroke_percentage', 'stroke_percentage']
    
    # Perform Chi-square test for independence
    chi2, p_value = stats.chi2_contingency(pd.crosstab(df[column_name], df['stroke']))[:2]
    
    return chi2,p_value,result_df

def analyze_categorical_column (column_name, df):
    """
    Analyzes a categorical column in relation to the target variable (stroke).
    
    Parameters:
    column_name (str): Name of the column to analyze
    df (pd.DataFrame): DataFrame containing the data
    
    Returns:
    tuple: 
        - Chi-square value
        - p-value
        - DataFrame summarizing the distribution of the column relative to stroke
        - DataFrame summarizing the overall category distribution

    """
    # Calculate overall category distribution
    category_distribution = df[column_name].value_counts(normalize=True) * 100
    category_distribution_df = category_distribution.reset_index().rename(
        columns={"index": column_name, column_name: "percentage"}
    )
    
    # Calculate distribution relative to stroke
    cross_tab = pd.crosstab(df[column_name], df['stroke'], normalize='index') * 100
    relative_distribution_df = cross_tab.reset_index().rename_axis(None, axis=1)
    relative_distribution_df.columns = [column_name, 'no_stroke_percentage', 'stroke_percentage']
    
    # Perform Chi-square test
    chi2, p_value = stats.chi2_contingency(pd.crosstab(df[column_name], df['stroke']))[:2]
    
    return chi2,p_value,relative_distribution_df, category_distribution_df

def analyze_numerical_column (column_name, df):
    """
    Analyzes a numerical column in relation to the target variable (stroke)
    and computes Point-Biserial Correlation.
    
    Parameters:
    column_name (str): Name of the numerical column
    df (pd.DataFrame): DataFrame
    
    Returns:
    corr (float): Point-Biserial Correlation coefficient
    p_value_corr (float): p-value for the correlation
    stats_desc (pd.Series): Descriptive statistics for the numerical column
    group_stats (pd.DataFrame): Descriptive statistics by stroke group
    t_stat (float): t-statistic for group comparison
    p_value_ttest (float): p-value for t-test
    """
    # Basic descriptive statistics
    stats_desc = df[column_name].describe()

    # Statistics by stroke group
    group_stats = df.groupby('stroke')[column_name].describe()

    # T-test for group differences
    stroke_yes = df[df['stroke'] == 1][column_name]
    stroke_no = df[df['stroke'] == 0][column_name]
    t_stat, p_value_ttest = stats.ttest_ind(stroke_yes, stroke_no)

    # Point-Biserial Correlation
    corr, p_value_corr = pointbiserialr(df['stroke'], df[column_name])

    return corr, p_value_corr, group_stats, stats_desc, t_stat, p_value_ttest

def analyze_column(column_name, df, column_type):
    """
    Main function that routes analysis based on column type
    
    Parameters:
    column_name (str): Name of the column
    df (pd.DataFrame): DataFrame
    """

    if column_type == 'binary':
        return analyze_binary_column(column_name, df)
    elif column_type == 'categorical':
        return analyze_categorical_column(column_name, df)
    else:  # numerical
        return analyze_numerical_column(column_name, df)

def analyze_modifiable_vs_nonmodifiable(corr_dict):
    """
    Compares cumulative impact of modifiable vs non-modifiable variables
    
    Parameters:
    corr_dict(dict)

    Returns:
    A tuple with the numerical variables:
    modifiable_impact, nonmodifiable_impact, ratio
    """
    modifiable = ['avg_glucose_level', 'bmi', 'smoking_status', 'work_type', 'Residence_type']
    nonmodifiable = ['age', 'gender', 'hypertension', 'heart_disease']

    # Filter to remain only existing keys from corr_dict
    modifiable_vars = [var for var in modifiable if var in corr_dict]
    nonmodifiable_vars = [var for var in nonmodifiable if var in corr_dict]
    
    # Calculate cumulative impact
    modifiable_impact = sum(corr_dict[col] #correlation value
                           for col in modifiable_vars)
    nonmodifiable_impact = sum(corr_dict[col] #correlation value
                           for col in nonmodifiable_vars)
    
    ratio= modifiable_impact/nonmodifiable_impact
    return modifiable_impact, nonmodifiable_impact, ratio

def calculate_category_analysis(df, gender_col, category_col):
    """
    Calculate stroke analysis for a given categorical variable by gender.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    gender_col (str): Column name for gender.
    category_col (str): Column name for the categorical variable to analyze.

    Returns:
    pd.DataFrame: Analysis DataFrame with total cases, stroke cases, and stroke rate.
    """
    analysis = df.groupby([gender_col, category_col]).agg({
        'stroke': ['count', 'sum', lambda x: (x.sum() / len(x)) * 100]
    }).round(2)
    analysis.columns = ['total_cases', 'stroke_cases', 'stroke_rate']
    return analysis

def analyze_by_gender(df):
    """
    Analyze stroke statistics by gender and different categories.
    Returns a dictionary containing DataFrames for each analysis.
    """
    results = {}

    # Age Analysis
    df_with_age_groups = create_age_groups(df)  # Already defined in data_cleaning.py
    results['age'] = calculate_category_analysis(df_with_age_groups, 'gender', 'age_group')

    # Smoking Status Analysis
    results['smoking'] = calculate_category_analysis(df, 'gender', 'smoking_status')

    # BMI Analysis - Convert BMI to categories first
    df['bmi_category'] = pd.cut(df['bmi'], 
                                bins=[0, 18.5, 24.9, 29.9, float('inf')],
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    results['bmi'] = calculate_category_analysis(df, 'gender', 'bmi_category')

    return results