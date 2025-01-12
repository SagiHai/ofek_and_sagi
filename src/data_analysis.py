import pandas as pd
import numpy as np
from data_cleaning import load_data, handle_missing_values, remove_outliers, create_age_groups

def analyze_stroke_by_age(df: pd.DataFrame) -> pd.DataFrame:
   """
   Analyze stroke occurrence by age groups
   """
   stroke_by_age = df.groupby('age_group', observed=True)['stroke'].agg([
       ('total_patients', 'count'),
       ('stroke_cases', 'sum'),
       ('stroke_rate', lambda x: (x.sum() / len(x)) * 100)
   ]).round(2)
   
   return stroke_by_age

def analyze_numerical_correlations(df: pd.DataFrame) -> pd.DataFrame:
   """
   Calculate correlations between numerical variables
   """
   numerical_cols = ['age', 'avg_glucose_level', 'bmi', 'stroke']
   correlations = df[numerical_cols].corr().round(3)
   return correlations

def analyze_risk_factors(df: pd.DataFrame) -> dict:
   """
   Analyze various risk factors and their relationship with stroke
   """
   risk_factors = {}
   
   # ניתוח לפי יתר לחץ דם
   risk_factors['hypertension'] = df.groupby('hypertension')['stroke'].agg([
       ('total_cases', 'count'),
       ('stroke_cases', 'sum'),
       ('stroke_rate', lambda x: (x.sum() / len(x)) * 100)
   ]).round(2)
   
   # ניתוח לפי מחלות לב
   risk_factors['heart_disease'] = df.groupby('heart_disease')['stroke'].agg([
       ('total_cases', 'count'),
       ('stroke_cases', 'sum'),
       ('stroke_rate', lambda x: (x.sum() / len(x)) * 100)
   ]).round(2)
   
   # ניתוח לפי סטטוס עישון
   risk_factors['smoking_status'] = df.groupby('smoking_status')['stroke'].agg([
       ('total_cases', 'count'),
       ('stroke_cases', 'sum'),
       ('stroke_rate', lambda x: (x.sum() / len(x)) * 100)
   ]).round(2)
   
   return risk_factors

def analyze_combined_risk_factors(df: pd.DataFrame) -> pd.DataFrame:
   """
   Analyze interaction between different risk factors
   """
   # בדיקת שילוב של יתר לחץ דם ומחלות לב
   combined_analysis = df.groupby(['hypertension', 'heart_disease'])['stroke'].agg([
       ('total_cases', 'count'),
       ('stroke_cases', 'sum'),
       ('stroke_rate', lambda x: (x.sum() / len(x)) * 100)
   ]).round(2)
   
   return combined_analysis

def analyze_glucose_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze glucose levels distribution and their relationship with stroke
    """
    # יצירת קבוצות של רמות סוכר
    df['glucose_category'] = pd.cut(df['avg_glucose_level'], 
                                  bins=[0, 80, 120, 160, 200, float('inf')],
                                  labels=['Very Low', 'Normal', 'Elevated', 'High', 'Very High'])
    
    glucose_analysis = df.groupby('glucose_category', observed=True)['stroke'].agg([
        ('total_cases', 'count'),
        ('stroke_cases', 'sum'),
        ('stroke_rate', lambda x: (x.sum() / len(x)) * 100)
    ]).round(2)
    
    return glucose_analysis

def analyze_bmi_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze BMI categories and their relationship with stroke
    """
    # יצירת קטגוריות BMI
    df['bmi_category'] = pd.cut(df['bmi'],
                               bins=[0, 18.5, 24.9, 29.9, float('inf')],
                               labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    bmi_analysis = df.groupby('bmi_category', observed=True)['stroke'].agg([
        ('total_cases', 'count'),
        ('stroke_cases', 'sum'),
        ('stroke_rate', lambda x: (x.sum() / len(x)) * 100)
    ]).round(2)
    
    return bmi_analysis

def calculate_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
   """
   Calculate descriptive statistics for numerical variables
   """
   numerical_cols = ['age', 'avg_glucose_level', 'bmi']
   stats = df[numerical_cols].describe()
   return stats

if __name__ == "__main__":
   # טען ונקה את הנתונים
   df = load_data('brain_stroke.csv')
   df_clean = handle_missing_values(df)
   
   # הסר ערכים חריגים מעמודות מספריות
   numeric_columns = ['age', 'avg_glucose_level', 'bmi']
   df_clean = remove_outliers(df_clean, numeric_columns)
   
   # צור קבוצות גיל
   df_clean = create_age_groups(df_clean)
   
   print("\n=== Basic Analysis ===")
   
   # ניתוח שבץ לפי קבוצות גיל
   print("\nStroke Analysis by Age Group:")
   age_analysis = analyze_stroke_by_age(df_clean)
   print(age_analysis)
   
   # ניתוח קורלציות
   print("\nCorrelations between Numerical Variables:")
   correlations = analyze_numerical_correlations(df_clean)
   print(correlations)
   
   # ניתוח גורמי סיכון
   print("\nRisk Factors Analysis:")
   risk_factors = analyze_risk_factors(df_clean)
   
   print("\nHypertension Analysis:")
   print(risk_factors['hypertension'])
   
   print("\nHeart Disease Analysis:")
   print(risk_factors['heart_disease'])
   
   print("\nSmoking Status Analysis:")
   print(risk_factors['smoking_status'])
   
   print("\n=== Additional Analysis ===")
   
   # סטטיסטיקה תיאורית
   print("\nDescriptive Statistics:")
   stats = calculate_descriptive_stats(df_clean)
   print(stats)
   
   # ניתוח משולב של גורמי סיכון
   print("\nCombined Risk Factors Analysis:")
   combined = analyze_combined_risk_factors(df_clean)
   print(combined)
   
   # ניתוח רמות סוכר
   print("\nGlucose Levels Analysis:")
   glucose = analyze_glucose_levels(df_clean)
   print(glucose)
   
   # ניתוח BMI
   print("\nBMI Analysis:")
   bmi = analyze_bmi_distribution(df_clean)
   print(bmi)

   import pandas as pd
import numpy as np
from data_cleaning import load_data, handle_missing_values, remove_outliers, create_age_groups

def analyze_stroke_by_age(df: pd.DataFrame) -> pd.DataFrame:
    stroke_by_age = df.groupby('age_group', observed=True)['stroke'].agg([
        ('total_patients', 'count'),
        ('stroke_cases', 'sum'),
        ('stroke_rate', lambda x: (x.sum() / len(x)) * 100)
    ]).round(2)
    return stroke_by_age

def analyze_numerical_correlations(df: pd.DataFrame) -> pd.DataFrame:
    numerical_cols = ['age', 'avg_glucose_level', 'bmi', 'stroke']
    correlations = df[numerical_cols].corr().round(3)
    return correlations

def analyze_risk_factors(df: pd.DataFrame) -> dict:
    risk_factors = {}
    risk_factors['hypertension'] = df.groupby('hypertension')['stroke'].agg([
        ('total_cases', 'count'),
        ('stroke_cases', 'sum'),
        ('stroke_rate', lambda x: (x.sum() / len(x)) * 100)
    ]).round(2)
    
    risk_factors['heart_disease'] = df.groupby('heart_disease')['stroke'].agg([
        ('total_cases', 'count'),
        ('stroke_cases', 'sum'),
        ('stroke_rate', lambda x: (x.sum() / len(x)) * 100)
    ]).round(2)
    
    risk_factors['smoking_status'] = df.groupby('smoking_status')['stroke'].agg([
        ('total_cases', 'count'),
        ('stroke_cases', 'sum'),
        ('stroke_rate', lambda x: (x.sum() / len(x)) * 100)
    ]).round(2)
    
    return risk_factors

def analyze_gender_differences(df: pd.DataFrame) -> dict:
    gender_analysis = {}
    
    gender_analysis['basic'] = df.groupby('gender')['stroke'].agg([
        ('total_cases', 'count'),
        ('stroke_cases', 'sum'),
        ('stroke_rate', lambda x: (x.sum() / len(x)) * 100)
    ]).round(2)
    
    gender_analysis['age'] = df.groupby(['gender', 'age_group'], observed=True)['stroke'].agg([
        ('total_cases', 'count'),
        ('stroke_cases', 'sum'),
        ('stroke_rate', lambda x: (x.sum() / len(x)) * 100)
    ]).round(2)
    
    gender_analysis['health'] = df.groupby(['gender', 'hypertension', 'heart_disease'])['stroke'].agg([
        ('total_cases', 'count'),
        ('stroke_cases', 'sum'),
        ('stroke_rate', lambda x: (x.sum() / len(x)) * 100)
    ]).round(2)
    
    return gender_analysis

def analyze_work_type_impact(df: pd.DataFrame) -> pd.DataFrame:
    work_analysis = df.groupby('work_type')['stroke'].agg([
        ('total_cases', 'count'),
        ('stroke_cases', 'sum'),
        ('stroke_rate', lambda x: (x.sum() / len(x)) * 100)
    ]).round(2)
    
    return work_analysis

def analyze_advanced_correlations(df: pd.DataFrame) -> dict:
    correlations = {}
    
    numerical_cols = ['age', 'avg_glucose_level', 'bmi', 'stroke']
    correlations['general'] = df[numerical_cols].corr().round(3)
    
    for gender in df['gender'].unique():
        gender_data = df[df['gender'] == gender]
        correlations[f'gender_{gender}'] = gender_data[numerical_cols].corr().round(3)
    
    return correlations

if __name__ == "__main__":
    # טען ונקה את הנתונים
    df = load_data('brain_stroke.csv')
    df_clean = handle_missing_values(df)
    
    # הסר ערכים חריגים מעמודות מספריות
    numeric_columns = ['age', 'avg_glucose_level', 'bmi']
    df_clean = remove_outliers(df_clean, numeric_columns)
    
    # צור קבוצות גיל
    df_clean = create_age_groups(df_clean)
    
    print("\n=== Basic Analysis ===")
    
    # ניתוח שבץ לפי קבוצות גיל
    print("\nStroke Analysis by Age Group:")
    age_analysis = analyze_stroke_by_age(df_clean)
    print(age_analysis)
    
    # ניתוח קורלציות
    print("\nCorrelations between Numerical Variables:")
    correlations = analyze_numerical_correlations(df_clean)
    print(correlations)
    
    # ניתוח גורמי סיכון
    print("\nRisk Factors Analysis:")
    risk_factors = analyze_risk_factors(df_clean)
    
    print("\nHypertension Analysis:")
    print(risk_factors['hypertension'])
    
    print("\nHeart Disease Analysis:")
    print(risk_factors['heart_disease'])
    
    print("\nSmoking Status Analysis:")
    print(risk_factors['smoking_status'])
    
    print("\n=== Additional Advanced Analysis ===")
    
    # ניתוח לפי מגדר
    print("\nGender Analysis:")
    gender_analysis = analyze_gender_differences(df_clean)
    print("\nBasic Gender Analysis:")
    print(gender_analysis['basic'])
    print("\nGender Analysis by Age:")
    print(gender_analysis['age'])
    
    # ניתוח לפי סוג עבודה
    print("\nWork Type Analysis:")
    work_analysis = analyze_work_type_impact(df_clean)
    print(work_analysis)
    
    # ניתוח קורלציות מתקדם
    print("\nAdvanced Correlations Analysis:")
    correlations_advanced = analyze_advanced_correlations(df_clean)
    print("\nGeneral Correlations:")
    print(correlations_advanced['general'])
    
    for gender in df_clean['gender'].unique():
        print(f"\nCorrelations for {gender}:")
        print(correlations_advanced[f'gender_{gender}'])