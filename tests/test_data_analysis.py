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