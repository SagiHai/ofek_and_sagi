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