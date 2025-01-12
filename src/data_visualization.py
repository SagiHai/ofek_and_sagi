import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # בחירת backend לא אינטראקטיבי
import matplotlib.pyplot as plt
import seaborn as sns
from data_cleaning import load_data, handle_missing_values, remove_outliers, create_age_groups

def plot_age_distribution(df: pd.DataFrame) -> None:
    """
    Plot age distribution of stroke cases
    """
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")  # שינוי מהקוד הקודם
    sns.barplot(data=df, x='age_group', y='stroke', ci=None)
    plt.title('Stroke Rate by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Stroke Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('stroke_by_age.png')
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Plot correlation heatmap for numerical variables
    """
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")  # שינוי מהקוד הקודם
    numerical_cols = ['age', 'avg_glucose_level', 'bmi', 'stroke']
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

def plot_risk_factors(df: pd.DataFrame) -> None:
    """
    Plot risk factors analysis
    """
    sns.set_theme(style="whitegrid")  # שינוי מהקוד הקודם
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Hypertension
    sns.barplot(data=df, x='hypertension', y='stroke', ax=axes[0])
    axes[0].set_title('Stroke Rate by Hypertension')
    axes[0].set_xlabel('Has Hypertension')
    
    # Heart Disease
    sns.barplot(data=df, x='heart_disease', y='stroke', ax=axes[1])
    axes[1].set_title('Stroke Rate by Heart Disease')
    axes[1].set_xlabel('Has Heart Disease')
    
    # Smoking Status
    sns.barplot(data=df, x='smoking_status', y='stroke', ax=axes[2])
    axes[2].set_title('Stroke Rate by Smoking Status')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('risk_factors.png')
    plt.close()

def plot_glucose_bmi_analysis(df: pd.DataFrame) -> None:
    """
    Plot glucose levels and BMI analysis
    """
    sns.set_theme(style="whitegrid")  # שינוי מהקוד הקודם
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Glucose Levels
    sns.boxplot(data=df, x='stroke', y='avg_glucose_level', ax=ax1)
    ax1.set_title('Glucose Levels Distribution by Stroke')
    ax1.set_xlabel('Had Stroke')
    
    # BMI
    sns.boxplot(data=df, x='stroke', y='bmi', ax=ax2)
    ax2.set_title('BMI Distribution by Stroke')
    ax2.set_xlabel('Had Stroke')
    
    plt.tight_layout()
    plt.savefig('glucose_bmi_analysis.png')
    plt.close()

def plot_gender_analysis(df: pd.DataFrame) -> None:
    """
    Plot gender-based analysis
    """
    sns.set_theme(style="whitegrid")  # שינוי מהקוד הקודם
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='gender', y='stroke', hue='age_group')
    plt.title('Stroke Rate by Gender and Age Group')
    plt.xlabel('Gender')
    plt.ylabel('Stroke Rate')
    plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('gender_analysis.png')
    plt.close()

def create_all_visualizations(df: pd.DataFrame) -> None:
    """
    Create all visualizations
    """
    print("Creating visualizations...")
    
    # Create visualizations
    plot_age_distribution(df)
    plot_correlation_heatmap(df)
    plot_risk_factors(df)
    plot_glucose_bmi_analysis(df)
    plot_gender_analysis(df)
    
    print("All visualizations have been created and saved!")

if __name__ == "__main__":
    # Load and clean data
    df = load_data('brain_stroke.csv')
    df_clean = handle_missing_values(df)
    
    # Remove outliers
    numeric_columns = ['age', 'avg_glucose_level', 'bmi']
    df_clean = remove_outliers(df_clean, numeric_columns)
    
    # Create age groups
    df_clean = create_age_groups(df_clean)
    
    # Create visualizations
    create_all_visualizations(df_clean)

    def plot_advanced_risk_analysis(df: pd.DataFrame) -> None:
     """
    יוצר תרשים מתקדם של גורמי סיכון משולבים
     """
    plt.figure(figsize=(15, 10))
    
    # יוצר רשת של תרשימים
    g = sns.FacetGrid(df, col='heart_disease', row='hypertension', hue='smoking_status', height=4)
    g.map(sns.scatterplot, 'age', 'avg_glucose_level', alpha=0.6)
    
    plt.suptitle('Risk Factors Interaction Analysis', y=1.02, size=16)
    plt.tight_layout()
    plt.savefig('advanced_risk_analysis.png')
    plt.close()

def plot_age_glucose_distribution(df: pd.DataFrame) -> None:
    """
    יוצר תרשים צפיפות דו-ממדי של גיל ורמת סוכר
    """
    plt.figure(figsize=(12, 8))
    
    # יוצר תרשים צפיפות דו-ממדי
    sns.kdeplot(data=df, x='age', y='avg_glucose_level', hue='stroke', 
                fill=True, thresh=.2, alpha=.6)
    
    plt.title('Age and Glucose Level Distribution by Stroke')
    plt.tight_layout()
    plt.savefig('age_glucose_density.png')
    plt.close()

def plot_radar_chart(df: pd.DataFrame) -> None:
    """
    יוצר תרשים רדאר להשוואת מאפיינים בין מקרי שבץ ולא שבץ
    """
    # מחשב ממוצעים עבור כל קבוצה
    stroke_means = df[df['stroke'] == 1][['age', 'avg_glucose_level', 'bmi']].mean()
    no_stroke_means = df[df['stroke'] == 0][['age', 'avg_glucose_level', 'bmi']].mean()
    
    # נרמול הנתונים
    max_values = df[['age', 'avg_glucose_level', 'bmi']].max()
    stroke_means_norm = stroke_means / max_values
    no_stroke_means_norm = no_stroke_means / max_values
    
    # הגדרת הפרמטרים לתרשים רדאר
    categories = ['Age', 'Glucose', 'BMI']
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    
    # יצירת התרשים
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, stroke_means_norm, 'o-', linewidth=2, label='Stroke')
    ax.fill(angles, stroke_means_norm, alpha=0.25)
    ax.plot(angles, no_stroke_means_norm, 'o-', linewidth=2, label='No Stroke')
    ax.fill(angles, no_stroke_means_norm, alpha=0.25)
    
    ax.set_xticks(angles)
    ax.set_xticklabels(categories)
    ax.set_title('Radar Chart of Key Metrics')
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('radar_chart.png')
    plt.close()

def plot_parallel_coordinates(df: pd.DataFrame) -> None:
    """
    יוצר תרשים קואורדינטות מקבילות להשוואת מאפיינים
    """
    plt.figure(figsize=(15, 8))
    
    # נרמול הנתונים
    df_norm = df.copy()
    for column in ['age', 'avg_glucose_level', 'bmi']:
        df_norm[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    
    # יצירת התרשים
    pd.plotting.parallel_coordinates(
        df_norm[['age', 'avg_glucose_level', 'bmi', 'stroke']].sample(n=1000, random_state=42),
        'stroke',
        colormap=plt.cm.viridis
    )
    
    plt.title('Parallel Coordinates Plot of Key Metrics')
    plt.tight_layout()
    plt.savefig('parallel_coordinates.png')
    plt.close()

# עדכון הפונקציה הראשית
def create_all_visualizations(df: pd.DataFrame) -> None:
    """
    Create all visualizations
    """
    print("Creating visualizations...")
    
    # Visualizations מהקוד הקודם
    plot_age_distribution(df)
    plot_correlation_heatmap(df)
    plot_risk_factors(df)
    plot_glucose_bmi_analysis(df)
    plot_gender_analysis(df)
    
    # Visualizations חדשות
    plot_advanced_risk_analysis(df)
    plot_age_glucose_distribution(df)
    plot_radar_chart(df)
    plot_parallel_coordinates(df)
    
    print("All visualizations have been created and saved!")