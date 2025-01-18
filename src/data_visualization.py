import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # בחירת backend לא אינטראקטיבי
import matplotlib.pyplot as plt
import seaborn as sns
from data_analysis import analyze_numerical_column

def plot_categorical_analysis(df, column_name):
    """
    Plots stroke rates for a categorical column.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns [category_name, no_stroke_percentage, stroke_percentage]
    column_name (str): Name of the categorical column
    """
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df[column_name]))
    width = 0.6
    
    # Create stacked bars
    bars1 = ax.bar(x, df['no_stroke_percentage'],
                  width,
                  label='No Stroke',
                  color='skyblue')
    
    bars2 = ax.bar(x, df['stroke_percentage'],
                  width,
                  bottom=df['no_stroke_percentage'],
                  label='Stroke',
                  color='salmon')
    
    # Add value labels on the bars
    for bars in [bars1, bars2]:
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:  # Only show non-zero values
                ax.text(bar.get_x() + bar.get_width()/2,
                       bar.get_y() + height/2,
                       f'{height:.1f}%',
                       ha='center',
                       va='center')
    
    # Customize the plot
    ax.set_ylabel('Percentage')
    ax.set_title(f'Stroke Distribution by {column_name.capitalize()}')
    ax.set_xticks(x)
    ax.set_xticklabels(df[column_name])
    ax.legend(title="Stroke Status")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'stroke_distribution_by_{column_name}.png', 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

def plot_numerical_analysis(group_stats, column_name):
   """
   Creates visualizations for numerical variables based on descriptive statistics by stroke groups.
   
   Parameters:
   group_stats (pd.DataFrame): DataFrame containing descriptive statistics by stroke group
   column_name (str): Name of the column being analyzed
   """
   # Set style
   sns.set_theme(style="whitegrid")
   
   # Create figure with two subplots
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
   
   # 1. Box Plot with additional statistics
   data_to_plot = {
       'No Stroke': [group_stats.loc[0, '25%'], group_stats.loc[0, '50%'], 
                    group_stats.loc[0, '75%'], group_stats.loc[0, 'min'], 
                    group_stats.loc[0, 'max']],
       'Stroke': [group_stats.loc[1, '25%'], group_stats.loc[1, '50%'], 
                 group_stats.loc[1, '75%'], group_stats.loc[1, 'min'], 
                 group_stats.loc[1, 'max']]
   }
   
   # Create box plot
   positions = [1, 2]
   labels = ['No Stroke', 'Stroke']
   bplot = ax1.boxplot([data_to_plot[label] for label in labels],
                      positions=positions,
                      labels=labels,
                      patch_artist=True,
                      medianprops=dict(color='red', linewidth=1.5))  # שינוי צבע וגודל קו החציון
   
   # Color the boxes
   for patch in bplot['boxes']:
       if patch == bplot['boxes'][0]:  # No Stroke
           patch.set_facecolor('skyblue')
       else:  # Stroke
           patch.set_facecolor('salmon')
   
   ax1.set_title(f'Box Plot of {column_name}')
   ax1.set_ylabel(f'{column_name} values')  # עדכון תווית ציר Y
   
   # 2. Bar plot for means with error bars (std)
   x = np.arange(2)
   means = [group_stats.loc[0, 'mean'], group_stats.loc[1, 'mean']]
   stds = [group_stats.loc[0, 'std'], group_stats.loc[1, 'std']]
   
   bars = ax2.bar(x, means, yerr=stds, 
                 color=['skyblue', 'salmon'],
                 capsize=5)
   
   # Add value labels on the bars
   for bar in bars:
       height = bar.get_height()
       ax2.text(bar.get_x() + bar.get_width()/2, height,
               f'{height:.2f}',
               ha='center', va='bottom')
   
   ax2.set_title(f'Mean {column_name} by Stroke')  # עדכון כותרת
   ax2.set_xticks(x)
   ax2.set_xticklabels(['No Stroke', 'Stroke'])
   ax2.set_ylabel(f'Mean {column_name} values')  # עדכון תווית ציר Y
   
   # Add overall title
   plt.suptitle(f'Analysis of {column_name}', fontsize=14, y=1.05)
   
   # Adjust layout and save
   plt.tight_layout()
   plt.savefig(f'numerical_analysis_{column_name}.png', 
               bbox_inches='tight',
               dpi=300)
   plt.close()

'''data = {
    'age': [2,15,35,50,60],
    'stroke': [0,0,1,0,1],
}
df = pd.DataFrame(data)
plot_numerical_analysis(analyze_numerical_column('age',df)[3], 'age')
'''
