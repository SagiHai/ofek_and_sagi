import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # בחירת backend לא אינטראקטיבי
import matplotlib.pyplot as plt
import seaborn as sns
from data_analysis import analyze_numerical_column

def plot_stroke_proportion(data, output_path="stroke_proportion_pie.png"):
    """
    Visualize the proportion of stroke and non-stroke cases using a pie chart.

    Parameters:
        data (pd.DataFrame): A DataFrame containing 'stroke' and 'proportion' columns.
        output_path (str): Path to save the pie chart plot.
    """
    # Validate input
    if 'stroke' not in data.columns or 'proportion' not in data.columns:
        raise ValueError("Input DataFrame must contain 'stroke' and 'proportion' columns.")
    
    # Prepare data
    labels = data['stroke'].replace({0: 'No Stroke', 1: 'Stroke'}).tolist()
    proportions = data['proportion']
    colors = sns.color_palette("pastel")[:len(labels)]
    
    # Plot pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(proportions, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title("Proportion of Stroke Cases")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()

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

def plot_variable_importance(variable_importance, output_path="variable_importance_heatmap.png"):
    """
    Visualizes the variable importance using a heatmap.

    Parameters:
        variable_importance (dict): A dictionary where keys are variable names, and values are their importance scores.
        output_path (str): Path to save the heatmap plot.
    """
    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(variable_importance, orient='index', columns=['Importance'])
    df.sort_values(by='Importance', ascending=False, inplace=True)
    
    # Set the theme
    sns.set_theme(style="whitegrid")
    
    # Create a heatmap
    plt.figure(figsize=(8, len(df) // 2))
    sns.heatmap(
        df,
        annot=True,
        cmap="YlGnBu",
        cbar_kws={'label': 'Importance Score'},
        fmt=".2f"
    )
    
    # Customize the plot
    plt.title("Variable Importance Heatmap")
    plt.ylabel("Variable")
    #plt.xlabel("Importance")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()

def plot_modifiable_vs_nonmodifiable_combined(modifiable_corr, non_modifiable_corr, ratio, output_path="modifiable_vs_nonmodifiable.png"):
    """
    Visualizes the correlation summary for modifiable vs. non-modifiable variables.
    Combines a bar chart (for correlations) and a text annotation (for ratio).

    Parameters:
        modifiable_corr (float): Cumulative correlation of modifiable variables.
        non_modifiable_corr (float): Cumulative correlation of non-modifiable variables.
        ratio (float): Ratio of modifiable to non-modifiable variables.
        output_path (str): Path to save the combined plot.
    """
    # Data preparation
    data = {
        "Variable Type": ["Modifiable", "Non-Modifiable"],
        "Cumulative Correlation": [modifiable_corr, non_modifiable_corr]
    }
    df = pd.DataFrame(data)
    
    # Set the theme
    sns.set_theme(style="whitegrid")
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Bar chart
    sns.barplot(
        x="Variable Type",
        y="Cumulative Correlation",
        data=df,
        palette="Set2",
        ax=ax
    )
    
    # Add value annotations
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", padding=3)
    
    # Add ratio annotation
    ratio_text = f"Ratio (Modifiable / Non-Modifiable): {ratio:.2f}"
    ax.text(
        0.5, 
        0.9, 
        ratio_text, 
        horizontalalignment='center', 
        verticalalignment='center', 
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray')
    )
    
    # Customize the plot
    ax.set_title("Modifiable vs. Non-Modifiable Variables - Correlation")
    ax.set_ylabel("Cumulative Correlation")
    ax.set_xlabel("Variable Type")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()

def visualize_stroke_data_by_gender(analysis_results, output_path = 'visualize_stroke_data_by_gender.png'):
    """
    Visualize stroke data by gender for age, smoking status, and BMI categories.

    Parameters:
    analysis_results (dict): Dictionary containing DataFrames with analysis results.
                              Expected keys: 'age', 'smoking', 'bmi'.
    output_path (str): Path to save the plot.
    """
    # Set up the figure and subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    plt.subplots_adjust(hspace=0.4)
    categories = ['age', 'smoking', 'bmi']

    for i, category in enumerate(categories):
        # Get DataFrame for the current category
        df_category = analysis_results[category]
        df_reset = df_category.reset_index()

        # Stroke cases bar chart
        ax1 = axes[i, 0]
        sns.barplot(
            x="age_group" if category == "age" else 
            "smoking_status" if category == "smoking" else "bmi_category",
            y="stroke_cases",
            hue="gender",
            data=df_reset,
            ax=ax1,
            palette="coolwarm"
        )
        ax1.set_title(f"Stroke Cases by Gender - {category.capitalize()}")
        ax1.set_xlabel(f"{category.capitalize()} Categories")
        ax1.set_ylabel("Stroke Cases")
        ax1.legend(title="Gender")

        # Stroke rate line chart
        ax2 = axes[i, 1]
        sns.lineplot(
            x="age_group" if category == "age" else 
            "smoking_status" if category == "smoking" else "bmi_category",
            y="stroke_rate",
            hue="gender",
            data=df_reset,
            ax=ax2,
            marker="o",
            palette="coolwarm"
        )
        ax2.set_title(f"Stroke Rate by Gender - {category.capitalize()}")
        ax2.set_xlabel(f"{category.capitalize()} Categories")
        ax2.set_ylabel("Stroke Rate (%)")
        ax2.legend(title="Gender")

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path)
    plt.close()