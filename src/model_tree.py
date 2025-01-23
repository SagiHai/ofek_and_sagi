import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def prepare_data_for_model(df):
    """
    Prepare data for the decision tree model
    """
    # Create copy of the dataframe
    df_model = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    non_numeric_df = df.select_dtypes(exclude=[float, int, 'number'])
    
    for col in non_numeric_df.columns:
        df_model[col] = le.fit_transform(df_model[col])
    
    # Separate features and target
    x = df_model.drop('stroke', axis=1)
    y = df_model['stroke']
    
    return x, y

def train_and_evaluate_model(x, y, output_dir="output"):
    """
    Train and evaluate a Random Forest model with SMOTE and weighted classes.
    
    Parameters:
    x (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    output_dir (str): Directory to save the output files (PNG and CSV).
    
    Returns:
    None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Data balancing with SMOTE
    smote = SMOTE(random_state=42)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

    # Train the model with weighted classes
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf.fit(x_train_balanced, y_train_balanced)

    # Predict
    y_pred = rf.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion Matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=np.unique(y))

    # Plot and save Confusion Matrix with accuracy
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix\nAccuracy: {accuracy:.2f}")
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()

    # Generate Classification Report and save to CSV
    classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
    classification_report_df = pd.DataFrame(classification_report_dict).transpose()
    classification_report_path = os.path.join(output_dir, "classification_report.csv")
    classification_report_df.to_csv(classification_report_path, index=True)