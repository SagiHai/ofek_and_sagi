import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # זה חייב להיות לפני import plt
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_cleaning import load_data, handle_missing_values, remove_outliers

def prepare_data_for_model(df: pd.DataFrame) -> tuple:
    """
    Prepare data for the decision tree model
    """
    # Create copy of the dataframe
    df_model = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    
    for col in categorical_cols:
        df_model[col] = le.fit_transform(df_model[col])
    
    # Separate features and target
    X = df_model.drop('stroke', axis=1)
    y = df_model['stroke']
    
    return X, y

def create_and_train_tree(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Create and train the decision tree model
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    tree = DecisionTreeClassifier(random_state=42, max_depth=5)
    tree.fit(X_train, y_train)
    
    # Calculate accuracy
    train_accuracy = tree.score(X_train, y_train)
    test_accuracy = tree.score(X_test, y_test)
    
    return tree, X_train, X_test, y_train, y_test, train_accuracy, test_accuracy

def plot_tree_visualization(tree: DecisionTreeClassifier, feature_names: list):
    """
    Plot decision tree visualization
    """
    plt.figure(figsize=(20,10))
    plot_tree(tree, feature_names=feature_names, class_names=['No Stroke', 'Stroke'], 
             filled=True, rounded=True)
    plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(tree: DecisionTreeClassifier, feature_names: list):
    """
    Plot feature importance
    """
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': tree.feature_importances_
    })
    importances = importances.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(importances['feature'], importances['importance'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Importance in Stroke Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

if __name__ == "__main__":
    # Load and clean data
    df = load_data('brain_stroke.csv')
    df_clean = handle_missing_values(df)
    
    # Prepare data
    X, y = prepare_data_for_model(df_clean)
    
    # Create and train model
    tree, X_train, X_test, y_train, y_test, train_acc, test_acc = create_and_train_tree(X, y)
    
    print("\n=== Model Performance ===")
    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Testing Accuracy: {test_acc:.3f}")
    
    # Create visualizations
    plot_tree_visualization(tree, X.columns.tolist())
    plot_feature_importance(tree, X.columns.tolist())
    
    print("\nVisualizations have been saved!")