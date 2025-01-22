from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
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


xy=prepare_data_for_model(df)
x, y = xy[0],xy[1]
# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=42)

# Data balancing with SMOTE
smote = SMOTE(random_state=42)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

# Train the model with weighted classes 
rf = RandomForestClassifier(class_weight='balanced',random_state=42)
rf.fit(x_train_balanced, y_train_balanced)

# Train the model
y_pred = rf.predict(x_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)

# Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=np.unique(y))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Weighted Model and SMOTE")
plt.show()

# Classification Report
print(metrics.classification_report(y_test, y_pred))