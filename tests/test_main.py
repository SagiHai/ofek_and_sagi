import pandas as pd
# Creating the test dataset
test_data = pd.DataFrame({
    "gender": ["Female", "Male", "Female", "Unknown", "Male", "Female", "Female"],
    "age": [78, 82, 25, 120, 67, 55, None],  # Including outlier (120) and missing value
    "hypertension": [0, 1, 0, 0, 0, 1, 1],
    "heart_disease": [0, 1, 0, 1, 0, 0, None],  # Including missing value
    "ever_married": ["Yes", "Yes", "No", "Yes", "Yes", "No", "Unknown"],  # Including "Unknown"
    "work_type": ["Private", "Self-employed", "children", "Govt_job", "Private", "Private", "Unknown"],
    "Residence_type": ["Urban", "Rural", "Urban", "Rural", "Urban", "Urban", "Rural"],
    "avg_glucose_level": [90.5, 200.3, 75.0, 305.0, None, 140.2, 85.4],  # Including outlier and missing value
    "bmi": [24.5, 36.0, 18.4, 45.3, 28.7, None, 22.1],  # Including missing value
    "smoking_status": ["never smoked", "formerly smoked", "smokes", "Unknown", "smokes", "never smoked", "Unknown"],
    "stroke": [1, 1, 0, 1, 0, 0, 0],
})

print(test_data)  # Displaying the first rows of the dataset
