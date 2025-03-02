import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle
import os
from pathlib import Path

# Create necessary directories
Path("models").mkdir(exist_ok=True)

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Exploratory Data Analysis (EDA)
sns.heatmap(df.isnull(), cmap="viridis", cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

print(df.head())
print(df.info())
print(df.shape)
print(df.describe())
print("Missing Values:\n", df.isnull().sum())

# Preprocessing - Label Encoding
le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])
df["smoking_history"] = le.fit_transform(df["smoking_history"])
print(df.dtypes)

# Data Splitting
x = df.drop(["diabetes"], axis=1)
y = df["diabetes"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Standardizing Features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Model Training (using Logistic Regression)
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(x_train, y_train)

# Evaluate the Model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy * 100)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()










# Save the Model and Scaler in the models directory
with open("models/Logistic_regression_Model_Diabetes.pkl", 'wb') as file:
    pickle.dump(model, file)
    print("Model saved successfully!")

with open("models/scaler.pkl", 'wb') as file:
    pickle.dump(scaler, file)
    print("Scaler saved successfully!")

# Verify the Saved Model and Scaler
try:
    loaded_model = pickle.load(open("models/Logistic_regression_Model_Diabetes.pkl", 'rb'))
    loaded_scaler = pickle.load(open("models/scaler.pkl", 'rb'))
    print("Model and Scaler loaded successfully!")

    # Re-evaluate the loaded model
    result = loaded_model.score(x_test, y_test)
    print(f"Model Evaluation Score (Loaded Model): {result * 100:.2f}%")

    if result == accuracy:
        print("Loaded model's accuracy matches the original model.")
    else:
        print("Loaded model's accuracy does not match the original model.")
except Exception as e:
    print(f"An error occurred: {e}")

# Prediction Function
def predict_diabetes(model, scaler):
    try:
        print("\nEnter the following details to predict diabetes:")
        gender = int(input("Gender (0 for Female, 1 for Male): "))
        age = int(input("Age: "))
        hypertension = int(input("Hypertension (0 for No, 1 for Yes): "))
        heart_disease = int(input("Heart Disease (0 for No, 1 for Yes): "))
        smoking_history = int(input("Smoking History (0 for No, 1 for Yes): "))
        bmi = float(input("BMI (Body Mass Index): "))
        HbA1c_level = float(input("HbA1c Level: "))
        blood_glucose_level = float(input("Blood Glucose Level: "))

        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'smoking_history': [smoking_history],
            'bmi': [bmi],
            'HbA1c_level': [HbA1c_level],
            'blood_glucose_level': [blood_glucose_level]
        })

        # Scale the input data
        scaled_input = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        # Display results
        if prediction[0] == 1:
            print("\nPrediction: The individual is likely to have diabetes.")
        else:
            print("\nPrediction: The individual is unlikely to have diabetes.")
        print(f"Prediction Probability: {prediction_proba}")
    except Exception as e:
        print("Error during prediction:", e)

# Call the prediction function
predict_diabetes(model, scaler)





import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

def perform_comprehensive_analysis(df, model, X_train, X_test, y_train, y_test):
    """Perform comprehensive data analysis and visualization"""
    
    # Set the style for all plots - using updated syntax
    sns.set_theme(style="whitegrid")  # This replaces plt.style.use('seaborn')
    
    # 1. Distribution Analysis
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.select_dtypes(include=['float64', 'int64']).columns, 1):
        plt.subplot(3, 3, i)
        sns.histplot(data=df, x=column, hue='diabetes', bins=30, alpha=0.6)
        plt.title(f'Distribution of {column}')
    plt.tight_layout()
    plt.show()  # Add explicit show command

    # 2. Correlation Matrix
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

    # 3. Pair Plot for Key Features
    numeric_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
    pair_plot = sns.pairplot(df[numeric_cols], hue='diabetes', diag_kind='hist')
    plt.show()

    # 4. Box Plots
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.select_dtypes(include=['float64', 'int64']).columns[:-1], 1):
        plt.subplot(3, 3, i)
        sns.boxplot(data=df, x='diabetes', y=column)
        plt.title(f'Box Plot of {column} by Diabetes Status')
    plt.tight_layout()
    plt.show()

    # 5. ROC Curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # 6. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    average_precision = average_precision_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

    # 7. Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns if hasattr(X_train, 'columns') else [f'Feature {i}' for i in range(X_train.shape[1])],
        'importance': abs(model.coef_[0])
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.show()

    # 8. Age vs BMI with Diabetes Status
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['age'], df['bmi'], c=df['diabetes'], 
                         cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('Age')
    plt.ylabel('BMI')
    plt.title('Age vs BMI with Diabetes Status')
    plt.show()

    # 9. Violin Plots for Key Metrics
    plt.figure(figsize=(15, 5))
    key_metrics = ['blood_glucose_level', 'HbA1c_level', 'bmi']
    for i, metric in enumerate(key_metrics, 1):
        plt.subplot(1, 3, i)
        sns.violinplot(data=df, x='diabetes', y=metric)
        plt.title(f'{metric} Distribution by Diabetes Status')
    plt.tight_layout()
    plt.show()

    # Print Statistical Summary
    print("\nStatistical Summary:")
    print("\nNumerical Features Summary:")
    print(df.describe())
    
    print("\nClass Distribution:")
    print(df['diabetes'].value_counts(normalize=True))

    # Calculate and print additional metrics
    print("\nFeature Importance Ranking:")
    print(feature_importance)

    print("\nROC AUC Score:", roc_auc)
    print("Average Precision Score:", average_precision)

# Here's how to integrate it with your existing code:
if __name__ == "__main__":
    # Load and preprocess your data as before
    df = pd.read_csv("diabetes.csv")
    
    # Your existing preprocessing code
    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"])
    df["smoking_history"] = le.fit_transform(df["smoking_history"])
    
    # Your existing model training code
    x = df.drop(["diabetes"], axis=1)
    y = df["diabetes"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(x_train_scaled, y_train)
    
    # Now call the analysis function
    perform_comprehensive_analysis(df, model, x_train, x_test, y_train, y_test)