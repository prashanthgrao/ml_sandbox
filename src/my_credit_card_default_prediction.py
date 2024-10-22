# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

# Set plot style
plt.style.use('ggplot')

def load_data(file_path):
    """Load and preprocess the dataset."""
    data = pd.read_excel(file_path)
    data = data.drop(labels=0, axis=0)  # Drop the first row if it's not needed
    return data

def exploratory_data_analysis(data):
    """Perform exploratory data analysis."""
    print("STATISTICS OF NUMERIC COLUMNS")
    print(data.describe().T)
    
    # Count unique values for categorical variables
    print(data.X2.value_counts())  # Gender counts
    print(data.X3.value_counts())  # Education categories
    print(data.X4.value_counts())  # Marriage categories

def visualize_data(data):
    """Visualize the data distributions."""
    plt.figure(figsize=(8, 6))
    data['Y'].value_counts().plot(kind='bar')
    plt.title('Frequency of Defaults')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xticks([0, 1], ['Defaults', 'Not Defaults'])
    plt.show()

def preprocess_data(df):
    """Preprocess the dataset for modeling."""
    df = df.copy()
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['X3', 'X4'], drop_first=True)
    
    # Scale features
    y = df['Y'].astype(int).copy()
    X = df.drop('Y', axis=1).copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return pd.DataFrame(X_scaled, columns=X.columns), y

def train_models(X_train, y_train):
    """Train various models and return predictions."""
    models = {
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(kernel='rbf', gamma='scale', random_state=42),
        'Perceptron': Perceptron(),
        'Random Forest': RandomForestClassifier()
    }
    
    predictions = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        
    return predictions

def evaluate_models(y_test, predictions):
    """Evaluate model performance using various metrics."""
    metrics = {
        'Accuracy': accuracy_score,
        'Precision': precision_score,
        'Recall': recall_score,
        'F1 Score': f1_score
    }
    
    for name, preds in predictions.items():
        print(f"Results for {name}:")
        for metric_name, metric_func in metrics.items():
            score = metric_func(y_test, preds)
            print(f"  {metric_name}: {score:.4f}")
        print()

if __name__ == "__main__":
    # Load data
    file_path = "default of credit card clients.xls"
    data = load_data(file_path)

    # Exploratory Data Analysis
    exploratory_data_analysis(data)

    # Visualize Data
    visualize_data(data)

    # Preprocess Data
    X, y = preprocess_data(data)

    # Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)

    # Train Models
    predictions = train_models(X_train, y_train)

    # Evaluate Models
    evaluate_models(y_test, predictions)
