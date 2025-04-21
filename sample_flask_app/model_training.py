import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

def load_and_prepare_data(file_path='equipment_anomaly_data.csv'):
    """Load and prepare the dataset for training"""
    try:
        # Load the actual CSV file
        df = pd.read_csv(file_path)
        
        # Convert faulty column to integer if it's in float format
        df['faulty'] = df['faulty'].astype(int)
        
        print(f"Data loaded successfully: {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def preprocess_data(df):
    """Preprocess the data by handling nulls and preparing for model training"""
    # Handle missing values
    for col in ['temperature', 'pressure', 'vibration', 'humidity']:
        df[col] = df[col].fillna(df[col].median())
    
    # Split features and target
    X = df.drop('faulty', axis=1)
    y = df['faulty']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define preprocessing for numerical and categorical features
    numerical_features = ['temperature', 'pressure', 'vibration', 'humidity']
    categorical_features = ['equipment', 'location']
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_model(X_train, y_train, preprocessor):
    """Train a Random Forest model on the preprocessed data"""
    # Create and train the pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }

def generate_visualization_data(df, model, X_test, y_test):
    """Generate data for visualizations"""
    # 1. Feature importance
    rf_model = model.named_steps['classifier']
    feature_names = list(df.drop('faulty', axis=1).columns)
    
    # Get feature names after preprocessing (including one-hot encoding)
    cat_cols = ['equipment', 'location']
    preprocessor = model.named_steps['preprocessor']
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_cats = cat_encoder.get_feature_names_out(cat_cols).tolist()
    
    # Replace original categorical features with encoded ones
    for cat in cat_cols:
        feature_names.remove(cat)
    feature_names.extend(encoded_cats)
    
    # Get feature importances
    importances = rf_model.feature_importances_
    
    # 2. Correlation matrix data (only for numerical features)
    numerical_features = ['temperature', 'pressure', 'vibration', 'humidity', 'faulty']
    corr_matrix = df[numerical_features].corr().round(2).to_dict()
    
    # 3. Fault distribution by equipment and location
    fault_by_equipment = df.groupby('equipment')['faulty'].mean().to_dict()
    fault_by_location = df.groupby('location')['faulty'].mean().to_dict()
    
    # Additional visualization: Feature value distributions by fault status
    feature_dist_by_fault = {}
    for feature in ['temperature', 'pressure', 'vibration', 'humidity']:
        feature_dist_by_fault[feature] = {
            'normal': df[df['faulty'] == 0][feature].describe().to_dict(),
            'faulty': df[df['faulty'] == 1][feature].describe().to_dict()
        }
    
    visualization_data = {
        'feature_importance': {
            'features': feature_names[:len(importances)],  # Ensure the lengths match
            'importance': importances.tolist()
        },
        'correlation_matrix': corr_matrix,
        'fault_by_equipment': fault_by_equipment,
        'fault_by_location': fault_by_location,
        'feature_distributions': feature_dist_by_fault
    }
    
    return visualization_data

def save_model(model, metrics, visualization_data):
    """Save the trained model and related data"""
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(model, 'models/rf_model.pkl')
    
    # Save other data for API use
    np.save('models/metrics.npy', metrics)
    np.save('models/visualization_data.npy', visualization_data)
    
    print("Model and related data saved successfully.")

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    df = load_and_prepare_data('equipment_anomaly_data.csv')
    
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Train model
    print("Training Random Forest model...")
    model = train_model(X_train, y_train, preprocessor)
    
    # Evaluate model
    print("Evaluating model performance...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Generate visualization data
    print("Generating visualization data...")
    visualization_data = generate_visualization_data(df, model, X_test, y_test)
    
    # Save model and data
    print("Saving model and related data...")
    save_model(model, metrics, visualization_data)
    
    return model, metrics, visualization_data

if __name__ == "__main__":
    main()