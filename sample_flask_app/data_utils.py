import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_csv_data(file_path='equipment_anomaly_data.csv'):
    """
    Load data from a CSV file
    """
    try:
        df = pd.read_csv(file_path)
        # Convert faulty to integer if it's float
        if 'faulty' in df.columns:
            df['faulty'] = df['faulty'].astype(int)
        return df, None
    except Exception as e:
        return None, str(e)

def validate_data(df):
    """
    Validate the data structure and contents
    """
    required_columns = ['temperature', 'pressure', 'vibration', 
                        'humidity', 'equipment', 'location', 'faulty']
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Validate data types
    numerical_cols = ['temperature', 'pressure', 'vibration', 'humidity']
    for col in numerical_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column '{col}' must contain numerical values"
    
    categorical_cols = ['equipment', 'location']
    for col in categorical_cols:
        if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
            return False, f"Column '{col}' must contain categorical values"
    
    # Check if 'faulty' contains only binary values
    if not set(df['faulty'].unique()).issubset({0, 1}):
        return False, "Column 'faulty' must contain only binary values (0 or 1)"
    
    return True, "Data validation passed"

def preprocess_data_for_prediction(df):
    """
    Preprocess incoming data for prediction
    """
    # Handle missing values
    for col in ['temperature', 'pressure', 'vibration', 'humidity']:
        if col in df.columns:
            # Use a reasonable default if value is missing
            df[col] = df[col].fillna(df[col].median() if not df[col].empty else 0)
    
    # Ensure categorical columns are strings
    categorical_cols = ['equipment', 'location']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df

def create_data_preprocessor():
    """
    Create a data preprocessor pipeline
    """
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
    
    return preprocessor

def get_data_summary(file_path='equipment_anomaly_data.csv'):
    """
    Get a summary of the dataset
    """
    try:
        df, error = load_csv_data(file_path)
        if error:
            return {
                'error': error
            }
        
        # Basic statistics
        total_records = len(df)
        fault_count = df['faulty'].sum()
        normal_count = total_records - fault_count
        fault_percentage = (fault_count / total_records) * 100
        
        # Equipment breakdown
        equipment_counts = df['equipment'].value_counts().to_dict()
        location_counts = df['location'].value_counts().to_dict()
        
        # Numerical column statistics
        numerical_stats = {}
        for col in ['temperature', 'pressure', 'vibration', 'humidity']:
            numerical_stats[col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median()
            }
        
        return {
            'total_records': total_records,
            'fault_count': int(fault_count),
            'normal_count': int(normal_count),
            'fault_percentage': fault_percentage,
            'equipment_counts': equipment_counts,
            'location_counts': location_counts,
            'numerical_stats': numerical_stats
        }
        
    except Exception as e:
        return {
            'error': str(e)
        }

def get_sample_for_prediction():
    """
    Get a random sample from the actual dataset for prediction testing
    """
    try:
        df, error = load_csv_data()
        if error or df is None:
            raise Exception(error)
        
        # Take a random sample
        sample = df.sample(1).drop('faulty', axis=1).iloc[0].to_dict()
        return sample
        
    except Exception as e:
        # If we can't load the actual data, return a reasonable sample
        return {
            'temperature': 85.5,
            'pressure': 42.3,
            'vibration': 2.1,
            'humidity': 65.7,
            'equipment': 'Turbine',
            'location': 'Chicago'
        }