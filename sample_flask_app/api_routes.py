from flask import Blueprint, jsonify, request
import joblib
import numpy as np
import pandas as pd
import os
from data_utils import preprocess_data_for_prediction, get_data_summary, get_sample_for_prediction

# Create API blueprint
api = Blueprint('api', __name__)

# Load model and visualization data
def load_model_data():
    model_path = 'models/rf_model.pkl'
    metrics_path = 'models/metrics.npy'
    viz_data_path = 'models/visualization_data.npy'
    
    if not (os.path.exists(model_path) and os.path.exists(metrics_path) and os.path.exists(viz_data_path)):
        return None, None, None
    
    model = joblib.load(model_path)
    metrics = np.load(metrics_path, allow_pickle=True).item()
    visualization_data = np.load(viz_data_path, allow_pickle=True).item()
    
    return model, metrics, visualization_data

# Route for model metrics
@api.route('/metrics', methods=['GET'])
def get_metrics():
    _, metrics, _ = load_model_data()
    
    if metrics is None:
        return jsonify({
            'error': 'Model metrics not found. Please train the model first.'
        }), 404
    
    return jsonify({
        'metrics': metrics
    })

# Route for visualization data
@api.route('/visualization-data', methods=['GET'])
def get_visualization_data():
    _, _, visualization_data = load_model_data()
    
    if visualization_data is None:
        return jsonify({
            'error': 'Visualization data not found. Please train the model first.'
        }), 404
    
    return jsonify(visualization_data)

# Route for feature importance
@api.route('/feature-importance', methods=['GET'])
def get_feature_importance():
    _, _, visualization_data = load_model_data()
    
    if visualization_data is None:
        return jsonify({
            'error': 'Visualization data not found. Please train the model first.'
        }), 404
    
    return jsonify(visualization_data['feature_importance'])

# Route for correlation matrix
@api.route('/correlation-matrix', methods=['GET'])
def get_correlation_matrix():
    _, _, visualization_data = load_model_data()
    
    if visualization_data is None:
        return jsonify({
            'error': 'Visualization data not found. Please train the model first.'
        }), 404
    
    return jsonify(visualization_data['correlation_matrix'])

# Route for fault distribution
@api.route('/fault-distribution', methods=['GET'])
def get_fault_distribution():
    _, _, visualization_data = load_model_data()
    
    if visualization_data is None:
        return jsonify({
            'error': 'Visualization data not found. Please train the model first.'
        }), 404
    
    return jsonify({
        'by_equipment': visualization_data['fault_by_equipment'],
        'by_location': visualization_data['fault_by_location']
    })

# Route for feature distributions
@api.route('/feature-distributions', methods=['GET'])
def get_feature_distributions():
    _, _, visualization_data = load_model_data()
    
    if visualization_data is None or 'feature_distributions' not in visualization_data:
        return jsonify({
            'error': 'Feature distribution data not found. Please train the model first.'
        }), 404
    
    return jsonify(visualization_data['feature_distributions'])

# Route for prediction
@api.route('/predict', methods=['POST'])
def predict():
    model, _, _ = load_model_data()
    
    if model is None:
        return jsonify({
            'error': 'Model not found. Please train the model first.'
        }), 404
    
    # Get data from request
    data = request.get_json()
    
    if not data:
        return jsonify({
            'error': 'No data provided for prediction'
        }), 400
    
    try:
        # Create DataFrame from input data
        input_data = pd.DataFrame([data])
        
        # Preprocess the input data
        input_data = preprocess_data_for_prediction(input_data)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0].tolist()
        
        return jsonify({
            'prediction': int(prediction),
            'prediction_label': 'Faulty' if prediction == 1 else 'Normal',
            'probability': {
                'normal': prediction_proba[0],
                'faulty': prediction_proba[1]
            },
            'input_data': data
        })
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

# Route for getting a sample for prediction
@api.route('/sample', methods=['GET'])
def get_sample():
    sample = get_sample_for_prediction()
    return jsonify(sample)

# Route for dataset summary
@api.route('/dataset-summary', methods=['GET'])
def dataset_summary():
    summary = get_data_summary()
    return jsonify(summary)

# Route for health check
@api.route('/health', methods=['GET'])
def health_check():
    model, metrics, visualization_data = load_model_data()
    
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'metrics_available': metrics is not None,
        'visualization_data_available': visualization_data is not None
    })