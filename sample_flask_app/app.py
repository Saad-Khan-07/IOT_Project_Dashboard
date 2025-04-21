from flask import Flask, jsonify, request
from api_routes import api
from flask_cors import CORS
import os
import sys

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    # Register blueprints
    app.register_blueprint(api, url_prefix='/api')
    
    # Root route
    @app.route('/', methods=['GET'])
    def index():
        return jsonify({
            'message': 'Welcome to the Predictive Maintenance API',
            'version': '1.0',
            'endpoints': {
                'GET /api/metrics': 'Get model performance metrics',
                'GET /api/visualization-data': 'Get data for all visualizations',
                'GET /api/feature-importance': 'Get feature importance data',
                'GET /api/correlation-matrix': 'Get correlation matrix data',
                'GET /api/fault-distribution': 'Get fault distribution data',
                'GET /api/feature-distributions': 'Get feature distribution data',
                'POST /api/predict': 'Make a prediction with new data',
                'GET /api/sample': 'Get a sample from the dataset for prediction testing',
                'GET /api/dataset-summary': 'Get a summary of the dataset',
                'GET /api/health': 'Health check endpoint'
            }
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def server_error(error):
        return jsonify({'error': 'Server error'}), 500
    
    return app

if __name__ == '__main__':
    # Check if model training is needed first
    if 'train' in sys.argv:
        print("Training model first...")
        from model_training import main as train_model
        train_model()
    
    # Create and run the app
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)