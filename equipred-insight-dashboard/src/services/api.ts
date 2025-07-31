
import axios from 'axios';

// API base URL
const API_URL = 'http://localhost:5000/api';

// Create axios instance
const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API service for fetching data
export const apiService = {
  // Health check
  checkHealth: () => apiClient.get('/health'),
  
  // Get model metrics
  getMetrics: () => apiClient.get('/metrics'),
  
  // Get feature importance data
  getFeatureImportance: () => apiClient.get('/feature-importance'),
  
  // Get fault distribution data
  getFaultDistribution: () => apiClient.get('/fault-distribution'),
  
  // Get visualization data
  getVisualizationData: () => apiClient.get('/visualization-data'),
  
  // Get correlation matrix
  getCorrelationMatrix: () => apiClient.get('/correlation-matrix'),
  
  // Get feature distributions
  getFeatureDistributions: () => apiClient.get('/feature-distributions'),
  
  // Get dataset summary
  getDatasetSummary: () => apiClient.get('/dataset-summary'),
  
  // Get sample for prediction
  getSample: () => apiClient.get('/sample'),
  
  // Make prediction
  predict: (data: any) => apiClient.post('/predict', data),
};

export default apiService;
