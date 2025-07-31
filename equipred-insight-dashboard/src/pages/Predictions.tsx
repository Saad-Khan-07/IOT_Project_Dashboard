
import { useState, useEffect } from 'react';
import DashboardLayout from '@/components/layout/DashboardLayout';
import ChartContainer from '@/components/dashboard/ChartContainer';
import apiService from '@/services/api';
import { AlertTriangle, Check, Loader } from 'lucide-react';

const Predictions = () => {
  // State variables
  const [predictionInput, setPredictionInput] = useState({
    temperature: 85.5,
    pressure: 42.3,
    vibration: 2.1,
    humidity: 65.7,
    equipment: 'Turbine',
    location: 'Chicago'
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingSample, setLoadingSample] = useState(true);
  const [error, setError] = useState(null);

  // Fetch sample data on component mount
  useEffect(() => {
    const fetchSample = async () => {
      try {
        setLoadingSample(true);
        
        // Get a sample for prediction form
        const sampleRes = await apiService.getSample();
        if (sampleRes.data) {
          setPredictionInput(sampleRes.data);
        }
        
        setLoadingSample(false);
      } catch (err) {
        console.error('Error fetching sample:', err);
        setError('Failed to load sample data. Please make sure the API is running.');
        setLoadingSample(false);
      }
    };

    fetchSample();
  }, []);

  // Handle input change
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    // Convert numerical inputs to numbers
    const newValue = ['temperature', 'pressure', 'vibration', 'humidity'].includes(name)
      ? parseFloat(value)
      : value;
    
    setPredictionInput(prev => ({
      ...prev,
      [name]: newValue
    }));
  };

  // Handle prediction submit
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.predict(predictionInput);
      setPredictionResult(response.data);
      setLoading(false);
    } catch (err) {
      console.error('Prediction error:', err);
      setError('Failed to make prediction. Please check your inputs and try again.');
      setLoading(false);
    }
  };

  if (loadingSample) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-64">
          <Loader className="h-8 w-8 animate-spin text-blue-500 mr-3" />
          <span>Loading sample data...</span>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout>
      <div className="max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold mb-6">Equipment Fault Prediction</h1>
        
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6 rounded">
            <div className="flex">
              <AlertTriangle className="h-6 w-6 text-red-500 mr-3" />
              <p className="text-red-700">{error}</p>
            </div>
          </div>
        )}
        
        <ChartContainer 
          title="Input Parameters" 
          description="Enter equipment parameters to predict potential faults"
          className="mb-6"
        >
          <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Numeric parameters */}
            <div className="space-y-4">
              <div>
                <label className="block mb-1 font-medium">Temperature (°C):</label>
                <input
                  type="number"
                  step="0.01"
                  name="temperature"
                  value={predictionInput.temperature}
                  onChange={handleInputChange}
                  className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block mb-1 font-medium">Pressure:</label>
                <input
                  type="number"
                  step="0.01"
                  name="pressure"
                  value={predictionInput.pressure}
                  onChange={handleInputChange}
                  className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block mb-1 font-medium">Vibration (mm/s):</label>
                <input
                  type="number"
                  step="0.01"
                  name="vibration"
                  value={predictionInput.vibration}
                  onChange={handleInputChange}
                  className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block mb-1 font-medium">Humidity (%):</label>
                <input
                  type="number"
                  step="0.01"
                  name="humidity"
                  value={predictionInput.humidity}
                  onChange={handleInputChange}
                  className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>
            </div>
            
            {/* Categorical parameters */}
            <div>
              <label className="block mb-1 font-medium">Equipment Type:</label>
              <select
                name="equipment"
                value={predictionInput.equipment}
                onChange={handleInputChange}
                className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                required
              >
                <option value="Compressor">Compressor</option>
                <option value="Turbine">Turbine</option>
                <option value="Pump">Pump</option>
                <option value="Valve">Valve</option>
                <option value="Other">Other</option>
              </select>
            </div>
            
            <div>
              <label className="block mb-1 font-medium">Location:</label>
              <select
                name="location"
                value={predictionInput.location}
                onChange={handleInputChange}
                className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                required
              >
                <option value="Atlanta">Atlanta</option>
                <option value="Chicago">Chicago</option>
                <option value="San Francisco">San Francisco</option>
                <option value="New York">New York</option>
                <option value="Dallas">Dallas</option>
                <option value="Other">Other</option>
              </select>
            </div>
            
            <div className="md:col-span-2">
              <button
                type="submit"
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 w-full flex items-center justify-center disabled:bg-blue-400"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <Loader className="h-5 w-5 animate-spin mr-2" />
                    Processing...
                  </>
                ) : 'Run Prediction'}
              </button>
            </div>
          </form>
        </ChartContainer>
        
        {/* Prediction Results */}
        {predictionResult && (
          <ChartContainer 
            title="Prediction Result" 
            className="mb-6"
          >
            {predictionResult.error ? (
              <div className="flex items-center text-red-600">
                <AlertTriangle className="h-6 w-6 mr-2" />
                <p>{predictionResult.error}</p>
              </div>
            ) : (
              <div className="p-4">
                <div className="flex items-center mb-4">
                  <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
                    predictionResult.prediction === 1 ? 'bg-red-100' : 'bg-green-100'
                  }`}>
                    {predictionResult.prediction === 1 ? (
                      <AlertTriangle className="h-8 w-8 text-red-600" />
                    ) : (
                      <Check className="h-8 w-8 text-green-600" />
                    )}
                  </div>
                  <div className="ml-4">
                    <h3 className="text-xl font-semibold">
                      Equipment Status: 
                      <span className={`ml-2 ${predictionResult.prediction === 1 ? 'text-red-600' : 'text-green-600'}`}>
                        {predictionResult.prediction_label || (predictionResult.prediction === 1 ? 'Fault Detected' : 'Normal Operation')}
                      </span>
                    </h3>
                    <p className="text-gray-600 mt-1">
                      {predictionResult.prediction === 1 
                        ? 'The model has detected a potential fault in the equipment' 
                        : 'The equipment appears to be operating normally'}
                    </p>
                  </div>
                </div>
                
                <div className="my-6">
                  <h4 className="font-medium mb-2">Confidence Level:</h4>
                  <div className="w-full bg-gray-200 rounded-full h-6">
                    <div 
                      className={`h-6 rounded-full flex items-center justify-center text-white text-sm ${predictionResult.prediction === 1 ? 'bg-red-600' : 'bg-green-600'}`}
                      style={{ width: `${(predictionResult.prediction === 1 
                        ? predictionResult.probability?.faulty || 0.75
                        : predictionResult.probability?.normal || 0.75) * 100}%` }}
                    >
                      {((predictionResult.prediction === 1 
                        ? predictionResult.probability?.faulty || 0.75 
                        : predictionResult.probability?.normal || 0.75) * 100).toFixed(2)}%
                    </div>
                  </div>
                </div>
                
                <div className="p-4 border rounded-lg bg-gray-50">
                  <h4 className="font-medium mb-2">Detailed Probability:</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-600">Fault Probability:</p>
                      <p className="text-lg font-semibold text-red-600">
                        {((predictionResult.probability?.faulty || 0) * 100).toFixed(2)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">Normal Operation Probability:</p>
                      <p className="text-lg font-semibold text-green-600">
                        {((predictionResult.probability?.normal || 0) * 100).toFixed(2)}%
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="mt-6 border-t pt-4">
                  <h4 className="font-medium mb-2">Recommendation:</h4>
                  <p className="text-gray-800">
                    {predictionResult.prediction === 1 
                      ? 'It is recommended to schedule a maintenance check for this equipment based on the detected fault probability.' 
                      : 'Regular maintenance schedule can be followed. No immediate action required.'}
                  </p>
                </div>
              </div>
            )}
          </ChartContainer>
        )}
      </div>
    </DashboardLayout>
  );
};

export default Predictions;
