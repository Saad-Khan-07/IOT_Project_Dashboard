
import { useState, useEffect } from 'react';
import DashboardLayout from '@/components/layout/DashboardLayout';
import ChartContainer from '@/components/dashboard/ChartContainer';
import apiService from '@/services/api';
import { 
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, 
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, ComposedChart
} from 'recharts';
import { COLORS } from '@/utils/chartUtils';
import { Loader, RefreshCcw } from 'lucide-react';

const Visualizations = () => {
  const [visualizationData, setVisualizationData] = useState(null);
  const [correlationMatrix, setCorrelationMatrix] = useState(null);
  const [featureDistributions, setFeatureDistributions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('correlations');

  // Fetch data on component mount
  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Get visualization data
      const visualRes = await apiService.getVisualizationData();
      setVisualizationData(visualRes.data);
      
      // Get correlation matrix
      const corrRes = await apiService.getCorrelationMatrix();
      setCorrelationMatrix(corrRes.data);
      
      // Get feature distributions
      const distRes = await apiService.getFeatureDistributions();
      setFeatureDistributions(distRes.data);
      
      setLoading(false);
    } catch (err) {
      console.error('Error fetching visualization data:', err);
      setError('Failed to load visualization data. Please make sure the API is running.');
      setLoading(false);
    }
  };

  // Prepare correlation data for heatmap
  const prepareCorrelationData = () => {
    if (!correlationMatrix || !correlationMatrix.features || !correlationMatrix.matrix) return [];
    
    const { features, matrix } = correlationMatrix;
    const data = [];
    
    for (let i = 0; i < features.length; i++) {
      const row = { feature: features[i] };
      for (let j = 0; j < features.length; j++) {
        row[features[j]] = matrix[i][j];
      }
      data.push(row);
    }
    
    return data;
  };

  // Prepare feature distributions data
  const prepareFeatureDistributionData = (featureName) => {
    if (!featureDistributions || !featureDistributions[featureName]) return [];
    
    return featureDistributions[featureName].map((item, index) => ({
      bin: index,
      count: item.count,
      range: item.range,
      fault_rate: item.fault_rate * 100,
    }));
  };

  // Helper for correlation color
  const getCorrelationColor = (value) => {
    if (value >= 0.7) return '#ef4444'; // Strong positive: red
    if (value >= 0.4) return '#f97316'; // Moderate positive: orange
    if (value >= 0.1) return '#84cc16'; // Weak positive: green
    if (value > -0.1) return '#d1d5db'; // Negligible: gray
    if (value > -0.4) return '#06b6d4'; // Weak negative: teal
    if (value > -0.7) return '#3b82f6'; // Moderate negative: blue
    return '#8b5cf6'; // Strong negative: purple
  };

  if (loading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-64">
          <Loader className="h-8 w-8 animate-spin text-blue-500 mr-3" />
          <span>Loading visualization data...</span>
        </div>
      </DashboardLayout>
    );
  }

  if (error) {
    return (
      <DashboardLayout>
        <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded mb-6">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-500" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-red-700">{error}</p>
              <div className="mt-2">
                <button 
                  onClick={fetchData} 
                  className="inline-flex items-center px-3 py-1 border border-red-500 text-red-500 rounded-md hover:bg-red-50"
                >
                  <RefreshCcw className="h-4 w-4 mr-1" />
                  Retry
                </button>
              </div>
            </div>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout>
      {/* Tabs for different visualizations */}
      <div className="mb-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-6">
            <button
              onClick={() => setActiveTab('correlations')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'correlations'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Feature Correlations
            </button>
            <button
              onClick={() => setActiveTab('distributions')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'distributions'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Feature Distributions
            </button>
            <button
              onClick={() => setActiveTab('performance')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'performance'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Performance Analysis
            </button>
          </nav>
        </div>
      </div>

      {/* Feature Correlations Tab */}
      {activeTab === 'correlations' && (
        <div>
          <ChartContainer 
            title="Feature Correlation Heatmap" 
            description="Visualizing relationships between different features"
            className="mb-6"
          >
            <div className="h-96">
              {correlationMatrix && correlationMatrix.features && (
                <div className="overflow-x-auto">
                  <table className="min-w-full border-collapse">
                    <thead>
                      <tr>
                        <th className="p-2 border bg-gray-50">Feature</th>
                        {correlationMatrix.features.map((feature, index) => (
                          <th key={index} className="p-2 border bg-gray-50 text-sm">
                            {feature.replace('num__scaler__', '').replace('cat__onehot__', '')}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {correlationMatrix.features.map((feature, rowIndex) => (
                        <tr key={rowIndex}>
                          <td className="p-2 border bg-gray-50 font-medium text-sm">
                            {feature.replace('num__scaler__', '').replace('cat__onehot__', '')}
                          </td>
                          {correlationMatrix.matrix[rowIndex].map((value, colIndex) => (
                            <td 
                              key={colIndex} 
                              className="p-2 border text-center relative group"
                              style={{ backgroundColor: getCorrelationColor(value) }}
                            >
                              <span className="text-white text-xs font-medium">
                                {value.toFixed(2)}
                              </span>
                              <div className="absolute z-10 invisible group-hover:visible bg-gray-900 text-white p-2 rounded text-xs -mt-8 ml-4">
                                {correlationMatrix.features[rowIndex].replace('num__scaler__', '').replace('cat__onehot__', '')} - 
                                {correlationMatrix.features[colIndex].replace('num__scaler__', '').replace('cat__onehot__', '')}: {value.toFixed(4)}
                              </div>
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
            <div className="mt-4 flex justify-center space-x-6">
              <div className="flex items-center">
                <div className="w-4 h-4 rounded bg-[#ef4444] mr-2"></div>
                <span className="text-xs">Strong Positive</span>
              </div>
              <div className="flex items-center">
                <div className="w-4 h-4 rounded bg-[#f97316] mr-2"></div>
                <span className="text-xs">Moderate Positive</span>
              </div>
              <div className="flex items-center">
                <div className="w-4 h-4 rounded bg-[#84cc16] mr-2"></div>
                <span className="text-xs">Weak Positive</span>
              </div>
              <div className="flex items-center">
                <div className="w-4 h-4 rounded bg-[#d1d5db] mr-2"></div>
                <span className="text-xs">Negligible</span>
              </div>
              <div className="flex items-center">
                <div className="w-4 h-4 rounded bg-[#06b6d4] mr-2"></div>
                <span className="text-xs">Weak Negative</span>
              </div>
              <div className="flex items-center">
                <div className="w-4 h-4 rounded bg-[#3b82f6] mr-2"></div>
                <span className="text-xs">Moderate Negative</span>
              </div>
              <div className="flex items-center">
                <div className="w-4 h-4 rounded bg-[#8b5cf6] mr-2"></div>
                <span className="text-xs">Strong Negative</span>
              </div>
            </div>
          </ChartContainer>
        </div>
      )}

      {/* Feature Distributions Tab */}
      {activeTab === 'distributions' && (
        <div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {featureDistributions && Object.keys(featureDistributions).slice(0, 4).map((feature, index) => (
              <ChartContainer 
                key={index}
                title={`${feature.replace('num__scaler__', '').replace('cat__onehot__', '')} Distribution`}
              >
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={prepareFeatureDistributionData(feature)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="range" />
                      <YAxis yAxisId="left" />
                      <YAxis yAxisId="right" orientation="right" />
                      <Tooltip />
                      <Legend />
                      <Bar yAxisId="left" dataKey="count" fill="#8884d8" name="Count" />
                      <Line yAxisId="right" type="monotone" dataKey="fault_rate" stroke="#ff7300" name="Fault Rate (%)" />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </ChartContainer>
            ))}
          </div>
        </div>
      )}

      {/* Performance Analysis Tab */}
      {activeTab === 'performance' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ChartContainer title="ROC Curve">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={visualizationData?.roc_curve || []}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="fpr" 
                    label={{ value: 'False Positive Rate', position: 'insideBottomRight', offset: -10 }} 
                  />
                  <YAxis 
                    label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft' }} 
                  />
                  <Tooltip formatter={(value) => typeof value === 'number' ? value.toFixed(4) : value} />
                  <Legend />
                  <Line type="monotone" dataKey="tpr" stroke="#8884d8" name="ROC Curve" />
                  <Line type="monotone" strokeDasharray="5 5" dataKey="baseline" stroke="#82ca9d" name="Baseline" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            {visualizationData?.auc && (
              <div className="mt-2 text-center">
                <span className="font-medium">AUC Score: </span>
                <span className="text-blue-600">{visualizationData.auc.toFixed(4)}</span>
              </div>
            )}
          </ChartContainer>

          <ChartContainer title="Precision-Recall Curve">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={visualizationData?.pr_curve || []}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="recall" 
                    label={{ value: 'Recall', position: 'insideBottomRight', offset: -10 }} 
                  />
                  <YAxis 
                    label={{ value: 'Precision', angle: -90, position: 'insideLeft' }} 
                  />
                  <Tooltip formatter={(value) => typeof value === 'number' ? value.toFixed(4) : value} />
                  <Legend />
                  <Line type="monotone" dataKey="precision" stroke="#ff7300" name="PR Curve" />
                  <Line type="monotone" strokeDasharray="5 5" dataKey="baseline" stroke="#82ca9d" name="Baseline" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            {visualizationData?.average_precision && (
              <div className="mt-2 text-center">
                <span className="font-medium">Average Precision: </span>
                <span className="text-blue-600">{visualizationData.average_precision.toFixed(4)}</span>
              </div>
            )}
          </ChartContainer>
        </div>
      )}
    </DashboardLayout>
  );
};

export default Visualizations;
