
import { useState, useEffect } from 'react';
import DashboardLayout from '@/components/layout/DashboardLayout';
import ChartContainer from '@/components/dashboard/ChartContainer';
import apiService from '@/services/api';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Cell, PieChart, Pie
} from 'recharts';
import { COLORS } from '@/utils/chartUtils';
import { Loader, Download, Database, FileBarChart, Filter } from 'lucide-react';

const Dataset = () => {
  const [datasetSummary, setDatasetSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch data on component mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Get dataset summary
        const summaryRes = await apiService.getDatasetSummary();
        setDatasetSummary(summaryRes.data);
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching dataset summary:', err);
        setError('Failed to load dataset summary. Please make sure the API is running.');
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Prepare data for class distribution chart
  const prepareClassDistributionData = () => {
    if (!datasetSummary || !datasetSummary.class_distribution) return [];
    
    return Object.entries(datasetSummary.class_distribution).map(([key, value]: [string, any]) => ({
      name: key === '0' ? 'Normal' : 'Fault',
      value: value
    }));
  };

  // Prepare data for numerical features summary
  const prepareNumericalFeaturesSummary = () => {
    if (!datasetSummary || !datasetSummary.numerical_features) return [];
    
    return Object.entries(datasetSummary.numerical_features).map(([key, value]: [string, any]) => ({
      name: key,
      min: value.min,
      max: value.max,
      mean: value.mean,
      median: value.median,
      std: value.std
    }));
  };

  // Prepare data for categorical features summary
  const prepareCategoricalFeaturesSummary = () => {
    if (!datasetSummary || !datasetSummary.categorical_features) return [];
    
    const result = [];
    
    Object.entries(datasetSummary.categorical_features).forEach(([feature, values]: [string, any]) => {
      Object.entries(values).forEach(([category, count]: [string, any]) => {
        result.push({
          feature,
          category,
          count
        });
      });
    });
    
    return result;
  };

  if (loading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-64">
          <Loader className="h-8 w-8 animate-spin text-blue-500 mr-3" />
          <span>Loading dataset information...</span>
        </div>
      </DashboardLayout>
    );
  }

  if (error) {
    return (
      <DashboardLayout>
        <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-500" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-red-700">{error}</p>
            </div>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout>
      {/* Dataset Overview */}
      <div className="mb-6">
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-800">Dataset Overview</h1>
              <p className="text-gray-600 mt-1">
                Equipment maintenance and fault prediction dataset
              </p>
            </div>
            <div className="mt-4 md:mt-0 flex items-center space-x-3">
              <span className="text-sm text-gray-500">
                {datasetSummary?.total_records || 0} records • 
                {datasetSummary?.numerical_features ? Object.keys(datasetSummary.numerical_features).length : 0} numerical features • 
                {datasetSummary?.categorical_features ? Object.keys(datasetSummary.categorical_features).length : 0} categorical features
              </span>
              <button className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
                <Download className="h-4 w-4 mr-2" />
                Export CSV
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="bg-white rounded-lg shadow-sm p-6 flex items-center">
          <div className="p-3 rounded-full bg-blue-100 text-blue-600 mr-4">
            <Database className="h-6 w-6" />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-600">Total Records</p>
            <p className="text-2xl font-semibold">{datasetSummary?.total_records || 0}</p>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow-sm p-6 flex items-center">
          <div className="p-3 rounded-full bg-green-100 text-green-600 mr-4">
            <FileBarChart className="h-6 w-6" />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-600">Features</p>
            <p className="text-2xl font-semibold">
              {(datasetSummary?.numerical_features ? Object.keys(datasetSummary.numerical_features).length : 0) + 
               (datasetSummary?.categorical_features ? Object.keys(datasetSummary.categorical_features).length : 0)}
            </p>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow-sm p-6 flex items-center">
          <div className="p-3 rounded-full bg-yellow-100 text-yellow-600 mr-4">
            <Filter className="h-6 w-6" />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-600">Fault Rate</p>
            <p className="text-2xl font-semibold">
              {datasetSummary?.class_distribution && 
               `${((datasetSummary.class_distribution['1'] / datasetSummary.total_records) * 100).toFixed(2)}%`}
            </p>
          </div>
        </div>
      </div>

      {/* Class Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <ChartContainer 
          title="Class Distribution" 
          description="Distribution of normal vs. fault records in the dataset"
        >
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={prepareClassDistributionData()}
                  cx="50%"
                  cy="50%"
                  labelLine={true}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  nameKey="name"
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                >
                  {prepareClassDistributionData().map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.name === 'Normal' ? '#10b981' : '#ef4444'} 
                    />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => value} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </ChartContainer>

        <ChartContainer 
          title="Numerical Features Summary" 
          description="Statistical summary of numerical features"
        >
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Feature</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Min</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Max</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Mean</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Median</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Std Dev</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {prepareNumericalFeaturesSummary().map((feature, index) => (
                  <tr key={index}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{feature.name}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{feature.min.toFixed(2)}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{feature.max.toFixed(2)}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{feature.mean.toFixed(2)}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{feature.median.toFixed(2)}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{feature.std.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </ChartContainer>
      </div>

      {/* Categorical Features Distribution */}
      <ChartContainer 
        title="Categorical Features Distribution" 
        description="Distribution of values in categorical features"
        className="mb-6"
      >
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={prepareCategoricalFeaturesSummary()}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="category" />
              <YAxis />
              <Tooltip formatter={(value) => value} />
              <Legend />
              <Bar dataKey="count" fill="#8884d8">
                {prepareCategoricalFeaturesSummary().map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartContainer>
    </DashboardLayout>
  );
};

export default Dataset;
