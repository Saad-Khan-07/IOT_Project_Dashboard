
import { useState, useEffect } from 'react';
import DashboardLayout from '@/components/layout/DashboardLayout';
import StatsCard from '@/components/dashboard/StatsCard';
import ChartContainer from '@/components/dashboard/ChartContainer';
import apiService from '@/services/api';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell
} from 'recharts';
import { 
  prepareMetricsData, 
  prepareFeatureImportanceData, 
  prepareFaultByEquipmentData, 
  prepareFaultByLocationData,
  COLORS
} from '@/utils/chartUtils';
import { AlertTriangle, Activity, Cpu, TrendingUp, Settings, Loader } from 'lucide-react';

const Index = () => {
  // State variables
  const [metrics, setMetrics] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [faultDistribution, setFaultDistribution] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [healthStatus, setHealthStatus] = useState('checking');

  // Fetch data on component mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Check API health
        const healthRes = await apiService.checkHealth();
        setHealthStatus(healthRes.data.status === 'ok' ? 'online' : 'offline');
        
        // Get model metrics
        const metricsRes = await apiService.getMetrics();
        setMetrics(metricsRes.data.metrics);
        
        // Get feature importance
        const featureRes = await apiService.getFeatureImportance();
        setFeatureImportance(featureRes.data);
        
        // Get fault distribution
        const faultRes = await apiService.getFaultDistribution();
        setFaultDistribution(faultRes.data);
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching data:', err);
        setHealthStatus('offline');
        setError('Failed to load data. Please make sure the API is running at http://localhost:5000');
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <Loader className="h-12 w-12 animate-spin mx-auto text-blue-500 mb-4" />
            <h2 className="text-xl font-semibold">Loading dashboard data...</h2>
            <p className="text-gray-500 mt-2">Connecting to the prediction API</p>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  if (error) {
    return (
      <DashboardLayout>
        <div className="bg-red-50 border-l-4 border-red-500 p-6 rounded-lg mb-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <AlertTriangle className="h-6 w-6 text-red-500" />
            </div>
            <div className="ml-3">
              <h3 className="text-lg font-medium text-red-800">API Connection Error</h3>
              <div className="mt-2 text-red-700">
                <p>{error}</p>
              </div>
            </div>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout>
      {/* API Status */}
      <div className={`mb-6 p-3 rounded-lg flex items-center ${
        healthStatus === 'online' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
      }`}>
        <div className={`h-3 w-3 rounded-full mr-2 ${
          healthStatus === 'online' ? 'bg-green-500' : 'bg-red-500'
        }`}></div>
        <span>
          API Status: <strong>{healthStatus === 'online' ? 'Connected' : 'Disconnected'}</strong>
        </span>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
        <StatsCard 
          title="Model Accuracy" 
          value={metrics ? `${(metrics.accuracy * 100).toFixed(2)}%` : 'N/A'} 
          icon={<TrendingUp className="h-6 w-6" />}
          trend="up"
          trendValue="2.5% from previous model"
        />
        <StatsCard 
          title="Prediction Confidence" 
          value={metrics ? `${(metrics.precision * 100).toFixed(2)}%` : 'N/A'} 
          icon={<Activity className="h-6 w-6" />}
        />
        <StatsCard 
          title="Fault Detection Rate" 
          value={metrics ? `${(metrics.recall * 100).toFixed(2)}%` : 'N/A'} 
          icon={<AlertTriangle className="h-6 w-6" />}
        />
        <StatsCard 
          title="F1 Score" 
          value={metrics ? `${(metrics.f1_score * 100).toFixed(2)}%` : 'N/A'} 
          icon={<Settings className="h-6 w-6" />}
        />
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Model Performance Metrics */}
        <ChartContainer title="Model Performance Metrics">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={prepareMetricsData(metrics)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 1]} />
                <Tooltip formatter={(value) => typeof value === 'number' ? value.toFixed(4) : value} />
                <Legend />
                <Bar dataKey="value" fill="#8884d8" name="Score" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartContainer>

        {/* Feature Importance */}
        <ChartContainer title="Top Feature Importance" description="Most influential factors in predictions">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={prepareFeatureImportanceData(featureImportance)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={120} />
                <Tooltip formatter={(value) => typeof value === 'number' ? value.toFixed(4) : value} />
                <Legend />
                <Bar dataKey="importance" fill="#82ca9d" name="Importance" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartContainer>
      </div>

      {/* Charts Row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Fault Rate by Equipment */}
        <ChartContainer title="Fault Rate by Equipment Type">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={prepareFaultByEquipmentData(faultDistribution)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: 'Fault Rate (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => typeof value === 'number' ? `${value.toFixed(2)}%` : value} />
                <Bar dataKey="faultRate" fill="#ff7300" name="Fault Rate (%)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartContainer>

        {/* Fault Rate by Location */}
        <ChartContainer title="Fault Rate by Location">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={prepareFaultByLocationData(faultDistribution)}
                  cx="50%"
                  cy="50%"
                  labelLine={true}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="faultRate"
                  nameKey="name"
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                >
                  {prepareFaultByLocationData(faultDistribution).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => typeof value === 'number' ? `${value.toFixed(2)}%` : value} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </ChartContainer>
      </div>
    </DashboardLayout>
  );
};

export default Index;
