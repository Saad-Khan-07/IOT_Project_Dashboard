
import DashboardLayout from '@/components/layout/DashboardLayout';
import { useState } from 'react';
import { Settings as SettingsIcon, Save, RefreshCcw, Server, Globe, Bell, Shield, Clock } from 'lucide-react';

const Settings = () => {
  const [apiUrl, setApiUrl] = useState('http://localhost:5000/api');
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(60);
  const [notifications, setNotifications] = useState(true);
  const [theme, setTheme] = useState('light');
  
  const handleSubmit = (e) => {
    e.preventDefault();
    // Save settings
    alert('Settings saved successfully!');
  };
  
  return (
    <DashboardLayout>
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center mb-6">
          <SettingsIcon className="h-6 w-6 mr-3 text-blue-600" />
          <h1 className="text-2xl font-bold">Dashboard Settings</h1>
        </div>
        
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <h2 className="text-lg font-medium mb-4">API Configuration</h2>
          <div className="space-y-4">
            <div>
              <label htmlFor="api-url" className="block text-sm font-medium text-gray-700 mb-1">
                API Base URL
              </label>
              <div className="flex">
                <div className="relative flex-grow">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Server className="h-5 w-5 text-gray-400" />
                  </div>
                  <input
                    type="text"
                    id="api-url"
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    className="pl-10 block w-full shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm border-gray-300 rounded-md p-2 border"
                    placeholder="Enter API URL"
                  />
                </div>
                <button
                  type="button"
                  className="ml-3 inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                >
                  Test Connection
                </button>
              </div>
              <p className="mt-1 text-sm text-gray-500">
                The base URL for your prediction API. Default: http://localhost:5000/api
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <h2 className="text-lg font-medium mb-4">Dashboard Preferences</h2>
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <RefreshCcw className="h-5 w-5 text-gray-400 mr-3" />
                <div>
                  <label htmlFor="auto-refresh" className="font-medium text-gray-700">
                    Auto-refresh Dashboard
                  </label>
                  <p className="text-sm text-gray-500">
                    Automatically refresh dashboard data at set intervals
                  </p>
                </div>
              </div>
              <div className="ml-4 flex-shrink-0">
                <div className="relative inline-block w-10 mr-2 align-middle select-none">
                  <input
                    type="checkbox"
                    id="auto-refresh"
                    checked={autoRefresh}
                    onChange={() => setAutoRefresh(!autoRefresh)}
                    className="sr-only"
                  />
                  <div className={`block h-6 rounded-full w-12 ${autoRefresh ? 'bg-blue-500' : 'bg-gray-300'}`}></div>
                  <div className={`dot absolute left-1 top-1 bg-white h-4 w-4 rounded-full transition ${autoRefresh ? 'transform translate-x-6' : ''}`}></div>
                </div>
              </div>
            </div>
            
            {autoRefresh && (
              <div className="pl-8">
                <label htmlFor="refresh-interval" className="block text-sm font-medium text-gray-700 mb-1">
                  Refresh Interval (seconds)
                </label>
                <input
                  type="number"
                  id="refresh-interval"
                  min="10"
                  max="3600"
                  value={refreshInterval}
                  onChange={(e) => setRefreshInterval(parseInt(e.target.value, 10))}
                  className="block w-full shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm border-gray-300 rounded-md p-2 border"
                />
              </div>
            )}
            
            <div className="flex items-center justify-between pt-4 border-t">
              <div className="flex items-center">
                <Bell className="h-5 w-5 text-gray-400 mr-3" />
                <div>
                  <label htmlFor="notifications" className="font-medium text-gray-700">
                    Prediction Alerts
                  </label>
                  <p className="text-sm text-gray-500">
                    Receive alerts for high-risk equipment fault predictions
                  </p>
                </div>
              </div>
              <div className="ml-4 flex-shrink-0">
                <div className="relative inline-block w-10 mr-2 align-middle select-none">
                  <input
                    type="checkbox"
                    id="notifications"
                    checked={notifications}
                    onChange={() => setNotifications(!notifications)}
                    className="sr-only"
                  />
                  <div className={`block h-6 rounded-full w-12 ${notifications ? 'bg-blue-500' : 'bg-gray-300'}`}></div>
                  <div className={`dot absolute left-1 top-1 bg-white h-4 w-4 rounded-full transition ${notifications ? 'transform translate-x-6' : ''}`}></div>
                </div>
              </div>
            </div>
            
            <div className="flex items-center justify-between pt-4 border-t">
              <div className="flex items-center">
                <Globe className="h-5 w-5 text-gray-400 mr-3" />
                <div>
                  <label className="font-medium text-gray-700">
                    Theme Preference
                  </label>
                  <p className="text-sm text-gray-500">
                    Choose between light and dark theme
                  </p>
                </div>
              </div>
              <div className="ml-4 flex-shrink-0">
                <select
                  id="theme"
                  value={theme}
                  onChange={(e) => setTheme(e.target.value)}
                  className="block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                >
                  <option value="light">Light</option>
                  <option value="dark">Dark</option>
                  <option value="system">System Default</option>
                </select>
              </div>
            </div>
          </div>
        </div>
        
        <div className="flex justify-end">
          <button
            type="button"
            className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 mr-3"
          >
            Reset to Defaults
          </button>
          <button
            type="button"
            onClick={handleSubmit}
            className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            <Save className="h-4 w-4 mr-2" />
            Save Settings
          </button>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default Settings;
