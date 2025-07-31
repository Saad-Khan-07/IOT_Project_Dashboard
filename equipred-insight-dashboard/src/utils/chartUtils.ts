
// Helper function to format feature names
export const formatFeatureName = (name: string): string => {
  return name
    .replace('cat__onehot__', '')
    .replace('num__scaler__', '')
    .replace(/_/g, ' ')
    .replace(/^./, str => str.toUpperCase());
};

// Random colors for charts
export const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#8dd1e1', '#a4de6c', '#d0ed57'];

// Prepare data for the metrics chart
export const prepareMetricsData = (metrics: any) => {
  if (!metrics) return [];
  
  return [
    { name: 'Accuracy', value: metrics.accuracy },
    { name: 'Precision', value: metrics.precision },
    { name: 'Recall', value: metrics.recall },
    { name: 'F1 Score', value: metrics.f1_score }
  ];
};

// Prepare feature importance data
export const prepareFeatureImportanceData = (featureImportance: any) => {
  if (!featureImportance || !featureImportance.features) return [];
  
  return featureImportance.features
    .map((feature: string, index: number) => ({
      name: formatFeatureName(feature),
      importance: featureImportance.importance[index]
    }))
    .sort((a: any, b: any) => b.importance - a.importance)
    .slice(0, 10); // Top 10 features
};

// Prepare fault distribution data by equipment
export const prepareFaultByEquipmentData = (faultDistribution: any) => {
  if (!faultDistribution || !faultDistribution.by_equipment) return [];
  
  return Object.entries(faultDistribution.by_equipment).map(([key, value]: [string, any]) => ({
    name: key,
    faultRate: value * 100 // Convert to percentage
  }));
};

// Prepare fault distribution data by location
export const prepareFaultByLocationData = (faultDistribution: any) => {
  if (!faultDistribution || !faultDistribution.by_location) return [];
  
  return Object.entries(faultDistribution.by_location).map(([key, value]: [string, any]) => ({
    name: key,
    faultRate: value * 100 // Convert to percentage
  }));
};
