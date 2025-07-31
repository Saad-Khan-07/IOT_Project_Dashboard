
# Equipment Fault Prediction Dashboard

A modern, responsive dashboard application built with Next.js to visualize and predict equipment faults using machine learning models exposed through a Flask API.

## Features

- **Real-time Dashboard**: View key metrics and model performance at a glance
- **Interactive Data Visualizations**: Explore feature importance, fault distributions, and correlations
- **Prediction Interface**: Make real-time predictions for equipment fault detection
- **Dataset Explorer**: Analyze the underlying dataset with statistics and visualizations
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## Prerequisites

- Node.js (14.x or higher)
- Flask API running on http://localhost:5000 (see API Endpoints below)

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open [http://localhost:8080](http://localhost:8080) in your browser

## API Endpoints

The dashboard connects to a Flask API running on `http://localhost:5000` with the following endpoints:

- `GET /api/metrics`: Get model performance metrics
- `GET /api/visualization-data`: Get data for all visualizations
- `GET /api/feature-importance`: Get feature importance data
- `GET /api/correlation-matrix`: Get correlation matrix data
- `GET /api/fault-distribution`: Get fault distribution data
- `GET /api/feature-distributions`: Get feature distribution data
- `POST /api/predict`: Make a prediction with new data
- `GET /api/sample`: Get a sample from the dataset for prediction testing
- `GET /api/dataset-summary`: Get a summary of the dataset
- `GET /api/health`: Health check endpoint

## Technologies Used

- **React**: Frontend library for building user interfaces
- **React Router**: For navigation between dashboard pages
- **Axios**: For API requests
- **Recharts**: For interactive data visualization
- **Tailwind CSS**: For styling and responsive design
- **Lucide Icons**: For beautiful, consistent iconography

## Project Structure

- `/src/components/layout`: Layout components (Dashboard, Sidebar, Header)
- `/src/components/dashboard`: Reusable dashboard components
- `/src/pages`: Main dashboard pages
- `/src/services`: API service for data fetching
- `/src/utils`: Utility functions for data formatting

## License

MIT
