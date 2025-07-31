
import { Home, BarChart2, Settings, Activity, Database, AlertTriangle } from 'lucide-react';
import { Link } from 'react-router-dom';

const Sidebar = () => {
  const menuItems = [
    { icon: Home, label: 'Dashboard', path: '/' },
    { icon: BarChart2, label: 'Visualizations', path: '/visualizations' },
    { icon: Activity, label: 'Predictions', path: '/predictions' },
    { icon: Database, label: 'Dataset', path: '/dataset' },
    { icon: Settings, label: 'Settings', path: '/settings' },
  ];

  return (
    <div className="bg-sidebar text-sidebar-foreground w-64 h-full flex flex-col shadow-lg overflow-hidden">
      <div className="p-6">
        <h2 className="text-2xl font-bold flex items-center">
          <AlertTriangle className="mr-2 h-6 w-6 text-yellow-500" />
          <span>EquipPred</span>
        </h2>
      </div>
      
      <nav className="flex-1 overflow-y-auto">
        <ul className="px-4 py-2">
          {menuItems.map((item, index) => (
            <li key={index} className="mb-2">
              <Link
                to={item.path}
                className="flex items-center p-3 rounded-lg hover:bg-sidebar-accent transition-colors duration-150"
              >
                <item.icon className="h-5 w-5 mr-3" />
                <span>{item.label}</span>
              </Link>
            </li>
          ))}
        </ul>
      </nav>
      
      <div className="p-4 border-t border-sidebar-border">
        <div className="flex items-center">
          <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center text-white">
            <span className="font-semibold">AP</span>
          </div>
          <div className="ml-3">
            <p className="font-medium">Admin Panel</p>
            <p className="text-xs text-sidebar-foreground/70">Equipment Insights</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
