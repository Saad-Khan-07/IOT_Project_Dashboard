
import { Bell, Search, MessageSquare } from 'lucide-react';

const Header = () => {
  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <h1 className="text-2xl font-semibold text-gray-800">Equipment Fault Prediction Dashboard</h1>
        </div>

        <div className="flex items-center space-x-4">
          <div className="relative">
            <input
              type="text"
              placeholder="Search..."
              className="py-2 pl-10 pr-4 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-64"
            />
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          </div>
          
          <button className="p-2 rounded-full hover:bg-gray-100 relative">
            <Bell className="h-5 w-5 text-gray-600" />
            <span className="absolute top-0 right-0 h-2 w-2 bg-red-500 rounded-full"></span>
          </button>
          
          <button className="p-2 rounded-full hover:bg-gray-100">
            <MessageSquare className="h-5 w-5 text-gray-600" />
          </button>
          
          <div className="flex items-center">
            <div className="h-9 w-9 rounded-full bg-blue-500 flex items-center justify-center text-white">
              <span className="font-medium">U</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
