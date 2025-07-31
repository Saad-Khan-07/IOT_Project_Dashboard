
import React from 'react';
import { cn } from '@/lib/utils';

interface StatsCardProps {
  title: string;
  value: string | number;
  description?: string;
  icon?: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  className?: string;
}

const StatsCard: React.FC<StatsCardProps> = ({
  title,
  value,
  description,
  icon,
  trend,
  trendValue,
  className,
}) => {
  return (
    <div className={cn("bg-white rounded-lg p-6 shadow-sm", className)}>
      <div className="flex justify-between items-start">
        <div>
          <p className="text-sm font-medium text-gray-500 mb-1">{title}</p>
          <h3 className="text-2xl font-bold">{value}</h3>
          {description && (
            <p className="text-sm text-gray-500 mt-1">{description}</p>
          )}
        </div>
        {icon && <div className="text-blue-500">{icon}</div>}
      </div>
      
      {trend && (
        <div className="mt-4 flex items-center">
          <span
            className={cn(
              "inline-flex items-center px-2 py-0.5 rounded text-xs font-medium",
              {
                "bg-green-100 text-green-800": trend === "up",
                "bg-red-100 text-red-800": trend === "down",
                "bg-gray-100 text-gray-800": trend === "neutral",
              }
            )}
          >
            {trend === "up" && <span className="mr-1">↑</span>}
            {trend === "down" && <span className="mr-1">↓</span>}
            {trendValue}
          </span>
        </div>
      )}
    </div>
  );
};

export default StatsCard;
