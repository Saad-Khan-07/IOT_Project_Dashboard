
import React from 'react';
import { cn } from '@/lib/utils';

interface ChartContainerProps {
  title: string;
  children: React.ReactNode;
  className?: string;
  actions?: React.ReactNode;
  description?: string;
}

const ChartContainer: React.FC<ChartContainerProps> = ({
  title,
  children,
  className,
  actions,
  description,
}) => {
  return (
    <div className={cn("bg-white rounded-lg shadow-sm", className)}>
      <div className="p-4 border-b border-gray-200">
        <div className="flex justify-between items-center">
          <div>
            <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
            {description && (
              <p className="text-sm text-gray-500 mt-1">{description}</p>
            )}
          </div>
          {actions && <div>{actions}</div>}
        </div>
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
};

export default ChartContainer;
