'use client';

import React, { memo, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Play, FileText, Calendar, Settings, Download, Users, BarChart3, Zap } from 'lucide-react';

interface QuickAction {
  id: string;
  label: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  onClick: () => void;
  disabled?: boolean;
}

interface QuickActionButtonsProps {
  actions?: QuickAction[];
  onActionClick?: (actionId: string) => void;
}

const QuickActionButtons = memo(({ actions, onActionClick }: QuickActionButtonsProps) => {
  const handleActionClick = useCallback(
    (actionId: string, onClick?: () => void) => {
      if (onClick) {
        onClick();
      } else if (onActionClick) {
        onActionClick(actionId);
      } else {
        console.log(`Action clicked: ${actionId}`);
      }
    },
    [onActionClick],
  );

  const defaultActions: QuickAction[] = [
    {
      id: 'start-assessment',
      label: 'Start New Assessment',
      description: 'Begin comprehensive neurological evaluation',
      icon: <Play className='h-5 w-5' />,
      color: '#007AFF',
      onClick: () => handleActionClick('start-assessment'),
    },
    {
      id: 'view-reports',
      label: 'View Reports',
      description: 'Access detailed analysis and insights',
      icon: <FileText className='h-5 w-5' />,
      color: '#34C759',
      onClick: () => handleActionClick('view-reports'),
    },
    {
      id: 'schedule-assessment',
      label: 'Schedule Assessment',
      description: 'Plan future evaluations and appointments',
      icon: <Calendar className='h-5 w-5' />,
      color: '#FF9500',
      onClick: () => handleActionClick('schedule-assessment'),
    },
    {
      id: 'system-settings',
      label: 'System Settings',
      description: 'Configure platform preferences',
      icon: <Settings className='h-5 w-5' />,
      color: '#86868B',
      onClick: () => handleActionClick('system-settings'),
    },
    {
      id: 'export-data',
      label: 'Export Data',
      description: 'Download assessment results and analytics',
      icon: <Download className='h-5 w-5' />,
      color: '#5856D6',
      onClick: () => handleActionClick('export-data'),
    },
    {
      id: 'manage-patients',
      label: 'Manage Patients',
      description: 'View and organize patient profiles',
      icon: <Users className='h-5 w-5' />,
      color: '#AF52DE',
      onClick: () => handleActionClick('manage-patients'),
    },
    {
      id: 'analytics',
      label: 'Analytics Dashboard',
      description: 'View performance metrics and trends',
      icon: <BarChart3 className='h-5 w-5' />,
      color: '#FF2D92',
      onClick: () => handleActionClick('analytics'),
    },
    {
      id: 'ai-insights',
      label: 'AI Insights',
      description: 'Get personalized recommendations',
      icon: <Zap className='h-5 w-5' />,
      color: '#FFD60A',
      onClick: () => handleActionClick('ai-insights'),
    },
  ];

  const actionList = actions || defaultActions;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className='rounded-2xl border p-6 backdrop-blur-xl'
      style={{
        backgroundColor: 'rgba(255, 255, 255, 0.6)',
        borderColor: 'rgba(0, 0, 0, 0.1)',
        backdropFilter: 'blur(20px)',
      }}
    >
      {/* Header */}
      <div className='mb-6'>
        <h3
          className='text-lg font-semibold'
          style={{
            color: '#1D1D1F',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          }}
        >
          Quick Actions
        </h3>
        <p className='mt-1 text-sm' style={{ color: '#86868B' }}>
          Frequently used tools and features
        </p>
      </div>

      {/* Action Grid */}
      <div className='grid grid-cols-2 gap-4 md:grid-cols-4'>
        {actionList.map((action, index) => (
          <motion.button
            key={action.id}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: index * 0.05 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => action.onClick()}
            disabled={action.disabled}
            className={`relative rounded-xl border p-4 backdrop-blur-xl transition-all duration-300 ${
              action.disabled ? 'cursor-not-allowed opacity-50' : 'hover:shadow-lg active:scale-95'
            } `}
            style={{
              backgroundColor: 'rgba(255, 255, 255, 0.8)',
              borderColor: 'rgba(0, 0, 0, 0.1)',
              backdropFilter: 'blur(20px)',
            }}
          >
            {/* Icon */}
            <div
              className='mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full'
              style={{ backgroundColor: `${action.color}20` }}
            >
              {React.cloneElement(action.icon as React.ReactElement, {
                style: { color: action.color },
              })}
            </div>

            {/* Label */}
            <div
              className='mb-1 text-center text-sm font-medium'
              style={{
                color: '#1D1D1F',
                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
              }}
            >
              {action.label}
            </div>

            {/* Description */}
            <div className='line-clamp-2 text-center text-xs' style={{ color: '#86868B' }}>
              {action.description}
            </div>

            {/* Hover Effect Overlay */}
            <motion.div
              className='pointer-events-none absolute inset-0 rounded-xl opacity-0'
              style={{ backgroundColor: `${action.color}10` }}
              whileHover={{ opacity: 1 }}
              transition={{ duration: 0.2 }}
            />
          </motion.button>
        ))}
      </div>

      {/* Bottom Action Bar */}
      <div className='mt-6 border-t pt-4' style={{ borderColor: 'rgba(0, 0, 0, 0.1)' }}>
        <div className='flex items-center justify-between'>
          <div className='text-sm' style={{ color: '#86868B' }}>
            Need help? Check our{' '}
            <button
              className='font-medium transition-opacity hover:opacity-70'
              style={{ color: '#007AFF' }}
              onClick={() => handleActionClick('help')}
            >
              documentation
            </button>
          </div>
          <button
            className='text-sm font-medium transition-opacity hover:opacity-70'
            style={{ color: '#007AFF' }}
            onClick={() => handleActionClick('customize')}
          >
            Customize Actions
          </button>
        </div>
      </div>
    </motion.div>
  );
});

export default QuickActionButtons;
