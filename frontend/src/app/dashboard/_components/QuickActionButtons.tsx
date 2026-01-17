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
      label: 'Start Assessment',
      description: 'Begin neurological evaluation',
      icon: <Play size={18} strokeWidth={1.5} />,
      color: '#3b82f6',
      onClick: () => handleActionClick('start-assessment'),
    },
    {
      id: 'view-reports',
      label: 'View Reports',
      description: 'Access analysis and insights',
      icon: <FileText size={18} strokeWidth={1.5} />,
      color: '#22c55e',
      onClick: () => handleActionClick('view-reports'),
    },
    {
      id: 'schedule-assessment',
      label: 'Schedule',
      description: 'Plan future evaluations',
      icon: <Calendar size={18} strokeWidth={1.5} />,
      color: '#f59e0b',
      onClick: () => handleActionClick('schedule-assessment'),
    },
    {
      id: 'system-settings',
      label: 'Settings',
      description: 'Configure preferences',
      icon: <Settings size={18} strokeWidth={1.5} />,
      color: '#64748b',
      onClick: () => handleActionClick('system-settings'),
    },
    {
      id: 'export-data',
      label: 'Export Data',
      description: 'Download results',
      icon: <Download size={18} strokeWidth={1.5} />,
      color: '#8b5cf6',
      onClick: () => handleActionClick('export-data'),
    },
    {
      id: 'manage-patients',
      label: 'Patients',
      description: 'Manage patient profiles',
      icon: <Users size={18} strokeWidth={1.5} />,
      color: '#ec4899',
      onClick: () => handleActionClick('manage-patients'),
    },
    {
      id: 'analytics',
      label: 'Analytics',
      description: 'View metrics and trends',
      icon: <BarChart3 size={18} strokeWidth={1.5} />,
      color: '#06b6d4',
      onClick: () => handleActionClick('analytics'),
    },
    {
      id: 'ai-insights',
      label: 'AI Insights',
      description: 'Get recommendations',
      icon: <Zap size={18} strokeWidth={1.5} />,
      color: '#eab308',
      onClick: () => handleActionClick('ai-insights'),
    },
  ];

  const actionList = actions || defaultActions;

  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4">
      {/* Header */}
      <div className="mb-4">
        <h3 className="text-[14px] font-medium text-zinc-900">
          Quick Actions
        </h3>
        <p className="mt-0.5 text-[12px] text-zinc-500">
          Frequently used tools
        </p>
      </div>

      {/* Action Grid */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {actionList.map((action, index) => (
          <motion.button
            key={action.id}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.15, delay: index * 0.03 }}
            onClick={() => action.onClick()}
            disabled={action.disabled}
            className={`relative rounded-lg border border-zinc-100 bg-white p-3 text-left transition-all duration-150 ${action.disabled
                ? 'cursor-not-allowed opacity-50'
                : 'hover:border-zinc-200 hover:bg-zinc-50'
              }`}
          >
            {/* Icon */}
            <div
              className="mb-2 flex h-8 w-8 items-center justify-center rounded-md"
              style={{ backgroundColor: `${action.color}15` }}
            >
              {React.cloneElement(action.icon as React.ReactElement, {
                style: { color: action.color },
              })}
            </div>

            {/* Label */}
            <div className="text-[12px] font-medium text-zinc-900">
              {action.label}
            </div>

            {/* Description */}
            <div className="text-[11px] text-zinc-500 line-clamp-1">
              {action.description}
            </div>
          </motion.button>
        ))}
      </div>

      {/* Bottom Action Bar */}
      <div className="mt-4 pt-3 border-t border-zinc-100">
        <div className="flex items-center justify-between">
          <div className="text-[11px] text-zinc-500">
            Need help?{' '}
            <button
              className="font-medium text-blue-500 hover:text-blue-600 transition-colors"
              onClick={() => handleActionClick('help')}
            >
              View docs
            </button>
          </div>
          <button
            className="text-[11px] font-medium text-blue-500 hover:text-blue-600 transition-colors"
            onClick={() => handleActionClick('customize')}
          >
            Customize
          </button>
        </div>
      </div>
    </div>
  );
});

QuickActionButtons.displayName = 'QuickActionButtons';

export default QuickActionButtons;
