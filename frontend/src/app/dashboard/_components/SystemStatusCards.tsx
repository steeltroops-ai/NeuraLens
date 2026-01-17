'use client';

import React, { cloneElement, memo } from 'react';
import { motion } from 'framer-motion';
import { Activity, Zap, Shield, Clock, CheckCircle, AlertTriangle, TrendingUp } from 'lucide-react';

interface SystemMetric {
  id: string;
  label: string;
  value: string | number;
  unit?: string;
  status: 'excellent' | 'good' | 'warning' | 'error';
  trend?: 'up' | 'down' | 'stable';
  icon: React.ReactNode;
}

interface SystemStatusCardsProps {
  metrics?: SystemMetric[];
}

const SystemStatusCards = memo(({ metrics }: SystemStatusCardsProps) => {
  const defaultMetrics: SystemMetric[] = [
    {
      id: 'accuracy',
      label: 'Model Accuracy',
      value: 95.2,
      unit: '%',
      status: 'excellent',
      trend: 'up',
      icon: <Shield size={18} strokeWidth={1.5} />,
    },
    {
      id: 'latency',
      label: 'Response Time',
      value: 87,
      unit: 'ms',
      status: 'excellent',
      trend: 'down',
      icon: <Zap size={18} strokeWidth={1.5} />,
    },
    {
      id: 'uptime',
      label: 'System Uptime',
      value: 99.9,
      unit: '%',
      status: 'excellent',
      trend: 'stable',
      icon: <Activity size={18} strokeWidth={1.5} />,
    },
    {
      id: 'processing',
      label: 'Active Sessions',
      value: 24,
      unit: '',
      status: 'good',
      trend: 'up',
      icon: <Clock size={18} strokeWidth={1.5} />,
    },
  ];

  const systemMetrics = metrics || defaultMetrics;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent':
        return '#22c55e';
      case 'good':
        return '#3b82f6';
      case 'warning':
        return '#f59e0b';
      case 'error':
        return '#ef4444';
      default:
        return '#71717a';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'excellent':
      case 'good':
        return <CheckCircle size={14} strokeWidth={1.5} style={{ color: getStatusColor(status) }} />;
      case 'warning':
      case 'error':
        return <AlertTriangle size={14} strokeWidth={1.5} style={{ color: getStatusColor(status) }} />;
      default:
        return <Activity size={14} strokeWidth={1.5} style={{ color: getStatusColor(status) }} />;
    }
  };

  const getTrendIcon = (trend: string | undefined, status: string) => {
    if (!trend) return null;

    const color = getStatusColor(status);
    switch (trend) {
      case 'up':
        return <TrendingUp size={12} strokeWidth={1.5} style={{ color }} />;
      case 'down':
        return <TrendingUp size={12} strokeWidth={1.5} className="rotate-180" style={{ color }} />;
      case 'stable':
        return <Activity size={12} strokeWidth={1.5} style={{ color }} />;
      default:
        return null;
    }
  };

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
      {systemMetrics.map((metric, index) => (
        <motion.div
          key={metric.id}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.15, delay: index * 0.05 }}
          className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm transition-all duration-150 hover:border-zinc-300 hover:shadow-md"
        >
          {/* Header with Icon and Status */}
          <div className="mb-3 flex items-center justify-between">
            <div
              className="flex h-8 w-8 items-center justify-center rounded-md"
              style={{ backgroundColor: `${getStatusColor(metric.status)}15` }}
            >
              {cloneElement(metric.icon as React.ReactElement, {
                style: { color: getStatusColor(metric.status) },
              })}
            </div>
            <div className="flex items-center gap-1">
              {getStatusIcon(metric.status)}
              {getTrendIcon(metric.trend, metric.status)}
            </div>
          </div>

          {/* Metric Value */}
          <div className="mb-1">
            <div className="flex items-baseline gap-0.5">
              <span className="text-[20px] font-semibold text-zinc-900">
                {metric.value}
              </span>
              {metric.unit && (
                <span className="text-[12px] font-medium text-zinc-500">
                  {metric.unit}
                </span>
              )}
            </div>
          </div>

          {/* Metric Label */}
          <div className="text-[12px] text-zinc-500">
            {metric.label}
          </div>

          {/* Mini Visualization Bar */}
          <div className="mt-3">
            <div className="h-1 rounded-full bg-zinc-100">
              <motion.div
                initial={{ width: 0 }}
                animate={{
                  width: typeof metric.value === 'number' ? `${Math.min(100, metric.value)}%` : '75%',
                }}
                transition={{ duration: 0.5, delay: index * 0.05 + 0.2 }}
                className="h-full rounded-full"
                style={{ backgroundColor: getStatusColor(metric.status) }}
              />
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
});

SystemStatusCards.displayName = 'SystemStatusCards';

export default SystemStatusCards;
