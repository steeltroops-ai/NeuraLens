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
      icon: <Shield className='h-5 w-5' />,
    },
    {
      id: 'latency',
      label: 'Response Time',
      value: 87,
      unit: 'ms',
      status: 'excellent',
      trend: 'down',
      icon: <Zap className='h-5 w-5' />,
    },
    {
      id: 'uptime',
      label: 'System Uptime',
      value: 99.9,
      unit: '%',
      status: 'excellent',
      trend: 'stable',
      icon: <Activity className='h-5 w-5' />,
    },
    {
      id: 'processing',
      label: 'Active Sessions',
      value: 24,
      unit: '',
      status: 'good',
      trend: 'up',
      icon: <Clock className='h-5 w-5' />,
    },
  ];

  const systemMetrics = metrics || defaultMetrics;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent':
        return '#34C759';
      case 'good':
        return '#007AFF';
      case 'warning':
        return '#FF9500';
      case 'error':
        return '#FF3B30';
      default:
        return '#86868B';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'excellent':
      case 'good':
        return <CheckCircle className='h-4 w-4' style={{ color: getStatusColor(status) }} />;
      case 'warning':
      case 'error':
        return <AlertTriangle className='h-4 w-4' style={{ color: getStatusColor(status) }} />;
      default:
        return <Activity className='h-4 w-4' style={{ color: getStatusColor(status) }} />;
    }
  };

  const getTrendIcon = (trend: string | undefined, status: string) => {
    if (!trend) return null;

    const color = getStatusColor(status);
    switch (trend) {
      case 'up':
        return <TrendingUp className='h-3 w-3' style={{ color }} />;
      case 'down':
        return <TrendingUp className='h-3 w-3 rotate-180' style={{ color }} />;
      case 'stable':
        return <Activity className='h-3 w-3' style={{ color }} />;
      default:
        return null;
    }
  };

  return (
    <div className='grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4'>
      {systemMetrics.map((metric, index) => (
        <motion.div
          key={metric.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: index * 0.1 }}
          className='rounded-2xl border p-6 backdrop-blur-xl transition-all duration-300 hover:scale-105'
          style={{
            backgroundColor: 'rgba(255, 255, 255, 0.6)',
            borderColor: 'rgba(0, 0, 0, 0.1)',
            backdropFilter: 'blur(20px)',
          }}
        >
          {/* Header with Icon and Status */}
          <div className='mb-4 flex items-center justify-between'>
            <div
              className='flex h-10 w-10 items-center justify-center rounded-full'
              style={{ backgroundColor: `${getStatusColor(metric.status)}20` }}
            >
              {cloneElement(metric.icon as React.ReactElement, {
                style: { color: getStatusColor(metric.status) },
              })}
            </div>
            <div className='flex items-center space-x-1'>
              {getStatusIcon(metric.status)}
              {getTrendIcon(metric.trend, metric.status)}
            </div>
          </div>

          {/* Metric Value */}
          <div className='mb-2'>
            <div className='flex items-baseline space-x-1'>
              <span className='text-2xl font-bold' style={{ color: '#1D1D1F' }}>
                {metric.value}
              </span>
              {metric.unit && (
                <span className='text-sm font-medium' style={{ color: '#86868B' }}>
                  {metric.unit}
                </span>
              )}
            </div>
          </div>

          {/* Metric Label */}
          <div
            className='text-sm font-medium'
            style={{
              color: '#86868B',
              fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            }}
          >
            {metric.label}
          </div>

          {/* Mini Visualization Bar */}
          <div className='mt-3'>
            <div className='h-1 rounded-full' style={{ backgroundColor: '#F5F5F7' }}>
              <motion.div
                initial={{ width: 0 }}
                animate={{
                  width:
                    typeof metric.value === 'number' ? `${Math.min(100, metric.value)}%` : '75%',
                }}
                transition={{ duration: 1, delay: index * 0.1 + 0.3 }}
                className='h-full rounded-full'
                style={{ backgroundColor: getStatusColor(metric.status) }}
              />
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
});

export default SystemStatusCards;
