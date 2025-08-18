'use client';

import React from 'react';
import { Card } from '@/components/ui';

export const SystemHealth: React.FC = () => {
  const healthMetrics = [
    { name: 'API Response Time', value: '145ms', status: 'excellent', target: '<200ms' },
    { name: 'System Uptime', value: '99.8%', status: 'excellent', target: '>99.5%' },
    { name: 'Error Rate', value: '0.12%', status: 'good', target: '<0.5%' },
    { name: 'Memory Usage', value: '68%', status: 'good', target: '<80%' },
    { name: 'CPU Usage', value: '42%', status: 'excellent', target: '<70%' },
    { name: 'Storage Usage', value: '34%', status: 'excellent', target: '<80%' },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'text-green-400';
      case 'good': return 'text-amber-400';
      case 'warning': return 'text-orange-400';
      case 'critical': return 'text-red-400';
      default: return 'text-neutral-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'excellent': return '🟢';
      case 'good': return '🟡';
      case 'warning': return '🟠';
      case 'critical': return '🔴';
      default: return '⚪';
    }
  };

  return (
    <Card className="p-8">
      <h3 className="text-2xl font-semibold text-text-primary mb-6">
        System Health Status
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {healthMetrics.map((metric) => (
          <div key={metric.name} className="bg-surface-secondary rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-text-secondary">{metric.name}</span>
              <span className="text-lg">{getStatusIcon(metric.status)}</span>
            </div>
            <div className="flex items-baseline space-x-2">
              <span className={`text-2xl font-bold ${getStatusColor(metric.status)}`}>
                {metric.value}
              </span>
              <span className="text-xs text-text-muted">
                Target: {metric.target}
              </span>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 p-4 bg-success/10 border border-success/20 rounded-lg">
        <div className="flex items-center space-x-3">
          <span className="text-2xl">✅</span>
          <div>
            <h4 className="font-semibold text-success">All Systems Operational</h4>
            <p className="text-sm text-text-secondary">
              All critical systems are functioning within normal parameters. 
              Last updated: {new Date().toLocaleTimeString()}
            </p>
          </div>
        </div>
      </div>
    </Card>
  );
};
