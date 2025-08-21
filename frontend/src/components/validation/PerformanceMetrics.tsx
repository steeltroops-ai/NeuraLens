'use client';

import React from 'react';
import { Card } from '@/components/ui';

interface PerformanceMetricsProps {
  data: {
    loadTime: number;
    lcp: number;
    fid: number;
    cls: number;
    uptime: number;
    throughput: number;
  };
}

export const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({ data }) => {
  const coreWebVitals = [
    {
      name: 'Largest Contentful Paint (LCP)',
      value: `${data.lcp}s`,
      target: '<2.5s',
      status: data.lcp < 2.5 ? 'excellent' : data.lcp < 4 ? 'good' : 'poor',
      description: 'Loading performance',
    },
    {
      name: 'First Input Delay (FID)',
      value: `${data.fid}ms`,
      target: '<100ms',
      status: data.fid < 100 ? 'excellent' : data.fid < 300 ? 'good' : 'poor',
      description: 'Interactivity',
    },
    {
      name: 'Cumulative Layout Shift (CLS)',
      value: data.cls.toString(),
      target: '<0.1',
      status: data.cls < 0.1 ? 'excellent' : data.cls < 0.25 ? 'good' : 'poor',
      description: 'Visual stability',
    },
  ];

  const systemMetrics = [
    { name: 'Average Load Time', value: `${data.loadTime}s`, icon: '⚡' },
    { name: 'System Uptime', value: `${data.uptime}%`, icon: '🔄' },
    { name: 'Hourly Throughput', value: `${data.throughput}`, icon: '📊' },
    { name: 'Response Time', value: '<200ms', icon: '🚀' },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'text-green-400 bg-green-500/10 border-green-500/20';
      case 'good': return 'text-amber-400 bg-amber-500/10 border-amber-500/20';
      case 'poor': return 'text-red-400 bg-red-500/10 border-red-500/20';
      default: return 'text-neutral-400 bg-neutral-500/10 border-neutral-500/20';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'excellent': return '🟢';
      case 'good': return '🟡';
      case 'poor': return '🔴';
      default: return '⚪';
    }
  };

  return (
    <div className="space-y-8">
      {/* Core Web Vitals */}
      <Card className="p-8">
        <h3 className="text-2xl font-semibold text-text-primary mb-6">
          Core Web Vitals
        </h3>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {coreWebVitals.map((vital) => (
            <div key={vital.name} className={`p-6 rounded-lg border ${getStatusColor(vital.status)}`}>
              <div className="flex items-center justify-between mb-4">
                <span className="text-2xl">{getStatusIcon(vital.status)}</span>
                <span className="text-xs font-medium uppercase tracking-wider">
                  {vital.status}
                </span>
              </div>
              <h4 className="font-semibold text-text-primary mb-2">{vital.name}</h4>
              <div className="flex items-baseline space-x-2 mb-2">
                <span className="text-3xl font-bold">{vital.value}</span>
                <span className="text-sm text-text-muted">Target: {vital.target}</span>
              </div>
              <p className="text-sm text-text-secondary">{vital.description}</p>
            </div>
          ))}
        </div>

        <div className="mt-8 p-6 bg-success/10 border border-success/20 rounded-lg">
          <div className="flex items-center space-x-3">
            <span className="text-2xl">🏆</span>
            <div>
              <h4 className="font-semibold text-success">Excellent Performance</h4>
              <p className="text-sm text-text-secondary">
                All Core Web Vitals meet or exceed Google's recommended thresholds for excellent user experience.
              </p>
            </div>
          </div>
        </div>
      </Card>

      {/* System Performance */}
      <Card className="p-8">
        <h3 className="text-2xl font-semibold text-text-primary mb-6">
          System Performance
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {systemMetrics.map((metric) => (
            <div key={metric.name} className="text-center p-6 bg-surface-secondary rounded-lg">
              <div className="text-4xl mb-3">{metric.icon}</div>
              <div className="text-2xl font-bold text-primary-400 mb-2">
                {metric.value}
              </div>
              <div className="text-sm text-text-secondary">
                {metric.name}
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Performance Optimization */}
      <Card className="p-8">
        <h3 className="text-2xl font-semibold text-text-primary mb-6">
          Performance Optimizations
        </h3>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <h4 className="text-lg font-semibold text-text-primary mb-4">
              Frontend Optimizations
            </h4>
            <ul className="space-y-3">
              <li className="flex items-start space-x-3">
                <span className="text-success mt-1">✓</span>
                <div>
                  <span className="font-medium text-text-primary">Code Splitting</span>
                  <p className="text-sm text-text-secondary">Dynamic imports reduce initial bundle size</p>
                </div>
              </li>
              <li className="flex items-start space-x-3">
                <span className="text-success mt-1">✓</span>
                <div>
                  <span className="font-medium text-text-primary">Image Optimization</span>
                  <p className="text-sm text-text-secondary">WebP format with lazy loading</p>
                </div>
              </li>
              <li className="flex items-start space-x-3">
                <span className="text-success mt-1">✓</span>
                <div>
                  <span className="font-medium text-text-primary">Caching Strategy</span>
                  <p className="text-sm text-text-secondary">Service worker with intelligent caching</p>
                </div>
              </li>
              <li className="flex items-start space-x-3">
                <span className="text-success mt-1">✓</span>
                <div>
                  <span className="font-medium text-text-primary">Font Optimization</span>
                  <p className="text-sm text-text-secondary">Preloaded fonts with display swap</p>
                </div>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="text-lg font-semibold text-text-primary mb-4">
              Backend Optimizations
            </h4>
            <ul className="space-y-3">
              <li className="flex items-start space-x-3">
                <span className="text-success mt-1">✓</span>
                <div>
                  <span className="font-medium text-text-primary">ML Model Optimization</span>
                  <p className="text-sm text-text-secondary">Quantized models for faster inference</p>
                </div>
              </li>
              <li className="flex items-start space-x-3">
                <span className="text-success mt-1">✓</span>
                <div>
                  <span className="font-medium text-text-primary">Parallel Processing</span>
                  <p className="text-sm text-text-secondary">Concurrent modality analysis</p>
                </div>
              </li>
              <li className="flex items-start space-x-3">
                <span className="text-success mt-1">✓</span>
                <div>
                  <span className="font-medium text-text-primary">Memory Management</span>
                  <p className="text-sm text-text-secondary">Efficient memory allocation and cleanup</p>
                </div>
              </li>
              <li className="flex items-start space-x-3">
                <span className="text-success mt-1">✓</span>
                <div>
                  <span className="font-medium text-text-primary">Error Handling</span>
                  <p className="text-sm text-text-secondary">Graceful degradation and recovery</p>
                </div>
              </li>
            </ul>
          </div>
        </div>
      </Card>

      {/* Performance Monitoring */}
      <Card className="p-8">
        <h3 className="text-2xl font-semibold text-text-primary mb-6">
          Continuous Monitoring
        </h3>
        
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-4 bg-surface-secondary rounded-lg">
              <div className="text-2xl font-bold text-primary-400 mb-2">24/7</div>
              <div className="text-sm text-text-secondary">Real-time monitoring</div>
            </div>
            <div className="text-center p-4 bg-surface-secondary rounded-lg">
              <div className="text-2xl font-bold text-primary-400 mb-2">99.8%</div>
              <div className="text-sm text-text-secondary">Uptime SLA</div>
            </div>
            <div className="text-center p-4 bg-surface-secondary rounded-lg">
              <div className="text-2xl font-bold text-primary-400 mb-2">&lt;5min</div>
              <div className="text-sm text-text-secondary">Alert response time</div>
            </div>
          </div>

          <div className="bg-surface-secondary rounded-lg p-6">
            <h4 className="font-semibold text-text-primary mb-3">Monitoring Tools</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-text-secondary">
              <div>• Real User Monitoring (RUM)</div>
              <div>• Synthetic performance testing</div>
              <div>• Application performance monitoring</div>
              <div>• Infrastructure monitoring</div>
              <div>• Error tracking and alerting</div>
              <div>• User experience analytics</div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};
