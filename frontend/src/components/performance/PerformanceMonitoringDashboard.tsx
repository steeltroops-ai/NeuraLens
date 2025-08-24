/**
 * Performance Monitoring Dashboard
 * Real-time performance metrics and optimization recommendations
 */

import React, { useState, useEffect, useCallback } from 'react';
import { optimizedApiClient } from '@/lib/api/optimizedApiClient';
import { frontendPerformanceOptimizer } from '@/lib/performance/frontendOptimizer';
import { comprehensiveCacheManager } from '@/lib/cache/comprehensiveCacheManager';
import {
  Activity,
  Zap,
  Database,
  Globe,
  Clock,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  BarChart3,
  Gauge,
} from 'lucide-react';

// Performance dashboard interfaces
interface PerformanceDashboardProps {
  refreshInterval?: number;
  showRecommendations?: boolean;
  className?: string;
}

interface PerformanceAlert {
  id: string;
  type: 'warning' | 'error' | 'info';
  message: string;
  timestamp: string;
  metric: string;
  value: number;
  threshold: number;
}

export function PerformanceMonitoringDashboard({
  refreshInterval = 5000,
  showRecommendations = true,
  className = '',
}: PerformanceDashboardProps) {
  const [apiMetrics, setApiMetrics] = useState(optimizedApiClient.getPerformanceSummary());
  const [frontendMetrics, setFrontendMetrics] = useState(
    frontendPerformanceOptimizer.getCurrentMetrics(),
  );
  const [cacheMetrics, setCacheMetrics] = useState(comprehensiveCacheManager.getMetrics());
  const [performanceReport, setPerformanceReport] = useState(
    frontendPerformanceOptimizer.generatePerformanceReport(),
  );
  const [alerts, setAlerts] = useState<PerformanceAlert[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(true);

  // Update metrics periodically
  useEffect(() => {
    if (!isMonitoring) return;

    const interval = setInterval(() => {
      updateAllMetrics();
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [isMonitoring, refreshInterval]);

  // Update all performance metrics
  const updateAllMetrics = useCallback(() => {
    const newApiMetrics = optimizedApiClient.getPerformanceSummary();
    const newFrontendMetrics = frontendPerformanceOptimizer.getCurrentMetrics();
    const newCacheMetrics = comprehensiveCacheManager.getMetrics();
    const newPerformanceReport = frontendPerformanceOptimizer.generatePerformanceReport();

    setApiMetrics(newApiMetrics);
    setFrontendMetrics(newFrontendMetrics);
    setCacheMetrics(newCacheMetrics);
    setPerformanceReport(newPerformanceReport);

    // Check for performance alerts
    checkPerformanceAlerts(newApiMetrics, newFrontendMetrics, newCacheMetrics);
  }, []);

  // Check for performance alerts
  const checkPerformanceAlerts = useCallback(
    (api: typeof apiMetrics, frontend: typeof frontendMetrics, cache: typeof cacheMetrics) => {
      const newAlerts: PerformanceAlert[] = [];

      // API performance alerts
      if (api.averageResponseTime > 1000) {
        newAlerts.push({
          id: `api-slow-${Date.now()}`,
          type: 'warning',
          message: 'API response times are above 1 second',
          timestamp: new Date().toISOString(),
          metric: 'API Response Time',
          value: api.averageResponseTime,
          threshold: 1000,
        });
      }

      if (api.errorRate > 5) {
        newAlerts.push({
          id: `api-errors-${Date.now()}`,
          type: 'error',
          message: 'API error rate is above 5%',
          timestamp: new Date().toISOString(),
          metric: 'API Error Rate',
          value: api.errorRate,
          threshold: 5,
        });
      }

      // Frontend performance alerts
      if (frontend.coreWebVitals.lcp > 2500) {
        newAlerts.push({
          id: `lcp-slow-${Date.now()}`,
          type: 'warning',
          message: 'Largest Contentful Paint is above 2.5 seconds',
          timestamp: new Date().toISOString(),
          metric: 'LCP',
          value: frontend.coreWebVitals.lcp,
          threshold: 2500,
        });
      }

      if (frontend.coreWebVitals.cls > 0.1) {
        newAlerts.push({
          id: `cls-high-${Date.now()}`,
          type: 'warning',
          message: 'Cumulative Layout Shift is above 0.1',
          timestamp: new Date().toISOString(),
          metric: 'CLS',
          value: frontend.coreWebVitals.cls,
          threshold: 0.1,
        });
      }

      // Cache performance alerts
      if (cache.hitRate < 70) {
        newAlerts.push({
          id: `cache-low-${Date.now()}`,
          type: 'info',
          message: 'Cache hit rate is below 70%',
          timestamp: new Date().toISOString(),
          metric: 'Cache Hit Rate',
          value: cache.hitRate,
          threshold: 70,
        });
      }

      // Memory usage alerts
      if (frontend.memoryUsage > 100) {
        newAlerts.push({
          id: `memory-high-${Date.now()}`,
          type: 'warning',
          message: 'Memory usage is above 100MB',
          timestamp: new Date().toISOString(),
          metric: 'Memory Usage',
          value: frontend.memoryUsage,
          threshold: 100,
        });
      }

      setAlerts(prev => [...newAlerts, ...prev.slice(0, 10)]); // Keep last 10 alerts
    },
    [],
  );

  // Get performance status color
  const getStatusColor = (score: number) => {
    if (score >= 90) return 'text-green-600 bg-green-100';
    if (score >= 70) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  // Get metric status icon
  const getMetricIcon = (value: number, threshold: number, higherIsBetter: boolean = true) => {
    const isGood = higherIsBetter ? value >= threshold : value <= threshold;
    return isGood ? (
      <CheckCircle className='h-4 w-4 text-green-600' />
    ) : (
      <AlertTriangle className='h-4 w-4 text-red-600' />
    );
  };

  return (
    <div className={`rounded-lg bg-white shadow-lg ${className}`}>
      {/* Header */}
      <div className='border-b border-gray-200 p-6'>
        <div className='flex items-center justify-between'>
          <div className='flex items-center gap-3'>
            <Activity className='h-6 w-6 text-blue-600' />
            <h2 className='text-xl font-semibold text-gray-900'>Performance Monitoring</h2>
          </div>

          <div className='flex items-center gap-3'>
            <div
              className={`flex items-center gap-2 rounded-full px-3 py-1 text-sm font-medium ${
                isMonitoring ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-700'
              }`}
            >
              <div
                className={`h-2 w-2 rounded-full ${isMonitoring ? 'bg-green-600' : 'bg-gray-400'}`}
              />
              {isMonitoring ? 'Live' : 'Paused'}
            </div>

            <button
              onClick={() => setIsMonitoring(!isMonitoring)}
              className='rounded-lg border border-blue-600 px-4 py-2 text-sm font-medium text-blue-600 transition-colors hover:bg-blue-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
            >
              {isMonitoring ? 'Pause' : 'Resume'}
            </button>
          </div>
        </div>
      </div>

      <div className='space-y-6 p-6'>
        {/* Overall Performance Score */}
        <div className='grid grid-cols-1 gap-4 md:grid-cols-4'>
          <div className='col-span-1 md:col-span-2'>
            <div className='rounded-lg bg-gradient-to-br from-blue-50 to-indigo-50 p-6 text-center'>
              <div className='mb-4 flex items-center justify-center'>
                <Gauge className='h-8 w-8 text-blue-600' />
              </div>
              <div className='mb-2 text-4xl font-bold text-gray-900'>{performanceReport.score}</div>
              <div className='mb-4 text-sm text-gray-600'>Overall Performance Score</div>
              <div
                className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-medium ${getStatusColor(
                  performanceReport.score,
                )}`}
              >
                Grade: {performanceReport.coreWebVitalsGrade}
              </div>
            </div>
          </div>

          <div className='space-y-4'>
            <div className='rounded-lg bg-gray-50 p-4'>
              <div className='mb-2 flex items-center justify-between'>
                <span className='text-sm font-medium text-gray-700'>API Performance</span>
                {getMetricIcon(apiMetrics.averageResponseTime, 200, false)}
              </div>
              <div className='text-2xl font-bold text-gray-900'>
                {apiMetrics.averageResponseTime.toFixed(0)}ms
              </div>
              <div className='text-xs text-gray-500'>Average Response Time</div>
            </div>

            <div className='rounded-lg bg-gray-50 p-4'>
              <div className='mb-2 flex items-center justify-between'>
                <span className='text-sm font-medium text-gray-700'>Cache Hit Rate</span>
                {getMetricIcon(cacheMetrics.hitRate, 80, true)}
              </div>
              <div className='text-2xl font-bold text-gray-900'>
                {cacheMetrics.hitRate.toFixed(1)}%
              </div>
              <div className='text-xs text-gray-500'>Cache Efficiency</div>
            </div>
          </div>

          <div className='space-y-4'>
            <div className='rounded-lg bg-gray-50 p-4'>
              <div className='mb-2 flex items-center justify-between'>
                <span className='text-sm font-medium text-gray-700'>Page Load Time</span>
                {getMetricIcon(frontendMetrics.loadTime, 3000, false)}
              </div>
              <div className='text-2xl font-bold text-gray-900'>
                {(frontendMetrics.loadTime / 1000).toFixed(1)}s
              </div>
              <div className='text-xs text-gray-500'>Total Load Time</div>
            </div>

            <div className='rounded-lg bg-gray-50 p-4'>
              <div className='mb-2 flex items-center justify-between'>
                <span className='text-sm font-medium text-gray-700'>Memory Usage</span>
                {getMetricIcon(frontendMetrics.memoryUsage, 50, false)}
              </div>
              <div className='text-2xl font-bold text-gray-900'>
                {frontendMetrics.memoryUsage.toFixed(1)}MB
              </div>
              <div className='text-xs text-gray-500'>JavaScript Heap</div>
            </div>
          </div>
        </div>

        {/* Core Web Vitals */}
        <div className='rounded-lg bg-gray-50 p-6'>
          <h3 className='mb-4 flex items-center gap-2 text-lg font-semibold text-gray-900'>
            <TrendingUp className='h-5 w-5' />
            Core Web Vitals
          </h3>

          <div className='grid grid-cols-1 gap-4 md:grid-cols-3'>
            <div className='rounded-lg bg-white p-4 text-center'>
              <div className='mb-1 text-2xl font-bold text-gray-900'>
                {(frontendMetrics.coreWebVitals.lcp / 1000).toFixed(2)}s
              </div>
              <div className='mb-2 text-sm font-medium text-gray-700'>Largest Contentful Paint</div>
              <div
                className={`inline-flex rounded px-2 py-1 text-xs font-medium ${
                  frontendMetrics.coreWebVitals.lcp <= 2500
                    ? 'bg-green-100 text-green-700'
                    : 'bg-red-100 text-red-700'
                }`}
              >
                Target: ≤2.5s
              </div>
            </div>

            <div className='rounded-lg bg-white p-4 text-center'>
              <div className='mb-1 text-2xl font-bold text-gray-900'>
                {frontendMetrics.coreWebVitals.fid.toFixed(0)}ms
              </div>
              <div className='mb-2 text-sm font-medium text-gray-700'>First Input Delay</div>
              <div
                className={`inline-flex rounded px-2 py-1 text-xs font-medium ${
                  frontendMetrics.coreWebVitals.fid <= 100
                    ? 'bg-green-100 text-green-700'
                    : 'bg-red-100 text-red-700'
                }`}
              >
                Target: ≤100ms
              </div>
            </div>

            <div className='rounded-lg bg-white p-4 text-center'>
              <div className='mb-1 text-2xl font-bold text-gray-900'>
                {frontendMetrics.coreWebVitals.cls.toFixed(3)}
              </div>
              <div className='mb-2 text-sm font-medium text-gray-700'>Cumulative Layout Shift</div>
              <div
                className={`inline-flex rounded px-2 py-1 text-xs font-medium ${
                  frontendMetrics.coreWebVitals.cls <= 0.1
                    ? 'bg-green-100 text-green-700'
                    : 'bg-red-100 text-red-700'
                }`}
              >
                Target: ≤0.1
              </div>
            </div>
          </div>
        </div>

        {/* Performance Alerts */}
        {alerts.length > 0 && (
          <div className='rounded-lg border border-yellow-200 bg-yellow-50 p-4'>
            <h3 className='mb-3 flex items-center gap-2 text-lg font-semibold text-yellow-800'>
              <AlertTriangle className='h-5 w-5' />
              Performance Alerts
            </h3>

            <div className='space-y-2'>
              {alerts.slice(0, 5).map(alert => (
                <div
                  key={alert.id}
                  className={`flex items-center gap-3 rounded-lg p-3 ${
                    alert.type === 'error'
                      ? 'bg-red-100 text-red-800'
                      : alert.type === 'warning'
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-blue-100 text-blue-800'
                  }`}
                >
                  <AlertTriangle className='h-4 w-4 flex-shrink-0' />
                  <div className='flex-1'>
                    <div className='font-medium'>{alert.message}</div>
                    <div className='text-sm opacity-75'>
                      {alert.metric}: {alert.value.toFixed(1)} (threshold: {alert.threshold})
                    </div>
                  </div>
                  <div className='text-xs opacity-75'>
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Performance Recommendations */}
        {showRecommendations && performanceReport.recommendations.length > 0 && (
          <div className='rounded-lg border border-blue-200 bg-blue-50 p-4'>
            <h3 className='mb-3 flex items-center gap-2 text-lg font-semibold text-blue-800'>
              <Zap className='h-5 w-5' />
              Optimization Recommendations
            </h3>

            <ul className='space-y-2'>
              {performanceReport.recommendations.map((recommendation, index) => (
                <li key={index} className='flex items-start gap-3 text-blue-800'>
                  <CheckCircle className='mt-0.5 h-4 w-4 flex-shrink-0' />
                  <span className='text-sm'>{recommendation}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Detailed Metrics */}
        <div className='grid grid-cols-1 gap-6 lg:grid-cols-3'>
          {/* API Metrics */}
          <div className='rounded-lg bg-gray-50 p-4'>
            <h4 className='mb-3 flex items-center gap-2 font-semibold text-gray-900'>
              <Globe className='h-4 w-4' />
              API Performance
            </h4>
            <div className='space-y-3'>
              <div className='flex justify-between'>
                <span className='text-sm text-gray-600'>Total Requests</span>
                <span className='font-medium'>{apiMetrics.totalRequests}</span>
              </div>
              <div className='flex justify-between'>
                <span className='text-sm text-gray-600'>Cache Hit Rate</span>
                <span className='font-medium'>{apiMetrics.cacheHitRate.toFixed(1)}%</span>
              </div>
              <div className='flex justify-between'>
                <span className='text-sm text-gray-600'>Error Rate</span>
                <span className='font-medium'>{apiMetrics.errorRate.toFixed(1)}%</span>
              </div>
              <div className='flex justify-between'>
                <span className='text-sm text-gray-600'>Slow Requests</span>
                <span className='font-medium'>{apiMetrics.slowRequestCount}</span>
              </div>
            </div>
          </div>

          {/* Cache Metrics */}
          <div className='rounded-lg bg-gray-50 p-4'>
            <h4 className='mb-3 flex items-center gap-2 font-semibold text-gray-900'>
              <Database className='h-4 w-4' />
              Cache Performance
            </h4>
            <div className='space-y-3'>
              <div className='flex justify-between'>
                <span className='text-sm text-gray-600'>Hit Rate</span>
                <span className='font-medium'>{cacheMetrics.hitRate.toFixed(1)}%</span>
              </div>
              <div className='flex justify-between'>
                <span className='text-sm text-gray-600'>Miss Rate</span>
                <span className='font-medium'>{cacheMetrics.missRate.toFixed(1)}%</span>
              </div>
              <div className='flex justify-between'>
                <span className='text-sm text-gray-600'>Cache Size</span>
                <span className='font-medium'>
                  {(cacheMetrics.cacheSize / 1024 / 1024).toFixed(1)}MB
                </span>
              </div>
              <div className='flex justify-between'>
                <span className='text-sm text-gray-600'>Evictions</span>
                <span className='font-medium'>{cacheMetrics.evictionCount}</span>
              </div>
            </div>
          </div>

          {/* Frontend Metrics */}
          <div className='rounded-lg bg-gray-50 p-4'>
            <h4 className='mb-3 flex items-center gap-2 font-semibold text-gray-900'>
              <BarChart3 className='h-4 w-4' />
              Frontend Performance
            </h4>
            <div className='space-y-3'>
              <div className='flex justify-between'>
                <span className='text-sm text-gray-600'>TTFB</span>
                <span className='font-medium'>
                  {frontendMetrics.coreWebVitals.ttfb.toFixed(0)}ms
                </span>
              </div>
              <div className='flex justify-between'>
                <span className='text-sm text-gray-600'>FCP</span>
                <span className='font-medium'>
                  {(frontendMetrics.coreWebVitals.fcp / 1000).toFixed(2)}s
                </span>
              </div>
              <div className='flex justify-between'>
                <span className='text-sm text-gray-600'>Render Time</span>
                <span className='font-medium'>
                  {(frontendMetrics.renderTime / 1000).toFixed(2)}s
                </span>
              </div>
              <div className='flex justify-between'>
                <span className='text-sm text-gray-600'>Interaction Time</span>
                <span className='font-medium'>
                  {(frontendMetrics.interactionTime / 1000).toFixed(2)}s
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
