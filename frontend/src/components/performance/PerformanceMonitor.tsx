'use client';

import { useEffect, useState } from 'react';

interface PerformanceMetrics {
  loadTime: number;
  renderTime: number;
  interactionTime: number;
  memoryUsage: number;
  bundleSize: number;
  coreWebVitals: {
    lcp: number;
    fid: number;
    cls: number;
    fcp: number;
    ttfb: number;
  };
}

export function PerformanceMonitor() {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);

  useEffect(() => {
    // Only run in development or when explicitly enabled
    if (process.env.NODE_ENV !== 'development' && !process.env.NEXT_PUBLIC_ENABLE_PERF_MONITOR) {
      return;
    }

    const measurePerformance = () => {
      const navigation = performance.getEntriesByType(
        'navigation',
      )[0] as PerformanceNavigationTiming;
      const memory = (performance as any).memory;

      const newMetrics: PerformanceMetrics = {
        loadTime: navigation ? navigation.loadEventEnd - navigation.fetchStart : 0,
        renderTime: navigation ? navigation.domContentLoadedEventEnd - navigation.fetchStart : 0,
        interactionTime: navigation ? navigation.domInteractive - navigation.fetchStart : 0,
        memoryUsage: memory ? memory.usedJSHeapSize / 1024 / 1024 : 0,
        bundleSize: 0, // Will be updated by build process
        coreWebVitals: {
          lcp: 0,
          fid: 0,
          cls: 0,
          fcp: 0,
          ttfb: navigation ? navigation.responseStart - navigation.requestStart : 0,
        },
      };

      setMetrics(newMetrics);

      // Log performance metrics for debugging
      console.group('🚀 Performance Metrics');
      console.log('Load Time:', `${newMetrics.loadTime.toFixed(2)}ms`);
      console.log('Render Time:', `${newMetrics.renderTime.toFixed(2)}ms`);
      console.log('Interaction Time:', `${newMetrics.interactionTime.toFixed(2)}ms`);
      console.log('Memory Usage:', `${newMetrics.memoryUsage.toFixed(2)}MB`);
      console.log('TTFB:', `${newMetrics.coreWebVitals.ttfb.toFixed(2)}ms`);
      console.groupEnd();
    };

    // Measure after page load
    if (document.readyState === 'complete') {
      measurePerformance();
    } else {
      window.addEventListener('load', measurePerformance);
    }

    return () => {
      window.removeEventListener('load', measurePerformance);
    };
  }, []);

  // Don't render in production unless explicitly enabled
  if (process.env.NODE_ENV === 'production' && !process.env.NEXT_PUBLIC_ENABLE_PERF_MONITOR) {
    return null;
  }

  if (!metrics) return null;

  return (
    <div className='fixed bottom-4 right-4 z-50 max-w-xs rounded-lg bg-black/80 p-3 font-mono text-xs text-white'>
      <div className='mb-2 font-bold'>⚡ Performance</div>
      <div className='space-y-1'>
        <div>Load: {metrics.loadTime.toFixed(0)}ms</div>
        <div>Render: {metrics.renderTime.toFixed(0)}ms</div>
        <div>Memory: {metrics.memoryUsage.toFixed(1)}MB</div>
        <div>TTFB: {metrics.coreWebVitals.ttfb.toFixed(0)}ms</div>
      </div>
    </div>
  );
}

// Hook for accessing performance metrics
export function usePerformanceMetrics() {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);

  useEffect(() => {
    const measurePerformance = () => {
      const navigation = performance.getEntriesByType(
        'navigation',
      )[0] as PerformanceNavigationTiming;
      const memory = (performance as any).memory;

      const newMetrics: PerformanceMetrics = {
        loadTime: navigation ? navigation.loadEventEnd - navigation.fetchStart : 0,
        renderTime: navigation ? navigation.domContentLoadedEventEnd - navigation.fetchStart : 0,
        interactionTime: navigation ? navigation.domInteractive - navigation.fetchStart : 0,
        memoryUsage: memory ? memory.usedJSHeapSize / 1024 / 1024 : 0,
        bundleSize: 0,
        coreWebVitals: {
          lcp: 0,
          fid: 0,
          cls: 0,
          fcp: 0,
          ttfb: navigation ? navigation.responseStart - navigation.requestStart : 0,
        },
      };

      setMetrics(newMetrics);
    };

    if (document.readyState === 'complete') {
      measurePerformance();
    } else {
      window.addEventListener('load', measurePerformance);
    }

    return () => {
      window.removeEventListener('load', measurePerformance);
    };
  }, []);

  return metrics;
}

export default PerformanceMonitor;
