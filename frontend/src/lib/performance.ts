/**
 * NeuroLens-X Performance Monitoring & Optimization
 * Core Web Vitals monitoring and performance optimization utilities
 * Designed for medical application performance requirements
 */

// Core Web Vitals thresholds
export const PERFORMANCE_THRESHOLDS = {
  // Largest Contentful Paint (LCP)
  LCP: {
    GOOD: 2500,
    NEEDS_IMPROVEMENT: 4000,
  },
  // First Input Delay (FID)
  FID: {
    GOOD: 100,
    NEEDS_IMPROVEMENT: 300,
  },
  // Cumulative Layout Shift (CLS)
  CLS: {
    GOOD: 0.1,
    NEEDS_IMPROVEMENT: 0.25,
  },
  // Time to First Byte (TTFB)
  TTFB: {
    GOOD: 800,
    NEEDS_IMPROVEMENT: 1800,
  },
  // First Contentful Paint (FCP)
  FCP: {
    GOOD: 1800,
    NEEDS_IMPROVEMENT: 3000,
  },
};

// Performance metric interface
export interface PerformanceMetric {
  name: string;
  value: number;
  rating: 'good' | 'needs-improvement' | 'poor';
  timestamp: number;
  url: string;
  id: string;
}

// Performance observer class
export class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private metrics: Map<string, PerformanceMetric[]> = new Map();
  private observers: PerformanceObserver[] = [];
  private analyticsEndpoint: string | null = null;

  private constructor() {
    this.initializeObservers();
  }

  public static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }

  // Initialize performance observers
  private initializeObservers(): void {
    if (typeof window === 'undefined') return;

    // Largest Contentful Paint (LCP)
    this.observeMetric('largest-contentful-paint', (entries) => {
      const lastEntry = entries[entries.length - 1];
      if (lastEntry) {
        this.recordMetric({
          name: 'LCP',
          value: lastEntry.startTime,
          rating: this.getRating('LCP', lastEntry.startTime),
          timestamp: Date.now(),
          url: window.location.href,
          id: this.generateId(),
        });
      }
    });

    // First Input Delay (FID)
    this.observeMetric('first-input', (entries) => {
      const firstEntry = entries[0];
      if (firstEntry) {
        this.recordMetric({
          name: 'FID',
          value: firstEntry.processingStart - firstEntry.startTime,
          rating: this.getRating('FID', firstEntry.processingStart - firstEntry.startTime),
          timestamp: Date.now(),
          url: window.location.href,
          id: this.generateId(),
        });
      }
    });

    // Cumulative Layout Shift (CLS)
    let clsValue = 0;
    this.observeMetric('layout-shift', (entries) => {
      for (const entry of entries) {
        if (!entry.hadRecentInput) {
          clsValue += entry.value;
        }
      }
      
      this.recordMetric({
        name: 'CLS',
        value: clsValue,
        rating: this.getRating('CLS', clsValue),
        timestamp: Date.now(),
        url: window.location.href,
        id: this.generateId(),
      });
    });

    // Navigation timing
    this.observeNavigationTiming();

    // Resource timing
    this.observeResourceTiming();
  }

  // Observe specific performance metrics
  private observeMetric(type: string, callback: (entries: any[]) => void): void {
    try {
      const observer = new PerformanceObserver((list) => {
        callback(list.getEntries());
      });
      
      observer.observe({ type, buffered: true });
      this.observers.push(observer);
    } catch (error) {
      console.warn(`Performance observer for ${type} not supported:`, error);
    }
  }

  // Observe navigation timing
  private observeNavigationTiming(): void {
    if (typeof window === 'undefined' || !window.performance?.getEntriesByType) return;

    const navigationEntries = window.performance.getEntriesByType('navigation') as PerformanceNavigationTiming[];
    
    if (navigationEntries.length > 0) {
      const entry = navigationEntries[0];
      
      // Time to First Byte (TTFB)
      const ttfb = entry.responseStart - entry.requestStart;
      this.recordMetric({
        name: 'TTFB',
        value: ttfb,
        rating: this.getRating('TTFB', ttfb),
        timestamp: Date.now(),
        url: window.location.href,
        id: this.generateId(),
      });

      // DOM Content Loaded
      const dcl = entry.domContentLoadedEventEnd - entry.navigationStart;
      this.recordMetric({
        name: 'DCL',
        value: dcl,
        rating: dcl < 1500 ? 'good' : dcl < 3000 ? 'needs-improvement' : 'poor',
        timestamp: Date.now(),
        url: window.location.href,
        id: this.generateId(),
      });

      // Load Complete
      const loadComplete = entry.loadEventEnd - entry.navigationStart;
      this.recordMetric({
        name: 'Load',
        value: loadComplete,
        rating: loadComplete < 2500 ? 'good' : loadComplete < 4000 ? 'needs-improvement' : 'poor',
        timestamp: Date.now(),
        url: window.location.href,
        id: this.generateId(),
      });
    }
  }

  // Observe resource timing
  private observeResourceTiming(): void {
    if (typeof window === 'undefined' || !window.performance?.getEntriesByType) return;

    const resourceEntries = window.performance.getEntriesByType('resource') as PerformanceResourceTiming[];
    
    // Analyze critical resources
    const criticalResources = resourceEntries.filter(entry => 
      entry.name.includes('.js') || 
      entry.name.includes('.css') || 
      entry.name.includes('/api/')
    );

    for (const resource of criticalResources) {
      const loadTime = resource.responseEnd - resource.startTime;
      
      this.recordMetric({
        name: 'Resource',
        value: loadTime,
        rating: loadTime < 500 ? 'good' : loadTime < 1000 ? 'needs-improvement' : 'poor',
        timestamp: Date.now(),
        url: resource.name,
        id: this.generateId(),
      });
    }
  }

  // Record performance metric
  private recordMetric(metric: PerformanceMetric): void {
    if (!this.metrics.has(metric.name)) {
      this.metrics.set(metric.name, []);
    }
    
    const metricArray = this.metrics.get(metric.name)!;
    metricArray.push(metric);
    
    // Keep only last 100 measurements per metric
    if (metricArray.length > 100) {
      metricArray.shift();
    }

    // Send to analytics if configured
    this.sendToAnalytics(metric);

    // Log performance issues
    if (metric.rating === 'poor') {
      console.warn(`Poor performance detected: ${metric.name} = ${metric.value}ms`);
    }
  }

  // Get performance rating
  private getRating(metricName: string, value: number): 'good' | 'needs-improvement' | 'poor' {
    const thresholds = PERFORMANCE_THRESHOLDS[metricName as keyof typeof PERFORMANCE_THRESHOLDS];
    
    if (!thresholds) return 'good';
    
    if (value <= thresholds.GOOD) return 'good';
    if (value <= thresholds.NEEDS_IMPROVEMENT) return 'needs-improvement';
    return 'poor';
  }

  // Generate unique ID
  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  // Send metric to analytics
  private sendToAnalytics(metric: PerformanceMetric): void {
    if (!this.analyticsEndpoint) return;

    // Use sendBeacon for reliability
    if (navigator.sendBeacon) {
      navigator.sendBeacon(
        this.analyticsEndpoint,
        JSON.stringify(metric)
      );
    } else {
      // Fallback to fetch
      fetch(this.analyticsEndpoint, {
        method: 'POST',
        body: JSON.stringify(metric),
        headers: {
          'Content-Type': 'application/json',
        },
        keepalive: true,
      }).catch(error => {
        console.warn('Failed to send performance metric:', error);
      });
    }
  }

  // Public methods
  public setAnalyticsEndpoint(endpoint: string): void {
    this.analyticsEndpoint = endpoint;
  }

  public getMetrics(metricName?: string): PerformanceMetric[] {
    if (metricName) {
      return this.metrics.get(metricName) || [];
    }
    
    const allMetrics: PerformanceMetric[] = [];
    for (const metrics of this.metrics.values()) {
      allMetrics.push(...metrics);
    }
    
    return allMetrics.sort((a, b) => b.timestamp - a.timestamp);
  }

  public getAverageMetric(metricName: string): number {
    const metrics = this.metrics.get(metricName) || [];
    if (metrics.length === 0) return 0;
    
    return metrics.reduce((sum, metric) => sum + metric.value, 0) / metrics.length;
  }

  public getPerformanceScore(): number {
    const lcpScore = this.getMetricScore('LCP');
    const fidScore = this.getMetricScore('FID');
    const clsScore = this.getMetricScore('CLS');
    
    // Weighted average (LCP: 25%, FID: 25%, CLS: 25%, others: 25%)
    return Math.round((lcpScore + fidScore + clsScore) / 3);
  }

  private getMetricScore(metricName: string): number {
    const metrics = this.metrics.get(metricName) || [];
    if (metrics.length === 0) return 100;
    
    const latestMetric = metrics[metrics.length - 1];
    
    switch (latestMetric.rating) {
      case 'good': return 100;
      case 'needs-improvement': return 75;
      case 'poor': return 50;
      default: return 100;
    }
  }

  public disconnect(): void {
    this.observers.forEach(observer => observer.disconnect());
    this.observers = [];
  }
}

// Performance optimization utilities
export class PerformanceOptimizer {
  // Preload critical resources
  public static preloadResource(href: string, as: string, crossorigin?: string): void {
    if (typeof document === 'undefined') return;

    const link = document.createElement('link');
    link.rel = 'preload';
    link.href = href;
    link.as = as;
    if (crossorigin) link.crossOrigin = crossorigin;
    
    document.head.appendChild(link);
  }

  // Prefetch resources for next navigation
  public static prefetchResource(href: string): void {
    if (typeof document === 'undefined') return;

    const link = document.createElement('link');
    link.rel = 'prefetch';
    link.href = href;
    
    document.head.appendChild(link);
  }

  // Lazy load images with intersection observer
  public static lazyLoadImages(): void {
    if (typeof window === 'undefined' || !('IntersectionObserver' in window)) return;

    const images = document.querySelectorAll('img[data-src]');
    
    const imageObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target as HTMLImageElement;
          img.src = img.dataset.src!;
          img.removeAttribute('data-src');
          imageObserver.unobserve(img);
        }
      });
    });

    images.forEach(img => imageObserver.observe(img));
  }

  // Optimize font loading
  public static optimizeFontLoading(): void {
    if (typeof document === 'undefined') return;

    // Preload critical fonts
    const criticalFonts = [
      '/fonts/inter-var.woff2',
    ];

    criticalFonts.forEach(font => {
      this.preloadResource(font, 'font', 'anonymous');
    });
  }

  // Measure and optimize bundle size
  public static measureBundleSize(): void {
    if (typeof window === 'undefined' || !window.performance?.getEntriesByType) return;

    const resources = window.performance.getEntriesByType('resource') as PerformanceResourceTiming[];
    const jsResources = resources.filter(r => r.name.includes('.js'));
    const cssResources = resources.filter(r => r.name.includes('.css'));

    const totalJSSize = jsResources.reduce((sum, r) => sum + (r.transferSize || 0), 0);
    const totalCSSSize = cssResources.reduce((sum, r) => sum + (r.transferSize || 0), 0);

    console.log('Bundle Analysis:', {
      totalJS: `${(totalJSSize / 1024).toFixed(2)} KB`,
      totalCSS: `${(totalCSSSize / 1024).toFixed(2)} KB`,
      jsFiles: jsResources.length,
      cssFiles: cssResources.length,
    });
  }
}

// Initialize performance monitoring
export function initializePerformanceMonitoring(): PerformanceMonitor {
  const monitor = PerformanceMonitor.getInstance();
  
  // Set analytics endpoint if available
  if (process.env.NEXT_PUBLIC_ANALYTICS_ENDPOINT) {
    monitor.setAnalyticsEndpoint(process.env.NEXT_PUBLIC_ANALYTICS_ENDPOINT);
  }
  
  return monitor;
}

// Export singleton instance
export const performanceMonitor = PerformanceMonitor.getInstance();
