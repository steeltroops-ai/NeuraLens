/**
 * Frontend Performance Optimizer
 * Comprehensive frontend optimization for Core Web Vitals and Lighthouse scores
 */

import React from 'react';

// Performance monitoring interfaces
export interface CoreWebVitals {
  lcp: number; // Largest Contentful Paint
  fid: number; // First Input Delay
  cls: number; // Cumulative Layout Shift
  fcp: number; // First Contentful Paint
  ttfb: number; // Time to First Byte
}

export interface PerformanceMetrics {
  coreWebVitals: CoreWebVitals;
  bundleSize: number;
  loadTime: number;
  renderTime: number;
  interactionTime: number;
  memoryUsage: number;
  timestamp: string;
}

export interface OptimizationConfig {
  enableCodeSplitting: boolean;
  enableLazyLoading: boolean;
  enableImageOptimization: boolean;
  enableServiceWorker: boolean;
  enableVirtualScrolling: boolean;
  enableMemoization: boolean;
  targetLighthouseScore: number;
  maxBundleSize: number; // in KB
}

// Frontend performance optimizer class
export class FrontendPerformanceOptimizer {
  private config: OptimizationConfig;
  private performanceObserver: PerformanceObserver | null = null;
  private metrics: PerformanceMetrics[] = [];
  private intersectionObserver: IntersectionObserver | null = null;

  constructor(config: Partial<OptimizationConfig> = {}) {
    this.config = {
      enableCodeSplitting: true,
      enableLazyLoading: true,
      enableImageOptimization: true,
      enableServiceWorker: true,
      enableVirtualScrolling: true,
      enableMemoization: true,
      targetLighthouseScore: 95,
      maxBundleSize: 250, // 250KB
      ...config,
    };

    this.initializePerformanceMonitoring();
  }

  /**
   * Initialize performance monitoring
   */
  private initializePerformanceMonitoring(): void {
    if (typeof window === 'undefined') return;

    // Monitor Core Web Vitals
    this.setupCoreWebVitalsMonitoring();

    // Monitor resource loading
    this.setupResourceMonitoring();

    // Monitor memory usage
    this.setupMemoryMonitoring();
  }

  /**
   * Setup Core Web Vitals monitoring
   */
  private setupCoreWebVitalsMonitoring(): void {
    if (!('PerformanceObserver' in window)) return;

    // Monitor LCP (Largest Contentful Paint)
    const lcpObserver = new PerformanceObserver(list => {
      const entries = list.getEntries();
      const lastEntry = entries[entries.length - 1] as any;
      this.updateCoreWebVital('lcp', lastEntry.startTime);
    });
    lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });

    // Monitor FID (First Input Delay)
    const fidObserver = new PerformanceObserver(list => {
      const entries = list.getEntries();
      entries.forEach((entry: any) => {
        this.updateCoreWebVital('fid', entry.processingStart - entry.startTime);
      });
    });
    fidObserver.observe({ entryTypes: ['first-input'] });

    // Monitor CLS (Cumulative Layout Shift)
    let clsValue = 0;
    const clsObserver = new PerformanceObserver(list => {
      const entries = list.getEntries();
      entries.forEach((entry: any) => {
        if (!entry.hadRecentInput) {
          clsValue += entry.value;
          this.updateCoreWebVital('cls', clsValue);
        }
      });
    });
    clsObserver.observe({ entryTypes: ['layout-shift'] });

    // Monitor FCP (First Contentful Paint)
    const fcpObserver = new PerformanceObserver(list => {
      const entries = list.getEntries();
      entries.forEach((entry: any) => {
        if (entry.name === 'first-contentful-paint') {
          this.updateCoreWebVital('fcp', entry.startTime);
        }
      });
    });
    fcpObserver.observe({ entryTypes: ['paint'] });
  }

  /**
   * Setup resource monitoring
   */
  private setupResourceMonitoring(): void {
    if (!('PerformanceObserver' in window)) return;

    const resourceObserver = new PerformanceObserver(list => {
      const entries = list.getEntries();
      entries.forEach((entry: any) => {
        // Monitor slow resources
        if (entry.duration > 1000) {
          console.warn(`Slow resource detected: ${entry.name} took ${entry.duration.toFixed(2)}ms`);
        }

        // Monitor large resources
        if (entry.transferSize > 500000) {
          // 500KB
          console.warn(
            `Large resource detected: ${entry.name} is ${(entry.transferSize / 1024).toFixed(2)}KB`,
          );
        }
      });
    });
    resourceObserver.observe({ entryTypes: ['resource'] });
  }

  /**
   * Setup memory monitoring
   */
  private setupMemoryMonitoring(): void {
    if (!('memory' in performance)) return;

    setInterval(() => {
      const memory = (performance as any).memory;
      if (memory) {
        const memoryUsage = memory.usedJSHeapSize / 1024 / 1024; // MB

        // Warn if memory usage is high
        if (memoryUsage > 100) {
          console.warn(`High memory usage detected: ${memoryUsage.toFixed(2)}MB`);
        }
      }
    }, 30000); // Check every 30 seconds
  }

  /**
   * Update Core Web Vital metric
   */
  private updateCoreWebVital(metric: keyof CoreWebVitals, value: number): void {
    const currentMetrics = this.getCurrentMetrics();
    currentMetrics.coreWebVitals[metric] = value;

    // Check if metrics meet targets
    this.validateCoreWebVitals(currentMetrics.coreWebVitals);
  }

  /**
   * Validate Core Web Vitals against targets
   */
  private validateCoreWebVitals(vitals: CoreWebVitals): void {
    const issues: string[] = [];

    if (vitals.lcp > 2500) {
      issues.push(`LCP too high: ${vitals.lcp.toFixed(2)}ms (target: <2500ms)`);
    }

    if (vitals.fid > 100) {
      issues.push(`FID too high: ${vitals.fid.toFixed(2)}ms (target: <100ms)`);
    }

    if (vitals.cls > 0.1) {
      issues.push(`CLS too high: ${vitals.cls.toFixed(3)} (target: <0.1)`);
    }

    if (issues.length > 0) {
      console.warn('Core Web Vitals issues detected:', issues);
    }
  }

  /**
   * Get current performance metrics
   */
  getCurrentMetrics(): PerformanceMetrics {
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    const memory = (performance as any).memory;

    return {
      coreWebVitals: {
        lcp: 0,
        fid: 0,
        cls: 0,
        fcp: 0,
        ttfb: navigation ? navigation.responseStart - navigation.requestStart : 0,
      },
      bundleSize: 0, // Will be updated by build process
      loadTime: navigation ? navigation.loadEventEnd - navigation.fetchStart : 0,
      renderTime: navigation ? navigation.domContentLoadedEventEnd - navigation.fetchStart : 0,
      interactionTime: navigation ? navigation.domInteractive - navigation.fetchStart : 0,
      memoryUsage: memory ? memory.usedJSHeapSize / 1024 / 1024 : 0,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Optimize images with lazy loading and modern formats
   */
  optimizeImages(): void {
    if (!this.config.enableImageOptimization) return;

    // Setup intersection observer for lazy loading
    if ('IntersectionObserver' in window && !this.intersectionObserver) {
      this.intersectionObserver = new IntersectionObserver(
        entries => {
          entries.forEach(entry => {
            if (entry.isIntersecting) {
              const img = entry.target as HTMLImageElement;
              if (img.dataset.src) {
                img.src = img.dataset.src;
                img.removeAttribute('data-src');
                this.intersectionObserver?.unobserve(img);
              }
            }
          });
        },
        {
          rootMargin: '50px 0px',
          threshold: 0.01,
        },
      );
    }

    // Observe all images with data-src attribute
    const lazyImages = document.querySelectorAll('img[data-src]');
    lazyImages.forEach(img => {
      this.intersectionObserver?.observe(img);
    });
  }

  /**
   * Implement virtual scrolling for large lists
   */
  createVirtualScrolling(
    container: HTMLElement,
    items: any[],
    itemHeight: number,
    renderItem: (item: any, index: number) => HTMLElement,
  ): {
    scrollToIndex: (index: number) => void;
    updateItems: (newItems: any[]) => void;
    destroy: () => void;
  } {
    if (!this.config.enableVirtualScrolling) {
      // Fallback to regular rendering
      items.forEach((item, index) => {
        container.appendChild(renderItem(item, index));
      });
      return {
        scrollToIndex: () => {},
        updateItems: () => {},
        destroy: () => {},
      };
    }

    const containerHeight = container.clientHeight;
    const visibleCount = Math.ceil(containerHeight / itemHeight) + 2; // Buffer
    let startIndex = 0;
    let endIndex = Math.min(visibleCount, items.length);

    const scrollContainer = document.createElement('div');
    scrollContainer.style.height = `${items.length * itemHeight}px`;
    scrollContainer.style.position = 'relative';

    const visibleContainer = document.createElement('div');
    visibleContainer.style.position = 'absolute';
    visibleContainer.style.top = '0';
    visibleContainer.style.width = '100%';

    scrollContainer.appendChild(visibleContainer);
    container.appendChild(scrollContainer);

    const renderVisibleItems = () => {
      visibleContainer.innerHTML = '';
      visibleContainer.style.transform = `translateY(${startIndex * itemHeight}px)`;

      for (let i = startIndex; i < endIndex; i++) {
        if (items[i]) {
          const itemElement = renderItem(items[i], i);
          itemElement.style.height = `${itemHeight}px`;
          visibleContainer.appendChild(itemElement);
        }
      }
    };

    const handleScroll = () => {
      const scrollTop = container.scrollTop;
      const newStartIndex = Math.floor(scrollTop / itemHeight);
      const newEndIndex = Math.min(newStartIndex + visibleCount, items.length);

      if (newStartIndex !== startIndex || newEndIndex !== endIndex) {
        startIndex = newStartIndex;
        endIndex = newEndIndex;
        renderVisibleItems();
      }
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    renderVisibleItems();

    return {
      scrollToIndex: (index: number) => {
        container.scrollTop = index * itemHeight;
      },
      updateItems: (newItems: any[]) => {
        items = newItems;
        scrollContainer.style.height = `${items.length * itemHeight}px`;
        endIndex = Math.min(startIndex + visibleCount, items.length);
        renderVisibleItems();
      },
      destroy: () => {
        container.removeEventListener('scroll', handleScroll);
        container.removeChild(scrollContainer);
      },
    };
  }

  /**
   * Optimize React component rendering
   */
  createMemoizedComponent<T extends Record<string, any>>(
    component: React.ComponentType<T>,
    propsAreEqual?: (prevProps: T, nextProps: T) => boolean,
  ): React.ComponentType<T> {
    if (!this.config.enableMemoization) return component;

    return React.memo(component, propsAreEqual) as unknown as React.ComponentType<T>;
  }

  /**
   * Debounce function for performance optimization
   */
  debounce<T extends (...args: any[]) => any>(
    func: T,
    wait: number,
    immediate?: boolean,
  ): (...args: Parameters<T>) => void {
    let timeout: NodeJS.Timeout | null = null;

    return (...args: Parameters<T>) => {
      const later = () => {
        timeout = null;
        if (!immediate) func(...args);
      };

      const callNow = immediate && !timeout;

      if (timeout) clearTimeout(timeout);
      timeout = setTimeout(later, wait);

      if (callNow) func(...args);
    };
  }

  /**
   * Throttle function for performance optimization
   */
  throttle<T extends (...args: any[]) => any>(
    func: T,
    limit: number,
  ): (...args: Parameters<T>) => void {
    let inThrottle: boolean;

    return (...args: Parameters<T>) => {
      if (!inThrottle) {
        func(...args);
        inThrottle = true;
        setTimeout(() => (inThrottle = false), limit);
      }
    };
  }

  /**
   * Preload critical resources
   */
  preloadCriticalResources(resources: Array<{ href: string; as: string; type?: string }>): void {
    resources.forEach(resource => {
      const link = document.createElement('link');
      link.rel = 'preload';
      link.href = resource.href;
      link.as = resource.as;
      if (resource.type) {
        link.type = resource.type;
      }
      document.head.appendChild(link);
    });
  }

  /**
   * Get performance recommendations
   */
  getPerformanceRecommendations(): string[] {
    const metrics = this.getCurrentMetrics();
    const recommendations: string[] = [];

    if (metrics.coreWebVitals.lcp > 2500) {
      recommendations.push(
        'Optimize Largest Contentful Paint: Consider image optimization, server response times, and resource loading',
      );
    }

    if (metrics.coreWebVitals.fid > 100) {
      recommendations.push(
        'Reduce First Input Delay: Minimize JavaScript execution time and use code splitting',
      );
    }

    if (metrics.coreWebVitals.cls > 0.1) {
      recommendations.push(
        'Improve Cumulative Layout Shift: Set dimensions for images and avoid inserting content above existing content',
      );
    }

    if (metrics.loadTime > 3000) {
      recommendations.push(
        'Reduce page load time: Optimize bundle size, enable compression, and use CDN',
      );
    }

    if (metrics.memoryUsage > 50) {
      recommendations.push(
        'Optimize memory usage: Check for memory leaks and optimize component lifecycle',
      );
    }

    return recommendations;
  }

  /**
   * Generate performance report
   */
  generatePerformanceReport(): {
    score: number;
    metrics: PerformanceMetrics;
    recommendations: string[];
    coreWebVitalsGrade: 'A' | 'B' | 'C' | 'D' | 'F';
  } {
    const metrics = this.getCurrentMetrics();
    const recommendations = this.getPerformanceRecommendations();

    // Calculate performance score (0-100)
    let score = 100;

    if (metrics.coreWebVitals.lcp > 2500) score -= 20;
    if (metrics.coreWebVitals.fid > 100) score -= 20;
    if (metrics.coreWebVitals.cls > 0.1) score -= 20;
    if (metrics.loadTime > 3000) score -= 20;
    if (metrics.memoryUsage > 50) score -= 20;

    // Determine Core Web Vitals grade
    let coreWebVitalsGrade: 'A' | 'B' | 'C' | 'D' | 'F' = 'A';
    const vitals = metrics.coreWebVitals;

    if (vitals.lcp > 4000 || vitals.fid > 300 || vitals.cls > 0.25) {
      coreWebVitalsGrade = 'F';
    } else if (vitals.lcp > 3000 || vitals.fid > 200 || vitals.cls > 0.2) {
      coreWebVitalsGrade = 'D';
    } else if (vitals.lcp > 2500 || vitals.fid > 100 || vitals.cls > 0.1) {
      coreWebVitalsGrade = 'C';
    } else if (vitals.lcp > 2000 || vitals.fid > 50 || vitals.cls > 0.05) {
      coreWebVitalsGrade = 'B';
    }

    return {
      score: Math.max(0, score),
      metrics,
      recommendations,
      coreWebVitalsGrade,
    };
  }

  /**
   * Cleanup performance monitoring
   */
  cleanup(): void {
    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
    }
    if (this.intersectionObserver) {
      this.intersectionObserver.disconnect();
    }
  }
}

// Export singleton instance
export const frontendPerformanceOptimizer = new FrontendPerformanceOptimizer();

// React performance utilities
export const usePerformanceOptimization = () => {
  const [metrics, setMetrics] = React.useState<PerformanceMetrics | null>(null);

  React.useEffect(() => {
    const updateMetrics = () => {
      setMetrics(frontendPerformanceOptimizer.getCurrentMetrics());
    };

    updateMetrics();
    const interval = setInterval(updateMetrics, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, []);

  return {
    metrics,
    optimizer: frontendPerformanceOptimizer,
  };
};
