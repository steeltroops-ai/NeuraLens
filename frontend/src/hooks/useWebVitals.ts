"use client";

import { useEffect, useCallback } from "react";

/**
 * Web Vitals Types
 */
interface WebVitalMetric {
  id: string;
  name: string;
  value: number;
  rating: "good" | "needs-improvement" | "poor";
  delta: number;
  navigationType: string;
}

/**
 * Web Vitals Thresholds (per Google recommendations)
 */
const THRESHOLDS: Record<string, { good: number; poor: number }> = {
  LCP: { good: 2500, poor: 4000 },
  FID: { good: 100, poor: 300 },
  CLS: { good: 0.1, poor: 0.25 },
  FCP: { good: 1800, poor: 3000 },
  TTFB: { good: 800, poor: 1800 },
  INP: { good: 200, poor: 500 },
};

/**
 * Get rating based on metric value and thresholds
 */
function getRating(
  name: string,
  value: number,
): "good" | "needs-improvement" | "poor" {
  const threshold = THRESHOLDS[name];
  if (!threshold) return "good";

  if (value <= threshold.good) return "good";
  if (value > threshold.poor) return "poor";
  return "needs-improvement";
}

/**
 * Report handler - can be customized for analytics
 */
function reportMetric(metric: WebVitalMetric) {
  // Log to console in development
  if (process.env.NODE_ENV === "development") {
    const style =
      metric.rating === "good"
        ? "color: green"
        : metric.rating === "poor"
          ? "color: red"
          : "color: orange";
    console.log(
      `%c[Web Vital] ${metric.name}: ${metric.value.toFixed(2)} (${metric.rating})`,
      style,
    );
  }

  // In production, you would send to analytics service
  // Example: navigator.sendBeacon('/api/analytics', JSON.stringify(metric));
}

/**
 * useWebVitals Hook
 *
 * Measures and reports Core Web Vitals using the Performance Observer API.
 * This implementation doesn't require the web-vitals library.
 *
 * Metrics measured:
 * - LCP (Largest Contentful Paint)
 * - FCP (First Contentful Paint)
 * - CLS (Cumulative Layout Shift)
 * - FID (First Input Delay)
 * - TTFB (Time to First Byte)
 */
export function useWebVitals() {
  const handleMetric = useCallback(
    (name: string, value: number, id: string = crypto.randomUUID()) => {
      reportMetric({
        id,
        name,
        value,
        delta: value,
        navigationType: "navigate",
        rating: getRating(name, value),
      });
    },
    [],
  );

  useEffect(() => {
    // Only run in browser
    if (typeof window === "undefined") return;
    if (typeof PerformanceObserver === "undefined") return;

    const observers: PerformanceObserver[] = [];

    // Use requestIdleCallback for non-blocking initialization
    const init = () => {
      try {
        // LCP - Largest Contentful Paint
        const lcpObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const lastEntry = entries[entries.length - 1];
          if (lastEntry) {
            handleMetric("LCP", lastEntry.startTime);
          }
        });
        lcpObserver.observe({
          type: "largest-contentful-paint",
          buffered: true,
        });
        observers.push(lcpObserver);

        // FCP - First Contentful Paint
        const fcpObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.name === "first-contentful-paint") {
              handleMetric("FCP", entry.startTime);
            }
          }
        });
        fcpObserver.observe({ type: "paint", buffered: true });
        observers.push(fcpObserver);

        // CLS - Cumulative Layout Shift
        let clsValue = 0;
        const clsObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            if (!(entry as any).hadRecentInput) {
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              clsValue += (entry as any).value;
            }
          }
        });
        clsObserver.observe({ type: "layout-shift", buffered: true });
        observers.push(clsObserver);

        // Report CLS on page hide
        const reportCLS = () => handleMetric("CLS", clsValue);
        window.addEventListener("visibilitychange", () => {
          if (document.visibilityState === "hidden") {
            reportCLS();
          }
        });

        // FID - First Input Delay
        const fidObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const fidEntry = entry as any;
            handleMetric("FID", fidEntry.processingStart - fidEntry.startTime);
          }
        });
        fidObserver.observe({ type: "first-input", buffered: true });
        observers.push(fidObserver);

        // TTFB - Time to First Byte
        const navEntries = performance.getEntriesByType("navigation");
        if (navEntries.length > 0) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const navEntry = navEntries[0] as any;
          handleMetric("TTFB", navEntry.responseStart - navEntry.requestStart);
        }
      } catch {
        // PerformanceObserver not fully supported
        if (process.env.NODE_ENV === "development") {
          console.debug("[Web Vitals] PerformanceObserver not fully supported");
        }
      }
    };

    if ("requestIdleCallback" in window) {
      window.requestIdleCallback(() => init());
    } else {
      setTimeout(init, 0);
    }

    return () => {
      observers.forEach((observer) => observer.disconnect());
    };
  }, [handleMetric]);
}

/**
 * Performance Mark Helper
 * Use for custom performance measurements
 */
export function markPerformance(name: string) {
  if (typeof performance !== "undefined" && performance.mark) {
    performance.mark(name);
  }
}

/**
 * Performance Measure Helper
 * Measure time between two marks
 */
export function measurePerformance(
  name: string,
  startMark: string,
  endMark?: string,
) {
  if (typeof performance !== "undefined" && performance.measure) {
    try {
      const measure = performance.measure(name, startMark, endMark);
      if (process.env.NODE_ENV === "development") {
        console.log(`[Performance] ${name}: ${measure.duration.toFixed(2)}ms`);
      }
      return measure.duration;
    } catch {
      return null;
    }
  }
  return null;
}

export default useWebVitals;
