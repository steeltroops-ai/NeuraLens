/**
 * NeuroLens-X Utility Functions
 * Core utilities for the Neuro-Minimalist UI system
 */

import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Combines class names with Tailwind CSS merge support
 * Prevents style conflicts and optimizes class application
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Formats numbers with appropriate precision for medical data
 */
export function formatNumber(
  value: number,
  options: {
    precision?: number;
    percentage?: boolean;
    currency?: boolean;
    compact?: boolean;
  } = {}
): string {
  const { precision = 2, percentage = false, currency = false, compact = false } = options;

  if (percentage) {
    return `${(value * 100).toFixed(precision)}%`;
  }

  if (currency) {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  }

  if (compact && value >= 1000) {
    return new Intl.NumberFormat('en-US', {
      notation: 'compact',
      maximumFractionDigits: precision,
    }).format(value);
  }

  return value.toFixed(precision);
}

/**
 * Formats time duration for assessment displays
 */
export function formatDuration(milliseconds: number): string {
  const seconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  }
  return `${seconds}s`;
}

/**
 * Formats processing time with appropriate units
 */
export function formatProcessingTime(milliseconds: number): string {
  if (milliseconds < 1) {
    return `${(milliseconds * 1000).toFixed(1)}μs`;
  }
  if (milliseconds < 1000) {
    return `${milliseconds.toFixed(1)}ms`;
  }
  return `${(milliseconds / 1000).toFixed(2)}s`;
}

/**
 * Determines risk category color based on score
 */
export function getRiskColor(score: number): string {
  if (score <= 25) return 'text-green-600';
  if (score <= 50) return 'text-yellow-600';
  if (score <= 75) return 'text-orange-600';
  return 'text-red-600';
}

/**
 * Determines risk category background color
 */
export function getRiskBgColor(score: number): string {
  if (score <= 25) return 'bg-green-100';
  if (score <= 50) return 'bg-yellow-100';
  if (score <= 75) return 'bg-orange-100';
  return 'bg-red-100';
}

/**
 * Gets risk category label
 */
export function getRiskCategory(score: number): string {
  if (score <= 25) return 'Low Risk';
  if (score <= 50) return 'Moderate Risk';
  if (score <= 75) return 'High Risk';
  return 'Very High Risk';
}

/**
 * Validates email format
 */
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Generates a unique session ID
 */
export function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Debounces function calls
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

/**
 * Throttles function calls
 */
export function throttle<T extends (...args: unknown[]) => unknown>(
  func: T,
  limit: number
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
 * Safely parses JSON with fallback
 */
export function safeJsonParse<T>(json: string, fallback: T): T {
  try {
    return JSON.parse(json) as T;
  } catch {
    return fallback;
  }
}

/**
 * Calculates confidence interval
 */
export function calculateConfidenceInterval(
  mean: number,
  standardError: number,
  confidenceLevel: number = 0.95
): [number, number] {
  const zScore = confidenceLevel === 0.95 ? 1.96 : 2.58; // 95% or 99%
  const margin = zScore * standardError;
  return [mean - margin, mean + margin];
}

/**
 * Normalizes score to 0-100 range
 */
export function normalizeScore(score: number, min: number, max: number): number {
  return Math.max(0, Math.min(100, ((score - min) / (max - min)) * 100));
}

/**
 * Calculates weighted average
 */
export function weightedAverage(values: number[], weights: number[]): number {
  if (values.length !== weights.length) {
    throw new Error('Values and weights arrays must have the same length');
  }
  
  const weightedSum = values.reduce((sum, value, index) => sum + value * weights[index]!, 0);
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  
  return weightedSum / totalWeight;
}

/**
 * Formats date for display
 */
export function formatDate(date: Date | string, format: 'short' | 'long' | 'time' = 'short'): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  
  switch (format) {
    case 'long':
      return dateObj.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    case 'time':
      return dateObj.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
      });
    default:
      return dateObj.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      });
  }
}

/**
 * Checks if device supports required features
 */
export function checkDeviceCapabilities(): {
  camera: boolean;
  microphone: boolean;
  accelerometer: boolean;
  gyroscope: boolean;
  webgl: boolean;
} {
  return {
    camera: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
    microphone: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
    accelerometer: 'DeviceMotionEvent' in window,
    gyroscope: 'DeviceOrientationEvent' in window,
    webgl: !!document.createElement('canvas').getContext('webgl'),
  };
}

/**
 * Measures performance metrics
 */
export function measurePerformance(name: string): () => number {
  const start = performance.now();
  return () => {
    const end = performance.now();
    const duration = end - start;
    console.log(`Performance [${name}]: ${duration.toFixed(2)}ms`);
    return duration;
  };
}

/**
 * Creates a delay promise for testing/demo purposes
 */
export function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Validates assessment data completeness
 */
export function validateAssessmentData(data: Record<string, unknown>): {
  isValid: boolean;
  missingFields: string[];
} {
  const requiredFields = ['sessionId', 'timestamp'];
  const missingFields = requiredFields.filter(field => !(field in data));
  
  return {
    isValid: missingFields.length === 0,
    missingFields,
  };
}
