'use client';

/**
 * Global Error Monitoring System for NeuraLens
 * Captures and handles runtime errors, unhandled promises, and navigation issues
 */

interface ErrorReport {
  type: 'javascript' | 'unhandledrejection' | 'navigation' | 'hydration' | 'chunk';
  message: string;
  stack?: string;
  url?: string;
  line?: number;
  column?: number;
  timestamp: number;
  userAgent: string;
  route: string;
}

class ErrorMonitoringService {
  private errors: ErrorReport[] = [];
  private maxErrors = 50; // Keep last 50 errors
  private isInitialized = false;

  initialize() {
    if (this.isInitialized || typeof window === 'undefined') {
      return;
    }

    this.isInitialized = true;

    // Global JavaScript error handler
    window.addEventListener('error', (event) => {
      this.captureError({
        type: 'javascript',
        message: event.message,
        stack: event.error?.stack,
        url: event.filename,
        line: event.lineno,
        column: event.colno,
        timestamp: Date.now(),
        userAgent: navigator.userAgent,
        route: window.location.pathname,
      });
    });

    // Unhandled promise rejection handler
    window.addEventListener('unhandledrejection', (event) => {
      this.captureError({
        type: 'unhandledrejection',
        message: event.reason?.message || String(event.reason),
        stack: event.reason?.stack,
        timestamp: Date.now(),
        userAgent: navigator.userAgent,
        route: window.location.pathname,
      });
    });

    // Chunk loading error detection
    const originalOnError = window.onerror;
    window.onerror = (message, source, lineno, colno, error) => {
      if (typeof message === 'string' && message.includes('Loading chunk')) {
        this.captureError({
          type: 'chunk',
          message: 'Dynamic import chunk failed to load',
          stack: error?.stack,
          url: source,
          line: lineno,
          column: colno,
          timestamp: Date.now(),
          userAgent: navigator.userAgent,
          route: window.location.pathname,
        });

        // Attempt to reload the page for chunk errors
        setTimeout(() => {
          window.location.reload();
        }, 1000);
      }

      // Call original handler if it exists
      if (originalOnError) {
        return originalOnError(message, source, lineno, colno, error);
      }
      return false;
    };

    // Navigation error detection
    this.monitorNavigation();

    console.log('🔍 Error monitoring initialized');
  }

  private captureError(errorReport: ErrorReport) {
    // Add to errors array
    this.errors.push(errorReport);

    // Keep only the last maxErrors
    if (this.errors.length > this.maxErrors) {
      this.errors = this.errors.slice(-this.maxErrors);
    }

    // Log error for development
    if (process.env.NODE_ENV === 'development') {
      console.error(`🚨 ${errorReport.type.toUpperCase()} Error:`, errorReport);
    }

    // Handle specific error types
    this.handleErrorType(errorReport);
  }

  private handleErrorType(errorReport: ErrorReport) {
    switch (errorReport.type) {
      case 'hydration':
        console.warn('🔥 Hydration mismatch detected - consider using ClientOnly wrapper');
        break;
      case 'chunk':
        console.warn('📦 Chunk loading failed - attempting page reload');
        break;
      case 'navigation':
        console.warn('🧭 Navigation error - check routing configuration');
        break;
      case 'unhandledrejection':
        console.warn('🔄 Unhandled promise rejection - check async operations');
        break;
    }
  }

  private monitorNavigation() {
    // Monitor for navigation failures
    const originalPushState = history.pushState;
    const originalReplaceState = history.replaceState;

    history.pushState = function(...args) {
      try {
        return originalPushState.apply(this, args);
      } catch (error) {
        errorMonitoring.captureError({
          type: 'navigation',
          message: `Navigation failed: ${error}`,
          stack: error instanceof Error ? error.stack : undefined,
          timestamp: Date.now(),
          userAgent: navigator.userAgent,
          route: window.location.pathname,
        });
        throw error;
      }
    };

    history.replaceState = function(...args) {
      try {
        return originalReplaceState.apply(this, args);
      } catch (error) {
        errorMonitoring.captureError({
          type: 'navigation',
          message: `Navigation failed: ${error}`,
          stack: error instanceof Error ? error.stack : undefined,
          timestamp: Date.now(),
          userAgent: navigator.userAgent,
          route: window.location.pathname,
        });
        throw error;
      }
    };
  }

  getErrors(): ErrorReport[] {
    return [...this.errors];
  }

  getErrorsByType(type: ErrorReport['type']): ErrorReport[] {
    return this.errors.filter(error => error.type === type);
  }

  clearErrors() {
    this.errors = [];
  }

  hasErrors(): boolean {
    return this.errors.length > 0;
  }

  getErrorSummary() {
    const summary = {
      total: this.errors.length,
      javascript: 0,
      unhandledrejection: 0,
      navigation: 0,
      hydration: 0,
      chunk: 0,
    };

    this.errors.forEach(error => {
      summary[error.type]++;
    });

    return summary;
  }
}

// Global instance
export const errorMonitoring = new ErrorMonitoringService();

// Auto-initialize on client side
if (typeof window !== 'undefined') {
  errorMonitoring.initialize();
}

// React hook for error monitoring
export function useErrorMonitoring() {
  return {
    errors: errorMonitoring.getErrors(),
    hasErrors: errorMonitoring.hasErrors(),
    clearErrors: errorMonitoring.clearErrors,
    getErrorSummary: errorMonitoring.getErrorSummary,
    getErrorsByType: errorMonitoring.getErrorsByType,
  };
}

// Error boundary integration
export function reportErrorToBoundary(error: Error, errorInfo: any) {
  errorMonitoring.captureError({
    type: 'javascript',
    message: error.message,
    stack: error.stack,
    timestamp: Date.now(),
    userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown',
    route: typeof window !== 'undefined' ? window.location.pathname : 'unknown',
  });
}

export default errorMonitoring;
