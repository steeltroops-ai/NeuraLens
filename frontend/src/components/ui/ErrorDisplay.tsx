/**
 * Error Display Components
 * User-friendly error messages and error boundaries
 */

import React from 'react';
import { AlertTriangle, RefreshCw, X } from 'lucide-react';

// Error message types
export type ErrorType = 'network' | 'validation' | 'server' | 'timeout' | 'unknown';

// Error display props
interface ErrorDisplayProps {
  error: string;
  type?: ErrorType;
  onRetry?: () => void;
  onDismiss?: () => void;
  className?: string;
}

// Main error display component
export function ErrorDisplay({
  error,
  type = 'unknown',
  onRetry,
  onDismiss,
  className = '',
}: ErrorDisplayProps) {
  const getErrorIcon = () => {
    switch (type) {
      case 'network':
        return <AlertTriangle className='h-5 w-5 text-orange-500' />;
      case 'validation':
        return <AlertTriangle className='h-5 w-5 text-yellow-500' />;
      case 'server':
        return <AlertTriangle className='h-5 w-5 text-red-500' />;
      case 'timeout':
        return <AlertTriangle className='h-5 w-5 text-blue-500' />;
      default:
        return <AlertTriangle className='h-5 w-5 text-gray-500' />;
    }
  };

  const getErrorColor = () => {
    switch (type) {
      case 'network':
        return 'border-orange-200 bg-orange-50 text-orange-800';
      case 'validation':
        return 'border-yellow-200 bg-yellow-50 text-yellow-800';
      case 'server':
        return 'border-red-200 bg-red-50 text-red-800';
      case 'timeout':
        return 'border-blue-200 bg-blue-50 text-blue-800';
      default:
        return 'border-gray-200 bg-gray-50 text-gray-800';
    }
  };

  return (
    <div
      className={`flex items-start gap-3 rounded-lg border p-4 ${getErrorColor()} ${className} `}
    >
      {getErrorIcon()}

      <div className='min-w-0 flex-1'>
        <p className='text-sm font-medium'>{getErrorTitle(type)}</p>
        <p className='mt-1 text-sm opacity-90'>{error}</p>
      </div>

      <div className='flex items-center gap-2'>
        {onRetry && (
          <button
            onClick={onRetry}
            className='flex items-center gap-1 rounded border border-current bg-white px-3 py-1 text-xs font-medium transition-colors hover:bg-opacity-80'
          >
            <RefreshCw className='h-3 w-3' />
            Retry
          </button>
        )}

        {onDismiss && (
          <button
            onClick={onDismiss}
            className='rounded p-1 transition-colors hover:bg-white hover:bg-opacity-50'
          >
            <X className='h-4 w-4' />
          </button>
        )}
      </div>
    </div>
  );
}

// Get error title based on type
function getErrorTitle(type: ErrorType): string {
  switch (type) {
    case 'network':
      return 'Connection Error';
    case 'validation':
      return 'Validation Error';
    case 'server':
      return 'Server Error';
    case 'timeout':
      return 'Request Timeout';
    default:
      return 'Error';
  }
}

// Inline error component for form fields
interface InlineErrorProps {
  error: string;
  className?: string;
}

export function InlineError({ error, className = '' }: InlineErrorProps) {
  return (
    <div className={`mt-1 flex items-center gap-2 text-sm text-red-600 ${className}`}>
      <AlertTriangle className='h-4 w-4' />
      <span>{error}</span>
    </div>
  );
}

// Enhanced error boundary component with logging and recovery
interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
  errorInfo?: React.ErrorInfo;
  errorId?: string;
}

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ComponentType<{
    error: Error;
    errorInfo?: React.ErrorInfo;
    resetError: () => void;
    errorId?: string;
  }>;
  onError?: (error: Error, errorInfo: React.ErrorInfo, errorId: string) => void;
  level?: 'app' | 'page' | 'component';
  name?: string;
  isolate?: boolean; // Prevent error from bubbling up
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  private errorId: string = '';

  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    const errorId = `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    return { hasError: true, error, errorId };
  }

  override componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    const errorId =
      this.state.errorId || `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Enhanced error logging
    const errorReport = {
      error: {
        name: error.name,
        message: error.message,
        stack: error.stack,
      },
      errorInfo: {
        componentStack: errorInfo.componentStack,
      },
      context: {
        level: this.props.level || 'component',
        name: this.props.name || 'Unknown',
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href,
      },
      errorId,
    };

    console.error('Error caught by boundary:', errorReport);

    // Report to error tracking service (e.g., Sentry)
    this.reportError(errorReport);

    this.setState({ errorInfo, errorId });
    this.props.onError?.(error, errorInfo, errorId);

    // Prevent error from bubbling up if isolate is true
    if (this.props.isolate) {
      return;
    }
  }

  private reportError = (errorReport: any) => {
    // In a real application, this would send to an error tracking service
    if (process.env.NODE_ENV === 'production') {
      // Example: Sentry.captureException(errorReport);
      console.log('Error reported to tracking service:', errorReport.errorId);
    }
  };

  resetError = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined, errorId: undefined });
  };

  override render() {
    if (this.state.hasError && this.state.error) {
      if (this.props.fallback) {
        const FallbackComponent = this.props.fallback;
        return (
          <FallbackComponent
            error={this.state.error}
            errorInfo={this.state.errorInfo}
            resetError={this.resetError}
            errorId={this.state.errorId}
          />
        );
      }

      return (
        <ErrorDisplay error={this.state.error.message} type='unknown' onRetry={this.resetError} />
      );
    }

    return this.props.children;
  }
}

// Default error fallback component
interface DefaultErrorFallbackProps {
  error: Error;
  errorInfo?: React.ErrorInfo;
  resetError: () => void;
  errorId?: string;
}

export function DefaultErrorFallback({
  error,
  errorInfo,
  resetError,
  errorId,
}: DefaultErrorFallbackProps) {
  const [showDetails, setShowDetails] = React.useState(false);

  return (
    <div
      className='flex min-h-[200px] flex-col items-center justify-center p-8 text-center'
      role='alert'
    >
      <AlertTriangle className='mb-4 h-12 w-12 text-red-500' />
      <h3 className='mb-2 text-lg font-semibold text-gray-900'>Something went wrong</h3>
      <p className='mb-4 max-w-md text-gray-600'>
        {error.message || 'An unexpected error occurred. Please try again.'}
      </p>

      {errorId && <p className='mb-4 text-xs text-gray-500'>Error ID: {errorId}</p>}

      <div className='mb-4 flex items-center gap-3'>
        <button
          onClick={resetError}
          className='flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
          aria-label='Try again'
        >
          <RefreshCw className='h-4 w-4' />
          Try Again
        </button>

        <button
          onClick={() => window.location.reload()}
          className='flex items-center gap-2 rounded-lg border border-gray-300 px-4 py-2 text-gray-600 transition-colors hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2'
          aria-label='Reload page'
        >
          Reload Page
        </button>
      </div>

      {(error.stack || errorInfo) && (
        <div className='w-full max-w-2xl'>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className='mb-2 text-sm text-gray-500 transition-colors hover:text-gray-700'
            aria-expanded={showDetails}
            aria-controls='error-details'
          >
            {showDetails ? 'Hide' : 'Show'} Error Details
          </button>

          {showDetails && (
            <div
              id='error-details'
              className='max-h-40 overflow-auto rounded-lg bg-gray-100 p-4 text-left font-mono text-xs'
            >
              <div className='mb-2'>
                <strong>Error:</strong> {error.name}: {error.message}
              </div>
              {error.stack && (
                <div className='mb-2'>
                  <strong>Stack:</strong>
                  <pre className='mt-1 whitespace-pre-wrap'>{error.stack}</pre>
                </div>
              )}
              {errorInfo?.componentStack && (
                <div>
                  <strong>Component Stack:</strong>
                  <pre className='mt-1 whitespace-pre-wrap'>{errorInfo.componentStack}</pre>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Loading error component (for when loading fails)
interface LoadingErrorProps {
  message?: string;
  onRetry?: () => void;
}

export function LoadingError({ message = 'Failed to load data', onRetry }: LoadingErrorProps) {
  return (
    <div className='flex flex-col items-center justify-center p-8 text-center'>
      <AlertTriangle className='mb-3 h-8 w-8 text-gray-400' />
      <p className='mb-4 text-gray-600'>{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className='flex items-center gap-2 rounded border border-blue-600 px-3 py-2 text-sm text-blue-600 transition-colors hover:bg-blue-50'
        >
          <RefreshCw className='h-4 w-4' />
          Retry
        </button>
      )}
    </div>
  );
}

// Assessment-specific error fallback
interface AssessmentErrorFallbackProps {
  error: Error;
  resetError: () => void;
  onRetryAssessment?: () => void;
  assessmentStep?: string;
}

export function AssessmentErrorFallback({
  error,
  resetError,
  onRetryAssessment,
  assessmentStep,
}: AssessmentErrorFallbackProps) {
  return (
    <div className='rounded-lg border border-red-200 bg-red-50 p-6 text-center' role='alert'>
      <AlertTriangle className='mx-auto mb-4 h-10 w-10 text-red-600' />
      <h3 className='mb-2 text-lg font-semibold text-red-900'>Assessment Error</h3>
      <p className='mb-4 text-red-700'>
        {assessmentStep
          ? `An error occurred during ${assessmentStep} processing: ${error.message}`
          : `Assessment failed: ${error.message}`}
      </p>

      <div className='flex items-center justify-center gap-3'>
        {onRetryAssessment && (
          <button
            onClick={onRetryAssessment}
            className='flex items-center gap-2 rounded-lg bg-red-600 px-4 py-2 text-white transition-colors hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2'
            aria-label='Retry assessment'
          >
            <RefreshCw className='h-4 w-4' />
            Retry Assessment
          </button>
        )}

        <button
          onClick={resetError}
          className='flex items-center gap-2 rounded-lg border border-red-600 px-4 py-2 text-red-600 transition-colors hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2'
          aria-label='Go back'
        >
          Go Back
        </button>
      </div>
    </div>
  );
}

// Network error fallback
export function NetworkErrorFallback({ onRetry }: { onRetry?: () => void }) {
  return (
    <div className='rounded-lg border border-orange-200 bg-orange-50 p-6 text-center' role='alert'>
      <AlertTriangle className='mx-auto mb-4 h-10 w-10 text-orange-600' />
      <h3 className='mb-2 text-lg font-semibold text-orange-900'>Connection Error</h3>
      <p className='mb-4 text-orange-700'>
        Unable to connect to the server. Please check your internet connection and try again.
      </p>

      {onRetry && (
        <button
          onClick={onRetry}
          className='mx-auto flex items-center gap-2 rounded-lg bg-orange-600 px-4 py-2 text-white transition-colors hover:bg-orange-700 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:ring-offset-2'
          aria-label='Retry connection'
        >
          <RefreshCw className='h-4 w-4' />
          Retry
        </button>
      )}
    </div>
  );
}

// Network error component (legacy)
export function NetworkError({ onRetry }: { onRetry?: () => void }) {
  return (
    <ErrorDisplay
      error='Unable to connect to the server. Please check your internet connection and try again.'
      type='network'
      onRetry={onRetry}
    />
  );
}

// Validation error component
export function ValidationError({ errors }: { errors: string[] }) {
  return (
    <div className='space-y-2'>
      {errors.map((error, index) => (
        <InlineError key={index} error={error} />
      ))}
    </div>
  );
}

// Utility function to determine error type from error message
export function getErrorType(error: string): ErrorType {
  const lowerError = error.toLowerCase();

  if (lowerError.includes('network') || lowerError.includes('connection')) {
    return 'network';
  }
  if (lowerError.includes('validation') || lowerError.includes('invalid')) {
    return 'validation';
  }
  if (lowerError.includes('server') || lowerError.includes('internal')) {
    return 'server';
  }
  if (lowerError.includes('timeout') || lowerError.includes('time out')) {
    return 'timeout';
  }

  return 'unknown';
}
