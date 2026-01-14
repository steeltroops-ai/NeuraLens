/**
 * Enhanced Error Display Components
 * MediLens Design System compliant error states
 * 
 * Requirements: 8.1, 8.2, 8.4
 * - WHEN an error occurs, THE System SHALL display a user-friendly error message
 * - THE System SHALL provide a retry button for failed operations
 * - IF an API call fails, THEN THE System SHALL show helpful suggestions for resolution
 */

import React from 'react';
import { AlertTriangle, RefreshCw, X, WifiOff, Server, Clock, AlertCircle, HelpCircle, Home, ArrowLeft } from 'lucide-react';
import { cn } from '@/utils/cn';

// Error message types
export type ErrorType = 'network' | 'validation' | 'server' | 'timeout' | 'unknown' | 'not-found' | 'permission';

// Error state interface for standardized error handling
export interface ErrorState {
  message: string;
  type: ErrorType;
  technicalDetails?: string;
  suggestions: string[];
  retryable: boolean;
}

// Error display props
interface ErrorDisplayProps {
  error: string;
  type?: ErrorType;
  onRetry?: () => void;
  onDismiss?: () => void;
  className?: string;
  suggestions?: string[];
  showSuggestions?: boolean;
}

/**
 * Get error icon based on error type
 */
function getErrorIcon(type: ErrorType) {
  const iconClasses = 'h-5 w-5';

  switch (type) {
    case 'network':
      return <WifiOff className={cn(iconClasses, 'text-[#FF9500]')} aria-hidden="true" />;
    case 'validation':
      return <AlertCircle className={cn(iconClasses, 'text-[#FFD60A]')} aria-hidden="true" />;
    case 'server':
      return <Server className={cn(iconClasses, 'text-[#FF3B30]')} aria-hidden="true" />;
    case 'timeout':
      return <Clock className={cn(iconClasses, 'text-[#5AC8FA]')} aria-hidden="true" />;
    case 'not-found':
      return <HelpCircle className={cn(iconClasses, 'text-[#8E8E93]')} aria-hidden="true" />;
    case 'permission':
      return <AlertTriangle className={cn(iconClasses, 'text-[#FF9500]')} aria-hidden="true" />;
    default:
      return <AlertTriangle className={cn(iconClasses, 'text-[#8E8E93]')} aria-hidden="true" />;
  }
}

/**
 * Get error styling based on type
 */
function getErrorStyles(type: ErrorType) {
  switch (type) {
    case 'network':
      return 'border-[#FF9500]/30 bg-[#FF9500]/10 text-[#CC7700]';
    case 'validation':
      return 'border-[#FFD60A]/30 bg-[#FFD60A]/10 text-[#997F00]';
    case 'server':
      return 'border-[#FF3B30]/30 bg-[#FF3B30]/10 text-[#CC2F26]';
    case 'timeout':
      return 'border-[#5AC8FA]/30 bg-[#5AC8FA]/10 text-[#0077B3]';
    case 'not-found':
      return 'border-[#8E8E93]/30 bg-[#8E8E93]/10 text-[#636366]';
    case 'permission':
      return 'border-[#FF9500]/30 bg-[#FF9500]/10 text-[#CC7700]';
    default:
      return 'border-[#8E8E93]/30 bg-[#8E8E93]/10 text-[#636366]';
  }
}

/**
 * Get error title based on type
 */
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
    case 'not-found':
      return 'Not Found';
    case 'permission':
      return 'Access Denied';
    default:
      return 'Error';
  }
}

/**
 * Get default suggestions based on error type
 */
function getDefaultSuggestions(type: ErrorType): string[] {
  switch (type) {
    case 'network':
      return [
        'Check your internet connection',
        'Try refreshing the page',
        'Disable VPN or proxy if enabled',
      ];
    case 'validation':
      return [
        'Review the input data for errors',
        'Ensure all required fields are filled',
        'Check for correct data formats',
      ];
    case 'server':
      return [
        'Wait a moment and try again',
        'The server may be temporarily unavailable',
        'Contact support if the issue persists',
      ];
    case 'timeout':
      return [
        'The request took too long to complete',
        'Try again with a smaller file or dataset',
        'Check your connection speed',
      ];
    case 'not-found':
      return [
        'The requested resource may have been moved',
        'Check the URL for typos',
        'Return to the dashboard and try again',
      ];
    case 'permission':
      return [
        'You may not have access to this resource',
        'Try logging in again',
        'Contact your administrator for access',
      ];
    default:
      return [
        'Try refreshing the page',
        'Clear your browser cache',
        'Contact support if the issue persists',
      ];
  }
}


/**
 * Main ErrorDisplay component
 * MediLens Design System compliant with user-friendly messages and suggestions
 */
export function ErrorDisplay({
  error,
  type = 'unknown',
  onRetry,
  onDismiss,
  className = '',
  suggestions,
  showSuggestions = true,
}: ErrorDisplayProps) {
  const displaySuggestions = suggestions || getDefaultSuggestions(type);

  return (
    <div
      className={cn(
        'rounded-2xl border p-6',
        getErrorStyles(type),
        className
      )}
      role="alert"
      aria-live="polite"
    >
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0 mt-0.5">
          {getErrorIcon(type)}
        </div>

        <div className="min-w-0 flex-1">
          <h3 className="text-[17px] font-semibold mb-1">
            {getErrorTitle(type)}
          </h3>
          <p className="text-[15px] opacity-90 mb-4">
            {error}
          </p>

          {/* Suggestions section */}
          {showSuggestions && displaySuggestions.length > 0 && (
            <div className="mb-4">
              <p className="text-[13px] font-medium mb-2 opacity-80">
                Suggestions:
              </p>
              <ul className="space-y-1">
                {displaySuggestions.map((suggestion, index) => (
                  <li
                    key={index}
                    className="text-[13px] opacity-75 flex items-start gap-2"
                  >
                    <span className="text-current mt-1">•</span>
                    <span>{suggestion}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Action buttons */}
          <div className="flex items-center gap-3">
            {onRetry && (
              <button
                onClick={onRetry}
                className={cn(
                  'inline-flex items-center gap-2 px-4 py-2 min-h-[44px]',
                  'bg-white/80 hover:bg-white',
                  'text-current font-medium text-[15px]',
                  'rounded-xl border border-current/20',
                  'transition-all duration-200',
                  'hover:-translate-y-0.5 active:scale-[0.98]',
                  'focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-current/40'
                )}
                aria-label="Retry the failed operation"
              >
                <RefreshCw className="h-4 w-4" aria-hidden="true" />
                Retry
              </button>
            )}

            {onDismiss && (
              <button
                onClick={onDismiss}
                className={cn(
                  'p-2 min-h-[44px] min-w-[44px]',
                  'rounded-xl',
                  'transition-colors duration-150',
                  'hover:bg-white/50',
                  'focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-current/40'
                )}
                aria-label="Dismiss error message"
              >
                <X className="h-5 w-5" aria-hidden="true" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Inline error component for form fields
 */
interface InlineErrorProps {
  error: string;
  className?: string;
}

export function InlineError({ error, className = '' }: InlineErrorProps) {
  return (
    <div
      className={cn(
        'mt-2 flex items-center gap-2 text-[13px] text-[#FF3B30]',
        className
      )}
      role="alert"
    >
      <AlertTriangle className='h-4 w-4' aria-hidden="true" />
      <span>{error}</span>
    </div>
  );
}


/**
 * Full Page Error Display
 * For critical errors that prevent page rendering
 */
interface FullPageErrorProps {
  error: string;
  type?: ErrorType;
  onRetry?: () => void;
  onGoHome?: () => void;
  onGoBack?: () => void;
  suggestions?: string[];
}

export function FullPageError({
  error,
  type = 'unknown',
  onRetry,
  onGoHome,
  onGoBack,
  suggestions,
}: FullPageErrorProps) {
  const displaySuggestions = suggestions || getDefaultSuggestions(type);

  return (
    <div
      className="min-h-[400px] flex flex-col items-center justify-center p-8 text-center"
      role="alert"
      aria-live="assertive"
    >
      {/* Error icon */}
      <div className={cn(
        'w-16 h-16 rounded-full flex items-center justify-center mb-6',
        type === 'server' ? 'bg-[#FF3B30]/10' :
          type === 'network' ? 'bg-[#FF9500]/10' :
            'bg-[#8E8E93]/10'
      )}>
        {type === 'server' && <Server className="h-8 w-8 text-[#FF3B30]" aria-hidden="true" />}
        {type === 'network' && <WifiOff className="h-8 w-8 text-[#FF9500]" aria-hidden="true" />}
        {type === 'timeout' && <Clock className="h-8 w-8 text-[#5AC8FA]" aria-hidden="true" />}
        {type === 'not-found' && <HelpCircle className="h-8 w-8 text-[#8E8E93]" aria-hidden="true" />}
        {!['server', 'network', 'timeout', 'not-found'].includes(type) && (
          <AlertTriangle className="h-8 w-8 text-[#8E8E93]" aria-hidden="true" />
        )}
      </div>

      {/* Error title */}
      <h2 className="text-[28px] font-semibold text-[#000000] mb-2">
        {getErrorTitle(type)}
      </h2>

      {/* Error message */}
      <p className="text-[17px] text-[#3C3C43] mb-6 max-w-md">
        {error}
      </p>

      {/* Suggestions */}
      {displaySuggestions.length > 0 && (
        <div className="mb-8 text-left max-w-md">
          <p className="text-[15px] font-medium text-[#3C3C43] mb-3">
            Here are some things you can try:
          </p>
          <ul className="space-y-2">
            {displaySuggestions.map((suggestion, index) => (
              <li
                key={index}
                className="text-[15px] text-[#8E8E93] flex items-start gap-2"
              >
                <span className="text-[#007AFF] mt-0.5">•</span>
                <span>{suggestion}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex items-center gap-4 flex-wrap justify-center">
        {onRetry && (
          <button
            onClick={onRetry}
            className={cn(
              'inline-flex items-center gap-2 px-6 py-3 min-h-[48px]',
              'bg-gradient-to-br from-[#007AFF] to-[#0062CC]',
              'text-white font-semibold text-[17px]',
              'rounded-xl',
              'shadow-[0_4px_14px_0_rgba(0,122,255,0.15)]',
              'hover:shadow-[0_8px_25px_0_rgba(0,122,255,0.25)]',
              'transition-all duration-200',
              'hover:-translate-y-0.5 active:scale-[0.98]',
              'focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-[#007AFF]/40'
            )}
            aria-label="Try again"
          >
            <RefreshCw className="h-5 w-5" aria-hidden="true" />
            Try Again
          </button>
        )}

        {onGoBack && (
          <button
            onClick={onGoBack}
            className={cn(
              'inline-flex items-center gap-2 px-6 py-3 min-h-[48px]',
              'bg-[#F2F2F7]',
              'text-[#000000] font-semibold text-[17px]',
              'rounded-xl border border-[#E5E5EA]',
              'transition-all duration-200',
              'hover:bg-[#E5E5EA] hover:-translate-y-0.5',
              'active:scale-[0.98]',
              'focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-[#007AFF]/40'
            )}
            aria-label="Go back to previous page"
          >
            <ArrowLeft className="h-5 w-5" aria-hidden="true" />
            Go Back
          </button>
        )}

        {onGoHome && (
          <button
            onClick={onGoHome}
            className={cn(
              'inline-flex items-center gap-2 px-6 py-3 min-h-[48px]',
              'text-[#007AFF] font-medium text-[17px]',
              'rounded-xl',
              'transition-colors duration-150',
              'hover:bg-[#007AFF]/10',
              'active:bg-[#007AFF]/20',
              'focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-[#007AFF]/40'
            )}
            aria-label="Return to home page"
          >
            <Home className="h-5 w-5" aria-hidden="true" />
            Go Home
          </button>
        )}
      </div>
    </div>
  );
}


/**
 * Loading Error component - For when data loading fails
 */
interface LoadingErrorProps {
  message?: string;
  onRetry?: () => void;
  suggestions?: string[];
}

export function LoadingError({
  message = 'Failed to load data',
  onRetry,
  suggestions = ['Check your connection and try again', 'The data may be temporarily unavailable']
}: LoadingErrorProps) {
  return (
    <div
      className='flex flex-col items-center justify-center p-8 text-center'
      role="alert"
    >
      <div className="w-12 h-12 rounded-full bg-[#8E8E93]/10 flex items-center justify-center mb-4">
        <AlertTriangle className='h-6 w-6 text-[#8E8E93]' aria-hidden="true" />
      </div>

      <p className='text-[17px] text-[#3C3C43] mb-4'>{message}</p>

      {suggestions.length > 0 && (
        <ul className="mb-4 space-y-1">
          {suggestions.map((suggestion, index) => (
            <li key={index} className="text-[13px] text-[#8E8E93]">
              {suggestion}
            </li>
          ))}
        </ul>
      )}

      {onRetry && (
        <button
          onClick={onRetry}
          className={cn(
            'inline-flex items-center gap-2 px-4 py-2 min-h-[44px]',
            'border border-[#007AFF] text-[#007AFF]',
            'font-medium text-[15px] rounded-xl',
            'transition-all duration-200',
            'hover:bg-[#007AFF]/10 hover:-translate-y-0.5',
            'active:scale-[0.98]',
            'focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-[#007AFF]/40'
          )}
          aria-label="Retry loading data"
        >
          <RefreshCw className='h-4 w-4' aria-hidden="true" />
          Retry
        </button>
      )}
    </div>
  );
}

/**
 * Network Error Fallback component
 */
export function NetworkErrorFallback({ onRetry }: { onRetry?: () => void }) {
  return (
    <FullPageError
      error="Unable to connect to the server. Please check your internet connection and try again."
      type="network"
      onRetry={onRetry}
      suggestions={[
        'Check your internet connection',
        'Try refreshing the page',
        'Disable VPN or proxy if enabled',
      ]}
    />
  );
}

/**
 * Network Error component (legacy compatibility)
 */
export function NetworkError({ onRetry }: { onRetry?: () => void }) {
  return (
    <ErrorDisplay
      error='Unable to connect to the server. Please check your internet connection and try again.'
      type='network'
      onRetry={onRetry}
    />
  );
}

/**
 * Validation Error component for multiple errors
 */
export function ValidationError({ errors }: { errors: string[] }) {
  return (
    <div className='space-y-2' role="alert">
      {errors.map((error, index) => (
        <InlineError key={index} error={error} />
      ))}
    </div>
  );
}


/**
 * Assessment Error Fallback - Specific to assessment workflows
 */
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
    <div
      className='rounded-2xl border border-[#FF3B30]/30 bg-[#FF3B30]/10 p-8 text-center'
      role='alert'
    >
      <div className="w-14 h-14 rounded-full bg-[#FF3B30]/20 flex items-center justify-center mx-auto mb-4">
        <AlertTriangle className='h-7 w-7 text-[#FF3B30]' aria-hidden="true" />
      </div>

      <h3 className='text-[22px] font-semibold text-[#CC2F26] mb-2'>
        Assessment Error
      </h3>

      <p className='text-[15px] text-[#CC2F26]/80 mb-6 max-w-md mx-auto'>
        {assessmentStep
          ? `An error occurred during ${assessmentStep} processing: ${error.message}`
          : `Assessment failed: ${error.message}`}
      </p>

      <div className="mb-6 text-left max-w-sm mx-auto">
        <p className="text-[13px] font-medium text-[#CC2F26]/70 mb-2">
          Suggestions:
        </p>
        <ul className="space-y-1">
          <li className="text-[13px] text-[#CC2F26]/60 flex items-start gap-2">
            <span>•</span>
            <span>Try the assessment again</span>
          </li>
          <li className="text-[13px] text-[#CC2F26]/60 flex items-start gap-2">
            <span>•</span>
            <span>Ensure your device has proper permissions</span>
          </li>
          <li className="text-[13px] text-[#CC2F26]/60 flex items-start gap-2">
            <span>•</span>
            <span>Check your internet connection</span>
          </li>
        </ul>
      </div>

      <div className='flex items-center justify-center gap-4'>
        {onRetryAssessment && (
          <button
            onClick={onRetryAssessment}
            className={cn(
              'inline-flex items-center gap-2 px-6 py-3 min-h-[48px]',
              'bg-[#FF3B30] hover:bg-[#CC2F26]',
              'text-white font-semibold text-[17px]',
              'rounded-xl',
              'transition-all duration-200',
              'hover:-translate-y-0.5 active:scale-[0.98]',
              'focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-[#FF3B30]/40'
            )}
            aria-label='Retry assessment'
          >
            <RefreshCw className='h-5 w-5' aria-hidden="true" />
            Retry Assessment
          </button>
        )}

        <button
          onClick={resetError}
          className={cn(
            'inline-flex items-center gap-2 px-6 py-3 min-h-[48px]',
            'border border-[#FF3B30] text-[#FF3B30]',
            'font-semibold text-[17px] rounded-xl',
            'transition-all duration-200',
            'hover:bg-[#FF3B30]/10 hover:-translate-y-0.5',
            'active:scale-[0.98]',
            'focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-[#FF3B30]/40'
          )}
          aria-label='Go back'
        >
          Go Back
        </button>
      </div>
    </div>
  );
}

/**
 * Enhanced error boundary component with logging and recovery
 */
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
  isolate?: boolean;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    const errorId = `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    return { hasError: true, error, errorId };
  }

  override componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    const errorId = this.state.errorId || `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    const errorReport = {
      error: { name: error.name, message: error.message, stack: error.stack },
      errorInfo: { componentStack: errorInfo.componentStack },
      context: {
        level: this.props.level || 'component',
        name: this.props.name || 'Unknown',
        timestamp: new Date().toISOString(),
        userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown',
        url: typeof window !== 'undefined' ? window.location.href : 'unknown',
      },
      errorId,
    };

    console.error('Error caught by boundary:', errorReport);
    this.setState({ errorInfo, errorId });
    this.props.onError?.(error, errorInfo, errorId);
  }

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

/**
 * Default error fallback component
 */
interface DefaultErrorFallbackProps {
  error: Error;
  errorInfo?: React.ErrorInfo;
  resetError: () => void;
  errorId?: string;
}

export function DefaultErrorFallback({
  error,
  resetError,
  errorId,
}: DefaultErrorFallbackProps) {
  return (
    <FullPageError
      error={error.message || 'An unexpected error occurred. Please try again.'}
      type="unknown"
      onRetry={resetError}
      suggestions={[
        'Try refreshing the page',
        'Clear your browser cache',
        errorId ? `Error ID: ${errorId}` : 'Contact support if the issue persists',
      ]}
    />
  );
}

/**
 * Utility function to determine error type from error message
 */
export function getErrorType(error: string): ErrorType {
  const lowerError = error.toLowerCase();

  // Check timeout FIRST since "timed out" messages may contain "connection"
  if (lowerError.includes('timeout') || lowerError.includes('time out') || lowerError.includes('timed out')) {
    return 'timeout';
  }
  // Network errors - includes "connect" for messages like "Unable to connect"
  if (lowerError.includes('network') || lowerError.includes('connection') || lowerError.includes('connect') || lowerError.includes('offline')) {
    return 'network';
  }
  if (lowerError.includes('validation') || lowerError.includes('invalid') || lowerError.includes('required')) {
    return 'validation';
  }
  if (lowerError.includes('server') || lowerError.includes('internal') || lowerError.includes('500')) {
    return 'server';
  }
  if (lowerError.includes('not found') || lowerError.includes('404')) {
    return 'not-found';
  }
  if (lowerError.includes('permission') || lowerError.includes('unauthorized') || lowerError.includes('forbidden') || lowerError.includes('403')) {
    return 'permission';
  }

  return 'unknown';
}

/**
 * Utility function to create standardized error state
 */
export function createErrorState(error: string | Error, type?: ErrorType): ErrorState {
  const message = typeof error === 'string' ? error : error.message;
  const errorType = type || getErrorType(message);

  return {
    message,
    type: errorType,
    technicalDetails: typeof error === 'object' ? error.stack : undefined,
    suggestions: getDefaultSuggestions(errorType),
    retryable: ['network', 'timeout', 'server'].includes(errorType),
  };
}
