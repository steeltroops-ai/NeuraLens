/**
 * ErrorAlert Component
 * 
 * Reusable error display component with:
 * - Severity levels
 * - Recovery steps
 * - Retry functionality with exponential backoff
 * - Accessibility support
 * 
 * Requirements: 13.1-13.4
 * 
 * @module components/common/ErrorAlert
 */

'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  AlertCircle, 
  AlertTriangle, 
  XCircle,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  X,
  HelpCircle,
  Wifi,
  WifiOff,
  Clock
} from 'lucide-react';

// ============================================================================
// Types
// ============================================================================

export type ErrorSeverity = 'info' | 'warning' | 'error' | 'critical';

export interface ErrorAlertProps {
  /** Error message to display */
  message: string;
  /** Severity level */
  severity?: ErrorSeverity;
  /** Technical details (collapsible) */
  details?: string;
  /** Recovery steps to show */
  recoverySteps?: string[];
  /** Show retry button */
  showRetry?: boolean;
  /** Called when retry is clicked */
  onRetry?: () => void | Promise<void>;
  /** Show dismiss button */
  dismissible?: boolean;
  /** Called when dismissed */
  onDismiss?: () => void;
  /** Error code (optional) */
  errorCode?: string;
  /** Max retry attempts */
  maxRetries?: number;
  /** Initial retry delay in ms */
  initialRetryDelay?: number;
  /** Show network status */
  showNetworkStatus?: boolean;
  /** Auto retry on reconnection */
  autoRetryOnReconnect?: boolean;
  /** Support link */
  supportLink?: string;
}

// ============================================================================
// Severity Configuration
// ============================================================================

const SEVERITY_CONFIG = {
  info: {
    icon: AlertCircle,
    bgClass: 'bg-blue-50',
    borderClass: 'border-blue-200',
    textClass: 'text-blue-900',
    textSecondaryClass: 'text-blue-700',
    iconClass: 'text-blue-600',
    buttonClass: 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-500',
  },
  warning: {
    icon: AlertTriangle,
    bgClass: 'bg-amber-50',
    borderClass: 'border-amber-200',
    textClass: 'text-amber-900',
    textSecondaryClass: 'text-amber-700',
    iconClass: 'text-amber-600',
    buttonClass: 'bg-amber-600 hover:bg-amber-700 focus:ring-amber-500',
  },
  error: {
    icon: AlertCircle,
    bgClass: 'bg-red-50',
    borderClass: 'border-red-200',
    textClass: 'text-red-900',
    textSecondaryClass: 'text-red-700',
    iconClass: 'text-red-600',
    buttonClass: 'bg-red-600 hover:bg-red-700 focus:ring-red-500',
  },
  critical: {
    icon: XCircle,
    bgClass: 'bg-red-100',
    borderClass: 'border-red-300',
    textClass: 'text-red-900',
    textSecondaryClass: 'text-red-800',
    iconClass: 'text-red-700',
    buttonClass: 'bg-red-700 hover:bg-red-800 focus:ring-red-600',
  },
};

// ============================================================================
// Network Status Hook
// ============================================================================

function useNetworkStatus() {
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    if (typeof window === 'undefined') return;

    setIsOnline(navigator.onLine);

    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return isOnline;
}

// ============================================================================
// Main Component
// ============================================================================

export function ErrorAlert({
  message,
  severity = 'error',
  details,
  recoverySteps,
  showRetry = true,
  onRetry,
  dismissible = false,
  onDismiss,
  errorCode,
  maxRetries = 3,
  initialRetryDelay = 1000,
  showNetworkStatus = false,
  autoRetryOnReconnect = false,
  supportLink = '/support',
}: ErrorAlertProps) {
  const [isVisible, setIsVisible] = useState(true);
  const [showDetails, setShowDetails] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [isRetrying, setIsRetrying] = useState(false);
  const [retryCountdown, setRetryCountdown] = useState(0);

  const isOnline = useNetworkStatus();
  const config = SEVERITY_CONFIG[severity];
  const Icon = config.icon;
  const canRetry = retryCount < maxRetries && !isRetrying;

  // ============================================================================
  // Exponential Backoff Retry
  // ============================================================================

  const calculateRetryDelay = useCallback((attempt: number) => {
    // Exponential backoff: 1s, 2s, 4s, 8s, etc.
    return Math.min(initialRetryDelay * Math.pow(2, attempt), 30000);
  }, [initialRetryDelay]);

  const handleRetry = useCallback(async () => {
    if (!canRetry || !onRetry) return;

    setIsRetrying(true);
    const delay = calculateRetryDelay(retryCount);

    // Show countdown
    setRetryCountdown(Math.ceil(delay / 1000));
    const countdownInterval = setInterval(() => {
      setRetryCountdown(prev => Math.max(0, prev - 1));
    }, 1000);

    // Wait for delay
    await new Promise(resolve => setTimeout(resolve, delay));
    clearInterval(countdownInterval);

    try {
      await onRetry();
    } catch (error) {
      setRetryCount(prev => prev + 1);
    } finally {
      setIsRetrying(false);
      setRetryCountdown(0);
    }
  }, [canRetry, onRetry, retryCount, calculateRetryDelay]);

  // ============================================================================
  // Auto Retry on Reconnect
  // ============================================================================

  useEffect(() => {
    if (autoRetryOnReconnect && isOnline && !isRetrying && retryCount < maxRetries && onRetry) {
      // Wait a moment to ensure stable connection
      const timeout = setTimeout(() => {
        handleRetry();
      }, 2000);
      return () => clearTimeout(timeout);
    }
    return undefined;
  }, [isOnline, autoRetryOnReconnect, isRetrying, retryCount, maxRetries, handleRetry, onRetry]);

  // ============================================================================
  // Dismiss Handler
  // ============================================================================

  const handleDismiss = useCallback(() => {
    setIsVisible(false);
    setTimeout(() => {
      onDismiss?.();
    }, 200);
  }, [onDismiss]);

  // ============================================================================
  // Render
  // ============================================================================

  if (!isVisible) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        className={`rounded-xl border ${config.borderClass} ${config.bgClass} p-4`}
        role="alert"
        aria-live={severity === 'critical' ? 'assertive' : 'polite'}
        aria-atomic="true"
      >
        <div className="flex items-start gap-3">
          {/* Icon */}
          <div className={`flex-shrink-0 mt-0.5 ${config.iconClass}`}>
            <Icon className="h-5 w-5" aria-hidden="true" />
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            {/* Header */}
            <div className="flex items-start justify-between gap-2">
              <div>
                <p className={`font-medium ${config.textClass}`}>
                  {message}
                </p>
                {errorCode && (
                  <p className={`text-xs mt-0.5 ${config.textSecondaryClass}`}>
                    Error code: {errorCode}
                  </p>
                )}
              </div>

              {/* Dismiss Button */}
              {dismissible && (
                <button
                  onClick={handleDismiss}
                  className={`flex-shrink-0 p-1 rounded-full hover:bg-black/5 transition-colors ${config.textSecondaryClass}`}
                  aria-label="Dismiss alert"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>

            {/* Network Status */}
            {showNetworkStatus && (
              <div className={`flex items-center gap-2 mt-2 text-sm ${config.textSecondaryClass}`}>
                {isOnline ? (
                  <>
                    <Wifi className="h-4 w-4" aria-hidden="true" />
                    <span>Network connected</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="h-4 w-4" aria-hidden="true" />
                    <span>No network connection - will retry when reconnected</span>
                  </>
                )}
              </div>
            )}

            {/* Recovery Steps */}
            {recoverySteps && recoverySteps.length > 0 && (
              <div className="mt-3">
                <p className={`text-sm font-medium ${config.textSecondaryClass} mb-1`}>
                  Try these steps:
                </p>
                <ul className="space-y-1">
                  {recoverySteps.map((step, index) => (
                    <li 
                      key={index}
                      className={`flex items-start gap-2 text-sm ${config.textSecondaryClass}`}
                    >
                      <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-current flex-shrink-0" />
                      {step}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Details (Collapsible) */}
            {details && (
              <div className="mt-3">
                <button
                  onClick={() => setShowDetails(!showDetails)}
                  className={`flex items-center gap-1 text-sm ${config.textSecondaryClass} hover:underline`}
                  aria-expanded={showDetails}
                  aria-controls="error-details"
                >
                  {showDetails ? (
                    <>
                      <ChevronUp className="h-4 w-4" />
                      Hide technical details
                    </>
                  ) : (
                    <>
                      <ChevronDown className="h-4 w-4" />
                      Show technical details
                    </>
                  )}
                </button>
                
                <AnimatePresence>
                  {showDetails && (
                    <motion.div
                      id="error-details"
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="overflow-hidden"
                    >
                      <pre className={`mt-2 p-2 rounded-lg bg-black/5 text-xs ${config.textSecondaryClass} overflow-x-auto whitespace-pre-wrap`}>
                        {details}
                      </pre>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )}

            {/* Actions */}
            <div className="flex flex-wrap items-center gap-3 mt-4">
              {/* Retry Button */}
              {showRetry && onRetry && (
                <button
                  onClick={handleRetry}
                  disabled={!canRetry}
                  className={`
                    inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-white
                    ${config.buttonClass}
                    transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2
                    disabled:opacity-50 disabled:cursor-not-allowed
                  `}
                  aria-label={
                    isRetrying 
                      ? `Retrying in ${retryCountdown} seconds` 
                      : `Retry, attempt ${retryCount + 1} of ${maxRetries}`
                  }
                >
                  {isRetrying ? (
                    <>
                      <Clock className="h-4 w-4 animate-pulse" />
                      Retrying in {retryCountdown}s...
                    </>
                  ) : (
                    <>
                      <RefreshCw className="h-4 w-4" />
                      Retry {retryCount > 0 ? `(${retryCount}/${maxRetries})` : ''}
                    </>
                  )}
                </button>
              )}

              {/* Support Link */}
              {severity === 'critical' && (
                <a
                  href={supportLink}
                  className={`
                    inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
                    border ${config.borderClass} ${config.textSecondaryClass}
                    hover:bg-black/5 transition-colors
                  `}
                >
                  <HelpCircle className="h-4 w-4" />
                  Contact Support
                </a>
              )}
            </div>

            {/* Retry Info */}
            {retryCount > 0 && !isRetrying && (
              <p className={`text-xs mt-2 ${config.textSecondaryClass}`}>
                {canRetry 
                  ? `${maxRetries - retryCount} ${maxRetries - retryCount === 1 ? 'retry' : 'retries'} remaining`
                  : 'Maximum retries reached. Please contact support.'}
              </p>
            )}
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

export default ErrorAlert;
