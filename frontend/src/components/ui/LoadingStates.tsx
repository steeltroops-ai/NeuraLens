/**
 * Loading State Components
 * Various loading indicators and progress displays
 */

import React from 'react';
import { Loader2, Upload, Brain, Eye, Activity, Zap, AlertTriangle } from 'lucide-react';

// Basic loading spinner
interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function LoadingSpinner({ size = 'md', className = '' }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
  };

  return <Loader2 className={`animate-spin ${sizeClasses[size]} ${className}`} />;
}

// Loading overlay
interface LoadingOverlayProps {
  message?: string;
  progress?: number;
  className?: string;
}

export function LoadingOverlay({
  message = 'Loading...',
  progress,
  className = '',
}: LoadingOverlayProps) {
  return (
    <div
      className={`absolute inset-0 flex flex-col items-center justify-center bg-white bg-opacity-90 ${className} `}
    >
      <LoadingSpinner size='lg' className='mb-4 text-blue-600' />
      <p className='mb-2 font-medium text-gray-700'>{message}</p>

      {progress !== undefined && (
        <div className='h-2 w-48 rounded-full bg-gray-200'>
          <div
            className='h-2 rounded-full bg-blue-600 transition-all duration-300'
            style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
          />
        </div>
      )}
    </div>
  );
}

// Assessment-specific loading states
interface AssessmentLoadingProps {
  type: 'speech' | 'retinal' | 'motor' | 'cognitive' | 'nri';
  message?: string;
  progress?: number;
}

export function AssessmentLoading({ type, message, progress }: AssessmentLoadingProps) {
  const getIcon = () => {
    switch (type) {
      case 'speech':
        return <Activity className='h-8 w-8 text-blue-600' />;
      case 'retinal':
        return <Eye className='h-8 w-8 text-green-600' />;
      case 'motor':
        return <Activity className='h-8 w-8 text-orange-600' />;
      case 'cognitive':
        return <Brain className='h-8 w-8 text-purple-600' />;
      case 'nri':
        return <Zap className='h-8 w-8 text-yellow-600' />;
    }
  };

  const getDefaultMessage = () => {
    switch (type) {
      case 'speech':
        return 'Analyzing speech patterns...';
      case 'retinal':
        return 'Processing retinal image...';
      case 'motor':
        return 'Evaluating motor function...';
      case 'cognitive':
        return 'Assessing cognitive performance...';
      case 'nri':
        return 'Computing neurological risk index...';
    }
  };

  return (
    <div className='flex flex-col items-center justify-center p-8 text-center'>
      <div className='relative mb-4'>
        {getIcon()}
        <div className='absolute -inset-2'>
          <div className='h-12 w-12 animate-spin rounded-full border-2 border-current border-t-transparent opacity-30' />
        </div>
      </div>

      <h3 className='mb-2 text-lg font-semibold text-gray-900'>
        {type.charAt(0).toUpperCase() + type.slice(1)} Analysis
      </h3>

      <p className='mb-4 text-gray-600'>{message || getDefaultMessage()}</p>

      {progress !== undefined && (
        <div className='mb-2 h-2 w-64 rounded-full bg-gray-200'>
          <div
            className='h-2 rounded-full bg-current transition-all duration-500'
            style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
          />
        </div>
      )}

      <p className='text-sm text-gray-500'>This may take a few moments...</p>
    </div>
  );
}

// File upload loading
interface UploadLoadingProps {
  fileName: string;
  progress: number;
  onCancel?: () => void;
}

export function UploadLoading({ fileName, progress, onCancel }: UploadLoadingProps) {
  return (
    <div className='flex items-center gap-4 rounded-lg border border-blue-200 bg-blue-50 p-4'>
      <Upload className='h-6 w-6 text-blue-600' />

      <div className='min-w-0 flex-1'>
        <p className='truncate text-sm font-medium text-gray-900'>Uploading {fileName}</p>

        <div className='mt-1 flex items-center gap-2'>
          <div className='h-2 flex-1 rounded-full bg-blue-200'>
            <div
              className='h-2 rounded-full bg-blue-600 transition-all duration-300'
              style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
            />
          </div>
          <span className='text-xs font-medium text-blue-600'>{Math.round(progress)}%</span>
        </div>
      </div>

      {onCancel && (
        <button onClick={onCancel} className='text-gray-400 transition-colors hover:text-gray-600'>
          ×
        </button>
      )}
    </div>
  );
}

// Multi-step progress indicator
interface ProgressStepsProps {
  steps: string[];
  currentStep: number;
  completedSteps: number[];
  className?: string;
}

export function ProgressSteps({
  steps,
  currentStep,
  completedSteps,
  className = '',
}: ProgressStepsProps) {
  return (
    <div className={`flex items-center justify-between ${className}`}>
      {steps.map((step, index) => {
        const isCompleted = completedSteps.includes(index);
        const isCurrent = currentStep === index;
        const isUpcoming = index > currentStep;

        return (
          <React.Fragment key={index}>
            <div className='flex flex-col items-center'>
              <div
                className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium ${
                  isCompleted
                    ? 'bg-green-600 text-white'
                    : isCurrent
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 text-gray-500'
                } `}
              >
                {isCompleted ? '✓' : index + 1}
              </div>

              <span
                className={`mt-2 max-w-16 text-center text-xs ${isCurrent ? 'font-medium text-blue-600' : 'text-gray-500'} `}
              >
                {step}
              </span>
            </div>

            {index < steps.length - 1 && (
              <div
                className={`mx-2 h-0.5 flex-1 ${completedSteps.includes(index) ? 'bg-green-600' : 'bg-gray-200'} `}
              />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}

// Skeleton loading for content
interface SkeletonProps {
  className?: string;
  lines?: number;
  width?: string;
  height?: string;
}

export function Skeleton({ className = '', lines = 1, width, height }: SkeletonProps) {
  const style = {
    ...(width && { width }),
    ...(height && { height }),
  };

  return (
    <div className={`animate-pulse ${className}`} style={style}>
      {Array.from({ length: lines }).map((_, index) => (
        <div
          key={index}
          className={`rounded bg-gray-200 ${lines > 1 ? 'mb-2 h-4 last:mb-0' : 'h-4'} ${index === lines - 1 && lines > 1 ? 'w-3/4' : 'w-full'} `}
        />
      ))}
    </div>
  );
}

// Card skeleton
export function CardSkeleton() {
  return (
    <div className='animate-pulse rounded-lg border border-gray-200 p-6'>
      <div className='mb-4 flex items-center gap-4'>
        <div className='h-12 w-12 rounded-full bg-gray-200' />
        <div className='flex-1'>
          <div className='mb-2 h-4 w-1/3 rounded bg-gray-200' />
          <div className='h-3 w-1/2 rounded bg-gray-200' />
        </div>
      </div>

      <div className='space-y-2'>
        <div className='h-3 rounded bg-gray-200' />
        <div className='h-3 w-5/6 rounded bg-gray-200' />
        <div className='h-3 w-4/6 rounded bg-gray-200' />
      </div>
    </div>
  );
}

// Button loading state
interface LoadingButtonProps {
  loading: boolean;
  children: React.ReactNode;
  className?: string;
  disabled?: boolean;
  onClick?: () => void;
  loadingText?: string;
  'aria-label'?: string;
}

export function LoadingButton({
  loading,
  children,
  className = '',
  disabled,
  onClick,
  loadingText,
  'aria-label': ariaLabel,
}: LoadingButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={loading || disabled}
      aria-label={ariaLabel || (loading ? loadingText || 'Loading...' : undefined)}
      aria-busy={loading}
      className={`flex items-center justify-center gap-2 transition-all duration-200 disabled:cursor-not-allowed disabled:opacity-50 ${className} `}
    >
      {loading && <LoadingSpinner size='sm' />}
      <span className={loading ? 'opacity-75' : ''}>
        {loading && loadingText ? loadingText : children}
      </span>
    </button>
  );
}

// Assessment results skeleton
export function AssessmentResultsSkeleton() {
  return (
    <div
      className='rounded-lg bg-white shadow-lg'
      role='status'
      aria-label='Loading assessment results'
    >
      {/* Header skeleton */}
      <div className='border-b border-gray-200 p-6'>
        <div className='flex items-center justify-between'>
          <div className='space-y-2'>
            <Skeleton width='200px' height='2rem' />
            <Skeleton width='150px' height='1rem' />
            <Skeleton width='180px' height='0.875rem' />
          </div>
          <div className='flex gap-2'>
            <Skeleton width='80px' height='2.5rem' />
            <Skeleton width='80px' height='2.5rem' />
            <Skeleton width='80px' height='2.5rem' />
          </div>
        </div>
      </div>

      {/* Risk assessment skeleton */}
      <div className='border-b border-gray-200 p-6'>
        <Skeleton width='200px' height='1.5rem' className='mb-6' />
        <div className='grid grid-cols-1 gap-6 md:grid-cols-3'>
          {[1, 2, 3].map(i => (
            <div key={i} className='text-center'>
              <Skeleton width='96px' height='96px' className='mx-auto mb-3 rounded-full' />
              <Skeleton width='80px' height='1rem' className='mx-auto mb-1' />
              <Skeleton width='120px' height='0.75rem' className='mx-auto' />
            </div>
          ))}
        </div>
      </div>

      {/* Modality results skeleton */}
      <div className='p-6'>
        <Skeleton width='180px' height='1.5rem' className='mb-6' />
        <div className='grid grid-cols-1 gap-6 lg:grid-cols-2'>
          {[1, 2, 3, 4].map(i => (
            <div key={i} className='rounded-lg border border-gray-200 p-4'>
              <div className='mb-4 flex items-center gap-3'>
                <Skeleton width='36px' height='36px' className='rounded-lg' />
                <div className='space-y-1'>
                  <Skeleton width='120px' height='1.25rem' />
                  <Skeleton width='100px' height='0.875rem' />
                </div>
              </div>
              <div className='space-y-2'>
                {[1, 2, 3].map(j => (
                  <div key={j} className='flex justify-between'>
                    <Skeleton width='80px' height='0.875rem' />
                    <Skeleton width='40px' height='0.875rem' />
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Assessment history skeleton
export function AssessmentHistorySkeleton() {
  return (
    <div className='space-y-4' role='status' aria-label='Loading assessment history'>
      {[1, 2, 3, 4, 5].map(i => (
        <div key={i} className='rounded-lg border border-gray-200 bg-white p-6'>
          <div className='mb-4 flex items-center justify-between'>
            <div className='space-y-2'>
              <Skeleton width='150px' height='1.25rem' />
              <Skeleton width='200px' height='0.875rem' />
            </div>
            <div className='flex items-center gap-2'>
              <Skeleton width='80px' height='1.5rem' className='rounded-full' />
              <Skeleton width='60px' height='0.875rem' />
            </div>
          </div>

          <div className='grid grid-cols-2 gap-4 md:grid-cols-4'>
            {[1, 2, 3, 4].map(j => (
              <div key={j} className='text-center'>
                <Skeleton width='40px' height='40px' className='mx-auto mb-2 rounded-lg' />
                <Skeleton width='60px' height='0.75rem' className='mx-auto mb-1' />
                <Skeleton width='50px' height='0.75rem' className='mx-auto' />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// Loading timeout handler
interface LoadingTimeoutProps {
  isLoading: boolean;
  timeout?: number;
  onTimeout?: () => void;
  children: React.ReactNode;
  timeoutMessage?: string;
}

export function LoadingTimeout({
  isLoading,
  timeout = 30000, // 30 seconds default
  onTimeout,
  children,
  timeoutMessage = 'This is taking longer than expected. Please check your connection and try again.',
}: LoadingTimeoutProps) {
  const [hasTimedOut, setHasTimedOut] = React.useState(false);

  React.useEffect(() => {
    if (!isLoading) {
      setHasTimedOut(false);
      return;
    }

    const timer = setTimeout(() => {
      setHasTimedOut(true);
      onTimeout?.();
    }, timeout);

    return () => clearTimeout(timer);
  }, [isLoading, timeout, onTimeout]);

  if (hasTimedOut) {
    return (
      <div className='p-8 text-center' role='alert'>
        <AlertTriangle className='mx-auto mb-4 h-12 w-12 text-yellow-500' />
        <h3 className='mb-2 text-lg font-semibold text-gray-900'>Loading Timeout</h3>
        <p className='mx-auto mb-4 max-w-md text-gray-600'>{timeoutMessage}</p>
        <button
          onClick={() => {
            setHasTimedOut(false);
            window.location.reload();
          }}
          className='rounded-lg bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700'
        >
          Retry
        </button>
      </div>
    );
  }

  return <>{children}</>;
}
