/**
 * NeuroLens-X Progress Component
 * Neural-themed progress indicators with accessibility
 * WCAG 2.1 AAA compliant with proper ARIA attributes
 */

import React from 'react';
import { cn } from '@/lib/utils';
import type { ProgressProps } from '@/types';

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  ({ 
    value, 
    max = 100, 
    variant = 'default', 
    size = 'md', 
    showLabel = false, 
    className,
    ...props 
  }, ref) => {
    const percentage = Math.min(100, Math.max(0, (value / max) * 100));
    const formattedValue = Math.round(percentage);

    const containerStyles = cn(
      'relative overflow-hidden rounded-full bg-neutral-200',
      {
        'h-1': size === 'sm',
        'h-2': size === 'md',
        'h-3': size === 'lg',
      },
      className
    );

    const barStyles = cn(
      'h-full transition-all duration-500 ease-out rounded-full',
      {
        // Default progress bar
        'bg-gradient-to-r from-primary-500 to-primary-600': variant === 'default',
        
        // Neural-themed with animated gradient
        'bg-gradient-to-r from-primary-500 via-secondary-500 to-primary-600 animate-pulse-slow': 
          variant === 'neural',
        
        // Success state
        'bg-gradient-to-r from-green-500 to-green-600': variant === 'success',
        
        // Warning state
        'bg-gradient-to-r from-yellow-500 to-orange-500': variant === 'warning',
        
        // Error state
        'bg-gradient-to-r from-red-500 to-red-600': variant === 'error',
      }
    );

    return (
      <div className="w-full space-y-2">
        {/* Progress label */}
        {showLabel && (
          <div className="flex justify-between items-center text-sm">
            <span className="text-neutral-700 font-medium">Progress</span>
            <span className="text-neutral-600">{formattedValue}%</span>
          </div>
        )}
        
        {/* Progress bar container */}
        <div
          ref={ref}
          className={containerStyles}
          role="progressbar"
          aria-valuenow={value}
          aria-valuemin={0}
          aria-valuemax={max}
          aria-label={`Progress: ${formattedValue}%`}
          {...props}
        >
          {/* Progress bar fill */}
          <div
            className={barStyles}
            style={{ width: `${percentage}%` }}
          />
          
          {/* Neural variant overlay effect */}
          {variant === 'neural' && (
            <div className="absolute inset-0 opacity-30">
              <div className="h-full bg-gradient-to-r from-transparent via-white to-transparent animate-neural-pulse" />
            </div>
          )}
        </div>
      </div>
    );
  }
);

Progress.displayName = 'Progress';

// Circular Progress Component
const CircularProgress = React.forwardRef<SVGSVGElement, 
  Omit<ProgressProps, 'size'> & { 
    size?: number;
    strokeWidth?: number;
  }
>(({ 
  value, 
  max = 100, 
  variant = 'default', 
  size = 64, 
  strokeWidth = 4,
  showLabel = false, 
  className,
  ...props 
}, ref) => {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  const formattedValue = Math.round(percentage);
  
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDasharray = circumference;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  const strokeColor = {
    default: '#3b82f6',
    neural: '#0d9488',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
  }[variant];

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg
        ref={ref}
        width={size}
        height={size}
        className={cn('transform -rotate-90', className)}
        role="progressbar"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={max}
        aria-label={`Circular progress: ${formattedValue}%`}
        {...props}
      >
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="#e5e7eb"
          strokeWidth={strokeWidth}
          fill="none"
        />
        
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={strokeColor}
          strokeWidth={strokeWidth}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={strokeDasharray}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-500 ease-out"
        />
      </svg>
      
      {/* Center label */}
      {showLabel && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-semibold text-neutral-700">
            {formattedValue}%
          </span>
        </div>
      )}
    </div>
  );
});

CircularProgress.displayName = 'CircularProgress';

// Step Progress Component for multi-step flows
const StepProgress = React.forwardRef<HTMLDivElement, {
  steps: Array<{ id: string; title: string; completed: boolean }>;
  currentStep: number;
  className?: string;
}>(({ steps, currentStep, className, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn('flex items-center justify-between w-full', className)}
      {...props}
    >
      {steps.map((step, index) => (
        <React.Fragment key={step.id}>
          {/* Step indicator */}
          <div className="flex flex-col items-center">
            <div
              className={cn(
                'w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-all duration-200',
                {
                  'bg-primary-500 text-white': index < currentStep || step.completed,
                  'bg-primary-100 text-primary-600 border-2 border-primary-500': index === currentStep,
                  'bg-neutral-200 text-neutral-500': index > currentStep && !step.completed,
                }
              )}
            >
              {step.completed ? (
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
              ) : (
                index + 1
              )}
            </div>
            <span className="mt-2 text-xs text-neutral-600 text-center max-w-20">
              {step.title}
            </span>
          </div>
          
          {/* Connector line */}
          {index < steps.length - 1 && (
            <div
              className={cn(
                'flex-1 h-0.5 mx-2 transition-all duration-200',
                {
                  'bg-primary-500': index < currentStep,
                  'bg-neutral-200': index >= currentStep,
                }
              )}
            />
          )}
        </React.Fragment>
      ))}
    </div>
  );
});

StepProgress.displayName = 'StepProgress';

export { Progress, CircularProgress, StepProgress };
