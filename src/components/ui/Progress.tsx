'use client';

import React from 'react';
import type { ProgressProps } from '@/types/design-system';
import { cn } from '@/utils/cn';
import { getRiskLevel, getRiskColor } from '@/types/design-system';

/**
 * Clinical-grade Progress component with risk-based coloring and accessibility
 */
export const Progress: React.FC<ProgressProps> = ({
  value,
  max = 100,
  variant = 'default',
  size = 'md',
  showLabel = false,
  animated = true,
  className,
  testId,
  ...props
}) => {
  // Normalize value to percentage
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

  // Base progress container classes
  const containerClasses = [
    'progress-bar',
    'w-full',
    'bg-neutral-800',
    'rounded-full',
    'overflow-hidden',
    'relative',
  ];

  // Size classes
  const sizeClasses = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3',
  };

  // Progress fill classes
  const fillBaseClasses = [
    'progress-fill',
    'h-full',
    'rounded-full',
    'transition-all',
    animated ? 'duration-slow' : 'duration-instant',
    'ease-out-quint',
  ];

  // Variant-specific fill classes
  const variantClasses = {
    default: 'bg-primary-500',
    risk: getRiskBasedClasses(percentage),
    clinical: 'bg-gradient-to-r from-primary-500 to-primary-600',
  };

  function getRiskBasedClasses(percent: number): string {
    const riskLevel = getRiskLevel(percent);
    const colorMap = {
      low: 'bg-success',
      moderate: 'bg-warning',
      high: 'bg-orange-500',
      critical: 'bg-error',
    };
    return colorMap[riskLevel];
  }

  // Combine container classes
  const progressContainerClasses = cn(
    ...containerClasses,
    sizeClasses[size],
    className
  );

  // Combine fill classes
  const progressFillClasses = cn(...fillBaseClasses, variantClasses[variant]);

  return (
    <div className="space-y-2">
      {/* Label */}
      {showLabel && (
        <div className="flex items-center justify-between text-sm">
          <span className="text-text-secondary">Progress</span>
          <span className="font-medium text-text-primary">
            {Math.round(percentage)}%
          </span>
        </div>
      )}

      {/* Progress Bar */}
      <div
        className={progressContainerClasses}
        role="progressbar"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={max}
        aria-label={`Progress: ${Math.round(percentage)}%`}
        data-testid={testId}
        {...props}
      >
        <div
          className={progressFillClasses}
          style={{ width: `${percentage}%` }}
        />

        {/* Animated shimmer effect for loading states */}
        {animated && (
          <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-white/10 to-transparent" />
        )}
      </div>
    </div>
  );
};

/**
 * Circular Progress component for compact displays
 */
interface CircularProgressProps extends Omit<ProgressProps, 'size'> {
  size?: number; // Size in pixels
  strokeWidth?: number;
  showPercentage?: boolean;
}

export const CircularProgress: React.FC<CircularProgressProps> = ({
  value,
  max = 100,
  variant = 'default',
  size = 120,
  strokeWidth = 8,
  showPercentage = true,
  animated = true,
  className,
  testId,
  ...props
}) => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const strokeDasharray = circumference;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  // Get color based on variant
  const getStrokeColor = (): string => {
    switch (variant) {
      case 'risk':
        const riskLevel = getRiskLevel(percentage);
        return getRiskColor(riskLevel);
      case 'clinical':
        return 'var(--primary-500)';
      default:
        return 'var(--primary-500)';
    }
  };

  return (
    <div
      className={cn(
        'relative inline-flex items-center justify-center',
        className
      )}
      style={{ width: size, height: size }}
      role="progressbar"
      aria-valuenow={value}
      aria-valuemin={0}
      aria-valuemax={max}
      aria-label={`Progress: ${Math.round(percentage)}%`}
      data-testid={testId}
      {...props}
    >
      {/* Background Circle */}
      <svg className="-rotate-90 transform" width={size} height={size}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="var(--neutral-800)"
          strokeWidth={strokeWidth}
          fill="transparent"
        />

        {/* Progress Circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={getStrokeColor()}
          strokeWidth={strokeWidth}
          fill="transparent"
          strokeDasharray={strokeDasharray}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className={cn(
            'transition-all',
            animated ? 'duration-slow' : 'duration-instant',
            'ease-out-quint'
          )}
        />
      </svg>

      {/* Center Content */}
      {showPercentage && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-2xl font-bold text-text-primary">
            {Math.round(percentage)}%
          </span>
        </div>
      )}
    </div>
  );
};

/**
 * Multi-step Progress component for assessment flow
 */
interface StepProgressProps {
  currentStep: number;
  totalSteps: number;
  steps?: Array<{
    id: string;
    label: string;
    completed?: boolean;
  }>;
  className?: string;
  testId?: string;
}

export const StepProgress: React.FC<StepProgressProps> = ({
  currentStep,
  totalSteps,
  steps,
  className,
  testId,
}) => {
  const progressPercentage = ((currentStep - 1) / (totalSteps - 1)) * 100;

  return (
    <div className={cn('space-y-4', className)} data-testid={testId}>
      {/* Progress Bar */}
      <div className="relative">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-sm text-text-secondary">
            Step {currentStep} of {totalSteps}
          </span>
          <span className="text-sm font-medium text-text-primary">
            {Math.round(progressPercentage)}% Complete
          </span>
        </div>

        <Progress
          value={progressPercentage}
          variant="clinical"
          size="md"
          animated={true}
        />
      </div>

      {/* Step Indicators */}
      {steps && (
        <div className="flex justify-between">
          {steps.map((step, index) => {
            const stepNumber = index + 1;
            const isCompleted = step.completed || stepNumber < currentStep;
            const isCurrent = stepNumber === currentStep;

            return (
              <div
                key={step.id}
                className="flex flex-col items-center space-y-2"
              >
                {/* Step Circle */}
                <div
                  className={cn(
                    'flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium transition-all duration-200',
                    isCompleted && 'bg-success text-white',
                    isCurrent &&
                      'bg-primary-500 text-white ring-2 ring-primary-500/30',
                    !isCompleted &&
                      !isCurrent &&
                      'bg-neutral-700 text-text-muted'
                  )}
                >
                  {isCompleted ? (
                    <svg
                      className="h-4 w-4"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                  ) : (
                    stepNumber
                  )}
                </div>

                {/* Step Label */}
                <span
                  className={cn(
                    'max-w-20 text-center text-xs',
                    isCurrent && 'font-medium text-text-primary',
                    isCompleted && 'text-text-secondary',
                    !isCompleted && !isCurrent && 'text-text-muted'
                  )}
                >
                  {step.label}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

/**
 * NRI Score Progress - Specialized progress for neurological risk index
 */
interface NRIProgressProps {
  score: number;
  confidence?: number;
  animated?: boolean;
  showDetails?: boolean;
  className?: string;
}

export const NRIProgress: React.FC<NRIProgressProps> = ({
  score,
  confidence,
  animated = true,
  showDetails = true,
  className,
}) => {
  const riskLevel = getRiskLevel(score);
  const riskLabels = {
    low: 'Low Risk',
    moderate: 'Moderate Risk',
    high: 'High Risk',
    critical: 'Critical Risk',
  };

  const riskDescriptions = {
    low: 'Routine monitoring recommended',
    moderate: 'Annual screening suggested',
    high: 'Specialist consultation advised',
    critical: 'Immediate medical attention required',
  };

  return (
    <div className={cn('space-y-4', className)}>
      {/* Score Display */}
      <div className="space-y-2 text-center">
        <div className="text-6xl font-black text-primary-500">
          {Math.round(score)}
        </div>
        <div className="text-lg text-text-muted">/ 100 NRI Score</div>
        {confidence && (
          <div className="text-sm text-text-muted">
            Confidence: ±{confidence}%
          </div>
        )}
      </div>

      {/* Risk Level Progress */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-text-primary">
            {riskLabels[riskLevel]}
          </span>
          <span className="text-sm text-text-secondary">
            {Math.round(score)}%
          </span>
        </div>

        <Progress value={score} variant="risk" size="lg" animated={animated} />

        {showDetails && (
          <p className="text-center text-sm text-text-secondary">
            {riskDescriptions[riskLevel]}
          </p>
        )}
      </div>

      {/* Risk Scale Reference */}
      {showDetails && (
        <div className="grid grid-cols-4 gap-2 text-xs">
          <div className="text-center">
            <div className="mb-1 h-2 w-full rounded bg-success" />
            <span className="text-text-muted">0-25</span>
          </div>
          <div className="text-center">
            <div className="mb-1 h-2 w-full rounded bg-warning" />
            <span className="text-text-muted">26-50</span>
          </div>
          <div className="text-center">
            <div className="mb-1 h-2 w-full rounded bg-orange-500" />
            <span className="text-text-muted">51-75</span>
          </div>
          <div className="text-center">
            <div className="mb-1 h-2 w-full rounded bg-error" />
            <span className="text-text-muted">76-100</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default Progress;
