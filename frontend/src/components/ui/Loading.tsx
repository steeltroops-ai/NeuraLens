'use client';

import React from 'react';

import { cn } from '@/utils/cn';

interface LoadingProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
  text?: string;
  fullScreen?: boolean;
}

const sizeClasses = {
  sm: 'w-4 h-4',
  md: 'w-8 h-8',
  lg: 'w-12 h-12',
  xl: 'w-16 h-16',
};

export const Loading: React.FC<LoadingProps> = ({
  size = 'md',
  className,
  text,
  fullScreen = false,
}) => {
  const spinner = (
    <div className='flex flex-col items-center space-y-4'>
      <div
        className={cn(
          'animate-spin rounded-full border-2 border-primary-500 border-t-transparent',
          sizeClasses[size],
          className,
        )}
        role='status'
        aria-label='Loading'
      />
      {text && <p className='text-sm font-medium text-text-secondary'>{text}</p>}
    </div>
  );

  if (fullScreen) {
    return (
      <div className='bg-surface-background/80 fixed inset-0 z-50 flex items-center justify-center backdrop-blur-sm'>
        {spinner}
      </div>
    );
  }

  return spinner;
};

/**
 * Skeleton loading component for content placeholders
 */
interface SkeletonProps {
  className?: string;
  width?: string | number;
  height?: string | number;
  rounded?: boolean;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className,
  width,
  height,
  rounded = false,
}) => {
  return (
    <div
      className={cn(
        'animate-pulse bg-neutral-800',
        rounded ? 'rounded-full' : 'rounded',
        className,
      )}
      style={{
        width: typeof width === 'number' ? `${width}px` : width,
        height: typeof height === 'number' ? `${height}px` : height,
      }}
      role='status'
      aria-label='Loading content'
    />
  );
};

/**
 * Loading overlay for components
 */
interface LoadingOverlayProps {
  isLoading: boolean;
  children: React.ReactNode;
  text?: string;
  className?: string;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  isLoading,
  children,
  text = 'Loading...',
  className,
}) => {
  return (
    <div className={cn('relative', className)}>
      {children}
      {isLoading && (
        <div className='bg-surface-background/80 absolute inset-0 z-10 flex items-center justify-center backdrop-blur-sm'>
          <Loading text={text} />
        </div>
      )}
    </div>
  );
};
