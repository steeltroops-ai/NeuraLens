'use client';

import { cn } from '@/utils/cn';

interface LoadingSkeletonProps {
  className?: string;
  variant?: 'default' | 'card' | 'text' | 'avatar' | 'button';
  lines?: number;
  width?: string;
  height?: string;
}

export function LoadingSkeleton({
  className,
  variant = 'default',
  lines = 1,
  width,
  height,
}: LoadingSkeletonProps) {
  const baseClasses = 'animate-pulse bg-gray-200 rounded';

  const variantClasses = {
    default: 'h-4 w-full',
    card: 'h-32 w-full',
    text: 'h-4',
    avatar: 'h-10 w-10 rounded-full',
    button: 'h-10 w-24 rounded-md',
  };

  const skeletonClasses = cn(baseClasses, variantClasses[variant], className);

  const style = {
    ...(width && { width }),
    ...(height && { height }),
  };

  if (variant === 'text' && lines > 1) {
    return (
      <div className='space-y-2'>
        {Array.from({ length: lines }).map((_, index) => (
          <div
            key={index}
            className={cn(
              skeletonClasses,
              index === lines - 1 && 'w-3/4', // Last line shorter
            )}
            style={style}
          />
        ))}
      </div>
    );
  }

  return <div className={skeletonClasses} style={style} />;
}

// Dashboard-specific loading skeletons
export function DashboardLoadingSkeleton() {
  return (
    <div className='space-y-6 p-6'>
      {/* Header skeleton */}
      <div className='flex items-center justify-between'>
        <LoadingSkeleton variant='text' width='200px' />
        <LoadingSkeleton variant='button' />
      </div>

      {/* Cards grid skeleton */}
      <div className='grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4'>
        {Array.from({ length: 4 }).map((_, index) => (
          <LoadingSkeleton key={index} variant='card' />
        ))}
      </div>

      {/* Content area skeleton */}
      <div className='grid grid-cols-1 gap-6 lg:grid-cols-3'>
        <div className='space-y-4 lg:col-span-2'>
          <LoadingSkeleton variant='card' height='300px' />
          <LoadingSkeleton variant='card' height='200px' />
        </div>
        <div className='space-y-4'>
          <LoadingSkeleton variant='card' height='250px' />
          <LoadingSkeleton variant='card' height='250px' />
        </div>
      </div>
    </div>
  );
}

// Assessment loading skeleton
export function AssessmentLoadingSkeleton() {
  return (
    <div className='space-y-4 p-4'>
      <LoadingSkeleton variant='text' lines={2} />
      <LoadingSkeleton variant='card' height='200px' />
      <div className='flex space-x-4'>
        <LoadingSkeleton variant='button' />
        <LoadingSkeleton variant='button' />
      </div>
    </div>
  );
}

export default LoadingSkeleton;
