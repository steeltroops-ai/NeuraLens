'use client';

/**
 * Enhanced Loading Skeleton Components
 * MediLens Design System compliant loading states
 * 
 * Requirements: 7.1, 7.4
 * - WHEN data is being fetched, THE System SHALL display skeleton loaders
 * - WHEN a page is loading, THE System SHALL display a consistent loading animation
 */

import { cn } from '@/utils/cn';

interface LoadingSkeletonProps {
  className?: string;
  variant?: 'default' | 'card' | 'text' | 'avatar' | 'button' | 'list-item' | 'full-page';
  lines?: number;
  width?: string;
  height?: string;
}

/**
 * Base LoadingSkeleton component with MediLens design system styling
 * Uses consistent animation and surface colors
 */
export function LoadingSkeleton({
  className,
  variant = 'default',
  lines = 1,
  width,
  height,
}: LoadingSkeletonProps) {
  // MediLens design system: consistent animation with surface-tertiary color
  const baseClasses = 'animate-pulse bg-[#E5E5EA] rounded';

  const variantClasses = {
    default: 'h-4 w-full',
    card: 'h-32 w-full rounded-2xl',
    text: 'h-4',
    avatar: 'h-10 w-10 rounded-full',
    button: 'h-12 w-24 rounded-xl',
    'list-item': 'h-16 w-full rounded-xl',
    'full-page': 'h-full w-full min-h-[400px] rounded-2xl',
  };

  const skeletonClasses = cn(baseClasses, variantClasses[variant], className);

  const style = {
    ...(width && { width }),
    ...(height && { height }),
  };

  if (variant === 'text' && lines > 1) {
    return (
      <div className='space-y-2' role="status" aria-label="Loading content">
        {Array.from({ length: lines }).map((_, index) => (
          <div
            key={index}
            className={cn(
              skeletonClasses,
              index === lines - 1 && 'w-3/4', // Last line shorter for natural look
            )}
            style={style}
          />
        ))}
        <span className="sr-only">Loading...</span>
      </div>
    );
  }

  return (
    <div
      className={skeletonClasses}
      style={style}
      role="status"
      aria-label="Loading"
    >
      <span className="sr-only">Loading...</span>
    </div>
  );
}

/**
 * Card Skeleton - For diagnostic cards and content cards
 * MediLens Design System: rounded-2xl, shadow-apple styling
 */
export function CardSkeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        'bg-white rounded-2xl p-6 shadow-[0_4px_6px_-1px_rgba(0,0,0,0.1),0_2px_4px_-1px_rgba(0,0,0,0.06)] border border-black/5',
        className
      )}
      role="status"
      aria-label="Loading card"
    >
      {/* Icon placeholder */}
      <div className="animate-pulse bg-[#E5E5EA] h-12 w-12 rounded-xl mb-4" />

      {/* Title placeholder */}
      <div className="animate-pulse bg-[#E5E5EA] h-5 w-3/4 rounded mb-2" />

      {/* Description placeholder */}
      <div className="space-y-2">
        <div className="animate-pulse bg-[#E5E5EA] h-4 w-full rounded" />
        <div className="animate-pulse bg-[#E5E5EA] h-4 w-2/3 rounded" />
      </div>

      {/* Badge placeholder */}
      <div className="animate-pulse bg-[#E5E5EA] h-6 w-20 rounded-full mt-4" />

      <span className="sr-only">Loading card...</span>
    </div>
  );
}

/**
 * List Skeleton - For lists of items
 */
export function ListSkeleton({
  items = 3,
  className
}: {
  items?: number;
  className?: string;
}) {
  return (
    <div
      className={cn('space-y-3', className)}
      role="status"
      aria-label="Loading list"
    >
      {Array.from({ length: items }).map((_, index) => (
        <div
          key={index}
          className="flex items-center gap-4 p-4 bg-white rounded-xl border border-black/5"
        >
          {/* Avatar/Icon placeholder */}
          <div className="animate-pulse bg-[#E5E5EA] h-10 w-10 rounded-full flex-shrink-0" />

          {/* Content placeholder */}
          <div className="flex-1 space-y-2">
            <div className="animate-pulse bg-[#E5E5EA] h-4 w-1/3 rounded" />
            <div className="animate-pulse bg-[#E5E5EA] h-3 w-2/3 rounded" />
          </div>

          {/* Action placeholder */}
          <div className="animate-pulse bg-[#E5E5EA] h-8 w-16 rounded-lg flex-shrink-0" />
        </div>
      ))}
      <span className="sr-only">Loading list...</span>
    </div>
  );
}

/**
 * Full Page Skeleton - For entire page loading states
 * Includes header, sidebar indication, and content area
 */
export function FullPageSkeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn('min-h-screen bg-[#F2F2F7] p-6', className)}
      role="status"
      aria-label="Loading page"
    >
      {/* Header skeleton */}
      <div className="flex items-center justify-between mb-8">
        <div className="space-y-2">
          <div className="animate-pulse bg-[#E5E5EA] h-8 w-48 rounded-lg" />
          <div className="animate-pulse bg-[#E5E5EA] h-4 w-64 rounded" />
        </div>
        <div className="animate-pulse bg-[#E5E5EA] h-12 w-32 rounded-xl" />
      </div>

      {/* Stats cards skeleton */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {Array.from({ length: 4 }).map((_, index) => (
          <div
            key={index}
            className="bg-white rounded-2xl p-6 shadow-[0_4px_6px_-1px_rgba(0,0,0,0.1)] border border-black/5"
          >
            <div className="animate-pulse bg-[#E5E5EA] h-4 w-20 rounded mb-2" />
            <div className="animate-pulse bg-[#E5E5EA] h-8 w-16 rounded mb-1" />
            <div className="animate-pulse bg-[#E5E5EA] h-3 w-24 rounded" />
          </div>
        ))}
      </div>

      {/* Main content skeleton */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-2xl p-6 shadow-[0_4px_6px_-1px_rgba(0,0,0,0.1)] border border-black/5">
            <div className="animate-pulse bg-[#E5E5EA] h-6 w-40 rounded mb-4" />
            <div className="animate-pulse bg-[#E5E5EA] h-64 w-full rounded-xl" />
          </div>
        </div>
        <div className="space-y-6">
          <div className="bg-white rounded-2xl p-6 shadow-[0_4px_6px_-1px_rgba(0,0,0,0.1)] border border-black/5">
            <div className="animate-pulse bg-[#E5E5EA] h-6 w-32 rounded mb-4" />
            <div className="space-y-3">
              {Array.from({ length: 4 }).map((_, index) => (
                <div key={index} className="animate-pulse bg-[#E5E5EA] h-12 w-full rounded-xl" />
              ))}
            </div>
          </div>
        </div>
      </div>

      <span className="sr-only">Loading page content...</span>
    </div>
  );
}

/**
 * Dashboard Loading Skeleton - Specific to dashboard layout
 */
export function DashboardLoadingSkeleton() {
  return (
    <div className='space-y-6 p-6' role="status" aria-label="Loading dashboard">
      {/* Header skeleton */}
      <div className='flex items-center justify-between'>
        <div className="space-y-2">
          <div className="animate-pulse bg-[#E5E5EA] h-8 w-48 rounded-lg" />
          <div className="animate-pulse bg-[#E5E5EA] h-4 w-64 rounded" />
        </div>
        <div className="animate-pulse bg-[#E5E5EA] h-12 w-32 rounded-xl" />
      </div>

      {/* Cards grid skeleton */}
      <div className='grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4'>
        {Array.from({ length: 4 }).map((_, index) => (
          <CardSkeleton key={index} />
        ))}
      </div>

      {/* Content area skeleton */}
      <div className='grid grid-cols-1 gap-6 lg:grid-cols-3'>
        <div className='space-y-4 lg:col-span-2'>
          <div className="bg-white rounded-2xl p-6 shadow-[0_4px_6px_-1px_rgba(0,0,0,0.1)] border border-black/5">
            <div className="animate-pulse bg-[#E5E5EA] h-6 w-40 rounded mb-4" />
            <div className="animate-pulse bg-[#E5E5EA] h-[300px] w-full rounded-xl" />
          </div>
          <div className="bg-white rounded-2xl p-6 shadow-[0_4px_6px_-1px_rgba(0,0,0,0.1)] border border-black/5">
            <div className="animate-pulse bg-[#E5E5EA] h-6 w-32 rounded mb-4" />
            <div className="animate-pulse bg-[#E5E5EA] h-[200px] w-full rounded-xl" />
          </div>
        </div>
        <div className='space-y-4'>
          <div className="bg-white rounded-2xl p-6 shadow-[0_4px_6px_-1px_rgba(0,0,0,0.1)] border border-black/5">
            <div className="animate-pulse bg-[#E5E5EA] h-6 w-28 rounded mb-4" />
            <div className="animate-pulse bg-[#E5E5EA] h-[250px] w-full rounded-xl" />
          </div>
          <div className="bg-white rounded-2xl p-6 shadow-[0_4px_6px_-1px_rgba(0,0,0,0.1)] border border-black/5">
            <div className="animate-pulse bg-[#E5E5EA] h-6 w-24 rounded mb-4" />
            <div className="animate-pulse bg-[#E5E5EA] h-[250px] w-full rounded-xl" />
          </div>
        </div>
      </div>

      <span className="sr-only">Loading dashboard...</span>
    </div>
  );
}

/**
 * Assessment Loading Skeleton - For assessment module pages
 */
export function AssessmentLoadingSkeleton() {
  return (
    <div className='space-y-6 p-6' role="status" aria-label="Loading assessment">
      {/* Title and description */}
      <div className="space-y-2">
        <div className="animate-pulse bg-[#E5E5EA] h-8 w-64 rounded-lg" />
        <div className="animate-pulse bg-[#E5E5EA] h-4 w-96 rounded" />
      </div>

      {/* Main assessment card */}
      <div className="bg-white rounded-2xl p-8 shadow-[0_4px_6px_-1px_rgba(0,0,0,0.1)] border border-black/5">
        <div className="animate-pulse bg-[#E5E5EA] h-6 w-40 rounded mb-6" />
        <div className="animate-pulse bg-[#E5E5EA] h-[200px] w-full rounded-xl mb-6" />

        {/* Progress indicator */}
        <div className="animate-pulse bg-[#E5E5EA] h-2 w-full rounded-full mb-4" />

        {/* Action buttons */}
        <div className='flex gap-4 justify-end'>
          <div className="animate-pulse bg-[#E5E5EA] h-12 w-24 rounded-xl" />
          <div className="animate-pulse bg-[#E5E5EA] h-12 w-32 rounded-xl" />
        </div>
      </div>

      <span className="sr-only">Loading assessment...</span>
    </div>
  );
}

/**
 * Diagnostic Grid Skeleton - For the diagnostic modules grid
 */
export function DiagnosticGridSkeleton({ count = 8 }: { count?: number }) {
  return (
    <div
      className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
      role="status"
      aria-label="Loading diagnostic modules"
    >
      {Array.from({ length: count }).map((_, index) => (
        <CardSkeleton key={index} />
      ))}
      <span className="sr-only">Loading diagnostic modules...</span>
    </div>
  );
}

/**
 * Table Skeleton - For data tables
 */
export function TableSkeleton({
  rows = 5,
  columns = 4,
  className
}: {
  rows?: number;
  columns?: number;
  className?: string;
}) {
  return (
    <div
      className={cn('bg-white rounded-2xl overflow-hidden border border-black/5', className)}
      role="status"
      aria-label="Loading table"
    >
      {/* Header row */}
      <div className="flex gap-4 p-4 bg-[#F2F2F7] border-b border-black/5">
        {Array.from({ length: columns }).map((_, index) => (
          <div
            key={index}
            className="animate-pulse bg-[#E5E5EA] h-4 flex-1 rounded"
          />
        ))}
      </div>

      {/* Data rows */}
      {Array.from({ length: rows }).map((_, rowIndex) => (
        <div
          key={rowIndex}
          className="flex gap-4 p-4 border-b border-black/5 last:border-b-0"
        >
          {Array.from({ length: columns }).map((_, colIndex) => (
            <div
              key={colIndex}
              className="animate-pulse bg-[#E5E5EA] h-4 flex-1 rounded"
            />
          ))}
        </div>
      ))}

      <span className="sr-only">Loading table data...</span>
    </div>
  );
}

export default LoadingSkeleton;
