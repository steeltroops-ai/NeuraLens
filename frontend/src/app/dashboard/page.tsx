'use client';

/**
 * Dashboard Overview Page
 * 
 * Main dashboard page displaying:
 * - User health overview
 * - System status cards
 * - Diagnostic module grid with all available and coming-soon modules
 * 
 * Implements lazy loading for non-critical components.
 * Follows MediLens Design System patterns.
 * 
 * Requirements: 4.3, 6.1, 12.1
 */

import { useState, useEffect, memo, lazy, Suspense } from 'react';
import { motion } from 'framer-motion';

// Lazy load dashboard components for optimal performance (Requirement 12.1)
const DiagnosticGrid = lazy(() => import('@/components/dashboard/DiagnosticGrid'));
const SystemStatusCards = lazy(() => import('@/components/dashboard/SystemStatusCards'));
const UserHealthOverview = lazy(() => import('@/components/dashboard/UserHealthOverview'));

/**
 * Loading skeleton for the diagnostic grid
 */
function DiagnosticGridSkeleton() {
  return (
    <div className="space-y-12">
      {/* Available section skeleton */}
      <div>
        <div className="mb-6 flex items-center gap-3">
          <div className="h-7 w-36 animate-pulse rounded bg-gray-200" />
          <div className="h-6 w-24 animate-pulse rounded-full bg-gray-200" />
        </div>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div
              key={`available-${i}`}
              className="h-[180px] animate-pulse rounded-2xl bg-gray-200"
            />
          ))}
        </div>
      </div>
      {/* Coming soon section skeleton */}
      <div>
        <div className="mb-6 flex items-center gap-3">
          <div className="h-7 w-36 animate-pulse rounded bg-gray-200" />
          <div className="h-6 w-24 animate-pulse rounded-full bg-gray-200" />
        </div>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <div
              key={`coming-${i}`}
              className="h-[180px] animate-pulse rounded-2xl bg-gray-200"
            />
          ))}
        </div>
      </div>
    </div>
  );
}

/**
 * Loading skeleton for user health overview
 */
function UserHealthSkeleton() {
  return (
    <div className="h-48 animate-pulse rounded-2xl bg-gray-200" />
  );
}

/**
 * Loading skeleton for system status cards
 */
function SystemStatusSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="h-32 animate-pulse rounded-2xl bg-gray-200" />
      ))}
    </div>
  );
}

/**
 * Dashboard Overview Page Component
 */
export default function DashboardPage() {
  const [systemStatus, setSystemStatus] = useState<'healthy' | 'warning' | 'error'>('healthy');

  // System health check
  useEffect(() => {
    const checkSystemHealth = async () => {
      try {
        const response = await fetch('/api/health', {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
          cache: 'no-cache',
        });
        if (response.ok) {
          setSystemStatus('healthy');
        } else {
          setSystemStatus('warning');
        }
      } catch (error) {
        console.error('Health check failed:', error);
        setSystemStatus('error');
      }
    };

    checkSystemHealth();
    const interval = setInterval(checkSystemHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
      className="space-y-8"
    >
      <Suspense fallback={<DashboardOverviewSkeleton />}>
        <DashboardOverview systemStatus={systemStatus} />
      </Suspense>
    </motion.div>
  );
}

/**
 * Full dashboard skeleton for initial load
 */
function DashboardOverviewSkeleton() {
  return (
    <div className="space-y-8">
      <UserHealthSkeleton />
      <SystemStatusSkeleton />
      <DiagnosticGridSkeleton />
    </div>
  );
}

/**
 * Dashboard Overview Component (Memoized for Performance)
 * 
 * Displays the main dashboard content with:
 * - User health overview at the top
 * - System status cards
 * - Diagnostic module grid with available and coming-soon modules
 */
const DashboardOverview = memo(
  ({ systemStatus: _systemStatus }: { systemStatus: 'healthy' | 'warning' | 'error' }) => {
    return (
      <div className="space-y-8">
        {/* Page Header - Title 1 (34px/700) for page titles */}
        <header>
          <h1 className="text-title1 text-[#000000]">
            Dashboard
          </h1>
          <p className="mt-2 text-body text-[#3C3C43] max-w-prose">
            Access AI-powered diagnostic tools and view your health insights
          </p>
        </header>

        {/* Primary Section - User Health Overview */}
        <section aria-labelledby="health-overview-heading">
          <h2 id="health-overview-heading" className="sr-only">
            Health Overview
          </h2>
          <Suspense fallback={<UserHealthSkeleton />}>
            <UserHealthOverview />
          </Suspense>
        </section>

        {/* System Status Cards */}
        <section aria-labelledby="system-status-heading">
          <h2 id="system-status-heading" className="sr-only">
            System Status
          </h2>
          <Suspense fallback={<SystemStatusSkeleton />}>
            <SystemStatusCards />
          </Suspense>
        </section>

        {/* Diagnostic Modules Grid (Requirement 6.1) */}
        <section aria-labelledby="diagnostic-modules-heading">
          {/* Title 2 (28px/600) for section headers */}
          <h2
            id="diagnostic-modules-heading"
            className="mb-6 text-title2 text-[#000000]"
          >
            Diagnostic Modules
          </h2>
          <Suspense fallback={<DiagnosticGridSkeleton />}>
            <DiagnosticGrid showSectionHeaders={true} />
          </Suspense>
        </section>
      </div>
    );
  }
);

DashboardOverview.displayName = 'DashboardOverview';
