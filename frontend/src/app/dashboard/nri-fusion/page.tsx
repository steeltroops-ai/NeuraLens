'use client';

/**
 * NRI Fusion Module Page
 * 
 * Dedicated page for the NRI Fusion Dashboard diagnostic module.
 * Implements lazy loading, loading states, and error boundaries.
 * 
 * Requirements: 4.1, 4.2, 7.1, 8.1
 */

import { Suspense, useState, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { motion } from 'framer-motion';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';

// Lazy load the NRIFusionDashboard component (Requirement 4.2)
const NRIFusionDashboard = dynamic(
    () => import('./_components/NRIFusionDashboard'),
    {
        ssr: false,
        loading: () => <NRIFusionSkeleton />,
    }
);

/**
 * Loading skeleton for NRI Fusion Dashboard (Requirement 7.1)
 */
function NRIFusionSkeleton() {
    return (
        <div className="space-y-6 animate-pulse">
            {/* Header skeleton */}
            <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm">
                <div className="flex items-center space-x-3 mb-4">
                    <div className="h-12 w-12 rounded-lg bg-zinc-200" />
                    <div className="space-y-2">
                        <div className="h-6 w-44 rounded bg-zinc-200" />
                        <div className="h-4 w-72 rounded bg-zinc-200" />
                    </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {Array.from({ length: 3 }).map((_, i) => (
                        <div key={i} className="h-10 rounded-lg bg-zinc-200" />
                    ))}
                </div>
            </div>

            {/* NRI Score display skeleton */}
            <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm">
                <div className="h-6 w-48 rounded bg-zinc-200 mb-6" />
                <div className="flex justify-center mb-6">
                    <div className="h-32 w-32 rounded-full bg-zinc-200" />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    {Array.from({ length: 4 }).map((_, i) => (
                        <div key={i} className="h-20 rounded-lg bg-zinc-200" />
                    ))}
                </div>
            </div>

            {/* Modality contributions skeleton */}
            <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm">
                <div className="h-6 w-52 rounded bg-zinc-200 mb-6" />
                <div className="space-y-4">
                    {Array.from({ length: 4 }).map((_, i) => (
                        <div key={i} className="flex items-center space-x-4">
                            <div className="h-5 w-24 rounded bg-zinc-200" />
                            <div className="flex-1 h-4 rounded-full bg-zinc-200" />
                            <div className="h-5 w-16 rounded bg-zinc-200" />
                        </div>
                    ))}
                </div>
            </div>

            {/* Coming soon skeleton */}
            <div className="rounded-xl border border-zinc-200 bg-zinc-100 p-6">
                <div className="flex flex-col items-center space-y-4">
                    <div className="h-16 w-16 rounded bg-zinc-200" />
                    <div className="h-6 w-72 rounded bg-zinc-200" />
                    <div className="h-4 w-96 rounded bg-zinc-200" />
                </div>
            </div>
        </div>
    );
}

/**
 * Error fallback component for NRI Fusion Dashboard (Requirement 8.1)
 */
function NRIFusionError({ error, resetError }: { error?: Error; resetError?: () => void }) {
    return (
        <div className="rounded-xl border border-red-200 bg-red-50 p-8 text-center">
            <div className="text-5xl mb-4">📈</div>
            <h2 className="text-xl font-semibold text-red-900 mb-2">
                NRI Fusion Dashboard Unavailable
            </h2>
            <p className="text-red-700 mb-4">
                {error?.message || 'An error occurred while loading the NRI fusion dashboard.'}
            </p>
            <div className="space-y-2 text-sm text-red-600 mb-6">
                <p>• Ensure assessment data is available</p>
                <p>• Check your internet connection</p>
                <p>• Try refreshing the page</p>
            </div>
            {resetError && (
                <button
                    onClick={resetError}
                    className="inline-flex items-center justify-center px-6 py-3 min-h-[48px] bg-red-600 text-white font-semibold rounded-xl hover:bg-red-700 transition-colors"
                >
                    Try Again
                </button>
            )}
        </div>
    );
}

/**
 * NRI Fusion Module Page Component
 */
export default function NRIFusionPage() {
    const [isProcessing, setIsProcessing] = useState(false);

    const handleProcessingChange = useCallback((processing: boolean) => {
        setIsProcessing(processing);
    }, []);

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
            className="space-y-6"
        >
            <ErrorBoundary
                fallback={<NRIFusionError />}
            >
                <Suspense fallback={<NRIFusionSkeleton />}>
                    <NRIFusionDashboard onProcessingChange={handleProcessingChange} />
                </Suspense>
            </ErrorBoundary>
        </motion.div>
    );
}
