'use client';

/**
 * Cognitive Testing Module Page
 * 
 * Dedicated page for the Cognitive Testing diagnostic module.
 * Implements lazy loading, loading states, and error boundaries.
 * 
 * Requirements: 4.1, 4.2, 7.1, 8.1
 */

import { Suspense, useState, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { motion } from 'framer-motion';
import { ErrorBoundary } from '@/components/ErrorBoundary';

// Lazy load the CognitiveAssessment component (Requirement 4.2)
const CognitiveAssessment = dynamic(
    () => import('@/components/dashboard/CognitiveAssessment'),
    {
        ssr: false,
        loading: () => <CognitiveAssessmentSkeleton />,
    }
);

/**
 * Loading skeleton for Cognitive Assessment (Requirement 7.1)
 */
function CognitiveAssessmentSkeleton() {
    return (
        <div className="space-y-6 animate-pulse">
            {/* Header skeleton */}
            <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                <div className="flex items-center space-x-3 mb-4">
                    <div className="h-12 w-12 rounded-lg bg-gray-200" />
                    <div className="space-y-2">
                        <div className="h-6 w-52 rounded bg-gray-200" />
                        <div className="h-4 w-72 rounded bg-gray-200" />
                    </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {Array.from({ length: 3 }).map((_, i) => (
                        <div key={i} className="h-10 rounded-lg bg-gray-200" />
                    ))}
                </div>
            </div>

            {/* Test battery skeleton */}
            <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                <div className="h-6 w-44 rounded bg-gray-200 mb-4" />
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Array.from({ length: 4 }).map((_, i) => (
                        <div key={i} className="rounded-lg border border-gray-200 p-4">
                            <div className="flex items-center space-x-3 mb-3">
                                <div className="h-10 w-10 rounded-lg bg-gray-200" />
                                <div className="h-5 w-36 rounded bg-gray-200" />
                            </div>
                            <div className="h-4 w-full rounded bg-gray-200 mb-3" />
                            <div className="h-10 w-full rounded-lg bg-gray-200" />
                        </div>
                    ))}
                </div>
            </div>

            {/* Coming soon skeleton */}
            <div className="rounded-xl border border-gray-200 bg-gray-100 p-6">
                <div className="flex flex-col items-center space-y-4">
                    <div className="h-16 w-16 rounded bg-gray-200" />
                    <div className="h-6 w-64 rounded bg-gray-200" />
                    <div className="h-4 w-96 rounded bg-gray-200" />
                </div>
            </div>
        </div>
    );
}

/**
 * Error fallback component for Cognitive Assessment (Requirement 8.1)
 */
function CognitiveAssessmentError({ error, resetError }: { error?: Error; resetError?: () => void }) {
    return (
        <div className="rounded-xl border border-red-200 bg-red-50 p-8 text-center">
            <div className="text-5xl mb-4">🧠</div>
            <h2 className="text-xl font-semibold text-red-900 mb-2">
                Cognitive Testing Unavailable
            </h2>
            <p className="text-red-700 mb-4">
                {error?.message || 'An error occurred while loading the cognitive testing module.'}
            </p>
            <div className="space-y-2 text-sm text-red-600 mb-6">
                <p>• Ensure your browser is up to date</p>
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
 * Cognitive Testing Module Page Component
 */
export default function CognitivePage() {
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
                fallback={<CognitiveAssessmentError />}
            >
                <Suspense fallback={<CognitiveAssessmentSkeleton />}>
                    <CognitiveAssessment onProcessingChange={handleProcessingChange} />
                </Suspense>
            </ErrorBoundary>
        </motion.div>
    );
}
