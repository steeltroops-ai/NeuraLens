'use client';

/**
 * Multi-Modal Assessment Module Page
 * 
 * Dedicated page for the Multi-Modal Assessment diagnostic module.
 * Implements lazy loading, loading states, and error boundaries.
 * 
 * Requirements: 4.1, 4.2, 7.1, 8.1
 */

import { Suspense, useState, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { motion } from 'framer-motion';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';

// Lazy load the MultiModalAssessment component (Requirement 4.2)
const MultiModalAssessment = dynamic(
    () => import('./_components/MultiModalAssessment'),
    {
        ssr: false,
        loading: () => <MultiModalAssessmentSkeleton />,
    }
);

/**
 * Loading skeleton for Multi-Modal Assessment (Requirement 7.1)
 */
function MultiModalAssessmentSkeleton() {
    return (
        <div className="space-y-6 animate-pulse">
            {/* Header skeleton */}
            <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                <div className="flex items-center space-x-3 mb-4">
                    <div className="h-12 w-12 rounded-lg bg-gray-200" />
                    <div className="space-y-2">
                        <div className="h-6 w-56 rounded bg-gray-200" />
                        <div className="h-4 w-80 rounded bg-gray-200" />
                    </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {Array.from({ length: 3 }).map((_, i) => (
                        <div key={i} className="h-10 rounded-lg bg-gray-200" />
                    ))}
                </div>
            </div>

            {/* Assessment control skeleton */}
            <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                <div className="flex items-center justify-between mb-6">
                    <div className="h-6 w-44 rounded bg-gray-200" />
                    <div className="h-12 w-56 rounded-lg bg-gray-200" />
                </div>
                <div className="space-y-4">
                    {Array.from({ length: 4 }).map((_, i) => (
                        <div key={i} className="rounded-lg border border-gray-200 p-4">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center space-x-3">
                                    <div className="h-10 w-10 rounded-lg bg-gray-200" />
                                    <div className="space-y-2">
                                        <div className="h-5 w-36 rounded bg-gray-200" />
                                        <div className="h-4 w-24 rounded bg-gray-200" />
                                    </div>
                                </div>
                                <div className="h-8 w-20 rounded bg-gray-200" />
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* NRI Results skeleton */}
            <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                <div className="h-6 w-56 rounded bg-gray-200 mb-6" />
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                    {Array.from({ length: 4 }).map((_, i) => (
                        <div key={i} className="text-center">
                            <div className="h-12 w-20 mx-auto rounded bg-gray-200 mb-2" />
                            <div className="h-4 w-24 mx-auto rounded bg-gray-200" />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

/**
 * Error fallback component for Multi-Modal Assessment (Requirement 8.1)
 */
function MultiModalAssessmentError({ error, resetError }: { error?: Error; resetError?: () => void }) {
    return (
        <div className="rounded-xl border border-red-200 bg-red-50 p-8 text-center">
            <div className="text-5xl mb-4">📊</div>
            <h2 className="text-xl font-semibold text-red-900 mb-2">
                Multi-Modal Assessment Unavailable
            </h2>
            <p className="text-red-700 mb-4">
                {error?.message || 'An error occurred while loading the multi-modal assessment module.'}
            </p>
            <div className="space-y-2 text-sm text-red-600 mb-6">
                <p>• Ensure all required permissions are granted</p>
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
 * Multi-Modal Assessment Module Page Component
 */
export default function MultiModalPage() {
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
                fallback={<MultiModalAssessmentError />}
            >
                <Suspense fallback={<MultiModalAssessmentSkeleton />}>
                    <MultiModalAssessment onProcessingChange={handleProcessingChange} />
                </Suspense>
            </ErrorBoundary>
        </motion.div>
    );
}
