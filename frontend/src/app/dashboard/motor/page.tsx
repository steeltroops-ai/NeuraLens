'use client';

/**
 * Motor Assessment Module Page
 * 
 * Dedicated page for the Motor Assessment diagnostic module.
 * Implements lazy loading, loading states, and error boundaries.
 * 
 * Requirements: 4.1, 4.2, 7.1, 8.1
 */

import { Suspense, useState, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { motion } from 'framer-motion';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';

// Lazy load the MotorAssessment component (Requirement 4.2)
const MotorAssessment = dynamic(
    () => import('./_components/MotorAssessment'),
    {
        ssr: false,
        loading: () => <MotorAssessmentSkeleton />,
    }
);

/**
 * Loading skeleton for Motor Assessment (Requirement 7.1)
 */
function MotorAssessmentSkeleton() {
    return (
        <div className="space-y-6 animate-pulse">
            {/* Header skeleton */}
            <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm">
                <div className="flex items-center space-x-3 mb-4">
                    <div className="h-12 w-12 rounded-lg bg-zinc-200" />
                    <div className="space-y-2">
                        <div className="h-6 w-56 rounded bg-zinc-200" />
                        <div className="h-4 w-72 rounded bg-zinc-200" />
                    </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {Array.from({ length: 3 }).map((_, i) => (
                        <div key={i} className="h-10 rounded-lg bg-zinc-200" />
                    ))}
                </div>
            </div>

            {/* Finger tapping test skeleton */}
            <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm">
                <div className="h-6 w-44 rounded bg-zinc-200 mb-6" />
                <div className="flex flex-col items-center space-y-6">
                    <div className="h-48 w-48 rounded-full bg-zinc-200" />
                    <div className="h-10 w-48 rounded-lg bg-zinc-200" />
                    <div className="grid grid-cols-2 gap-4 w-full max-w-md">
                        <div className="h-20 rounded-lg bg-zinc-200" />
                        <div className="h-20 rounded-lg bg-zinc-200" />
                    </div>
                </div>
            </div>

            {/* Results skeleton */}
            <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm">
                <div className="h-6 w-48 rounded bg-zinc-200 mb-6" />
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                    {Array.from({ length: 4 }).map((_, i) => (
                        <div key={i} className="h-28 rounded-lg bg-zinc-200" />
                    ))}
                </div>
                <div className="h-24 rounded-lg bg-zinc-200" />
            </div>
        </div>
    );
}

/**
 * Error fallback component for Motor Assessment (Requirement 8.1)
 */
function MotorAssessmentError({ error, resetError }: { error?: Error; resetError?: () => void }) {
    return (
        <div className="rounded-xl border border-red-200 bg-red-50 p-8 text-center">
            <div className="text-5xl mb-4">✋</div>
            <h2 className="text-xl font-semibold text-red-900 mb-2">
                Motor Assessment Unavailable
            </h2>
            <p className="text-red-700 mb-4">
                {error?.message || 'An error occurred while loading the motor assessment module.'}
            </p>
            <div className="space-y-2 text-sm text-red-600 mb-6">
                <p>• Ensure your device supports touch or click interactions</p>
                <p>• Check that JavaScript is enabled</p>
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
 * Motor Assessment Module Page Component
 */
export default function MotorPage() {
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
                fallback={<MotorAssessmentError />}
            >
                <Suspense fallback={<MotorAssessmentSkeleton />}>
                    <MotorAssessment onProcessingChange={handleProcessingChange} />
                </Suspense>
            </ErrorBoundary>
        </motion.div>
    );
}
