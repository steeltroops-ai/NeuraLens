'use client';

/**
 * Speech Analysis Module Page
 * 
 * Dedicated page for the Speech Analysis diagnostic module.
 * Implements lazy loading, loading states, and error boundaries.
 * 
 * Requirements: 4.1, 4.2, 7.1, 8.1
 */

import { Suspense, useState, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { motion } from 'framer-motion';
import { ErrorBoundary } from '@/components/ErrorBoundary';

// Lazy load the SpeechAssessment component (Requirement 4.2)
const SpeechAssessment = dynamic(
    () => import('@/components/dashboard/SpeechAssessment'),
    {
        ssr: false,
        loading: () => <SpeechAssessmentSkeleton />,
    }
);

/**
 * Loading skeleton for Speech Assessment (Requirement 7.1)
 */
function SpeechAssessmentSkeleton() {
    return (
        <div className="space-y-6 animate-pulse">
            {/* Header skeleton */}
            <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                <div className="flex items-center space-x-3 mb-4">
                    <div className="h-12 w-12 rounded-lg bg-gray-200" />
                    <div className="space-y-2">
                        <div className="h-6 w-48 rounded bg-gray-200" />
                        <div className="h-4 w-64 rounded bg-gray-200" />
                    </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    {Array.from({ length: 4 }).map((_, i) => (
                        <div key={i} className="h-10 rounded-lg bg-gray-200" />
                    ))}
                </div>
            </div>

            {/* Recording interface skeleton */}
            <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                <div className="h-6 w-40 rounded bg-gray-200 mb-4" />
                <div className="flex flex-col items-center space-y-4">
                    <div className="h-32 w-32 rounded-full bg-gray-200" />
                    <div className="h-10 w-48 rounded-lg bg-gray-200" />
                </div>
            </div>

            {/* Results skeleton */}
            <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                <div className="h-6 w-32 rounded bg-gray-200 mb-4" />
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                        <div className="h-20 rounded-lg bg-gray-200" />
                        <div className="h-20 rounded-lg bg-gray-200" />
                    </div>
                    <div className="space-y-4">
                        <div className="h-20 rounded-lg bg-gray-200" />
                        <div className="h-20 rounded-lg bg-gray-200" />
                    </div>
                </div>
            </div>
        </div>
    );
}

/**
 * Error fallback component for Speech Assessment (Requirement 8.1)
 */
function SpeechAssessmentError({ error, resetError }: { error?: Error; resetError?: () => void }) {
    return (
        <div className="rounded-xl border border-red-200 bg-red-50 p-8 text-center">
            <div className="text-5xl mb-4">🎤</div>
            <h2 className="text-xl font-semibold text-red-900 mb-2">
                Speech Analysis Unavailable
            </h2>
            <p className="text-red-700 mb-4">
                {error?.message || 'An error occurred while loading the speech analysis module.'}
            </p>
            <div className="space-y-2 text-sm text-red-600 mb-6">
                <p>• Check your microphone permissions</p>
                <p>• Ensure your browser supports audio recording</p>
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
 * Speech Analysis Module Page Component
 */
export default function SpeechPage() {
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
                fallback={<SpeechAssessmentError />}
            >
                <Suspense fallback={<SpeechAssessmentSkeleton />}>
                    <SpeechAssessment onProcessingChange={handleProcessingChange} />
                </Suspense>
            </ErrorBoundary>
        </motion.div>
    );
}
