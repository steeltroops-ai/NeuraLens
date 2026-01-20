"use client";

/**
 * Cardiology Analysis Page
 *
 * AI-powered ECG analysis for cardiac assessment.
 * Implements the CardioPredict AI diagnostic module with:
 * - ECG signal upload and analysis
 * - Arrhythmia classification
 * - Heart rate variability (HRV) metrics
 * - Risk score calculation with clinical recommendations
 */

import React, { Suspense, useState, useCallback } from "react";
import { motion } from "framer-motion";
import {
  Heart,
  Target,
  Clock,
  Shield,
  Activity,
  AlertCircle,
} from "lucide-react";
import dynamic from "next/dynamic";
import { ErrorBoundary } from "@/components/common/ErrorBoundary";

// Lazy load the CardiologyAssessment component
const CardiologyAssessment = dynamic(
  () =>
    import("./_components/CardiologyAssessment").then(
      (mod) => mod.CardiologyAssessment,
    ),
  {
    ssr: false,
    loading: () => <CardiologyAssessmentSkeleton />,
  },
);

/**
 * Loading skeleton for Cardiology Assessment - Dark Theme
 */
function CardiologyAssessmentSkeleton() {
  return (
    <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-6 animate-pulse">
      <div className="h-6 w-48 bg-zinc-800 rounded mb-4" />
      <div className="space-y-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="flex gap-2">
            <div className="h-4 w-4 bg-zinc-800 rounded" />
            <div className="h-4 w-full bg-zinc-800 rounded" />
          </div>
        ))}
      </div>
      <div className="mt-6 h-32 bg-zinc-800/50 rounded-xl" />
    </div>
  );
}

/**
 * Error fallback component - Dark Theme
 */
function CardiologyAssessmentError() {
  return (
    <div className="bg-red-500/10 rounded-xl border border-red-500/30 p-6 text-center">
      <AlertCircle className="h-10 w-10 text-red-500 mx-auto mb-3" />
      <h3 className="text-lg font-medium text-red-400 mb-1">
        Cardiology Analysis Unavailable
      </h3>
      <p className="text-sm text-red-400/80">
        An error occurred while loading the cardiology analysis module. Please
        try refreshing the page.
      </p>
    </div>
  );
}

export default function CardiologyPage() {
  const [isProcessing, setIsProcessing] = useState(false);

  // Callback to update local state when assessment processing changes
  const handleProcessingChange = useCallback((processing: boolean) => {
    setIsProcessing(processing);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="space-y-4 pb-6"
    >
      {/* Header Card */}
      <div className="relative overflow-hidden rounded-xl bg-zinc-900 border border-zinc-800 p-5">
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-20 -right-20 w-64 h-64 bg-gradient-to-bl from-rose-500/10 to-transparent rounded-full blur-3xl" />
        </div>
        <div className="relative z-10">
          <div className="flex items-start gap-4">
            <div className="p-3 rounded-lg bg-rose-500/15">
              <Heart className="h-6 w-6 text-rose-400" strokeWidth={1.5} />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-semibold text-zinc-100">
                  CardioPredict AI
                </h1>
                <span className="px-2 py-0.5 bg-rose-500/20 text-rose-400 text-[10px] font-medium rounded-full">
                  v2.0
                </span>
                {isProcessing && (
                  <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 text-[10px] font-medium rounded-full animate-pulse">
                    PROCESSING
                  </span>
                )}
              </div>
              <p className="text-[13px] text-zinc-400 mt-1">
                AI-powered ECG analysis for arrhythmia classification, atrial
                fibrillation detection, and heart rate variability metrics.
              </p>
            </div>
          </div>

          {/* Stats Row */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-4">
            <div className="flex items-center gap-2 p-3 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
              <Target className="h-4 w-4 text-emerald-500" />
              <div>
                <div className="text-[13px] font-medium text-zinc-100">
                  99.8%
                </div>
                <div className="text-[10px] text-zinc-500">Accuracy</div>
              </div>
            </div>
            <div className="flex items-center gap-2 p-3 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
              <Clock className="h-4 w-4 text-blue-500" />
              <div>
                <div className="text-[13px] font-medium text-zinc-100">
                  &lt;2s
                </div>
                <div className="text-[10px] text-zinc-500">Processing</div>
              </div>
            </div>
            <div className="flex items-center gap-2 p-3 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
              <Activity className="h-4 w-4 text-rose-500" />
              <div>
                <div className="text-[13px] font-medium text-zinc-100">15+</div>
                <div className="text-[10px] text-zinc-500">Biomarkers</div>
              </div>
            </div>
            <div className="flex items-center gap-2 p-3 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
              <Shield className="h-4 w-4 text-amber-500" />
              <div>
                <div className="text-[13px] font-medium text-zinc-100">
                  HIPAA
                </div>
                <div className="text-[10px] text-zinc-500">Compliant</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <ErrorBoundary fallback={<CardiologyAssessmentError />}>
        <Suspense fallback={<CardiologyAssessmentSkeleton />}>
          <CardiologyAssessment onProcessingChange={handleProcessingChange} />
        </Suspense>
      </ErrorBoundary>

      {/* Info Panel - Dark Theme */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <div className="p-1.5 rounded bg-blue-500/15">
            <Activity size={14} className="text-blue-400" />
          </div>
          <div className="text-[12px] text-zinc-400">
            <p className="font-medium text-zinc-300 mb-1">
              About Cardiology Analysis
            </p>
            <p className="leading-relaxed">
              This module uses deep learning to analyze ECG waveforms for early
              detection of cardiac abnormalities. Results should be verified by
              a cardiologist using standard 12-lead ECG equipment and are not
              intended as a standalone diagnosis.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
