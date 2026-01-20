"use client";

/**
 * Retinal Analysis Page
 *
 * AI-powered retinal analysis for diabetic retinopathy grading.
 * Implements the RetinaScan AI diagnostic module with:
 * - Fundus image upload and analysis
 * - DR grading (ETDRS standard)
 * - Vessel segmentation and biomarker extraction
 * - Risk score calculation with clinical recommendations
 */

import React, { Suspense, useState, useCallback } from "react";
import { motion } from "framer-motion";
import {
  Eye,
  Target,
  Clock,
  Brain,
  Shield,
  AlertCircle,
  Activity,
} from "lucide-react";
import dynamic from "next/dynamic";
import { ErrorBoundary } from "@/components/common/ErrorBoundary";

// Lazy load the RetinalAssessment component
const RetinalAssessment = dynamic(
  () => import("./_components/RetinalAssessment"),
  {
    ssr: false,
    loading: () => <RetinalAssessmentSkeleton />,
  },
);

/**
 * Loading skeleton for Retinal Assessment - Dark Theme
 */
function RetinalAssessmentSkeleton() {
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
function RetinalAssessmentError() {
  return (
    <div className="bg-red-500/10 rounded-xl border border-red-500/30 p-6 text-center">
      <AlertCircle className="h-10 w-10 text-red-500 mx-auto mb-3" />
      <h3 className="text-lg font-medium text-red-400 mb-1">
        Retinal Analysis Unavailable
      </h3>
      <p className="text-sm text-red-400/80">
        An error occurred while loading the retinal analysis module. Please try
        refreshing the page.
      </p>
    </div>
  );
}

export default function RetinalPage() {
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
          <div className="absolute -top-20 -right-20 w-64 h-64 bg-gradient-to-bl from-cyan-500/10 to-transparent rounded-full blur-3xl" />
        </div>
        <div className="relative z-10">
          <div className="flex items-start gap-4">
            <div className="p-3 rounded-lg bg-cyan-500/15">
              <Eye className="h-6 w-6 text-cyan-400" strokeWidth={1.5} />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-semibold text-zinc-100">
                  RetinaScan AI
                </h1>
                <span className="px-2 py-0.5 bg-cyan-500/20 text-cyan-400 text-[10px] font-medium rounded-full">
                  v4.0
                </span>
                {isProcessing && (
                  <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 text-[10px] font-medium rounded-full animate-pulse">
                    PROCESSING
                  </span>
                )}
              </div>
              <p className="text-[13px] text-zinc-400 mt-1">
                AI-powered retinal analysis for diabetic retinopathy grading,
                vessel segmentation, and biomarker extraction.
              </p>
            </div>
          </div>

          {/* Stats Row */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-4">
            <div className="flex items-center gap-2 p-3 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
              <Target className="h-4 w-4 text-emerald-500" />
              <div>
                <div className="text-[13px] font-medium text-zinc-100">93%</div>
                <div className="text-[10px] text-zinc-500">DR Accuracy</div>
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
              <Brain className="h-4 w-4 text-cyan-500" />
              <div>
                <div className="text-[13px] font-medium text-zinc-100">12</div>
                <div className="text-[10px] text-zinc-500">Biomarkers</div>
              </div>
            </div>
            <div className="flex items-center gap-2 p-3 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
              <Shield className="h-4 w-4 text-amber-500" />
              <div>
                <div className="text-[13px] font-medium text-zinc-100">
                  ETDRS
                </div>
                <div className="text-[10px] text-zinc-500">Standards</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <ErrorBoundary fallback={<RetinalAssessmentError />}>
        <Suspense fallback={<RetinalAssessmentSkeleton />}>
          <RetinalAssessment onProcessingChange={handleProcessingChange} />
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
              About Retinal Analysis
            </p>
            <p className="leading-relaxed">
              This module uses deep learning to analyze fundus images for signs
              of diabetic retinopathy, extracting key biomarkers like vessel
              tortuosity, AV ratio, and lesion detection. It adheres to the
              ETDRS grading standards and should be reviewed by an
              ophthalmologist.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
