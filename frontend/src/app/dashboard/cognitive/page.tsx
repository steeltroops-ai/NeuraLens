"use client";

/**
 * Cognitive Testing Module Page
 *
 * Verified: 2026-01-19
 * Architecture: Client-side Page + Server-side Layout (Metadata)
 */

import { Suspense, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import { Brain, Info, Shield, Clock, Target, Activity } from "lucide-react";
import { ErrorBoundary } from "@/components/common/ErrorBoundary";

// Lazy load the CognitiveAssessment component
const CognitiveAssessment = dynamic(
  () => import("./_components/CognitiveAssessment"),
  {
    ssr: false,
    loading: () => <CognitiveAssessmentSkeleton />,
  },
);

/**
 * Loading skeleton for Cognitive Assessment
 */
function CognitiveAssessmentSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 ">
        <div className="h-6 w-44 rounded bg-zinc-200 mb-4" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="h-40 rounded-lg bg-zinc-200" />
          ))}
        </div>
      </div>
    </div>
  );
}

/**
 * Error fallback component
 */
function CognitiveAssessmentError() {
  return (
    <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-8 text-center">
      <div className="text-5xl mb-4">🧠</div>
      <h2 className="text-xl font-semibold text-zinc-100 mb-2">
        Cognitive Testing Unavailable
      </h2>
      <p className="text-red-400">
        An error occurred while loading the cognitive testing module.
      </p>
    </div>
  );
}

export default function CognitivePage() {
  const [isProcessing, setIsProcessing] = useState(false);

  const handleProcessingChange = useCallback((processing: boolean) => {
    setIsProcessing(processing);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-lg bg-amber-500/15">
            <Brain className="h-6 w-6 text-amber-400" strokeWidth={1.5} />
          </div>
          <div className="flex-1">
            <h1 className="text-[20px] font-semibold text-zinc-100">
              Cognitive Assessment
            </h1>
            <p className="text-[13px] text-zinc-500 mt-1">
              Comprehensive evaluation of memory, attention, executive function,
              and processing speed for early detection of cognitive decline.
            </p>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
          <div className="flex items-center gap-2 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <Target className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-100">92.1%</div>
              <div className="text-[11px] text-zinc-500">Accuracy</div>
            </div>
          </div>
          <div className="flex items-center gap-2 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <Clock className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-100">15m</div>
              <div className="text-[11px] text-zinc-500">Duration</div>
            </div>
          </div>
          <div className="flex items-center gap-2 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <Shield className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-100">HIPAA</div>
              <div className="text-[11px] text-zinc-500">Compliant</div>
            </div>
          </div>
          <div className="flex items-center gap-2 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <Activity className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-100">4</div>
              <div className="text-[11px] text-zinc-500">Domains</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <ErrorBoundary fallback={<CognitiveAssessmentError />}>
        <Suspense fallback={<CognitiveAssessmentSkeleton />}>
          <CognitiveAssessment onProcessingChange={handleProcessingChange} />
        </Suspense>
      </ErrorBoundary>

      {/* Info Panel */}
      <div className="bg-zinc-800/50 rounded-lg border border-zinc-700/50 border border-zinc-200 p-6">
        <div className="flex items-start gap-3">
          <Info className="h-4 w-4 text-zinc-500 flex-shrink-0 mt-0.5" />
          <div className="text-[12px] text-zinc-500">
            <p className="font-medium text-zinc-300 mb-1">
              About Cognitive Testing
            </p>
            <p>
              Digital biomarkers are used to assess cognitive performance.
              Results are compared against normative data for age and education
              level.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
