"use client";

/**
 * Multi-Modal Assessment Module Page
 *
 * Verified: 2026-01-19
 * Architecture: Client-side Page + Server-side Layout (Metadata)
 */

import { Suspense, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import { Activity, Info, Shield, Clock, Target, Zap } from "lucide-react";
import { ErrorBoundary } from "@/components/common/ErrorBoundary";

// Lazy load the MultiModalAssessment component
const MultiModalAssessment = dynamic(
  () => import("./_components/MultiModalAssessment"),
  {
    ssr: false,
    loading: () => <MultiModalAssessmentSkeleton />,
  },
);

/**
 * Loading skeleton for Multi-Modal Assessment
 */
function MultiModalAssessmentSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 ">
        <div className="flex items-center justify-between mb-6">
          <div className="h-6 w-44 rounded bg-zinc-200" />
          <div className="h-12 w-56 rounded-lg bg-zinc-200" />
        </div>
        <div className="space-y-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="rounded-lg border border-zinc-200 p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="h-10 w-10 rounded-lg bg-zinc-200" />
                  <div className="space-y-2">
                    <div className="h-5 w-36 rounded bg-zinc-200" />
                    <div className="h-4 w-24 rounded bg-zinc-200" />
                  </div>
                </div>
                <div className="h-8 w-20 rounded bg-zinc-200" />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 ">
        <div className="h-6 w-56 rounded bg-zinc-200 mb-6" />
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="text-center">
              <div className="h-12 w-20 mx-auto rounded bg-zinc-200 mb-2" />
              <div className="h-4 w-24 mx-auto rounded bg-zinc-200" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/**
 * Error fallback component
 */
function MultiModalAssessmentError() {
  return (
    <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-8 text-center">
      <div className="text-5xl mb-4">📊</div>
      <h2 className="text-xl font-semibold text-zinc-100 mb-2">
        Multi-Modal Assessment Unavailable
      </h2>
      <p className="text-red-400">
        An error occurred while loading the multi-modal assessment module.
      </p>
    </div>
  );
}

export default function MultiModalPage() {
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
          <div className="p-3 rounded-lg bg-violet-500/15">
            <Activity className="h-6 w-6 text-violet-400" strokeWidth={1.5} />
          </div>
          <div className="flex-1">
            <h1 className="text-[20px] font-semibold text-zinc-100">
              Multi-Modal Assessment
            </h1>
            <p className="text-[13px] text-zinc-500 mt-1">
              Integrates Speech, Retinal, Motor, and Cognitive analysis for a
              complete neurological health profile using NRI Fusion.
            </p>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
          <div className="flex items-center gap-2 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <Activity className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-100">4</div>
              <div className="text-[11px] text-zinc-500">Modalities</div>
            </div>
          </div>
          <div className="flex items-center gap-2 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <Target className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-100">96.8%</div>
              <div className="text-[11px] text-zinc-500">Accuracy</div>
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
            <Zap className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-100">
                Real-time
              </div>
              <div className="text-[11px] text-zinc-500">Fusion</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <ErrorBoundary fallback={<MultiModalAssessmentError />}>
        <Suspense fallback={<MultiModalAssessmentSkeleton />}>
          <MultiModalAssessment onProcessingChange={handleProcessingChange} />
        </Suspense>
      </ErrorBoundary>

      {/* Info Panel */}
      <div className="bg-zinc-800/50 rounded-lg border border-zinc-700/50 border border-zinc-200 p-6">
        <div className="flex items-start gap-3">
          <Info className="h-4 w-4 text-zinc-500 flex-shrink-0 mt-0.5" />
          <div className="text-[12px] text-zinc-500">
            <p className="font-medium text-zinc-300 mb-1">
              About Multi-Modal Fusion
            </p>
            <p>
              Digital biomarkers from varying modalities are fused to create a
              composite Neurological Risk Index (NRI). This provides a more
              robust and accurate risk profile than isolated assessments.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
