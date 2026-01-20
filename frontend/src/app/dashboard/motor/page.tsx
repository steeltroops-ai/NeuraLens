"use client";

/**
 * Motor Assessment Module Page
 *
 * Verified: 2026-01-19
 * Architecture: Client-side Page + Server-side Layout (Metadata)
 */

import { Suspense, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import { Hand, Info, Shield, Clock, Target, Activity } from "lucide-react";
import { ErrorBoundary } from "@/components/common/ErrorBoundary";

// Lazy load the MotorAssessment component
const MotorAssessment = dynamic(() => import("./_components/MotorAssessment"), {
  ssr: false,
  loading: () => <MotorAssessmentSkeleton />,
});

/**
 * Loading skeleton for Motor Assessment
 */
function MotorAssessmentSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 ">
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
    </div>
  );
}

/**
 * Error fallback component
 */
function MotorAssessmentError() {
  return (
    <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-8 text-center">
      <div className="text-5xl mb-4">✋</div>
      <h2 className="text-xl font-semibold text-zinc-100 mb-2">
        Motor Assessment Unavailable
      </h2>
      <p className="text-red-400">
        An error occurred while loading the motor assessment module.
      </p>
    </div>
  );
}

export default function MotorPage() {
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
            <Hand className="h-6 w-6 text-violet-400" strokeWidth={1.5} />
          </div>
          <div className="flex-1">
            <h1 className="text-[20px] font-semibold text-zinc-100">
              Motor Function Assessment
            </h1>
            <p className="text-[13px] text-zinc-500 mt-1">
              Real-time digital biomarker analysis for Parkinson's disease,
              essential tremor, and motor coordination.
            </p>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
          <div className="flex items-center gap-2 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <Target className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-100">93.5%</div>
              <div className="text-[11px] text-zinc-500">Accuracy</div>
            </div>
          </div>
          <div className="flex items-center gap-2 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <Clock className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-100">15s</div>
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
              <div className="text-[13px] font-medium text-zinc-100">UPDRS</div>
              <div className="text-[11px] text-zinc-500">Scale Aligned</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <ErrorBoundary fallback={<MotorAssessmentError />}>
        <Suspense fallback={<MotorAssessmentSkeleton />}>
          <MotorAssessment onProcessingChange={handleProcessingChange} />
        </Suspense>
      </ErrorBoundary>

      {/* Info Panel */}
      <div className="bg-zinc-800/50 rounded-lg border border-zinc-700/50 border border-zinc-200 p-6">
        <div className="flex items-start gap-3">
          <Info className="h-4 w-4 text-zinc-500 flex-shrink-0 mt-0.5" />
          <div className="text-[12px] text-zinc-500">
            <p className="font-medium text-zinc-300 mb-1">
              About Motor Assessment
            </p>
            <p>
              The finger tapping test is a standard neurological examination for
              upper limb motor function. It assesses bradykinesia (slowness of
              movement) and rhythm consistency.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
