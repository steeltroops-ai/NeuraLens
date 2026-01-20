"use client";

/**
 * Retinal Analysis Page
 *
 * Architecture: Client-side Page + Client-side Logic Component
 */

import React, { Suspense } from "react";
import { motion } from "framer-motion";
import { Eye, Target, Clock, Brain, Shield, Info } from "lucide-react";
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
 * Loading skeleton for Retinal Assessment
 */
function RetinalAssessmentSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white rounded-xl border border-zinc-200 p-8 flex flex-col items-center justify-center h-64">
          <div className="h-12 w-12 rounded-full bg-zinc-200 mb-4" />
          <div className="h-4 w-48 rounded bg-zinc-200 mb-2" />
          <div className="h-3 w-32 rounded bg-zinc-200" />
        </div>
        <div className="bg-zinc-900 rounded-xl border border-zinc-700 p-4 h-64">
          <div className="h-4 w-24 rounded bg-zinc-800 mb-4" />
          <div className="space-y-2">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-3 w-full rounded bg-zinc-800" />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Error fallback component
 */
function RetinalAssessmentError() {
  return (
    <div className="rounded-xl border border-red-200 bg-red-50 p-8 text-center">
      <div className="text-5xl mb-4">👁️</div>
      <h2 className="text-xl font-semibold text-red-900 mb-2">
        Retinal Analysis Unavailable
      </h2>
      <p className="text-red-700">
        An error occurred while loading the retinal analysis module.
      </p>
    </div>
  );
}

export default function RetinalPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="bg-white rounded-xl border border-zinc-200 p-6">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-lg bg-cyan-50">
            <Eye className="h-6 w-6 text-cyan-600" strokeWidth={1.5} />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <h1 className="text-[20px] font-semibold text-zinc-900">
                Retinal Analysis
              </h1>
              <span className="px-2 py-0.5 bg-cyan-100 text-cyan-700 text-[10px] font-medium rounded-full">
                v4.0 MODULAR
              </span>
            </div>
            <p className="text-[13px] text-zinc-500 mt-1">
              AI-powered retinal analysis for diabetic retinopathy grading,
              vessel segmentation, and biomarker extraction.
            </p>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
          <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
            <Target className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-900">
                93% DR
              </div>
              <div className="text-[11px] text-zinc-500">Accuracy</div>
            </div>
          </div>
          <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
            <Clock className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-900">
                &lt;2s
              </div>
              <div className="text-[11px] text-zinc-500">Processing</div>
            </div>
          </div>
          <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
            <Brain className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-900">12</div>
              <div className="text-[11px] text-zinc-500">Biomarkers</div>
            </div>
          </div>
          <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
            <Shield className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-900">ETDRS</div>
              <div className="text-[11px] text-zinc-500">Standards</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <ErrorBoundary fallback={<RetinalAssessmentError />}>
        <Suspense fallback={<RetinalAssessmentSkeleton />}>
          <RetinalAssessment />
        </Suspense>
      </ErrorBoundary>

      {/* Info Panel */}
      <div className="bg-zinc-50 rounded-xl border border-zinc-200 p-6">
        <div className="flex items-start gap-3">
          <Info className="h-4 w-4 text-zinc-500 flex-shrink-0 mt-0.5" />
          <div className="text-[12px] text-zinc-500">
            <p className="font-medium text-zinc-700 mb-1">
              About Retinal Analysis
            </p>
            <p className="mb-2">
              This module uses deep learning to analyze fundus images for signs
              of diabetic retinopathy, extracting key biomarkers like vessel
              tortuosity, AV ratio, and lesion detection. It adheres to the
              ETDRS grading standards.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
