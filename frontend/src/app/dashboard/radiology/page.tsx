"use client";

/**
 * Radiology Analysis Page
 *
 * AI-powered chest X-ray analysis using TorchXRayVision.
 * Implements the ChestXplorer AI diagnostic module with:
 * - Chest X-ray image upload and analysis
 * - Detection of 18 pulmonary and cardiac conditions
 * - Grad-CAM explainability visualizations
 * - Risk score calculation with clinical recommendations
 */

import React, { Suspense, useState, useCallback } from "react";
import { motion } from "framer-motion";
import {
  Scan,
  Target,
  Clock,
  Shield,
  Activity,
  AlertCircle,
} from "lucide-react";
import dynamic from "next/dynamic";
import { ErrorBoundary } from "@/components/common/ErrorBoundary";

// Lazy load the RadiologyAssessment component
const RadiologyAssessment = dynamic(
  () =>
    import("./_components/RadiologyAssessment").then(
      (mod) => mod.RadiologyAssessment,
    ),
  {
    ssr: false,
    loading: () => <RadiologyAssessmentSkeleton />,
  },
);

/**
 * Loading skeleton for Radiology Assessment - Dark Theme
 */
function RadiologyAssessmentSkeleton() {
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
function RadiologyAssessmentError() {
  return (
    <div className="bg-red-500/10 rounded-xl border border-red-500/30 p-6 text-center">
      <AlertCircle className="h-10 w-10 text-red-500 mx-auto mb-3" />
      <h3 className="text-lg font-medium text-red-400 mb-1">
        Radiology Analysis Unavailable
      </h3>
      <p className="text-sm text-red-400/80">
        An error occurred while loading the radiology analysis module. Please
        try refreshing the page.
      </p>
    </div>
  );
}

export default function RadiologyPage() {
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
          <div className="absolute -top-20 -right-20 w-64 h-64 bg-gradient-to-bl from-amber-500/10 to-transparent rounded-full blur-3xl" />
        </div>
        <div className="relative z-10">
          <div className="flex items-start gap-4">
            <div className="p-3 rounded-lg bg-amber-500/15">
              <Scan className="h-6 w-6 text-amber-400" strokeWidth={1.5} />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-semibold text-zinc-100">
                  ChestXplorer AI
                </h1>
                <span className="px-2 py-0.5 bg-amber-500/20 text-amber-400 text-[10px] font-medium rounded-full">
                  v2.0
                </span>
                {isProcessing && (
                  <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 text-[10px] font-medium rounded-full animate-pulse">
                    ANALYZING
                  </span>
                )}
              </div>
              <p className="text-[13px] text-zinc-400 mt-1">
                AI-powered chest X-ray analysis using TorchXRayVision
                DenseNet121. Detects pneumonia, cardiomegaly, effusion,
                pneumothorax, and 14 more conditions.
              </p>
            </div>
          </div>

          {/* Stats Row */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-4">
            <div className="flex items-center gap-2 p-3 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
              <Target className="h-4 w-4 text-emerald-500" />
              <div>
                <div className="text-[13px] font-medium text-zinc-100">92%</div>
                <div className="text-[10px] text-zinc-500">Accuracy</div>
              </div>
            </div>
            <div className="flex items-center gap-2 p-3 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
              <Clock className="h-4 w-4 text-blue-500" />
              <div>
                <div className="text-[13px] font-medium text-zinc-100">
                  &lt;3s
                </div>
                <div className="text-[10px] text-zinc-500">Processing</div>
              </div>
            </div>
            <div className="flex items-center gap-2 p-3 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
              <Activity className="h-4 w-4 text-amber-500" />
              <div>
                <div className="text-[13px] font-medium text-zinc-100">18</div>
                <div className="text-[10px] text-zinc-500">Pathologies</div>
              </div>
            </div>
            <div className="flex items-center gap-2 p-3 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
              <Shield className="h-4 w-4 text-emerald-500" />
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
      <ErrorBoundary fallback={<RadiologyAssessmentError />}>
        <Suspense fallback={<RadiologyAssessmentSkeleton />}>
          <RadiologyAssessment onProcessingChange={handleProcessingChange} />
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
              About Chest Radiography Analysis
            </p>
            <p className="leading-relaxed">
              This module utilizes TorchXRayVision's DenseNet121 model trained
              on 800,000+ images from 8 major medical datasets including NIH
              ChestX-ray14, CheXpert, and MIMIC-CXR. It screens for 18
              conditions simultaneously with Grad-CAM explainability. Results
              require radiologist validation.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
