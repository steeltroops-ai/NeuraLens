"use client";

import React, { Suspense } from "react";
import { motion } from "framer-motion";
import {
  Heart,
  Info,
  Shield,
  Clock,
  Target,
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

function CardiologyAssessmentSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-zinc-200 rounded-xl p-6 h-96"></div>
        <div className="space-y-4">
          <div className="bg-white border border-zinc-200 rounded-xl p-6 h-48"></div>
          <div className="bg-white border border-zinc-200 rounded-xl p-6 h-48"></div>
        </div>
      </div>
    </div>
  );
}

function CardiologyAssessmentError() {
  return (
    <div className="bg-red-50 rounded-xl border border-red-200 p-6 text-center">
      <AlertCircle className="h-10 w-10 text-red-500 mx-auto mb-3" />
      <h3 className="text-lg font-medium text-red-900 mb-1">
        Cardiology Analysis Unavailable
      </h3>
      <p className="text-sm text-red-700">
        An error occurred while loading the cardiology analysis module. Please
        try refreshing the page.
      </p>
    </div>
  );
}

export default function CardiologyPage() {
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
          <div className="p-3 rounded-lg bg-red-50">
            <Heart className="h-6 w-6 text-red-600" strokeWidth={1.5} />
          </div>
          <div className="flex-1">
            <h1 className="text-[20px] font-semibold text-zinc-900">
              CardioPredict AI
            </h1>
            <p className="text-[13px] text-zinc-500 mt-1">
              AI-powered ECG analysis for arrhythmia classification, atrial
              fibrillation detection, and heart rate variability (HRV) metrics.
            </p>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
          <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
            <Target className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-900">99.8%</div>
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
            <Shield className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-900">HIPAA</div>
              <div className="text-[11px] text-zinc-500">Compliant</div>
            </div>
          </div>
          <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
            <Activity className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-900">15+</div>
              <div className="text-[11px] text-zinc-500">Biomarkers</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <ErrorBoundary fallback={<CardiologyAssessmentError />}>
        <Suspense fallback={<CardiologyAssessmentSkeleton />}>
          <CardiologyAssessment />
        </Suspense>
      </ErrorBoundary>

      {/* Info Panel */}
      <div className="bg-zinc-50 rounded-xl border border-zinc-200 p-6">
        <div className="flex items-start gap-3">
          <Info className="h-4 w-4 text-zinc-500 flex-shrink-0 mt-0.5" />
          <div className="text-[12px] text-zinc-500">
            <p className="font-medium text-zinc-700 mb-1">
              About Cardiology Analysis
            </p>
            <p>
              This module uses deep learning to analyze ECG waveforms for early
              detection of cardiac abnormalities. Results should be verified by
              a cardiologist using standard 12-lead ECG equipment.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
