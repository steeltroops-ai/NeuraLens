"use client";

import React, { Suspense } from "react";
import { motion } from "framer-motion";
import {
  Scan,
  Info,
  Shield,
  Clock,
  Target,
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

function RadiologyAssessmentSkeleton() {
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

function RadiologyAssessmentError() {
  return (
    <div className="bg-red-50 rounded-xl border border-red-200 p-6 text-center">
      <AlertCircle className="h-10 w-10 text-red-500 mx-auto mb-3" />
      <h3 className="text-lg font-medium text-red-900 mb-1">
        Radiology Analysis Unavailable
      </h3>
      <p className="text-sm text-red-700">
        An error occurred while loading the radiology analysis module. Please
        try refreshing the page.
      </p>
    </div>
  );
}

export default function RadiologyPage() {
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
          <div className="p-3 rounded-lg bg-blue-50">
            <Scan className="h-6 w-6 text-blue-600" strokeWidth={1.5} />
          </div>
          <div className="flex-1">
            <h1 className="text-[20px] font-semibold text-zinc-900">
              ChestXplorer AI
            </h1>
            <p className="text-[13px] text-zinc-500 mt-1">
              AI-powered chest X-ray analysis for pneumonia, COVID-19,
              tuberculosis, lung cancer, and other thoracic conditions.
            </p>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
          <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
            <Target className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-900">97.8%</div>
              <div className="text-[11px] text-zinc-500">Accuracy</div>
            </div>
          </div>
          <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
            <Clock className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-900">
                &lt;2.5s
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
              <div className="text-[13px] font-medium text-zinc-900">8+</div>
              <div className="text-[11px] text-zinc-500">Conditions</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <ErrorBoundary fallback={<RadiologyAssessmentError />}>
        <Suspense fallback={<RadiologyAssessmentSkeleton />}>
          <RadiologyAssessment />
        </Suspense>
      </ErrorBoundary>

      {/* Info Panel */}
      <div className="bg-zinc-50 rounded-xl border border-zinc-200 p-6">
        <div className="flex items-start gap-3">
          <Info className="h-4 w-4 text-zinc-500 flex-shrink-0 mt-0.5" />
          <div className="text-[12px] text-zinc-500">
            <p className="font-medium text-zinc-700 mb-1">
              About Chest Radiography Analysis
            </p>
            <p>
              This module utilizes deep convolutional neural networks to detect
              abnormalities in chest X-rays. It screens for multiple conditions
              simultaneously but requires radiologist validation.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
