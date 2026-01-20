"use client";

/**
 * NRI Fusion Module Page
 *
 * Verified: 2026-01-19
 * Architecture: Client-side Page + Server-side Layout (Metadata)
 */

import { Suspense, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import { TrendingUp, Zap, Activity, Clock, Shield, Target } from "lucide-react";
import { ErrorBoundary } from "@/components/common/ErrorBoundary";

// Lazy load the NRIFusionDashboard component
const NRIFusionDashboard = dynamic(
  () => import("./_components/NRIFusionDashboard"),
  {
    ssr: false,
    loading: () => <NRIFusionSkeleton />,
  },
);

/**
 * Loading skeleton for NRI Fusion Dashboard
 */
function NRIFusionSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 ">
        <div className="h-6 w-52 rounded bg-zinc-200 mb-6" />
        <div className="space-y-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="flex items-center space-x-4">
              <div className="h-5 w-24 rounded bg-zinc-200" />
              <div className="flex-1 h-4 rounded-full bg-zinc-200" />
              <div className="h-5 w-16 rounded bg-zinc-200" />
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-xl border border-zinc-200 bg-zinc-800 p-6">
        <div className="flex flex-col items-center space-y-4">
          <div className="h-16 w-16 rounded bg-zinc-200" />
          <div className="h-6 w-72 rounded bg-zinc-200" />
          <div className="h-4 w-96 rounded bg-zinc-200" />
        </div>
      </div>
    </div>
  );
}

/**
 * Error fallback component
 */
function NRIFusionError() {
  return (
    <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-8 text-center">
      <div className="text-5xl mb-4">📈</div>
      <h2 className="text-xl font-semibold text-zinc-100 mb-2">
        NRI Fusion Dashboard Unavailable
      </h2>
      <p className="text-red-400">
        An error occurred while loading the NRI fusion dashboard.
      </p>
    </div>
  );
}

export default function NRIFusionPage() {
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
            <TrendingUp className="h-6 w-6 text-amber-400" strokeWidth={1.5} />
          </div>
          <div className="flex-1">
            <h1 className="text-[20px] font-semibold text-zinc-100">
              NRI Fusion Engine
            </h1>
            <p className="text-[13px] text-zinc-500 mt-1">
              Advanced Neurological Risk Index based on Bayesian fusion of
              multi-modal biomarkers.
            </p>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
          <div className="flex items-center gap-2 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <Clock className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-100">
                ~0.3ms
              </div>
              <div className="text-[11px] text-zinc-500">Processing</div>
            </div>
          </div>
          <div className="flex items-center gap-2 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <Target className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-100">97.2%</div>
              <div className="text-[11px] text-zinc-500">Accuracy</div>
            </div>
          </div>
          <div className="flex items-center gap-2 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <Shield className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-100">
                Verify
              </div>
              <div className="text-[11px] text-zinc-500">Validations</div>
            </div>
          </div>
          <div className="flex items-center gap-2 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <Zap className="h-4 w-4 text-zinc-500" />
            <div>
              <div className="text-[13px] font-medium text-zinc-100">
                Fusion
              </div>
              <div className="text-[11px] text-zinc-500">Bayesian</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <ErrorBoundary fallback={<NRIFusionError />}>
        <Suspense fallback={<NRIFusionSkeleton />}>
          <NRIFusionDashboard onProcessingChange={handleProcessingChange} />
        </Suspense>
      </ErrorBoundary>
    </motion.div>
  );
}
