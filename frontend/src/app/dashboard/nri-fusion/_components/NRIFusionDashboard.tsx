"use client";

import { TrendingUp, Zap, Activity, Clock } from "lucide-react";
import React from "react";

interface NRIFusionDashboardProps {
  onProcessingChange: (isProcessing: boolean) => void;
}

export default function NRIFusionDashboard({
  onProcessingChange,
}: NRIFusionDashboardProps) {
  return (
    <div className="space-y-6">
      {/* Coming Soon */}
      <div className="bg-zinc-900 rounded-lg border border-amber-500/30 p-8">
        <div className="text-center">
          <div className="inline-flex p-4 rounded-full bg-amber-500/15 mb-4">
            <TrendingUp
              className="h-12 w-12 text-amber-400"
              strokeWidth={1.5}
            />
          </div>
          <h2 className="mb-2 text-[18px] font-semibold text-zinc-100">
            NRI Fusion Dashboard Coming Soon
          </h2>
          <p className="mb-6 text-[13px] text-zinc-400 max-w-2xl mx-auto leading-relaxed">
            Advanced analytics dashboard for Neurological Risk Index fusion,
            featuring real-time risk calculation, uncertainty quantification,
            and predictive modeling capabilities.
          </p>
          <div className="grid grid-cols-1 gap-4 text-sm md:grid-cols-2">
            <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-3">
              <h3 className="mb-1 font-medium text-zinc-200">Features</h3>
              <ul className="space-y-1 text-zinc-400">
                <li>* Bayesian Risk Fusion</li>
                <li>* Uncertainty Quantification</li>
                <li>* Cross-Modal Consistency</li>
              </ul>
            </div>
            <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-3">
              <h3 className="mb-1 font-medium text-zinc-200">Analytics</h3>
              <ul className="space-y-1 text-zinc-400">
                <li>* Risk Trend Analysis</li>
                <li>* Modality Contributions</li>
                <li>* Predictive Modeling</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
