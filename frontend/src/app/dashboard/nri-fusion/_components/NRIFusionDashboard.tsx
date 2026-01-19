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
      {/* Header removed and lifted to page.tsx */}

      {/* Coming Soon */}
      <div className="rounded-xl border border-yellow-200 bg-yellow-50 p-8">
        <div className="text-center">
          <div className="inline-flex p-4 rounded-full bg-yellow-100 mb-4">
            <TrendingUp
              className="h-12 w-12 text-yellow-600"
              strokeWidth={1.5}
            />
          </div>
          <h2 className="mb-2 text-[18px] font-semibold text-yellow-900">
            NRI Fusion Dashboard Coming Soon
          </h2>
          <p className="mb-6 text-[13px] text-yellow-700 max-w-2xl mx-auto leading-relaxed">
            Advanced analytics dashboard for Neurological Risk Index fusion,
            featuring real-time risk calculation, uncertainty quantification,
            and predictive modeling capabilities.
          </p>
          <div className="grid grid-cols-1 gap-4 text-sm md:grid-cols-2">
            <div className="rounded-lg bg-white/50 p-3">
              <h3 className="mb-1 font-medium text-yellow-900">Features</h3>
              <ul className="space-y-1 text-yellow-700">
                <li>• Bayesian Risk Fusion</li>
                <li>• Uncertainty Quantification</li>
                <li>• Cross-Modal Consistency</li>
              </ul>
            </div>
            <div className="rounded-lg bg-white/50 p-3">
              <h3 className="mb-1 font-medium text-yellow-900">Analytics</h3>
              <ul className="space-y-1 text-yellow-700">
                <li>• Risk Trend Analysis</li>
                <li>• Modality Contributions</li>
                <li>• Predictive Modeling</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
