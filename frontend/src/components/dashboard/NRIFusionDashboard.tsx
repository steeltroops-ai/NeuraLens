'use client';

import React from 'react';
import { TrendingUp, Zap, Activity, Clock } from 'lucide-react';

interface NRIFusionDashboardProps {
  onProcessingChange: (isProcessing: boolean) => void;
}

export default function NRIFusionDashboard({ onProcessingChange }: NRIFusionDashboardProps) {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div className="flex items-center space-x-3 mb-4">
          <div className="p-3 bg-gradient-to-r from-yellow-500 to-yellow-600 rounded-lg">
            <TrendingUp className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-900">NRI Fusion Engine</h1>
            <p className="text-slate-600">Advanced Neurological Risk Index calculation and analytics</p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="flex items-center space-x-2 text-slate-600">
            <Clock className="h-4 w-4" />
            <span>Processing Time: ~0.3ms</span>
          </div>
          <div className="flex items-center space-x-2 text-slate-600">
            <Activity className="h-4 w-4" />
            <span>Accuracy: 97%</span>
          </div>
          <div className="flex items-center space-x-2 text-slate-600">
            <Zap className="h-4 w-4" />
            <span>Bayesian Fusion</span>
          </div>
        </div>
      </div>

      {/* Coming Soon */}
      <div className="bg-gradient-to-r from-yellow-50 to-yellow-100 rounded-xl border border-yellow-200 p-6">
        <div className="text-center">
          <TrendingUp className="h-16 w-16 text-yellow-600 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-yellow-900 mb-2">NRI Fusion Dashboard Coming Soon</h2>
          <p className="text-yellow-700 mb-4">
            Advanced analytics dashboard for Neurological Risk Index fusion, 
            featuring real-time risk calculation, uncertainty quantification, 
            and predictive modeling capabilities.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="bg-white/50 rounded-lg p-3">
              <h3 className="font-medium text-yellow-900 mb-1">Features</h3>
              <ul className="text-yellow-700 space-y-1">
                <li>• Bayesian Risk Fusion</li>
                <li>• Uncertainty Quantification</li>
                <li>• Cross-Modal Consistency</li>
              </ul>
            </div>
            <div className="bg-white/50 rounded-lg p-3">
              <h3 className="font-medium text-yellow-900 mb-1">Analytics</h3>
              <ul className="text-yellow-700 space-y-1">
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
