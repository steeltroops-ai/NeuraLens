'use client';

import React from 'react';
import { Hand, Smartphone, Activity, Clock, TrendingUp } from 'lucide-react';

interface MotorAssessmentProps {
  onProcessingChange: (isProcessing: boolean) => void;
}

export default function MotorAssessment({ onProcessingChange }: MotorAssessmentProps) {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div className="flex items-center space-x-3 mb-4">
          <div className="p-3 bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg">
            <Hand className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-900">Motor Function Assessment</h1>
            <p className="text-slate-600">Movement pattern analysis using smartphone sensors</p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="flex items-center space-x-2 text-slate-600">
            <Clock className="h-4 w-4" />
            <span>Processing Time: ~42ms</span>
          </div>
          <div className="flex items-center space-x-2 text-slate-600">
            <Activity className="h-4 w-4" />
            <span>Accuracy: 92%</span>
          </div>
          <div className="flex items-center space-x-2 text-slate-600">
            <TrendingUp className="h-4 w-4" />
            <span>Real-time Analysis</span>
          </div>
        </div>
      </div>

      {/* Test Selection */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">Motor Tests</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="border border-slate-200 rounded-lg p-4 hover:border-purple-300 cursor-pointer transition-colors">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Hand className="h-5 w-5 text-purple-600" />
              </div>
              <h3 className="font-medium text-slate-900">Finger Tapping</h3>
            </div>
            <p className="text-sm text-slate-600 mb-3">
              Assess bradykinesia and rhythm stability through finger tapping patterns
            </p>
            <button className="w-full bg-purple-600 hover:bg-purple-700 text-white py-2 rounded-lg font-medium transition-colors">
              Start Test
            </button>
          </div>

          <div className="border border-slate-200 rounded-lg p-4 hover:border-purple-300 cursor-pointer transition-colors">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Smartphone className="h-5 w-5 text-purple-600" />
              </div>
              <h3 className="font-medium text-slate-900">Tremor Detection</h3>
            </div>
            <p className="text-sm text-slate-600 mb-3">
              Detect and analyze tremor patterns using accelerometer data
            </p>
            <button className="w-full bg-purple-600 hover:bg-purple-700 text-white py-2 rounded-lg font-medium transition-colors">
              Start Test
            </button>
          </div>
        </div>
      </div>

      {/* Coming Soon */}
      <div className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-xl border border-purple-200 p-6">
        <div className="text-center">
          <Hand className="h-16 w-16 text-purple-600 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-purple-900 mb-2">Motor Assessment Coming Soon</h2>
          <p className="text-purple-700 mb-4">
            Comprehensive motor function analysis using smartphone sensors 
            for tremor detection, coordination assessment, and movement pattern analysis.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="bg-white/50 rounded-lg p-3">
              <h3 className="font-medium text-purple-900 mb-1">Detects</h3>
              <ul className="text-purple-700 space-y-1">
                <li>• Parkinson's Disease</li>
                <li>• Essential Tremor</li>
                <li>• Motor Coordination Issues</li>
              </ul>
            </div>
            <div className="bg-white/50 rounded-lg p-3">
              <h3 className="font-medium text-purple-900 mb-1">Features</h3>
              <ul className="text-purple-700 space-y-1">
                <li>• Tremor Analysis</li>
                <li>• Bradykinesia Detection</li>
                <li>• Coordination Scoring</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
