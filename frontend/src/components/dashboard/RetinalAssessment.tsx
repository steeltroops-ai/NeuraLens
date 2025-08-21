'use client';

import React from 'react';
import { Eye, Upload, Activity, Clock, TrendingUp } from 'lucide-react';

interface RetinalAssessmentProps {
  onProcessingChange: (isProcessing: boolean) => void;
}

export default function RetinalAssessment({ onProcessingChange }: RetinalAssessmentProps) {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div className="flex items-center space-x-3 mb-4">
          <div className="p-3 bg-gradient-to-r from-green-500 to-green-600 rounded-lg">
            <Eye className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-900">Retinal Analysis</h1>
            <p className="text-slate-600">Fundus image analysis for vascular health assessment</p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="flex items-center space-x-2 text-slate-600">
            <Clock className="h-4 w-4" />
            <span>Processing Time: ~145ms</span>
          </div>
          <div className="flex items-center space-x-2 text-slate-600">
            <Activity className="h-4 w-4" />
            <span>Accuracy: 89%</span>
          </div>
          <div className="flex items-center space-x-2 text-slate-600">
            <TrendingUp className="h-4 w-4" />
            <span>Vascular Analysis</span>
          </div>
        </div>
      </div>

      {/* Upload Interface */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">Fundus Image Upload</h2>
        
        <div className="border-2 border-dashed border-slate-300 rounded-lg p-8 text-center">
          <Upload className="h-12 w-12 text-slate-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-slate-900 mb-2">Upload Retinal Image</h3>
          <p className="text-slate-600 mb-4">
            Upload a fundus photograph for automated retinal analysis
          </p>
          <button className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg font-medium transition-colors">
            Choose Image File
          </button>
          <p className="text-xs text-slate-500 mt-2">
            Supports JPG, PNG files up to 10MB
          </p>
        </div>
      </div>

      {/* Coming Soon */}
      <div className="bg-gradient-to-r from-green-50 to-green-100 rounded-xl border border-green-200 p-6">
        <div className="text-center">
          <Eye className="h-16 w-16 text-green-600 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-green-900 mb-2">Retinal Analysis Coming Soon</h2>
          <p className="text-green-700 mb-4">
            Advanced fundus image analysis with vessel tortuosity detection, 
            A/V ratio calculation, and glaucoma screening capabilities.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="bg-white/50 rounded-lg p-3">
              <h3 className="font-medium text-green-900 mb-1">Detects</h3>
              <ul className="text-green-700 space-y-1">
                <li>• Diabetic Retinopathy</li>
                <li>• Glaucoma Risk</li>
                <li>• Hypertensive Changes</li>
              </ul>
            </div>
            <div className="bg-white/50 rounded-lg p-3">
              <h3 className="font-medium text-green-900 mb-1">Features</h3>
              <ul className="text-green-700 space-y-1">
                <li>• Vessel Analysis</li>
                <li>• Cup-Disc Ratio</li>
                <li>• Hemorrhage Detection</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
