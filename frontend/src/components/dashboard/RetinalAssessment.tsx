'use client';

import { Eye, Upload, Activity, Clock, TrendingUp } from 'lucide-react';
import React from 'react';

interface RetinalAssessmentProps {
  onProcessingChange: (isProcessing: boolean) => void;
}

export default function RetinalAssessment({ onProcessingChange }: RetinalAssessmentProps) {
  return (
    <div className='space-y-6'>
      {/* Header */}
      <div className='rounded-xl border border-slate-200 bg-white p-6 shadow-sm'>
        <div className='mb-4 flex items-center space-x-3'>
          <div className='rounded-lg bg-gradient-to-r from-green-500 to-green-600 p-3'>
            <Eye className='h-6 w-6 text-white' />
          </div>
          <div>
            <h1 className='text-2xl font-bold text-slate-900'>Retinal Analysis</h1>
            <p className='text-slate-600'>Fundus image analysis for vascular health assessment</p>
          </div>
        </div>

        <div className='grid grid-cols-1 gap-4 text-sm md:grid-cols-3'>
          <div className='flex items-center space-x-2 text-slate-600'>
            <Clock className='h-4 w-4' />
            <span>Processing Time: ~145ms</span>
          </div>
          <div className='flex items-center space-x-2 text-slate-600'>
            <Activity className='h-4 w-4' />
            <span>Accuracy: 89%</span>
          </div>
          <div className='flex items-center space-x-2 text-slate-600'>
            <TrendingUp className='h-4 w-4' />
            <span>Vascular Analysis</span>
          </div>
        </div>
      </div>

      {/* Upload Interface */}
      <div className='rounded-xl border border-slate-200 bg-white p-6 shadow-sm'>
        <h2 className='mb-4 text-lg font-semibold text-slate-900'>Fundus Image Upload</h2>

        <div className='rounded-lg border-2 border-dashed border-slate-300 p-8 text-center'>
          <Upload className='mx-auto mb-4 h-12 w-12 text-slate-400' />
          <h3 className='mb-2 text-lg font-medium text-slate-900'>Upload Retinal Image</h3>
          <p className='mb-4 text-slate-600'>
            Upload a fundus photograph for automated retinal analysis
          </p>
          <button className='rounded-lg bg-green-600 px-6 py-2 font-medium text-white transition-colors hover:bg-green-700'>
            Choose Image File
          </button>
          <p className='mt-2 text-xs text-slate-500'>Supports JPG, PNG files up to 10MB</p>
        </div>
      </div>

      {/* Coming Soon */}
      <div className='rounded-xl border border-green-200 bg-gradient-to-r from-green-50 to-green-100 p-6'>
        <div className='text-center'>
          <Eye className='mx-auto mb-4 h-16 w-16 text-green-600' />
          <h2 className='mb-2 text-xl font-bold text-green-900'>Retinal Analysis Coming Soon</h2>
          <p className='mb-4 text-green-700'>
            Advanced fundus image analysis with vessel tortuosity detection, A/V ratio calculation,
            and glaucoma screening capabilities.
          </p>
          <div className='grid grid-cols-1 gap-4 text-sm md:grid-cols-2'>
            <div className='rounded-lg bg-white/50 p-3'>
              <h3 className='mb-1 font-medium text-green-900'>Detects</h3>
              <ul className='space-y-1 text-green-700'>
                <li>• Diabetic Retinopathy</li>
                <li>• Glaucoma Risk</li>
                <li>• Hypertensive Changes</li>
              </ul>
            </div>
            <div className='rounded-lg bg-white/50 p-3'>
              <h3 className='mb-1 font-medium text-green-900'>Features</h3>
              <ul className='space-y-1 text-green-700'>
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
