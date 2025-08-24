'use client';

import { Hand, Smartphone, Activity, Clock, TrendingUp } from 'lucide-react';
import React from 'react';

interface MotorAssessmentProps {
  onProcessingChange: (isProcessing: boolean) => void;
}

export default function MotorAssessment({ onProcessingChange }: MotorAssessmentProps) {
  return (
    <div className='space-y-6'>
      {/* Header */}
      <div className='rounded-xl border border-slate-200 bg-white p-6 shadow-sm'>
        <div className='mb-4 flex items-center space-x-3'>
          <div className='rounded-lg bg-gradient-to-r from-purple-500 to-purple-600 p-3'>
            <Hand className='h-6 w-6 text-white' />
          </div>
          <div>
            <h1 className='text-2xl font-bold text-slate-900'>Motor Function Assessment</h1>
            <p className='text-slate-600'>Movement pattern analysis using smartphone sensors</p>
          </div>
        </div>

        <div className='grid grid-cols-1 gap-4 text-sm md:grid-cols-3'>
          <div className='flex items-center space-x-2 text-slate-600'>
            <Clock className='h-4 w-4' />
            <span>Processing Time: ~42ms</span>
          </div>
          <div className='flex items-center space-x-2 text-slate-600'>
            <Activity className='h-4 w-4' />
            <span>Accuracy: 92%</span>
          </div>
          <div className='flex items-center space-x-2 text-slate-600'>
            <TrendingUp className='h-4 w-4' />
            <span>Real-time Analysis</span>
          </div>
        </div>
      </div>

      {/* Test Selection */}
      <div className='rounded-xl border border-slate-200 bg-white p-6 shadow-sm'>
        <h2 className='mb-4 text-lg font-semibold text-slate-900'>Motor Tests</h2>

        <div className='grid grid-cols-1 gap-4 md:grid-cols-2'>
          <div className='cursor-pointer rounded-lg border border-slate-200 p-4 transition-colors hover:border-purple-300'>
            <div className='mb-3 flex items-center space-x-3'>
              <div className='rounded-lg bg-purple-100 p-2'>
                <Hand className='h-5 w-5 text-purple-600' />
              </div>
              <h3 className='font-medium text-slate-900'>Finger Tapping</h3>
            </div>
            <p className='mb-3 text-sm text-slate-600'>
              Assess bradykinesia and rhythm stability through finger tapping patterns
            </p>
            <button className='w-full rounded-lg bg-purple-600 py-2 font-medium text-white transition-colors hover:bg-purple-700'>
              Start Test
            </button>
          </div>

          <div className='cursor-pointer rounded-lg border border-slate-200 p-4 transition-colors hover:border-purple-300'>
            <div className='mb-3 flex items-center space-x-3'>
              <div className='rounded-lg bg-purple-100 p-2'>
                <Smartphone className='h-5 w-5 text-purple-600' />
              </div>
              <h3 className='font-medium text-slate-900'>Tremor Detection</h3>
            </div>
            <p className='mb-3 text-sm text-slate-600'>
              Detect and analyze tremor patterns using accelerometer data
            </p>
            <button className='w-full rounded-lg bg-purple-600 py-2 font-medium text-white transition-colors hover:bg-purple-700'>
              Start Test
            </button>
          </div>
        </div>
      </div>

      {/* Coming Soon */}
      <div className='rounded-xl border border-purple-200 bg-gradient-to-r from-purple-50 to-purple-100 p-6'>
        <div className='text-center'>
          <Hand className='mx-auto mb-4 h-16 w-16 text-purple-600' />
          <h2 className='mb-2 text-xl font-bold text-purple-900'>Motor Assessment Coming Soon</h2>
          <p className='mb-4 text-purple-700'>
            Comprehensive motor function analysis using smartphone sensors for tremor detection,
            coordination assessment, and movement pattern analysis.
          </p>
          <div className='grid grid-cols-1 gap-4 text-sm md:grid-cols-2'>
            <div className='rounded-lg bg-white/50 p-3'>
              <h3 className='mb-1 font-medium text-purple-900'>Detects</h3>
              <ul className='space-y-1 text-purple-700'>
                <li>• Parkinson's Disease</li>
                <li>• Essential Tremor</li>
                <li>• Motor Coordination Issues</li>
              </ul>
            </div>
            <div className='rounded-lg bg-white/50 p-3'>
              <h3 className='mb-1 font-medium text-purple-900'>Features</h3>
              <ul className='space-y-1 text-purple-700'>
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
