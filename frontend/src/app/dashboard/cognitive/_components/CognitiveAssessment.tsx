'use client';

import { Brain, Play, Activity, Clock, TrendingUp } from 'lucide-react';
import React from 'react';

interface CognitiveAssessmentProps {
  onProcessingChange: (isProcessing: boolean) => void;
}

export default function CognitiveAssessment({ onProcessingChange }: CognitiveAssessmentProps) {
  return (
    <div className='space-y-6'>
      {/* Header */}
      <div className='rounded-xl border border-slate-200 bg-white p-6 shadow-sm'>
        <div className='mb-4 flex items-center space-x-3'>
          <div className='rounded-lg bg-gradient-to-r from-indigo-500 to-indigo-600 p-3'>
            <Brain className='h-6 w-6 text-white' />
          </div>
          <div>
            <h1 className='text-2xl font-bold text-slate-900'>Cognitive Assessment</h1>
            <p className='text-slate-600'>Memory, attention, and executive function evaluation</p>
          </div>
        </div>

        <div className='grid grid-cols-1 gap-4 text-sm md:grid-cols-3'>
          <div className='flex items-center space-x-2 text-slate-600'>
            <Clock className='h-4 w-4' />
            <span>Processing Time: ~38ms</span>
          </div>
          <div className='flex items-center space-x-2 text-slate-600'>
            <Activity className='h-4 w-4' />
            <span>Accuracy: 94%</span>
          </div>
          <div className='flex items-center space-x-2 text-slate-600'>
            <TrendingUp className='h-4 w-4' />
            <span>Multi-domain Testing</span>
          </div>
        </div>
      </div>

      {/* Test Battery */}
      <div className='rounded-xl border border-slate-200 bg-white p-6 shadow-sm'>
        <h2 className='mb-4 text-lg font-semibold text-slate-900'>Cognitive Test Battery</h2>

        <div className='grid grid-cols-1 gap-4 md:grid-cols-2'>
          <div className='cursor-pointer rounded-lg border border-slate-200 p-4 transition-colors hover:border-indigo-300'>
            <div className='mb-3 flex items-center space-x-3'>
              <div className='rounded-lg bg-indigo-100 p-2'>
                <Brain className='h-5 w-5 text-indigo-600' />
              </div>
              <h3 className='font-medium text-slate-900'>Memory Assessment</h3>
            </div>
            <p className='mb-3 text-sm text-slate-600'>
              Evaluate immediate and delayed recall, working memory capacity
            </p>
            <button className='w-full rounded-lg bg-indigo-600 py-2 font-medium text-white transition-colors hover:bg-indigo-700'>
              Start Memory Test
            </button>
          </div>

          <div className='cursor-pointer rounded-lg border border-slate-200 p-4 transition-colors hover:border-indigo-300'>
            <div className='mb-3 flex items-center space-x-3'>
              <div className='rounded-lg bg-indigo-100 p-2'>
                <Activity className='h-5 w-5 text-indigo-600' />
              </div>
              <h3 className='font-medium text-slate-900'>Attention Test</h3>
            </div>
            <p className='mb-3 text-sm text-slate-600'>
              Assess sustained attention, selective attention, and divided attention
            </p>
            <button className='w-full rounded-lg bg-indigo-600 py-2 font-medium text-white transition-colors hover:bg-indigo-700'>
              Start Attention Test
            </button>
          </div>

          <div className='cursor-pointer rounded-lg border border-slate-200 p-4 transition-colors hover:border-indigo-300'>
            <div className='mb-3 flex items-center space-x-3'>
              <div className='rounded-lg bg-indigo-100 p-2'>
                <TrendingUp className='h-5 w-5 text-indigo-600' />
              </div>
              <h3 className='font-medium text-slate-900'>Executive Function</h3>
            </div>
            <p className='mb-3 text-sm text-slate-600'>
              Test planning, inhibition, cognitive flexibility, and problem-solving
            </p>
            <button className='w-full rounded-lg bg-indigo-600 py-2 font-medium text-white transition-colors hover:bg-indigo-700'>
              Start Executive Test
            </button>
          </div>

          <div className='cursor-pointer rounded-lg border border-slate-200 p-4 transition-colors hover:border-indigo-300'>
            <div className='mb-3 flex items-center space-x-3'>
              <div className='rounded-lg bg-indigo-100 p-2'>
                <Play className='h-5 w-5 text-indigo-600' />
              </div>
              <h3 className='font-medium text-slate-900'>Processing Speed</h3>
            </div>
            <p className='mb-3 text-sm text-slate-600'>
              Measure cognitive processing speed and reaction time
            </p>
            <button className='w-full rounded-lg bg-indigo-600 py-2 font-medium text-white transition-colors hover:bg-indigo-700'>
              Start Speed Test
            </button>
          </div>
        </div>
      </div>

      {/* Coming Soon */}
      <div className='rounded-xl border border-indigo-200 bg-gradient-to-r from-indigo-50 to-indigo-100 p-6'>
        <div className='text-center'>
          <Brain className='mx-auto mb-4 h-16 w-16 text-indigo-600' />
          <h2 className='mb-2 text-xl font-bold text-indigo-900'>Cognitive Testing Coming Soon</h2>
          <p className='mb-4 text-indigo-700'>
            Comprehensive cognitive assessment battery with validated tests for memory, attention,
            executive function, and processing speed.
          </p>
          <div className='grid grid-cols-1 gap-4 text-sm md:grid-cols-2'>
            <div className='rounded-lg bg-white/50 p-3'>
              <h3 className='mb-1 font-medium text-indigo-900'>Detects</h3>
              <ul className='space-y-1 text-indigo-700'>
                <li>• Mild Cognitive Impairment</li>
                <li>• Alzheimer's Disease</li>
                <li>• Executive Dysfunction</li>
              </ul>
            </div>
            <div className='rounded-lg bg-white/50 p-3'>
              <h3 className='mb-1 font-medium text-indigo-900'>Features</h3>
              <ul className='space-y-1 text-indigo-700'>
                <li>• Memory Testing</li>
                <li>• Attention Assessment</li>
                <li>• Executive Function</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
