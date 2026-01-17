'use client';

import { Brain, Play, Activity, Clock, TrendingUp } from 'lucide-react';
import React from 'react';

interface CognitiveAssessmentProps {
  onProcessingChange: (isProcessing: boolean) => void;
}

export default function CognitiveAssessment({ onProcessingChange }: CognitiveAssessmentProps) {
  return (
    <div className='space-y-6'>
      {/* Header - Enhanced with cognitive theme */}
      <div className='relative overflow-hidden bg-white rounded-2xl border border-zinc-200/80 p-8'>
        {/* Gradient background */}
        <div className='absolute inset-0 bg-gradient-to-br from-amber-50/40 via-transparent to-orange-50/30 pointer-events-none' />

        <div className='relative'>
          <div className='flex items-start justify-between'>
            <div className='flex items-start gap-4'>
              <div className='p-3 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 shadow-lg shadow-amber-500/20'>
                <Brain className='h-7 w-7 text-white' strokeWidth={2} />
              </div>
              <div>
                <div className='flex items-center gap-3 mb-2'>
                  <h1 className='text-[24px] font-semibold text-zinc-900'>Cognitive Assessment</h1>
                  <span className='px-2.5 py-1 bg-amber-100 text-amber-700 text-[11px] font-medium rounded-full'>
                    Multi-Domain
                  </span>
                </div>
                <p className='text-[14px] text-zinc-600 max-w-xl'>
                  Comprehensive evaluation of memory, attention, and executive function
                </p>
              </div>
            </div>
          </div>

          {/* Feature pills */}
          <div className='flex flex-wrap gap-2 mt-6'>
            <div className='flex items-center gap-2 px-3 py-2 bg-white border border-zinc-200 rounded-lg'>
              <Clock className='h-4 w-4 text-amber-600' strokeWidth={2} />
              <span className='text-[12px] font-medium text-zinc-700'>Processing: ~38ms</span>
            </div>
            <div className='flex items-center gap-2 px-3 py-2 bg-white border border-zinc-200 rounded-lg'>
              <Activity className='h-4 w-4 text-orange-600' strokeWidth={2} />
              <span className='text-[12px] font-medium text-zinc-700'>Accuracy: 94%</span>
            </div>
            <div className='flex items-center gap-2 px-3 py-2 bg-white border border-zinc-200 rounded-lg'>
              <TrendingUp className='h-4 w-4 text-yellow-600' strokeWidth={2} />
              <span className='text-[12px] font-medium text-zinc-700'>Multi-domain Testing</span>
            </div>
          </div>
        </div>
      </div>

      {/* Test Battery */}
      <div className='rounded-xl border border-zinc-200 bg-white p-6'>
        <h2 className='mb-4 text-[16px] font-semibold text-zinc-900'>Cognitive Test Battery</h2>

        <div className='grid grid-cols-1 gap-4 md:grid-cols-2'>
          <div className='cursor-pointer rounded-lg border border-zinc-200 p-4 transition-all hover:border-amber-300 hover:shadow-sm'>
            <div className='mb-3 flex items-center gap-3'>
              <div className='p-2 rounded-lg bg-amber-50'>
                <Brain className='h-5 w-5 text-amber-600' strokeWidth={1.5} />
              </div>
              <h3 className='text-[14px] font-medium text-zinc-900'>Memory Assessment</h3>
            </div>
            <p className='mb-3 text-[12px] text-zinc-600 leading-relaxed'>
              Evaluate immediate and delayed recall, working memory capacity
            </p>
            <button className='w-full rounded-lg bg-amber-600 py-2.5 text-[13px] font-medium text-white transition-colors hover:bg-amber-700'>
              Start Memory Test
            </button>
          </div>

          <div className='cursor-pointer rounded-lg border border-zinc-200 p-4 transition-all hover:border-amber-300 hover:shadow-sm'>
            <div className='mb-3 flex items-center gap-3'>
              <div className='p-2 rounded-lg bg-amber-50'>
                <Activity className='h-5 w-5 text-amber-600' strokeWidth={1.5} />
              </div>
              <h3 className='text-[14px] font-medium text-zinc-900'>Attention Test</h3>
            </div>
            <p className='mb-3 text-[12px] text-zinc-600 leading-relaxed'>
              Assess sustained attention, selective attention, and divided attention
            </p>
            <button className='w-full rounded-lg bg-amber-600 py-2.5 text-[13px] font-medium text-white transition-colors hover:bg-amber-700'>
              Start Attention Test
            </button>
          </div>

          <div className='cursor-pointer rounded-lg border border-zinc-200 p-4 transition-all hover:border-amber-300 hover:shadow-sm'>
            <div className='mb-3 flex items-center gap-3'>
              <div className='p-2 rounded-lg bg-amber-50'>
                <TrendingUp className='h-5 w-5 text-amber-600' strokeWidth={1.5} />
              </div>
              <h3 className='text-[14px] font-medium text-zinc-900'>Executive Function</h3>
            </div>
            <p className='mb-3 text-[12px] text-zinc-600 leading-relaxed'>
              Test planning, inhibition, cognitive flexibility, and problem-solving
            </p>
            <button className='w-full rounded-lg bg-amber-600 py-2.5 text-[13px] font-medium text-white transition-colors hover:bg-amber-700'>
              Start Executive Test
            </button>
          </div>

          <div className='cursor-pointer rounded-lg border border-zinc-200 p-4 transition-all hover:border-amber-300 hover:shadow-sm'>
            <div className='mb-3 flex items-center gap-3'>
              <div className='p-2 rounded-lg bg-amber-50'>
                <Play className='h-5 w-5 text-amber-600' strokeWidth={1.5} />
              </div>
              <h3 className='text-[14px] font-medium text-zinc-900'>Processing Speed</h3>
            </div>
            <p className='mb-3 text-[12px] text-zinc-600 leading-relaxed'>
              Measure cognitive processing speed and reaction time
            </p>
            <button className='w-full rounded-lg bg-amber-600 py-2.5 text-[13px] font-medium text-white transition-colors hover:bg-amber-700'>
              Start Speed Test
            </button>
          </div>
        </div>
      </div>

      {/* Coming Soon */}
      <div className='rounded-xl border border-amber-200 bg-amber-50 p-8'>
        <div className='text-center'>
          <div className='inline-flex p-4 rounded-full bg-amber-100 mb-4'>
            <Brain className='h-12 w-12 text-amber-600' strokeWidth={1.5} />
          </div>
          <h2 className='mb-2 text-[18px] font-semibold text-amber-900'>Cognitive Testing Coming Soon</h2>
          <p className='mb-6 text-[13px] text-amber-700 max-w-2xl mx-auto leading-relaxed'>
            Comprehensive cognitive assessment battery with validated tests for memory, attention,
            executive function, and processing speed.
          </p>
          <div className='grid grid-cols-1 gap-4 md:grid-cols-2 max-w-2xl mx-auto'>
            <div className='rounded-lg bg-white border border-amber-100 p-4'>
              <h3 className='mb-2 text-[14px] font-medium text-amber-900'>Detects</h3>
              <ul className='space-y-1 text-[12px] text-amber-700 text-left'>
                <li>• Mild Cognitive Impairment</li>
                <li>• Alzheimer's Disease</li>
                <li>• Executive Dysfunction</li>
              </ul>
            </div>
            <div className='rounded-lg bg-white border border-amber-100 p-4'>
              <h3 className='mb-2 text-[14px] font-medium text-amber-900'>Features</h3>
              <ul className='space-y-1 text-[12px] text-amber-700 text-left'>
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
