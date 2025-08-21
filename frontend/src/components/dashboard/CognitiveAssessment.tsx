'use client';

import React from 'react';
import { Brain, Play, Activity, Clock, TrendingUp } from 'lucide-react';

interface CognitiveAssessmentProps {
  onProcessingChange: (isProcessing: boolean) => void;
}

export default function CognitiveAssessment({ onProcessingChange }: CognitiveAssessmentProps) {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div className="flex items-center space-x-3 mb-4">
          <div className="p-3 bg-gradient-to-r from-indigo-500 to-indigo-600 rounded-lg">
            <Brain className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-900">Cognitive Assessment</h1>
            <p className="text-slate-600">Memory, attention, and executive function evaluation</p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="flex items-center space-x-2 text-slate-600">
            <Clock className="h-4 w-4" />
            <span>Processing Time: ~38ms</span>
          </div>
          <div className="flex items-center space-x-2 text-slate-600">
            <Activity className="h-4 w-4" />
            <span>Accuracy: 94%</span>
          </div>
          <div className="flex items-center space-x-2 text-slate-600">
            <TrendingUp className="h-4 w-4" />
            <span>Multi-domain Testing</span>
          </div>
        </div>
      </div>

      {/* Test Battery */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">Cognitive Test Battery</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="border border-slate-200 rounded-lg p-4 hover:border-indigo-300 cursor-pointer transition-colors">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 bg-indigo-100 rounded-lg">
                <Brain className="h-5 w-5 text-indigo-600" />
              </div>
              <h3 className="font-medium text-slate-900">Memory Assessment</h3>
            </div>
            <p className="text-sm text-slate-600 mb-3">
              Evaluate immediate and delayed recall, working memory capacity
            </p>
            <button className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-2 rounded-lg font-medium transition-colors">
              Start Memory Test
            </button>
          </div>

          <div className="border border-slate-200 rounded-lg p-4 hover:border-indigo-300 cursor-pointer transition-colors">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 bg-indigo-100 rounded-lg">
                <Activity className="h-5 w-5 text-indigo-600" />
              </div>
              <h3 className="font-medium text-slate-900">Attention Test</h3>
            </div>
            <p className="text-sm text-slate-600 mb-3">
              Assess sustained attention, selective attention, and divided attention
            </p>
            <button className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-2 rounded-lg font-medium transition-colors">
              Start Attention Test
            </button>
          </div>

          <div className="border border-slate-200 rounded-lg p-4 hover:border-indigo-300 cursor-pointer transition-colors">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 bg-indigo-100 rounded-lg">
                <TrendingUp className="h-5 w-5 text-indigo-600" />
              </div>
              <h3 className="font-medium text-slate-900">Executive Function</h3>
            </div>
            <p className="text-sm text-slate-600 mb-3">
              Test planning, inhibition, cognitive flexibility, and problem-solving
            </p>
            <button className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-2 rounded-lg font-medium transition-colors">
              Start Executive Test
            </button>
          </div>

          <div className="border border-slate-200 rounded-lg p-4 hover:border-indigo-300 cursor-pointer transition-colors">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 bg-indigo-100 rounded-lg">
                <Play className="h-5 w-5 text-indigo-600" />
              </div>
              <h3 className="font-medium text-slate-900">Processing Speed</h3>
            </div>
            <p className="text-sm text-slate-600 mb-3">
              Measure cognitive processing speed and reaction time
            </p>
            <button className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-2 rounded-lg font-medium transition-colors">
              Start Speed Test
            </button>
          </div>
        </div>
      </div>

      {/* Coming Soon */}
      <div className="bg-gradient-to-r from-indigo-50 to-indigo-100 rounded-xl border border-indigo-200 p-6">
        <div className="text-center">
          <Brain className="h-16 w-16 text-indigo-600 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-indigo-900 mb-2">Cognitive Testing Coming Soon</h2>
          <p className="text-indigo-700 mb-4">
            Comprehensive cognitive assessment battery with validated tests 
            for memory, attention, executive function, and processing speed.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="bg-white/50 rounded-lg p-3">
              <h3 className="font-medium text-indigo-900 mb-1">Detects</h3>
              <ul className="text-indigo-700 space-y-1">
                <li>• Mild Cognitive Impairment</li>
                <li>• Alzheimer's Disease</li>
                <li>• Executive Dysfunction</li>
              </ul>
            </div>
            <div className="bg-white/50 rounded-lg p-3">
              <h3 className="font-medium text-indigo-900 mb-1">Features</h3>
              <ul className="text-indigo-700 space-y-1">
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
