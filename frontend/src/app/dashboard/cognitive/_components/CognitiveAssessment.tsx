"use client";

import { Brain, Play, Activity, Clock, TrendingUp } from "lucide-react";
import React from "react";

interface CognitiveAssessmentProps {
  onProcessingChange: (isProcessing: boolean) => void;
}

export default function CognitiveAssessment({
  onProcessingChange,
}: CognitiveAssessmentProps) {
  return (
    <div className="space-y-6">
      {/* Test Battery */}
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6">
        <h2 className="mb-4 text-[16px] font-semibold text-zinc-100">
          Cognitive Test Battery
        </h2>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="cursor-pointer rounded-lg border border-zinc-700 p-4 transition-all hover:border-amber-500/50 bg-zinc-800/50">
            <div className="mb-3 flex items-center gap-3">
              <div className="p-2 rounded-lg bg-amber-500/15">
                <Brain className="h-5 w-5 text-amber-400" strokeWidth={1.5} />
              </div>
              <h3 className="text-[14px] font-medium text-zinc-100">
                Memory Assessment
              </h3>
            </div>
            <p className="mb-3 text-[12px] text-zinc-400 leading-relaxed">
              Evaluate immediate and delayed recall, working memory capacity
            </p>
            <button className="w-full rounded-lg bg-amber-600 py-2.5 text-[13px] font-medium text-white transition-colors hover:bg-amber-700">
              Start Memory Test
            </button>
          </div>

          <div className="cursor-pointer rounded-lg border border-zinc-700 p-4 transition-all hover:border-amber-500/50 bg-zinc-800/50">
            <div className="mb-3 flex items-center gap-3">
              <div className="p-2 rounded-lg bg-amber-500/15">
                <Activity
                  className="h-5 w-5 text-amber-400"
                  strokeWidth={1.5}
                />
              </div>
              <h3 className="text-[14px] font-medium text-zinc-100">
                Attention Test
              </h3>
            </div>
            <p className="mb-3 text-[12px] text-zinc-400 leading-relaxed">
              Assess sustained attention, selective attention, and divided
              attention
            </p>
            <button className="w-full rounded-lg bg-amber-600 py-2.5 text-[13px] font-medium text-white transition-colors hover:bg-amber-700">
              Start Attention Test
            </button>
          </div>

          <div className="cursor-pointer rounded-lg border border-zinc-700 p-4 transition-all hover:border-amber-500/50 bg-zinc-800/50">
            <div className="mb-3 flex items-center gap-3">
              <div className="p-2 rounded-lg bg-amber-500/15">
                <TrendingUp
                  className="h-5 w-5 text-amber-400"
                  strokeWidth={1.5}
                />
              </div>
              <h3 className="text-[14px] font-medium text-zinc-100">
                Executive Function
              </h3>
            </div>
            <p className="mb-3 text-[12px] text-zinc-400 leading-relaxed">
              Test planning, inhibition, cognitive flexibility, and
              problem-solving
            </p>
            <button className="w-full rounded-lg bg-amber-600 py-2.5 text-[13px] font-medium text-white transition-colors hover:bg-amber-700">
              Start Executive Test
            </button>
          </div>

          <div className="cursor-pointer rounded-lg border border-zinc-700 p-4 transition-all hover:border-amber-500/50 bg-zinc-800/50">
            <div className="mb-3 flex items-center gap-3">
              <div className="p-2 rounded-lg bg-amber-500/15">
                <Play className="h-5 w-5 text-amber-400" strokeWidth={1.5} />
              </div>
              <h3 className="text-[14px] font-medium text-zinc-100">
                Processing Speed
              </h3>
            </div>
            <p className="mb-3 text-[12px] text-zinc-400 leading-relaxed">
              Measure cognitive processing speed and reaction time
            </p>
            <button className="w-full rounded-lg bg-amber-600 py-2.5 text-[13px] font-medium text-white transition-colors hover:bg-amber-700">
              Start Speed Test
            </button>
          </div>
        </div>
      </div>

      {/* Coming Soon */}
      <div className="bg-zinc-900 rounded-lg border border-amber-500/30 p-8">
        <div className="text-center">
          <div className="inline-flex p-4 rounded-full bg-amber-500/15 mb-4">
            <Brain className="h-12 w-12 text-amber-400" strokeWidth={1.5} />
          </div>
          <h2 className="mb-2 text-[18px] font-semibold text-zinc-100">
            Cognitive Testing Coming Soon
          </h2>
          <p className="mb-6 text-[13px] text-zinc-400 max-w-2xl mx-auto leading-relaxed">
            Comprehensive cognitive assessment battery with validated tests for
            memory, attention, executive function, and processing speed.
          </p>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 max-w-2xl mx-auto">
            <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-4">
              <h3 className="mb-2 text-[14px] font-medium text-zinc-200">
                Detects
              </h3>
              <ul className="space-y-1 text-[12px] text-zinc-400 text-left">
                <li>* Mild Cognitive Impairment</li>
                <li>* Alzheimer's Disease</li>
                <li>* Executive Dysfunction</li>
              </ul>
            </div>
            <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-4">
              <h3 className="mb-2 text-[14px] font-medium text-zinc-200">
                Features
              </h3>
              <ul className="space-y-1 text-[12px] text-zinc-400 text-left">
                <li>* Memory Testing</li>
                <li>* Attention Assessment</li>
                <li>* Executive Function</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
