"use client";

/**
 * Analytics Page - Dark Theme Enterprise Dashboard
 *
 * Professional analytics dashboard with realistic neurological data
 * matching the dark theme design philosophy.
 */

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  BarChart3,
  TrendingUp,
  Activity,
  Calendar,
  Clock,
  Users,
  Brain,
  AlertTriangle,
} from "lucide-react";

/**
 * Analytics Page Component
 */
export default function AnalyticsPage() {
  const [analyticsData, setAnalyticsData] = useState({
    totalAssessments: 0,
    averageNRIScore: 0,
    thisMonth: 0,
    improvement: 0,
  });

  // Load realistic analytics data
  useEffect(() => {
    // Simulate realistic neurological analytics data
    setAnalyticsData({
      totalAssessments: 156,
      averageNRIScore: 72,
      thisMonth: 23,
      improvement: 15,
    });
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-lg bg-blue-500/15">
            <BarChart3 className="h-6 w-6 text-blue-400" strokeWidth={1.5} />
          </div>
          <div className="flex-1">
            <h1 className="text-[20px] font-semibold text-zinc-100">
              Neurological Analytics
            </h1>
            <p className="text-[13px] text-zinc-400 mt-1">
              Track assessment trends and neurological health insights over time
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4">
          <div className="flex items-center gap-2 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <TrendingUp className="h-4 w-4 text-zinc-400" />
            <span className="text-[13px] text-zinc-300">Trend Analysis</span>
          </div>
          <div className="flex items-center gap-2 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <Activity className="h-4 w-4 text-zinc-400" />
            <span className="text-[13px] text-zinc-300">
              Performance Metrics
            </span>
          </div>
          <div className="flex items-center gap-2 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <Calendar className="h-4 w-4 text-zinc-400" />
            <span className="text-[13px] text-zinc-300">Historical Data</span>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-[13px] font-medium text-zinc-300">
              Total Assessments
            </span>
            <div className="p-2 rounded-lg bg-blue-500/15">
              <Activity className="h-4 w-4 text-blue-400" strokeWidth={1.5} />
            </div>
          </div>
          <div className="text-[24px] font-semibold text-zinc-100">
            {analyticsData.totalAssessments}
          </div>
          <p className="text-[11px] text-emerald-400 mt-1">+12 this week</p>
        </div>

        <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-[13px] font-medium text-zinc-300">
              Average NRI Score
            </span>
            <div className="p-2 rounded-lg bg-emerald-500/15">
              <TrendingUp
                className="h-4 w-4 text-emerald-400"
                strokeWidth={1.5}
              />
            </div>
          </div>
          <div className="text-[24px] font-semibold text-zinc-100">
            {analyticsData.averageNRIScore}%
          </div>
          <p className="text-[11px] text-emerald-400 mt-1">+5% improvement</p>
        </div>

        <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-[13px] font-medium text-zinc-300">
              This Month
            </span>
            <div className="p-2 rounded-lg bg-violet-500/15">
              <Calendar className="h-4 w-4 text-violet-400" strokeWidth={1.5} />
            </div>
          </div>
          <div className="text-[24px] font-semibold text-zinc-100">
            {analyticsData.thisMonth}
          </div>
          <p className="text-[11px] text-zinc-500 mt-1">
            Assessments completed
          </p>
        </div>

        <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-[13px] font-medium text-zinc-300">
              Patient Improvement
            </span>
            <div className="p-2 rounded-lg bg-amber-500/15">
              <BarChart3 className="h-4 w-4 text-amber-400" strokeWidth={1.5} />
            </div>
          </div>
          <div className="text-[24px] font-semibold text-zinc-100">
            {analyticsData.improvement}%
          </div>
          <p className="text-[11px] text-emerald-400 mt-1">Above baseline</p>
        </div>
      </div>

      {/* Assessment Trends */}
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6">
        <h2 className="text-[16px] font-semibold text-zinc-100 mb-4">
          Assessment Trends (Last 30 Days)
        </h2>
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-cyan-500/15">
                <Brain className="h-5 w-5 text-cyan-400" />
              </div>
              <div>
                <p className="text-[14px] font-medium text-zinc-100">
                  Retinal Scans
                </p>
                <p className="text-[12px] text-zinc-500">
                  Diabetic retinopathy screening
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-[16px] font-semibold text-zinc-100">
                8 assessments
              </p>
              <p className="text-[11px] text-emerald-400">+2 from last month</p>
            </div>
          </div>

          <div className="flex items-center justify-between p-4 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-500/15">
                <Activity className="h-5 w-5 text-blue-400" />
              </div>
              <div>
                <p className="text-[14px] font-medium text-zinc-100">
                  Speech Analysis
                </p>
                <p className="text-[12px] text-zinc-500">
                  Parkinson's disease monitoring
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-[16px] font-semibold text-zinc-100">
                12 assessments
              </p>
              <p className="text-[11px] text-emerald-400">+5 from last month</p>
            </div>
          </div>

          <div className="flex items-center justify-between p-4 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-red-500/15">
                <AlertTriangle className="h-5 w-5 text-red-400" />
              </div>
              <div>
                <p className="text-[14px] font-medium text-zinc-100">
                  Cognitive Testing
                </p>
                <p className="text-[12px] text-zinc-500">
                  Dementia and MS screening
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-[16px] font-semibold text-zinc-100">
                3 assessments
              </p>
              <p className="text-[11px] text-amber-400">Same as last month</p>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Distribution */}
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6">
        <h2 className="text-[16px] font-semibold text-zinc-100 mb-4">
          Risk Score Distribution
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-emerald-500/10 rounded-lg border border-emerald-500/30">
            <div className="text-[20px] font-semibold text-emerald-400">12</div>
            <div className="text-[12px] text-emerald-400 font-medium">
              Low Risk (0-25%)
            </div>
            <div className="text-[11px] text-emerald-500/70 mt-1">
              52% of patients
            </div>
          </div>
          <div className="text-center p-4 bg-amber-500/10 rounded-lg border border-amber-500/30">
            <div className="text-[20px] font-semibold text-amber-400">7</div>
            <div className="text-[12px] text-amber-400 font-medium">
              Moderate (26-50%)
            </div>
            <div className="text-[11px] text-amber-500/70 mt-1">
              30% of patients
            </div>
          </div>
          <div className="text-center p-4 bg-orange-500/10 rounded-lg border border-orange-500/30">
            <div className="text-[20px] font-semibold text-orange-400">3</div>
            <div className="text-[12px] text-orange-400 font-medium">
              High (51-75%)
            </div>
            <div className="text-[11px] text-orange-500/70 mt-1">
              13% of patients
            </div>
          </div>
          <div className="text-center p-4 bg-red-500/10 rounded-lg border border-red-500/30">
            <div className="text-[20px] font-semibold text-red-400">1</div>
            <div className="text-[12px] text-red-400 font-medium">
              Critical (76-100%)
            </div>
            <div className="text-[11px] text-red-500/70 mt-1">
              4% of patients
            </div>
          </div>
        </div>
      </div>

      {/* Coming Soon Features */}
      <div className="bg-zinc-900 rounded-lg border border-cyan-500/30 p-6">
        <div className="text-center">
          <div className="inline-flex p-3 rounded-full bg-cyan-500/15 mb-3">
            <BarChart3 className="h-8 w-8 text-cyan-400" strokeWidth={1.5} />
          </div>
          <h2 className="mb-2 text-[16px] font-semibold text-zinc-100">
            Advanced Analytics Coming Soon
          </h2>
          <p className="mb-4 text-[13px] text-zinc-400 max-w-2xl mx-auto leading-relaxed">
            Interactive charts, longitudinal tracking, and predictive analytics
            for neurological conditions.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 max-w-2xl mx-auto">
            <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-3">
              <h3 className="mb-1 text-[13px] font-medium text-zinc-200">
                Interactive Charts
              </h3>
              <p className="text-[11px] text-zinc-500">Visual trend analysis</p>
            </div>
            <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-3">
              <h3 className="mb-1 text-[13px] font-medium text-zinc-200">
                Predictive Models
              </h3>
              <p className="text-[11px] text-zinc-500">Risk forecasting</p>
            </div>
            <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-3">
              <h3 className="mb-1 text-[13px] font-medium text-zinc-200">
                Population Health
              </h3>
              <p className="text-[11px] text-zinc-500">Cohort comparisons</p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
