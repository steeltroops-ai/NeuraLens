"use client";

/**
 * Reports Page - Dark Theme Enterprise Dashboard
 *
 * Professional clinical reports dashboard matching the dark theme design philosophy.
 */

import { motion } from "framer-motion";
import {
  FileText,
  Download,
  Share2,
  Printer,
  Clock,
  CheckCircle,
  Eye,
  Brain,
  Activity,
} from "lucide-react";

/**
 * Reports Page Component
 */
export default function ReportsPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-8">
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-4">
            <div className="p-3 rounded-lg bg-emerald-500/15">
              <FileText className="h-7 w-7 text-emerald-400" strokeWidth={2} />
            </div>
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h1 className="text-[24px] font-semibold text-zinc-100">
                  Clinical Reports
                </h1>
                <span className="px-2.5 py-1 bg-emerald-500/15 text-emerald-400 text-[11px] font-medium rounded-full">
                  HIPAA Compliant
                </span>
              </div>
              <p className="text-[14px] text-zinc-400 max-w-xl">
                Generate, export, and share professional medical assessment
                reports
              </p>
            </div>
          </div>

          <button className="px-4 py-2 text-[13px] font-medium text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 rounded-lg transition-colors">
            View Archive
          </button>
        </div>

        {/* Feature pills */}
        <div className="flex flex-wrap gap-2 mt-6">
          <div className="flex items-center gap-2 px-3 py-2 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
            <Download className="h-4 w-4 text-emerald-400" strokeWidth={2} />
            <span className="text-[12px] font-medium text-zinc-300">
              PDF Export
            </span>
          </div>
          <div className="flex items-center gap-2 px-3 py-2 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
            <Share2 className="h-4 w-4 text-blue-400" strokeWidth={2} />
            <span className="text-[12px] font-medium text-zinc-300">
              Provider Sharing
            </span>
          </div>
          <div className="flex items-center gap-2 px-3 py-2 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
            <Printer className="h-4 w-4 text-violet-400" strokeWidth={2} />
            <span className="text-[12px] font-medium text-zinc-300">
              Print Ready
            </span>
          </div>
          <div className="flex items-center gap-2 px-3 py-2 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
            <CheckCircle className="h-4 w-4 text-emerald-400" strokeWidth={2} />
            <span className="text-[12px] font-medium text-zinc-300">
              Secure Storage
            </span>
          </div>
        </div>
      </div>

      {/* Coming Soon Content */}
      <div className="bg-zinc-900 rounded-lg border border-emerald-500/30 p-8">
        <div className="text-center">
          <div className="inline-flex p-4 rounded-full bg-emerald-500/15 mb-4">
            <FileText
              className="h-12 w-12 text-emerald-400"
              strokeWidth={1.5}
            />
          </div>
          <h2 className="mb-2 text-[18px] font-semibold text-zinc-100">
            Reports Center Coming Soon
          </h2>
          <p className="mb-6 text-[13px] text-zinc-400 max-w-2xl mx-auto leading-relaxed">
            Generate comprehensive clinical reports from your assessment data.
            Export to PDF, share with healthcare providers, and maintain a
            complete record of your neurological health assessments.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl mx-auto">
            <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-4">
              <div className="inline-flex p-2 rounded-lg bg-emerald-500/15 mb-3">
                <Download
                  className="h-6 w-6 text-emerald-400"
                  strokeWidth={1.5}
                />
              </div>
              <h3 className="mb-1 text-[14px] font-medium text-zinc-200">
                PDF Export
              </h3>
              <p className="text-[12px] text-zinc-500 leading-relaxed">
                Download clinical-grade PDF reports
              </p>
            </div>
            <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-4">
              <div className="inline-flex p-2 rounded-lg bg-emerald-500/15 mb-3">
                <Share2
                  className="h-6 w-6 text-emerald-400"
                  strokeWidth={1.5}
                />
              </div>
              <h3 className="mb-1 text-[14px] font-medium text-zinc-200">
                Provider Sharing
              </h3>
              <p className="text-[12px] text-zinc-500 leading-relaxed">
                Securely share with healthcare providers
              </p>
            </div>
            <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-4">
              <div className="inline-flex p-2 rounded-lg bg-emerald-500/15 mb-3">
                <CheckCircle
                  className="h-6 w-6 text-emerald-400"
                  strokeWidth={1.5}
                />
              </div>
              <h3 className="mb-1 text-[14px] font-medium text-zinc-200">
                HIPAA Compliant
              </h3>
              <p className="text-[12px] text-zinc-500 leading-relaxed">
                Secure, compliant data handling
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Reports Placeholder */}
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-[18px] font-semibold text-zinc-100">
            Recent Reports
          </h2>
          <button className="px-3 py-1.5 text-[12px] font-medium text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 rounded-lg transition-colors">
            View All
          </button>
        </div>

        <div className="space-y-3">
          {/* Empty state */}
          <div className="text-center py-16">
            <div className="inline-flex p-4 rounded-2xl bg-zinc-800/50 border border-zinc-700/50 mb-4">
              <FileText className="h-10 w-10 text-zinc-500" strokeWidth={1.5} />
            </div>
            <p className="text-[14px] font-medium text-zinc-200 mb-1">
              No reports generated yet
            </p>
            <p className="text-[13px] text-zinc-500">
              Complete an assessment to generate your first clinical report
            </p>
          </div>
        </div>
      </div>

      {/* Report Types */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 hover:border-zinc-700 transition-colors">
          <div className="flex items-start gap-3 mb-4">
            <div className="p-2.5 rounded-lg bg-blue-500/15">
              <FileText className="h-6 w-6 text-blue-400" strokeWidth={2} />
            </div>
            <div className="flex-1">
              <h3 className="text-[15px] font-semibold text-zinc-100 mb-1">
                Assessment Summary
              </h3>
              <p className="text-[13px] text-zinc-400 leading-relaxed">
                Comprehensive summary of individual assessment results with
                biomarkers and clinical recommendations.
              </p>
            </div>
          </div>
          <button
            disabled
            className="w-full rounded-lg bg-zinc-800 py-2.5 text-[13px] font-medium text-zinc-500 cursor-not-allowed border border-zinc-700"
          >
            Coming Soon
          </button>
        </div>

        <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 hover:border-zinc-700 transition-colors">
          <div className="flex items-start gap-3 mb-4">
            <div className="p-2.5 rounded-lg bg-violet-500/15">
              <Clock className="h-6 w-6 text-violet-400" strokeWidth={2} />
            </div>
            <div className="flex-1">
              <h3 className="text-[15px] font-semibold text-zinc-100 mb-1">
                Progress Report
              </h3>
              <p className="text-[13px] text-zinc-400 leading-relaxed">
                Track changes over time with detailed progress analysis and
                longitudinal trend visualization.
              </p>
            </div>
          </div>
          <button
            disabled
            className="w-full rounded-lg bg-zinc-800 py-2.5 text-[13px] font-medium text-zinc-500 cursor-not-allowed border border-zinc-700"
          >
            Coming Soon
          </button>
        </div>
      </div>
    </motion.div>
  );
}
