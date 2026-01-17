'use client';

/**
 * Reports Page - Neurologist Focus
 * 
 * Professional clinical reports dashboard with realistic neurological data
 * for hackathon demonstration.
 */

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FileText, Download, Share2, Printer, Clock, CheckCircle, Eye, Brain, Activity } from 'lucide-react';

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
            {/* Header - Enhanced with document theme */}
            <div className="relative overflow-hidden bg-white rounded-2xl border border-zinc-200/80 p-8">
                {/* Gradient background */}
                <div className="absolute inset-0 bg-gradient-to-br from-emerald-50/40 via-transparent to-teal-50/30 pointer-events-none" />

                <div className="relative">
                    <div className="flex items-start justify-between">
                        <div className="flex items-start gap-4">
                            <div className="p-3 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 shadow-lg shadow-emerald-500/20">
                                <FileText className="h-7 w-7 text-white" strokeWidth={2} />
                            </div>
                            <div>
                                <div className="flex items-center gap-3 mb-2">
                                    <h1 className="text-[24px] font-semibold text-zinc-900">Clinical Reports</h1>
                                    <span className="px-2.5 py-1 bg-emerald-100 text-emerald-700 text-[11px] font-medium rounded-full">
                                        HIPAA Compliant
                                    </span>
                                </div>
                                <p className="text-[14px] text-zinc-600 max-w-xl">
                                    Generate, export, and share professional medical assessment reports
                                </p>
                            </div>
                        </div>

                        <button className="px-4 py-2 text-[13px] font-medium text-zinc-700 hover:text-zinc-900 hover:bg-zinc-50 rounded-lg transition-colors">
                            View Archive
                        </button>
                    </div>

                    {/* Feature pills */}
                    <div className="flex flex-wrap gap-2 mt-6">
                        <div className="flex items-center gap-2 px-3 py-2 bg-white border border-zinc-200 rounded-lg">
                            <Download className="h-4 w-4 text-emerald-600" strokeWidth={2} />
                            <span className="text-[12px] font-medium text-zinc-700">PDF Export</span>
                        </div>
                        <div className="flex items-center gap-2 px-3 py-2 bg-white border border-zinc-200 rounded-lg">
                            <Share2 className="h-4 w-4 text-blue-600" strokeWidth={2} />
                            <span className="text-[12px] font-medium text-zinc-700">Provider Sharing</span>
                        </div>
                        <div className="flex items-center gap-2 px-3 py-2 bg-white border border-zinc-200 rounded-lg">
                            <Printer className="h-4 w-4 text-purple-600" strokeWidth={2} />
                            <span className="text-[12px] font-medium text-zinc-700">Print Ready</span>
                        </div>
                        <div className="flex items-center gap-2 px-3 py-2 bg-white border border-zinc-200 rounded-lg">
                            <CheckCircle className="h-4 w-4 text-green-600" strokeWidth={2} />
                            <span className="text-[12px] font-medium text-zinc-700">Secure Storage</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Coming Soon Content */}
            <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-8">
                <div className="text-center">
                    <div className="inline-flex p-4 rounded-full bg-emerald-100 mb-4">
                        <FileText className="h-12 w-12 text-emerald-600" strokeWidth={1.5} />
                    </div>
                    <h2 className="mb-2 text-[18px] font-semibold text-emerald-900">
                        Reports Center Coming Soon
                    </h2>
                    <p className="mb-6 text-[13px] text-emerald-700 max-w-2xl mx-auto leading-relaxed">
                        Generate comprehensive clinical reports from your assessment data. Export to PDF,
                        share with healthcare providers, and maintain a complete record of your
                        neurological health assessments.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl mx-auto">
                        <div className="rounded-lg bg-white border border-emerald-100 p-4">
                            <div className="inline-flex p-2 rounded-lg bg-emerald-50 mb-3">
                                <Download className="h-6 w-6 text-emerald-600" strokeWidth={1.5} />
                            </div>
                            <h3 className="mb-1 text-[14px] font-medium text-emerald-900">PDF Export</h3>
                            <p className="text-[12px] text-emerald-700 leading-relaxed">
                                Download clinical-grade PDF reports
                            </p>
                        </div>
                        <div className="rounded-lg bg-white border border-emerald-100 p-4">
                            <div className="inline-flex p-2 rounded-lg bg-emerald-50 mb-3">
                                <Share2 className="h-6 w-6 text-emerald-600" strokeWidth={1.5} />
                            </div>
                            <h3 className="mb-1 text-[14px] font-medium text-emerald-900">Provider Sharing</h3>
                            <p className="text-[12px] text-emerald-700 leading-relaxed">
                                Securely share with healthcare providers
                            </p>
                        </div>
                        <div className="rounded-lg bg-white border border-emerald-100 p-4">
                            <div className="inline-flex p-2 rounded-lg bg-emerald-50 mb-3">
                                <CheckCircle className="h-6 w-6 text-emerald-600" strokeWidth={1.5} />
                            </div>
                            <h3 className="mb-1 text-[14px] font-medium text-emerald-900">HIPAA Compliant</h3>
                            <p className="text-[12px] text-emerald-700 leading-relaxed">
                                Secure, compliant data handling
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Recent Reports Placeholder - Enhanced */}
            <div className="bg-white rounded-2xl border border-zinc-200/80 p-6">
                <div className="flex items-center justify-between mb-5">
                    <h2 className="text-[18px] font-semibold text-zinc-900">Recent Reports</h2>
                    <button className="px-3 py-1.5 text-[12px] font-medium text-zinc-600 hover:text-zinc-900 hover:bg-zinc-50 rounded-lg transition-colors">
                        View All
                    </button>
                </div>

                <div className="space-y-3">
                    {/* Empty state */}
                    <div className="text-center py-16">
                        <div className="inline-flex p-4 rounded-2xl bg-gradient-to-br from-zinc-50 to-zinc-100 mb-4">
                            <FileText className="h-10 w-10 text-zinc-400" strokeWidth={1.5} />
                        </div>
                        <p className="text-[14px] font-medium text-zinc-900 mb-1">No reports generated yet</p>
                        <p className="text-[13px] text-zinc-500">
                            Complete an assessment to generate your first clinical report
                        </p>
                    </div>
                </div>
            </div>

            {/* Report Types - Enhanced with better visual design */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="group relative overflow-hidden rounded-2xl border border-zinc-200/80 bg-white p-6 hover:shadow-lg hover:shadow-blue-500/5 transition-all duration-300">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-blue-500/10 to-transparent rounded-full -mr-16 -mt-16" />
                    <div className="relative">
                        <div className="flex items-start gap-3 mb-4">
                            <div className="p-2.5 rounded-xl bg-blue-50 group-hover:bg-blue-100 transition-colors">
                                <FileText className="h-6 w-6 text-blue-600" strokeWidth={2} />
                            </div>
                            <div className="flex-1">
                                <h3 className="text-[15px] font-semibold text-zinc-900 mb-1">Assessment Summary</h3>
                                <p className="text-[13px] text-zinc-600 leading-relaxed">
                                    Comprehensive summary of individual assessment results with biomarkers and clinical recommendations.
                                </p>
                            </div>
                        </div>
                        <button
                            disabled
                            className="w-full rounded-lg bg-zinc-100 py-2.5 text-[13px] font-medium text-zinc-400 cursor-not-allowed"
                        >
                            Coming Soon
                        </button>
                    </div>
                </div>

                <div className="group relative overflow-hidden rounded-2xl border border-zinc-200/80 bg-white p-6 hover:shadow-lg hover:shadow-purple-500/5 transition-all duration-300">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-purple-500/10 to-transparent rounded-full -mr-16 -mt-16" />
                    <div className="relative">
                        <div className="flex items-start gap-3 mb-4">
                            <div className="p-2.5 rounded-xl bg-purple-50 group-hover:bg-purple-100 transition-colors">
                                <Clock className="h-6 w-6 text-purple-600" strokeWidth={2} />
                            </div>
                            <div className="flex-1">
                                <h3 className="text-[15px] font-semibold text-zinc-900 mb-1">Progress Report</h3>
                                <p className="text-[13px] text-zinc-600 leading-relaxed">
                                    Track changes over time with detailed progress analysis and longitudinal trend visualization.
                                </p>
                            </div>
                        </div>
                        <button
                            disabled
                            className="w-full rounded-lg bg-zinc-100 py-2.5 text-[13px] font-medium text-zinc-400 cursor-not-allowed"
                        >
                            Coming Soon
                        </button>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}
