'use client';

/**
 * Reports Page
 * 
 * Placeholder page for reports functionality.
 * 
 * Requirements: 4.1
 */

import { motion } from 'framer-motion';
import { FileText, Download, Share2, Printer, Clock, CheckCircle } from 'lucide-react';

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
            <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                <div className="flex items-center space-x-3 mb-4">
                    <div className="rounded-lg bg-gradient-to-r from-emerald-500 to-emerald-600 p-3">
                        <FileText className="h-6 w-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-gray-900">Reports</h1>
                        <p className="text-gray-600">
                            Generate and manage clinical assessment reports
                        </p>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div className="flex items-center space-x-2 text-gray-600">
                        <Download className="h-4 w-4" />
                        <span>PDF Export</span>
                    </div>
                    <div className="flex items-center space-x-2 text-gray-600">
                        <Share2 className="h-4 w-4" />
                        <span>Share with Providers</span>
                    </div>
                    <div className="flex items-center space-x-2 text-gray-600">
                        <Printer className="h-4 w-4" />
                        <span>Print Ready</span>
                    </div>
                </div>
            </div>

            {/* Coming Soon Content */}
            <div className="rounded-xl border border-emerald-200 bg-gradient-to-r from-emerald-50 to-emerald-100 p-8">
                <div className="text-center">
                    <FileText className="mx-auto mb-4 h-16 w-16 text-emerald-600" />
                    <h2 className="mb-2 text-xl font-bold text-emerald-900">
                        Reports Center Coming Soon
                    </h2>
                    <p className="mb-6 text-emerald-700 max-w-2xl mx-auto">
                        Generate comprehensive clinical reports from your assessment data. Export to PDF,
                        share with healthcare providers, and maintain a complete record of your
                        neurological health assessments.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm max-w-3xl mx-auto">
                        <div className="rounded-lg bg-white/50 p-4">
                            <Download className="mx-auto mb-2 h-8 w-8 text-emerald-600" />
                            <h3 className="mb-1 font-medium text-emerald-900">PDF Export</h3>
                            <p className="text-emerald-700 text-xs">
                                Download clinical-grade PDF reports
                            </p>
                        </div>
                        <div className="rounded-lg bg-white/50 p-4">
                            <Share2 className="mx-auto mb-2 h-8 w-8 text-emerald-600" />
                            <h3 className="mb-1 font-medium text-emerald-900">Provider Sharing</h3>
                            <p className="text-emerald-700 text-xs">
                                Securely share with healthcare providers
                            </p>
                        </div>
                        <div className="rounded-lg bg-white/50 p-4">
                            <CheckCircle className="mx-auto mb-2 h-8 w-8 text-emerald-600" />
                            <h3 className="mb-1 font-medium text-emerald-900">HIPAA Compliant</h3>
                            <p className="text-emerald-700 text-xs">
                                Secure, compliant data handling
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Recent Reports Placeholder */}
            <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Recent Reports</h2>

                <div className="space-y-3">
                    {/* Empty state */}
                    <div className="text-center py-8">
                        <FileText className="mx-auto mb-3 h-12 w-12 text-gray-300" />
                        <p className="text-gray-500 mb-2">No reports generated yet</p>
                        <p className="text-sm text-gray-400">
                            Complete an assessment to generate your first report
                        </p>
                    </div>
                </div>
            </div>

            {/* Report Types */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                    <div className="flex items-center space-x-3 mb-3">
                        <div className="rounded-lg bg-blue-100 p-2">
                            <FileText className="h-5 w-5 text-blue-600" />
                        </div>
                        <h3 className="font-medium text-gray-900">Assessment Summary</h3>
                    </div>
                    <p className="text-sm text-gray-600 mb-4">
                        Comprehensive summary of individual assessment results with biomarkers and recommendations.
                    </p>
                    <button
                        disabled
                        className="w-full rounded-lg bg-gray-100 py-2 text-sm font-medium text-gray-400 cursor-not-allowed"
                    >
                        Coming Soon
                    </button>
                </div>

                <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                    <div className="flex items-center space-x-3 mb-3">
                        <div className="rounded-lg bg-purple-100 p-2">
                            <Clock className="h-5 w-5 text-purple-600" />
                        </div>
                        <h3 className="font-medium text-gray-900">Progress Report</h3>
                    </div>
                    <p className="text-sm text-gray-600 mb-4">
                        Track changes over time with detailed progress analysis and trend visualization.
                    </p>
                    <button
                        disabled
                        className="w-full rounded-lg bg-gray-100 py-2 text-sm font-medium text-gray-400 cursor-not-allowed"
                    >
                        Coming Soon
                    </button>
                </div>
            </div>
        </motion.div>
    );
}
