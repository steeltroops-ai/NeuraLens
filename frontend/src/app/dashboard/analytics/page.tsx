'use client';

/**
 * Analytics Page
 * 
 * Placeholder page for analytics functionality.
 * 
 * Requirements: 4.1
 */

import { motion } from 'framer-motion';
import { BarChart3, TrendingUp, Activity, Calendar, Clock, Users } from 'lucide-react';

/**
 * Analytics Page Component
 */
export default function AnalyticsPage() {
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
                    <div className="rounded-lg bg-gradient-to-r from-blue-500 to-blue-600 p-3">
                        <BarChart3 className="h-6 w-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
                        <p className="text-gray-600">
                            Track assessment trends and health insights over time
                        </p>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div className="flex items-center space-x-2 text-gray-600">
                        <TrendingUp className="h-4 w-4" />
                        <span>Trend Analysis</span>
                    </div>
                    <div className="flex items-center space-x-2 text-gray-600">
                        <Activity className="h-4 w-4" />
                        <span>Performance Metrics</span>
                    </div>
                    <div className="flex items-center space-x-2 text-gray-600">
                        <Calendar className="h-4 w-4" />
                        <span>Historical Data</span>
                    </div>
                </div>
            </div>

            {/* Coming Soon Content */}
            <div className="rounded-xl border border-blue-200 bg-gradient-to-r from-blue-50 to-blue-100 p-8">
                <div className="text-center">
                    <BarChart3 className="mx-auto mb-4 h-16 w-16 text-blue-600" />
                    <h2 className="mb-2 text-xl font-bold text-blue-900">
                        Analytics Dashboard Coming Soon
                    </h2>
                    <p className="mb-6 text-blue-700 max-w-2xl mx-auto">
                        Comprehensive analytics dashboard for tracking assessment trends, health insights,
                        and performance metrics over time. Visualize your neurological health journey with
                        interactive charts and detailed reports.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm max-w-3xl mx-auto">
                        <div className="rounded-lg bg-white/50 p-4">
                            <TrendingUp className="mx-auto mb-2 h-8 w-8 text-blue-600" />
                            <h3 className="mb-1 font-medium text-blue-900">Trend Analysis</h3>
                            <p className="text-blue-700 text-xs">
                                Track NRI scores and biomarkers over time
                            </p>
                        </div>
                        <div className="rounded-lg bg-white/50 p-4">
                            <Users className="mx-auto mb-2 h-8 w-8 text-blue-600" />
                            <h3 className="mb-1 font-medium text-blue-900">Comparative Insights</h3>
                            <p className="text-blue-700 text-xs">
                                Compare results across assessment sessions
                            </p>
                        </div>
                        <div className="rounded-lg bg-white/50 p-4">
                            <Clock className="mx-auto mb-2 h-8 w-8 text-blue-600" />
                            <h3 className="mb-1 font-medium text-blue-900">Progress Tracking</h3>
                            <p className="text-blue-700 text-xs">
                                Monitor improvement and health milestones
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Placeholder Stats */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-600">Total Assessments</span>
                        <Activity className="h-5 w-5 text-blue-500" />
                    </div>
                    <div className="text-2xl font-bold text-gray-900">--</div>
                    <p className="text-xs text-gray-500 mt-1">Data available soon</p>
                </div>

                <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-600">Average NRI Score</span>
                        <TrendingUp className="h-5 w-5 text-green-500" />
                    </div>
                    <div className="text-2xl font-bold text-gray-900">--</div>
                    <p className="text-xs text-gray-500 mt-1">Data available soon</p>
                </div>

                <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-600">This Month</span>
                        <Calendar className="h-5 w-5 text-purple-500" />
                    </div>
                    <div className="text-2xl font-bold text-gray-900">--</div>
                    <p className="text-xs text-gray-500 mt-1">Data available soon</p>
                </div>

                <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-600">Improvement</span>
                        <BarChart3 className="h-5 w-5 text-orange-500" />
                    </div>
                    <div className="text-2xl font-bold text-gray-900">--</div>
                    <p className="text-xs text-gray-500 mt-1">Data available soon</p>
                </div>
            </div>
        </motion.div>
    );
}
