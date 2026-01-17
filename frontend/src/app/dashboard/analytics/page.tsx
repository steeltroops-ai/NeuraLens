'use client';

/**
 * Analytics Page - Neurologist Focus
 * 
 * Professional analytics dashboard with realistic neurological data
 * for hackathon demonstration.
 */

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { BarChart3, TrendingUp, Activity, Calendar, Clock, Users, Brain, AlertTriangle } from 'lucide-react';

/**
 * Analytics Page Component
 */
export default function AnalyticsPage() {
    const [analyticsData, setAnalyticsData] = useState({
        totalAssessments: 0,
        averageNRIScore: 0,
        thisMonth: 0,
        improvement: 0
    });

    // Load realistic analytics data
    useEffect(() => {
        // Simulate realistic neurological analytics data
        setAnalyticsData({
            totalAssessments: 156,
            averageNRIScore: 72,
            thisMonth: 23,
            improvement: 15
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
            <div className="rounded-xl border border-zinc-200 bg-white p-6">
                <div className="flex items-start gap-4">
                    <div className="p-3 rounded-lg bg-blue-50">
                        <BarChart3 className="h-6 w-6 text-blue-600" strokeWidth={1.5} />
                    </div>
                    <div className="flex-1">
                        <h1 className="text-[20px] font-semibold text-zinc-900">Neurological Analytics</h1>
                        <p className="text-[13px] text-zinc-600 mt-1">
                            Track assessment trends and neurological health insights over time
                        </p>
                    </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4">
                    <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
                        <TrendingUp className="h-4 w-4 text-zinc-500" />
                        <span className="text-[13px] text-zinc-700">Trend Analysis</span>
                    </div>
                    <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
                        <Activity className="h-4 w-4 text-zinc-500" />
                        <span className="text-[13px] text-zinc-700">Performance Metrics</span>
                    </div>
                    <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
                        <Calendar className="h-4 w-4 text-zinc-500" />
                        <span className="text-[13px] text-zinc-700">Historical Data</span>
                    </div>
                </div>
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="rounded-xl border border-zinc-200 bg-white p-5">
                    <div className="flex items-center justify-between mb-3">
                        <span className="text-[13px] font-medium text-zinc-700">Total Assessments</span>
                        <div className="p-2 rounded-lg bg-blue-50">
                            <Activity className="h-4 w-4 text-blue-600" strokeWidth={1.5} />
                        </div>
                    </div>
                    <div className="text-[24px] font-semibold text-zinc-900">{analyticsData.totalAssessments}</div>
                    <p className="text-[11px] text-green-600 mt-1">+12 this week</p>
                </div>

                <div className="rounded-xl border border-zinc-200 bg-white p-5">
                    <div className="flex items-center justify-between mb-3">
                        <span className="text-[13px] font-medium text-zinc-700">Average NRI Score</span>
                        <div className="p-2 rounded-lg bg-green-50">
                            <TrendingUp className="h-4 w-4 text-green-600" strokeWidth={1.5} />
                        </div>
                    </div>
                    <div className="text-[24px] font-semibold text-zinc-900">{analyticsData.averageNRIScore}%</div>
                    <p className="text-[11px] text-green-600 mt-1">+5% improvement</p>
                </div>

                <div className="rounded-xl border border-zinc-200 bg-white p-5">
                    <div className="flex items-center justify-between mb-3">
                        <span className="text-[13px] font-medium text-zinc-700">This Month</span>
                        <div className="p-2 rounded-lg bg-purple-50">
                            <Calendar className="h-4 w-4 text-purple-600" strokeWidth={1.5} />
                        </div>
                    </div>
                    <div className="text-[24px] font-semibold text-zinc-900">{analyticsData.thisMonth}</div>
                    <p className="text-[11px] text-zinc-500 mt-1">Assessments completed</p>
                </div>

                <div className="rounded-xl border border-zinc-200 bg-white p-5">
                    <div className="flex items-center justify-between mb-3">
                        <span className="text-[13px] font-medium text-zinc-700">Patient Improvement</span>
                        <div className="p-2 rounded-lg bg-orange-50">
                            <BarChart3 className="h-4 w-4 text-orange-600" strokeWidth={1.5} />
                        </div>
                    </div>
                    <div className="text-[24px] font-semibold text-zinc-900">{analyticsData.improvement}%</div>
                    <p className="text-[11px] text-green-600 mt-1">Above baseline</p>
                </div>
            </div>

            {/* Assessment Trends */}
            <div className="bg-white rounded-xl border border-zinc-200 p-6">
                <h2 className="text-[16px] font-semibold text-zinc-900 mb-4">Assessment Trends (Last 30 Days)</h2>
                <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 bg-zinc-50 rounded-lg">
                        <div className="flex items-center gap-3">
                            <div className="p-2 rounded-lg bg-cyan-50">
                                <Brain className="h-5 w-5 text-cyan-600" />
                            </div>
                            <div>
                                <p className="text-[14px] font-medium text-zinc-900">Retinal Scans</p>
                                <p className="text-[12px] text-zinc-600">Diabetic retinopathy screening</p>
                            </div>
                        </div>
                        <div className="text-right">
                            <p className="text-[16px] font-semibold text-zinc-900">8 assessments</p>
                            <p className="text-[11px] text-green-600">+2 from last month</p>
                        </div>
                    </div>

                    <div className="flex items-center justify-between p-4 bg-zinc-50 rounded-lg">
                        <div className="flex items-center gap-3">
                            <div className="p-2 rounded-lg bg-blue-50">
                                <Activity className="h-5 w-5 text-blue-600" />
                            </div>
                            <div>
                                <p className="text-[14px] font-medium text-zinc-900">Speech Analysis</p>
                                <p className="text-[12px] text-zinc-600">Parkinson's disease monitoring</p>
                            </div>
                        </div>
                        <div className="text-right">
                            <p className="text-[16px] font-semibold text-zinc-900">12 assessments</p>
                            <p className="text-[11px] text-green-600">+5 from last month</p>
                        </div>
                    </div>

                    <div className="flex items-center justify-between p-4 bg-zinc-50 rounded-lg">
                        <div className="flex items-center gap-3">
                            <div className="p-2 rounded-lg bg-red-50">
                                <AlertTriangle className="h-5 w-5 text-red-600" />
                            </div>
                            <div>
                                <p className="text-[14px] font-medium text-zinc-900">Cognitive Testing</p>
                                <p className="text-[12px] text-zinc-600">Dementia and MS screening</p>
                            </div>
                        </div>
                        <div className="text-right">
                            <p className="text-[16px] font-semibold text-zinc-900">3 assessments</p>
                            <p className="text-[11px] text-amber-600">Same as last month</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Risk Distribution */}
            <div className="bg-white rounded-xl border border-zinc-200 p-6">
                <h2 className="text-[16px] font-semibold text-zinc-900 mb-4">Risk Score Distribution</h2>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
                        <div className="text-[20px] font-semibold text-green-700">12</div>
                        <div className="text-[12px] text-green-600 font-medium">Low Risk (0-25%)</div>
                        <div className="text-[11px] text-green-500 mt-1">52% of patients</div>
                    </div>
                    <div className="text-center p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                        <div className="text-[20px] font-semibold text-yellow-700">7</div>
                        <div className="text-[12px] text-yellow-600 font-medium">Moderate (26-50%)</div>
                        <div className="text-[11px] text-yellow-500 mt-1">30% of patients</div>
                    </div>
                    <div className="text-center p-4 bg-orange-50 rounded-lg border border-orange-200">
                        <div className="text-[20px] font-semibold text-orange-700">3</div>
                        <div className="text-[12px] text-orange-600 font-medium">High (51-75%)</div>
                        <div className="text-[11px] text-orange-500 mt-1">13% of patients</div>
                    </div>
                    <div className="text-center p-4 bg-red-50 rounded-lg border border-red-200">
                        <div className="text-[20px] font-semibold text-red-700">1</div>
                        <div className="text-[12px] text-red-600 font-medium">Critical (76-100%)</div>
                        <div className="text-[11px] text-red-500 mt-1">4% of patients</div>
                    </div>
                </div>
            </div>

            {/* Coming Soon Features */}
            <div className="rounded-xl border border-blue-200 bg-blue-50 p-6">
                <div className="text-center">
                    <div className="inline-flex p-3 rounded-full bg-blue-100 mb-3">
                        <BarChart3 className="h-8 w-8 text-blue-600" strokeWidth={1.5} />
                    </div>
                    <h2 className="mb-2 text-[16px] font-semibold text-blue-900">
                        Advanced Analytics Coming Soon
                    </h2>
                    <p className="mb-4 text-[13px] text-blue-700 max-w-2xl mx-auto leading-relaxed">
                        Interactive charts, longitudinal tracking, and predictive analytics for neurological conditions.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 max-w-2xl mx-auto">
                        <div className="rounded-lg bg-white border border-blue-100 p-3">
                            <h3 className="mb-1 text-[13px] font-medium text-blue-900">Interactive Charts</h3>
                            <p className="text-[11px] text-blue-700">Visual trend analysis</p>
                        </div>
                        <div className="rounded-lg bg-white border border-blue-100 p-3">
                            <h3 className="mb-1 text-[13px] font-medium text-blue-900">Predictive Models</h3>
                            <p className="text-[11px] text-blue-700">Risk forecasting</p>
                        </div>
                        <div className="rounded-lg bg-white border border-blue-100 p-3">
                            <h3 className="mb-1 text-[13px] font-medium text-blue-900">Population Health</h3>
                            <p className="text-[11px] text-blue-700">Cohort comparisons</p>
                        </div>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}
