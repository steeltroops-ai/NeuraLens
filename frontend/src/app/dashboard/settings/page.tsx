'use client';

/**
 * Settings Page - Neurologist Focus
 * 
 * Professional settings for neurologists with clinical preferences
 * and diagnostic tool configurations.
 */

import { motion } from 'framer-motion';
import {
    Settings,
    Shield,
    Brain,
    Stethoscope,
    FileText,
    Clock,
    AlertTriangle,
    LogOut
} from 'lucide-react';

/**
 * Settings Page Component
 */
export default function SettingsPage() {
    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
            className="space-y-6"
        >
            {/* Header */}
            <div className="relative overflow-hidden bg-white rounded-2xl border border-zinc-200/80 p-8">
                <div className="absolute inset-0 bg-gradient-to-br from-zinc-50/60 via-transparent to-slate-50/40 pointer-events-none" />

                <div className="relative">
                    <div className="flex items-start gap-4">
                        <div className="p-3 rounded-xl bg-gradient-to-br from-zinc-700 to-zinc-900 shadow-lg shadow-zinc-500/20">
                            <Settings className="h-7 w-7 text-white" strokeWidth={2} />
                        </div>
                        <div className="flex-1">
                            <h1 className="text-[24px] font-semibold text-zinc-900 mb-2">Clinical Settings</h1>
                            <p className="text-[14px] text-zinc-600 max-w-xl">
                                Configure your neurological practice preferences and diagnostic tool settings
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Settings Sections */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Main Settings */}
                <div className="lg:col-span-2 space-y-4">
                    {/* Professional Profile */}
                    <div className="group relative overflow-hidden rounded-2xl border border-zinc-200/80 bg-white p-6 hover:shadow-lg hover:shadow-blue-500/5 transition-all duration-300">
                        <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-blue-500/10 to-transparent rounded-full -mr-16 -mt-16" />
                        <div className="relative">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="p-2.5 rounded-xl bg-blue-50 group-hover:bg-blue-100 transition-colors">
                                    <Stethoscope className="h-5 w-5 text-blue-600" strokeWidth={2} />
                                </div>
                                <h2 className="text-[17px] font-semibold text-zinc-900">Professional Profile</h2>
                            </div>
                            <p className="text-[13px] text-zinc-600 mb-5 leading-relaxed">
                                Manage your medical credentials and practice information.
                            </p>
                            <div className="space-y-3">
                                <div className="flex items-center justify-between py-3.5 border-b border-zinc-100">
                                    <span className="text-[13px] font-medium text-zinc-700">Medical License</span>
                                    <span className="text-[13px] text-green-600 font-medium">MD-NY-2019-4567</span>
                                </div>
                                <div className="flex items-center justify-between py-3.5 border-b border-zinc-100">
                                    <span className="text-[13px] font-medium text-zinc-700">Specialty</span>
                                    <span className="text-[13px] text-zinc-900 font-medium">Neurology</span>
                                </div>
                                <div className="flex items-center justify-between py-3.5 border-b border-zinc-100">
                                    <span className="text-[13px] font-medium text-zinc-700">Institution</span>
                                    <span className="text-[13px] text-zinc-500">Mount Sinai Hospital</span>
                                </div>
                                <div className="flex items-center justify-between py-3.5">
                                    <span className="text-[13px] font-medium text-zinc-700">Years of Practice</span>
                                    <span className="text-[13px] text-zinc-500">12 years</span>
                                </div>
                            </div>
                            <button
                                disabled
                                className="mt-5 w-full rounded-lg bg-zinc-100 py-2.5 text-[13px] font-medium text-zinc-400 cursor-not-allowed"
                            >
                                Update Credentials (Coming Soon)
                            </button>
                        </div>
                    </div>

                    {/* Diagnostic Preferences */}
                    <div className="group relative overflow-hidden rounded-2xl border border-zinc-200/80 bg-white p-6 hover:shadow-lg hover:shadow-purple-500/5 transition-all duration-300">
                        <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-purple-500/10 to-transparent rounded-full -mr-16 -mt-16" />
                        <div className="relative">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="p-2.5 rounded-xl bg-purple-50 group-hover:bg-purple-100 transition-colors">
                                    <Brain className="h-5 w-5 text-purple-600" strokeWidth={2} />
                                </div>
                                <h2 className="text-[17px] font-semibold text-zinc-900">Diagnostic Preferences</h2>
                            </div>
                            <p className="text-[13px] text-zinc-600 mb-5 leading-relaxed">
                                Configure your preferred diagnostic thresholds and assessment protocols.
                            </p>
                            <div className="space-y-3">
                                <div className="flex items-center justify-between py-3.5 border-b border-zinc-100">
                                    <span className="text-[13px] font-medium text-zinc-700">Risk Score Threshold</span>
                                    <span className="text-[13px] text-zinc-900 font-medium">75% (High Sensitivity)</span>
                                </div>
                                <div className="flex items-center justify-between py-3.5 border-b border-zinc-100">
                                    <span className="text-[13px] font-medium text-zinc-700">Auto-Generate Reports</span>
                                    <div className="h-6 w-11 rounded-full bg-blue-500 relative">
                                        <div className="absolute right-1 top-1 h-4 w-4 rounded-full bg-white transition-transform" />
                                    </div>
                                </div>
                                <div className="flex items-center justify-between py-3.5 border-b border-zinc-100">
                                    <span className="text-[13px] font-medium text-zinc-700">Critical Alert Notifications</span>
                                    <div className="h-6 w-11 rounded-full bg-blue-500 relative">
                                        <div className="absolute right-1 top-1 h-4 w-4 rounded-full bg-white transition-transform" />
                                    </div>
                                </div>
                                <div className="flex items-center justify-between py-3.5">
                                    <span className="text-[13px] font-medium text-zinc-700">Default Assessment Duration</span>
                                    <span className="text-[13px] text-zinc-500">15 minutes</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Clinical Alerts */}
                    <div className="group relative overflow-hidden rounded-2xl border border-zinc-200/80 bg-white p-6 hover:shadow-lg hover:shadow-red-500/5 transition-all duration-300">
                        <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-red-500/10 to-transparent rounded-full -mr-16 -mt-16" />
                        <div className="relative">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="p-2.5 rounded-xl bg-red-50 group-hover:bg-red-100 transition-colors">
                                    <AlertTriangle className="h-5 w-5 text-red-600" strokeWidth={2} />
                                </div>
                                <h2 className="text-[17px] font-semibold text-zinc-900">Clinical Alerts</h2>
                            </div>
                            <p className="text-[13px] text-zinc-600 mb-5 leading-relaxed">
                                Configure when and how you receive critical patient alerts.
                            </p>
                            <div className="space-y-3">
                                <div className="flex items-center justify-between py-3.5 border-b border-zinc-100">
                                    <span className="text-[13px] font-medium text-zinc-700">Stroke Risk Alerts</span>
                                    <div className="h-6 w-11 rounded-full bg-blue-500 relative">
                                        <div className="absolute right-1 top-1 h-4 w-4 rounded-full bg-white transition-transform" />
                                    </div>
                                </div>
                                <div className="flex items-center justify-between py-3.5 border-b border-zinc-100">
                                    <span className="text-[13px] font-medium text-zinc-700">Parkinson's Progression</span>
                                    <div className="h-6 w-11 rounded-full bg-blue-500 relative">
                                        <div className="absolute right-1 top-1 h-4 w-4 rounded-full bg-white transition-transform" />
                                    </div>
                                </div>
                                <div className="flex items-center justify-between py-3.5 border-b border-zinc-100">
                                    <span className="text-[13px] font-medium text-zinc-700">Cognitive Decline</span>
                                    <div className="h-6 w-11 rounded-full bg-blue-500 relative">
                                        <div className="absolute right-1 top-1 h-4 w-4 rounded-full bg-white transition-transform" />
                                    </div>
                                </div>
                                <div className="flex items-center justify-between py-3.5">
                                    <span className="text-[13px] font-medium text-zinc-700">Emergency Referrals</span>
                                    <div className="h-6 w-11 rounded-full bg-blue-500 relative">
                                        <div className="absolute right-1 top-1 h-4 w-4 rounded-full bg-white transition-transform" />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Sidebar Settings */}
                <div className="space-y-4">
                    {/* Report Templates */}
                    <div className="group relative overflow-hidden rounded-2xl border border-zinc-200/80 bg-white p-6 hover:shadow-lg hover:shadow-green-500/5 transition-all duration-300">
                        <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-green-500/10 to-transparent rounded-full -mr-16 -mt-16" />
                        <div className="relative">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="p-2.5 rounded-xl bg-green-50 group-hover:bg-green-100 transition-colors">
                                    <FileText className="h-5 w-5 text-green-600" strokeWidth={2} />
                                </div>
                                <h2 className="text-[17px] font-semibold text-zinc-900">Report Templates</h2>
                            </div>
                            <div className="space-y-3">
                                <div className="flex items-center justify-between py-3">
                                    <span className="text-[13px] font-medium text-zinc-700">Default Template</span>
                                    <span className="text-[13px] text-zinc-500">Neurological Assessment</span>
                                </div>
                                <div className="flex items-center justify-between py-3">
                                    <span className="text-[13px] font-medium text-zinc-700">Include Recommendations</span>
                                    <div className="h-6 w-11 rounded-full bg-blue-500 relative">
                                        <div className="absolute right-1 top-1 h-4 w-4 rounded-full bg-white transition-transform" />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Schedule Settings */}
                    <div className="group relative overflow-hidden rounded-2xl border border-zinc-200/80 bg-white p-6 hover:shadow-lg hover:shadow-orange-500/5 transition-all duration-300">
                        <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-orange-500/10 to-transparent rounded-full -mr-16 -mt-16" />
                        <div className="relative">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="p-2.5 rounded-xl bg-orange-50 group-hover:bg-orange-100 transition-colors">
                                    <Clock className="h-5 w-5 text-orange-600" strokeWidth={2} />
                                </div>
                                <h2 className="text-[17px] font-semibold text-zinc-900">Schedule</h2>
                            </div>
                            <div className="space-y-3">
                                <div className="flex items-center justify-between py-3">
                                    <span className="text-[13px] font-medium text-zinc-700">Working Hours</span>
                                    <span className="text-[13px] text-zinc-500">9 AM - 5 PM</span>
                                </div>
                                <div className="flex items-center justify-between py-3">
                                    <span className="text-[13px] font-medium text-zinc-700">Time Zone</span>
                                    <span className="text-[13px] text-zinc-500">EST</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Privacy & Security */}
                    <div className="group relative overflow-hidden rounded-2xl border border-zinc-200/80 bg-white p-6 hover:shadow-lg hover:shadow-indigo-500/5 transition-all duration-300">
                        <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-indigo-500/10 to-transparent rounded-full -mr-16 -mt-16" />
                        <div className="relative">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="p-2.5 rounded-xl bg-indigo-50 group-hover:bg-indigo-100 transition-colors">
                                    <Shield className="h-5 w-5 text-indigo-600" strokeWidth={2} />
                                </div>
                                <h2 className="text-[17px] font-semibold text-zinc-900">Security</h2>
                            </div>
                            <div className="space-y-2">
                                <div className="flex items-center justify-between py-2.5">
                                    <span className="text-[13px] font-medium text-zinc-700">HIPAA Compliance</span>
                                    <span className="text-[11px] text-green-600 bg-green-50 px-2.5 py-1 rounded-full font-medium">Active</span>
                                </div>
                                <div className="flex items-center justify-between py-2.5">
                                    <span className="text-[13px] font-medium text-zinc-700">Data Encryption</span>
                                    <span className="text-[11px] text-green-600 bg-green-50 px-2.5 py-1 rounded-full font-medium">AES-256</span>
                                </div>
                                <div className="flex items-center justify-between py-2.5">
                                    <span className="text-[13px] font-medium text-zinc-700">Audit Logging</span>
                                    <span className="text-[11px] text-green-600 bg-green-50 px-2.5 py-1 rounded-full font-medium">Enabled</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Sign Out */}
                    <div className="rounded-2xl border border-red-200/80 bg-gradient-to-br from-red-50/50 to-red-50/30 p-6">
                        <button
                            disabled
                            className="w-full flex items-center justify-center gap-2 text-red-500 cursor-not-allowed"
                        >
                            <LogOut className="h-5 w-5" strokeWidth={2} />
                            <span className="text-[13px] font-semibold">Sign Out (Coming Soon)</span>
                        </button>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}
