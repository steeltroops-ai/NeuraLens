'use client';

/**
 * Settings Page
 * 
 * Placeholder page for settings functionality.
 * 
 * Requirements: 4.1
 */

import { motion } from 'framer-motion';
import { Settings, User, Bell, Shield, Palette, Globe, HelpCircle, LogOut } from 'lucide-react';

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
            <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                <div className="flex items-center space-x-3 mb-4">
                    <div className="rounded-lg bg-gradient-to-r from-gray-500 to-gray-600 p-3">
                        <Settings className="h-6 w-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
                        <p className="text-gray-600">
                            Manage your account and application preferences
                        </p>
                    </div>
                </div>
            </div>

            {/* Settings Sections */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Main Settings */}
                <div className="lg:col-span-2 space-y-4">
                    {/* Profile Settings */}
                    <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                        <div className="flex items-center space-x-3 mb-4">
                            <div className="rounded-lg bg-blue-100 p-2">
                                <User className="h-5 w-5 text-blue-600" />
                            </div>
                            <h2 className="text-lg font-semibold text-gray-900">Profile</h2>
                        </div>
                        <p className="text-sm text-gray-600 mb-4">
                            Manage your personal information and account details.
                        </p>
                        <div className="space-y-3">
                            <div className="flex items-center justify-between py-2 border-b border-gray-100">
                                <span className="text-sm text-gray-700">Display Name</span>
                                <span className="text-sm text-gray-400">Not set</span>
                            </div>
                            <div className="flex items-center justify-between py-2 border-b border-gray-100">
                                <span className="text-sm text-gray-700">Email</span>
                                <span className="text-sm text-gray-400">Not set</span>
                            </div>
                            <div className="flex items-center justify-between py-2">
                                <span className="text-sm text-gray-700">Date of Birth</span>
                                <span className="text-sm text-gray-400">Not set</span>
                            </div>
                        </div>
                        <button
                            disabled
                            className="mt-4 w-full rounded-lg bg-gray-100 py-2 text-sm font-medium text-gray-400 cursor-not-allowed"
                        >
                            Edit Profile (Coming Soon)
                        </button>
                    </div>

                    {/* Notification Settings */}
                    <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                        <div className="flex items-center space-x-3 mb-4">
                            <div className="rounded-lg bg-purple-100 p-2">
                                <Bell className="h-5 w-5 text-purple-600" />
                            </div>
                            <h2 className="text-lg font-semibold text-gray-900">Notifications</h2>
                        </div>
                        <p className="text-sm text-gray-600 mb-4">
                            Configure how you receive updates and alerts.
                        </p>
                        <div className="space-y-3">
                            <div className="flex items-center justify-between py-2 border-b border-gray-100">
                                <span className="text-sm text-gray-700">Assessment Reminders</span>
                                <div className="h-6 w-11 rounded-full bg-gray-200 cursor-not-allowed" />
                            </div>
                            <div className="flex items-center justify-between py-2 border-b border-gray-100">
                                <span className="text-sm text-gray-700">Results Ready</span>
                                <div className="h-6 w-11 rounded-full bg-gray-200 cursor-not-allowed" />
                            </div>
                            <div className="flex items-center justify-between py-2">
                                <span className="text-sm text-gray-700">Health Insights</span>
                                <div className="h-6 w-11 rounded-full bg-gray-200 cursor-not-allowed" />
                            </div>
                        </div>
                    </div>

                    {/* Privacy & Security */}
                    <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                        <div className="flex items-center space-x-3 mb-4">
                            <div className="rounded-lg bg-green-100 p-2">
                                <Shield className="h-5 w-5 text-green-600" />
                            </div>
                            <h2 className="text-lg font-semibold text-gray-900">Privacy & Security</h2>
                        </div>
                        <p className="text-sm text-gray-600 mb-4">
                            Manage your data privacy and security settings.
                        </p>
                        <div className="space-y-3">
                            <div className="flex items-center justify-between py-2 border-b border-gray-100">
                                <span className="text-sm text-gray-700">Two-Factor Authentication</span>
                                <span className="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded">Coming Soon</span>
                            </div>
                            <div className="flex items-center justify-between py-2 border-b border-gray-100">
                                <span className="text-sm text-gray-700">Data Export</span>
                                <span className="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded">Coming Soon</span>
                            </div>
                            <div className="flex items-center justify-between py-2">
                                <span className="text-sm text-gray-700">Delete Account</span>
                                <span className="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded">Coming Soon</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Sidebar Settings */}
                <div className="space-y-4">
                    {/* Appearance */}
                    <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                        <div className="flex items-center space-x-3 mb-4">
                            <div className="rounded-lg bg-orange-100 p-2">
                                <Palette className="h-5 w-5 text-orange-600" />
                            </div>
                            <h2 className="text-lg font-semibold text-gray-900">Appearance</h2>
                        </div>
                        <div className="space-y-3">
                            <div className="flex items-center justify-between py-2">
                                <span className="text-sm text-gray-700">Theme</span>
                                <span className="text-sm text-gray-400">Light</span>
                            </div>
                        </div>
                    </div>

                    {/* Language */}
                    <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                        <div className="flex items-center space-x-3 mb-4">
                            <div className="rounded-lg bg-cyan-100 p-2">
                                <Globe className="h-5 w-5 text-cyan-600" />
                            </div>
                            <h2 className="text-lg font-semibold text-gray-900">Language</h2>
                        </div>
                        <div className="space-y-3">
                            <div className="flex items-center justify-between py-2">
                                <span className="text-sm text-gray-700">Display Language</span>
                                <span className="text-sm text-gray-400">English</span>
                            </div>
                        </div>
                    </div>

                    {/* Help & Support */}
                    <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                        <div className="flex items-center space-x-3 mb-4">
                            <div className="rounded-lg bg-indigo-100 p-2">
                                <HelpCircle className="h-5 w-5 text-indigo-600" />
                            </div>
                            <h2 className="text-lg font-semibold text-gray-900">Help & Support</h2>
                        </div>
                        <div className="space-y-2">
                            <button
                                disabled
                                className="w-full text-left text-sm text-gray-400 py-2 cursor-not-allowed"
                            >
                                Documentation
                            </button>
                            <button
                                disabled
                                className="w-full text-left text-sm text-gray-400 py-2 cursor-not-allowed"
                            >
                                Contact Support
                            </button>
                            <button
                                disabled
                                className="w-full text-left text-sm text-gray-400 py-2 cursor-not-allowed"
                            >
                                Privacy Policy
                            </button>
                        </div>
                    </div>

                    {/* Sign Out */}
                    <div className="rounded-xl border border-red-200 bg-red-50 p-6">
                        <button
                            disabled
                            className="w-full flex items-center justify-center space-x-2 text-red-400 cursor-not-allowed"
                        >
                            <LogOut className="h-5 w-5" />
                            <span className="font-medium">Sign Out (Coming Soon)</span>
                        </button>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}
