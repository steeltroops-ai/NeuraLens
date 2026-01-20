"use client";

/**
 * Settings Page - Dark Theme Enterprise Dashboard
 *
 * Professional settings for neurologists matching the dark theme design philosophy.
 */

import { motion } from "framer-motion";
import {
  Settings,
  Shield,
  Brain,
  Stethoscope,
  FileText,
  Clock,
  AlertTriangle,
  LogOut,
} from "lucide-react";

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
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-8">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-lg bg-zinc-700">
            <Settings className="h-7 w-7 text-zinc-200" strokeWidth={2} />
          </div>
          <div className="flex-1">
            <h1 className="text-[24px] font-semibold text-zinc-100 mb-2">
              Clinical Settings
            </h1>
            <p className="text-[14px] text-zinc-400 max-w-xl">
              Configure your neurological practice preferences and diagnostic
              tool settings
            </p>
          </div>
        </div>
      </div>

      {/* Settings Sections */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Settings */}
        <div className="lg:col-span-2 space-y-4">
          {/* Professional Profile */}
          <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 hover:border-zinc-700 transition-colors">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2.5 rounded-lg bg-blue-500/15">
                <Stethoscope
                  className="h-5 w-5 text-blue-400"
                  strokeWidth={2}
                />
              </div>
              <h2 className="text-[17px] font-semibold text-zinc-100">
                Professional Profile
              </h2>
            </div>
            <p className="text-[13px] text-zinc-400 mb-5 leading-relaxed">
              Manage your medical credentials and practice information.
            </p>
            <div className="space-y-3">
              <div className="flex items-center justify-between py-3.5 border-b border-zinc-800">
                <span className="text-[13px] font-medium text-zinc-300">
                  Medical License
                </span>
                <span className="text-[13px] text-emerald-400 font-medium">
                  MD-NY-2019-4567
                </span>
              </div>
              <div className="flex items-center justify-between py-3.5 border-b border-zinc-800">
                <span className="text-[13px] font-medium text-zinc-300">
                  Specialty
                </span>
                <span className="text-[13px] text-zinc-100 font-medium">
                  Neurology
                </span>
              </div>
              <div className="flex items-center justify-between py-3.5 border-b border-zinc-800">
                <span className="text-[13px] font-medium text-zinc-300">
                  Institution
                </span>
                <span className="text-[13px] text-zinc-400">
                  Mount Sinai Hospital
                </span>
              </div>
              <div className="flex items-center justify-between py-3.5">
                <span className="text-[13px] font-medium text-zinc-300">
                  Years of Practice
                </span>
                <span className="text-[13px] text-zinc-400">12 years</span>
              </div>
            </div>
            <button
              disabled
              className="mt-5 w-full rounded-lg bg-zinc-800 border border-zinc-700 py-2.5 text-[13px] font-medium text-zinc-500 cursor-not-allowed"
            >
              Update Credentials (Coming Soon)
            </button>
          </div>

          {/* Diagnostic Preferences */}
          <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 hover:border-zinc-700 transition-colors">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2.5 rounded-lg bg-violet-500/15">
                <Brain className="h-5 w-5 text-violet-400" strokeWidth={2} />
              </div>
              <h2 className="text-[17px] font-semibold text-zinc-100">
                Diagnostic Preferences
              </h2>
            </div>
            <p className="text-[13px] text-zinc-400 mb-5 leading-relaxed">
              Configure your preferred diagnostic thresholds and assessment
              protocols.
            </p>
            <div className="space-y-3">
              <div className="flex items-center justify-between py-3.5 border-b border-zinc-800">
                <span className="text-[13px] font-medium text-zinc-300">
                  Risk Score Threshold
                </span>
                <span className="text-[13px] text-zinc-100 font-medium">
                  75% (High Sensitivity)
                </span>
              </div>
              <div className="flex items-center justify-between py-3.5 border-b border-zinc-800">
                <span className="text-[13px] font-medium text-zinc-300">
                  Auto-Generate Reports
                </span>
                <div className="h-6 w-11 rounded-full bg-cyan-500 relative">
                  <div className="absolute right-1 top-1 h-4 w-4 rounded-full bg-white transition-transform" />
                </div>
              </div>
              <div className="flex items-center justify-between py-3.5 border-b border-zinc-800">
                <span className="text-[13px] font-medium text-zinc-300">
                  Critical Alert Notifications
                </span>
                <div className="h-6 w-11 rounded-full bg-cyan-500 relative">
                  <div className="absolute right-1 top-1 h-4 w-4 rounded-full bg-white transition-transform" />
                </div>
              </div>
              <div className="flex items-center justify-between py-3.5">
                <span className="text-[13px] font-medium text-zinc-300">
                  Default Assessment Duration
                </span>
                <span className="text-[13px] text-zinc-400">15 minutes</span>
              </div>
            </div>
          </div>

          {/* Clinical Alerts */}
          <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 hover:border-zinc-700 transition-colors">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2.5 rounded-lg bg-red-500/15">
                <AlertTriangle
                  className="h-5 w-5 text-red-400"
                  strokeWidth={2}
                />
              </div>
              <h2 className="text-[17px] font-semibold text-zinc-100">
                Clinical Alerts
              </h2>
            </div>
            <p className="text-[13px] text-zinc-400 mb-5 leading-relaxed">
              Configure when and how you receive critical patient alerts.
            </p>
            <div className="space-y-3">
              <div className="flex items-center justify-between py-3.5 border-b border-zinc-800">
                <span className="text-[13px] font-medium text-zinc-300">
                  Stroke Risk Alerts
                </span>
                <div className="h-6 w-11 rounded-full bg-cyan-500 relative">
                  <div className="absolute right-1 top-1 h-4 w-4 rounded-full bg-white transition-transform" />
                </div>
              </div>
              <div className="flex items-center justify-between py-3.5 border-b border-zinc-800">
                <span className="text-[13px] font-medium text-zinc-300">
                  Parkinson's Progression
                </span>
                <div className="h-6 w-11 rounded-full bg-cyan-500 relative">
                  <div className="absolute right-1 top-1 h-4 w-4 rounded-full bg-white transition-transform" />
                </div>
              </div>
              <div className="flex items-center justify-between py-3.5 border-b border-zinc-800">
                <span className="text-[13px] font-medium text-zinc-300">
                  Cognitive Decline
                </span>
                <div className="h-6 w-11 rounded-full bg-cyan-500 relative">
                  <div className="absolute right-1 top-1 h-4 w-4 rounded-full bg-white transition-transform" />
                </div>
              </div>
              <div className="flex items-center justify-between py-3.5">
                <span className="text-[13px] font-medium text-zinc-300">
                  Emergency Referrals
                </span>
                <div className="h-6 w-11 rounded-full bg-cyan-500 relative">
                  <div className="absolute right-1 top-1 h-4 w-4 rounded-full bg-white transition-transform" />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar Settings */}
        <div className="space-y-4">
          {/* Report Templates */}
          <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 hover:border-zinc-700 transition-colors">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2.5 rounded-lg bg-emerald-500/15">
                <FileText
                  className="h-5 w-5 text-emerald-400"
                  strokeWidth={2}
                />
              </div>
              <h2 className="text-[17px] font-semibold text-zinc-100">
                Report Templates
              </h2>
            </div>
            <div className="space-y-3">
              <div className="flex items-center justify-between py-3">
                <span className="text-[13px] font-medium text-zinc-300">
                  Default Template
                </span>
                <span className="text-[13px] text-zinc-400">
                  Neurological Assessment
                </span>
              </div>
              <div className="flex items-center justify-between py-3">
                <span className="text-[13px] font-medium text-zinc-300">
                  Include Recommendations
                </span>
                <div className="h-6 w-11 rounded-full bg-cyan-500 relative">
                  <div className="absolute right-1 top-1 h-4 w-4 rounded-full bg-white transition-transform" />
                </div>
              </div>
            </div>
          </div>

          {/* Schedule Settings */}
          <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 hover:border-zinc-700 transition-colors">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2.5 rounded-lg bg-amber-500/15">
                <Clock className="h-5 w-5 text-amber-400" strokeWidth={2} />
              </div>
              <h2 className="text-[17px] font-semibold text-zinc-100">
                Schedule
              </h2>
            </div>
            <div className="space-y-3">
              <div className="flex items-center justify-between py-3">
                <span className="text-[13px] font-medium text-zinc-300">
                  Working Hours
                </span>
                <span className="text-[13px] text-zinc-400">9 AM - 5 PM</span>
              </div>
              <div className="flex items-center justify-between py-3">
                <span className="text-[13px] font-medium text-zinc-300">
                  Time Zone
                </span>
                <span className="text-[13px] text-zinc-400">EST</span>
              </div>
            </div>
          </div>

          {/* Privacy & Security */}
          <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 hover:border-zinc-700 transition-colors">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2.5 rounded-lg bg-indigo-500/15">
                <Shield className="h-5 w-5 text-indigo-400" strokeWidth={2} />
              </div>
              <h2 className="text-[17px] font-semibold text-zinc-100">
                Security
              </h2>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between py-2.5">
                <span className="text-[13px] font-medium text-zinc-300">
                  HIPAA Compliance
                </span>
                <span className="text-[11px] text-emerald-400 bg-emerald-500/15 px-2.5 py-1 rounded-full font-medium">
                  Active
                </span>
              </div>
              <div className="flex items-center justify-between py-2.5">
                <span className="text-[13px] font-medium text-zinc-300">
                  Data Encryption
                </span>
                <span className="text-[11px] text-emerald-400 bg-emerald-500/15 px-2.5 py-1 rounded-full font-medium">
                  AES-256
                </span>
              </div>
              <div className="flex items-center justify-between py-2.5">
                <span className="text-[13px] font-medium text-zinc-300">
                  Audit Logging
                </span>
                <span className="text-[11px] text-emerald-400 bg-emerald-500/15 px-2.5 py-1 rounded-full font-medium">
                  Enabled
                </span>
              </div>
            </div>
          </div>

          {/* Sign Out */}
          <div className="bg-zinc-900 rounded-lg border border-red-500/30 p-6">
            <button
              disabled
              className="w-full flex items-center justify-center gap-2 text-red-400 cursor-not-allowed"
            >
              <LogOut className="h-5 w-5" strokeWidth={2} />
              <span className="text-[13px] font-semibold">
                Sign Out (Coming Soon)
              </span>
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
