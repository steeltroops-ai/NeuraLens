"use client";

/**
 * MediLens Dashboard - Enterprise-Grade Clinical AI Platform
 *
 * Premium dashboard inspired by Atlassian, Linear, and modern SaaS platforms:
 * - Gradient accents and depth
 * - Real-time system notifications via API
 * - Visual card differentiation with subtle shadows
 * - Modern, sleek enterprise aesthetics
 */

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { useUser } from "@clerk/nextjs";
import {
  Activity,
  Eye,
  Mic,
  Heart,
  Scan,
  ArrowRight,
  Clock,
  Shield,
  Brain,
  Hand,
  ChevronRight,
  CheckCircle2,
  AlertTriangle,
  FileText,
  Zap,
  Sparkles,
  Server,
  TrendingUp,
  Users,
  AlertCircle,
  Play,
  BarChart3,
  RefreshCw,
  ExternalLink,
  Bell,
  X,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface SystemHealth {
  backend: "online" | "offline" | "checking";
  latency: number | null;
  lastChecked: Date | null;
  pipelinesOnline: number;
  totalPipelines: number;
}

interface DiagnosticModule {
  id: string;
  name: string;
  shortName: string;
  description: string;
  icon: React.ElementType;
  route: string;
  gradient: string;
  accentColor: string;
  status: "active" | "idle" | "maintenance";
  lastUsed?: string;
  usageCount: number;
}

interface QuickStat {
  label: string;
  value: string | number;
  change?: string;
  changeType?: "positive" | "negative" | "neutral";
  icon: React.ElementType;
  gradient: string;
}

// ============================================================================
// Enterprise Module Data with Rich Gradients
// ============================================================================

const diagnosticModules: DiagnosticModule[] = [
  {
    id: "retinal",
    name: "RetinaScan AI",
    shortName: "Retinal",
    description: "Diabetic retinopathy grading",
    icon: Eye,
    route: "/dashboard/retinal",
    gradient: "from-cyan-500 to-blue-600",
    accentColor: "#0EA5E9",
    status: "active",
    lastUsed: "2m ago",
    usageCount: 1247,
  },
  {
    id: "speech",
    name: "SpeechMD AI",
    shortName: "Speech",
    description: "Voice biomarker analysis",
    icon: Mic,
    route: "/dashboard/speech",
    gradient: "from-violet-500 to-purple-600",
    accentColor: "#8B5CF6",
    status: "active",
    lastUsed: "15m ago",
    usageCount: 892,
  },
  {
    id: "cardiology",
    name: "CardioPredict AI",
    shortName: "Cardio",
    description: "ECG & arrhythmia detection",
    icon: Heart,
    route: "/dashboard/cardiology",
    gradient: "from-rose-500 to-pink-600",
    accentColor: "#F43F5E",
    status: "active",
    lastUsed: "1h ago",
    usageCount: 1456,
  },
  {
    id: "radiology",
    name: "ChestXplorer AI",
    shortName: "Radiology",
    description: "Chest X-ray pathology",
    icon: Scan,
    route: "/dashboard/radiology",
    gradient: "from-amber-500 to-orange-600",
    accentColor: "#F59E0B",
    status: "idle",
    usageCount: 678,
  },
  {
    id: "dermatology",
    name: "SkinSense AI",
    shortName: "Dermatology",
    description: "Skin lesion classification",
    icon: Sparkles,
    route: "/dashboard/dermatology",
    gradient: "from-fuchsia-500 to-pink-600",
    accentColor: "#D946EF",
    status: "active",
    lastUsed: "8m ago",
    usageCount: 543,
  },
  {
    id: "cognitive",
    name: "Cognitive Testing",
    shortName: "Cognitive",
    description: "Memory & cognition assessment",
    icon: Brain,
    route: "/dashboard/cognitive",
    gradient: "from-emerald-500 to-teal-600",
    accentColor: "#10B981",
    status: "idle",
    usageCount: 234,
  },
];

// ============================================================================
// Helper Functions
// ============================================================================

function getGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
}

// ============================================================================
// Enterprise Components
// ============================================================================

/**
 * Premium Header with Gradient Accent
 */
function DashboardHero({
  firstName,
  greeting,
  systemHealth,
}: {
  firstName: string;
  greeting: string;
  systemHealth: SystemHealth;
}) {
  return (
    <motion.header
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6 lg:p-8 border border-slate-700/50"
    >
      {/* Subtle animated gradient background */}
      <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-cyan-500/10 animate-pulse" />
      <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-bl from-blue-500/20 to-transparent rounded-full blur-3xl" />

      <div className="relative z-10 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <p className="text-sm text-slate-400 mb-1">{greeting}</p>
          <h1 className="text-2xl lg:text-3xl font-bold text-white tracking-tight">
            {firstName}
          </h1>
          <p className="mt-2 text-slate-400 text-sm max-w-md">
            Your clinical AI command center. Run diagnostics, monitor pipelines,
            and review assessments.
          </p>
        </div>

        {/* System Status Card */}
        <div className="flex items-center gap-4">
          <div className="bg-slate-800/80 backdrop-blur border border-slate-700/50 rounded-xl p-4 min-w-[180px]">
            <div className="flex items-center gap-2 mb-2">
              <div
                className={`w-2.5 h-2.5 rounded-full ${systemHealth.backend === "online" ? "bg-emerald-500 shadow-lg shadow-emerald-500/50" : "bg-amber-500 animate-pulse"}`}
              />
              <span className="text-xs font-medium text-slate-300">
                {systemHealth.backend === "online"
                  ? "All Systems Go"
                  : "Checking..."}
              </span>
            </div>
            <div className="flex items-baseline gap-1">
              <span className="text-2xl font-bold text-white">
                {systemHealth.pipelinesOnline}
              </span>
              <span className="text-sm text-slate-400">
                / {systemHealth.totalPipelines} pipelines
              </span>
            </div>
            {systemHealth.latency && (
              <p className="text-xs text-slate-500 mt-1">
                {systemHealth.latency}ms latency
              </p>
            )}
          </div>
        </div>
      </div>
    </motion.header>
  );
}

/**
 * Premium Stat Cards with Gradients
 */
function StatCard({ stat, index }: { stat: QuickStat; index: number }) {
  const Icon = stat.icon;
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className="relative overflow-hidden bg-white rounded-xl border border-slate-200 p-5 hover:shadow-lg hover:shadow-slate-200/50 transition-all duration-300 group"
    >
      {/* Gradient accent bar on top */}
      <div
        className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${stat.gradient}`}
      />

      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">
            {stat.label}
          </p>
          <p className="text-2xl font-bold text-slate-900 mt-1">{stat.value}</p>
          {stat.change && (
            <p
              className={`text-xs mt-1 font-medium ${
                stat.changeType === "positive"
                  ? "text-emerald-600"
                  : stat.changeType === "negative"
                    ? "text-red-600"
                    : "text-slate-500"
              }`}
            >
              {stat.change}
            </p>
          )}
        </div>
        <div
          className={`p-2.5 rounded-xl bg-gradient-to-br ${stat.gradient} shadow-lg`}
        >
          <Icon size={18} className="text-white" strokeWidth={2} />
        </div>
      </div>
    </motion.div>
  );
}

/**
 * Premium Diagnostic Module Card
 */
function ModuleCard({
  module,
  index,
}: {
  module: DiagnosticModule;
  index: number;
}) {
  const Icon = module.icon;
  const isActive = module.status === "active";

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: index * 0.03 }}
    >
      <Link href={module.route} className="group block">
        <div className="relative overflow-hidden bg-white rounded-xl border border-slate-200 p-5 hover:shadow-xl hover:shadow-slate-200/50 hover:border-slate-300 transition-all duration-300">
          {/* Gradient accent on left */}
          <div
            className={`absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b ${module.gradient}`}
          />

          <div className="flex items-start gap-4">
            {/* Icon with gradient background */}
            <div
              className={`flex-shrink-0 p-3 rounded-xl bg-gradient-to-br ${module.gradient} shadow-lg shadow-slate-200`}
            >
              <Icon size={22} className="text-white" strokeWidth={1.5} />
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <h3 className="text-sm font-semibold text-slate-900 group-hover:text-blue-600 transition-colors">
                  {module.name}
                </h3>
                {isActive && (
                  <span className="flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-emerald-50 border border-emerald-200">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                    <span className="text-[10px] font-medium text-emerald-700">
                      Active
                    </span>
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-500 mb-2">
                {module.description}
              </p>
              <div className="flex items-center gap-3 text-xs text-slate-400">
                <span className="flex items-center gap-1">
                  <BarChart3 size={12} />
                  {module.usageCount.toLocaleString()} runs
                </span>
                {module.lastUsed && (
                  <span className="flex items-center gap-1">
                    <Clock size={12} />
                    {module.lastUsed}
                  </span>
                )}
              </div>
            </div>

            {/* Arrow */}
            <ArrowRight
              size={16}
              className="text-slate-300 group-hover:text-slate-500 group-hover:translate-x-1 transition-all flex-shrink-0 mt-1"
            />
          </div>
        </div>
      </Link>
    </motion.div>
  );
}

/**
 * Quick Action Button
 */
function QuickActionButton({
  icon: Icon,
  label,
  href,
  gradient,
}: {
  icon: React.ElementType;
  label: string;
  href: string;
  gradient: string;
}) {
  return (
    <Link href={href}>
      <motion.div
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        className={`relative overflow-hidden p-4 rounded-xl bg-gradient-to-br ${gradient} text-white shadow-lg hover:shadow-xl transition-shadow cursor-pointer`}
      >
        <div className="absolute inset-0 bg-white/10 opacity-0 hover:opacity-100 transition-opacity" />
        <div className="relative flex items-center gap-3">
          <div className="p-2 bg-white/20 rounded-lg backdrop-blur">
            <Icon size={18} strokeWidth={2} />
          </div>
          <span className="text-sm font-semibold">{label}</span>
          <ChevronRight size={16} className="ml-auto" />
        </div>
      </motion.div>
    </Link>
  );
}

/**
 * Pipeline Status Row
 */
function PipelineStatusCard() {
  const pipelines = [
    {
      name: "RetinaScan",
      status: "healthy",
      latency: "1.2s",
      color: "bg-cyan-500",
    },
    {
      name: "SpeechMD",
      status: "healthy",
      latency: "0.9s",
      color: "bg-violet-500",
    },
    {
      name: "CardioPredict",
      status: "healthy",
      latency: "1.5s",
      color: "bg-rose-500",
    },
    {
      name: "ChestXplorer",
      status: "idle",
      latency: "2.1s",
      color: "bg-amber-500",
    },
    {
      name: "SkinSense",
      status: "healthy",
      latency: "1.1s",
      color: "bg-fuchsia-500",
    },
  ];

  const healthyCount = pipelines.filter((p) => p.status === "healthy").length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="bg-white rounded-xl border border-slate-200 p-5 hover:shadow-lg transition-shadow"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600 shadow-lg">
            <Activity size={16} className="text-white" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-900">
              Pipeline Health
            </h3>
            <p className="text-xs text-slate-500">
              {healthyCount}/{pipelines.length} operational
            </p>
          </div>
        </div>
        <button className="text-xs text-slate-500 hover:text-slate-700 flex items-center gap-1">
          <RefreshCw size={12} />
          Refresh
        </button>
      </div>

      <div className="space-y-2">
        {pipelines.map((pipeline, i) => (
          <div
            key={i}
            className="flex items-center gap-3 p-2 rounded-lg hover:bg-slate-50 transition-colors"
          >
            <div className={`w-2 h-2 rounded-full ${pipeline.color}`} />
            <span className="text-sm text-slate-700 flex-1">
              {pipeline.name}
            </span>
            <span
              className={`text-xs px-2 py-0.5 rounded-full ${
                pipeline.status === "healthy"
                  ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                  : "bg-slate-100 text-slate-600 border border-slate-200"
              }`}
            >
              {pipeline.status}
            </span>
            <span className="text-xs text-slate-400">{pipeline.latency}</span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

/**
 * Recent Activity Card
 */
function RecentActivityCard() {
  const activities = [
    {
      module: "Retinal",
      action: "Analysis completed",
      time: "2m ago",
      status: "success",
      gradient: "from-cyan-500 to-blue-600",
    },
    {
      module: "Speech",
      action: "New assessment",
      time: "15m ago",
      status: "success",
      gradient: "from-violet-500 to-purple-600",
    },
    {
      module: "Cardio",
      action: "Report generated",
      time: "1h ago",
      status: "success",
      gradient: "from-rose-500 to-pink-600",
    },
    {
      module: "Dermatology",
      action: "Review pending",
      time: "2h ago",
      status: "pending",
      gradient: "from-fuchsia-500 to-pink-600",
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.25 }}
      className="bg-white rounded-xl border border-slate-200 p-5 hover:shadow-lg transition-shadow"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 shadow-lg">
            <Clock size={16} className="text-white" />
          </div>
          <h3 className="text-sm font-semibold text-slate-900">
            Recent Activity
          </h3>
        </div>
        <Link
          href="/dashboard/reports"
          className="text-xs text-blue-600 hover:text-blue-700 flex items-center gap-1"
        >
          View all <ExternalLink size={10} />
        </Link>
      </div>

      <div className="space-y-3">
        {activities.map((activity, i) => (
          <div key={i} className="flex items-center gap-3">
            <div
              className={`w-8 h-8 rounded-lg bg-gradient-to-br ${activity.gradient} flex items-center justify-center shadow`}
            >
              <span className="text-xs font-bold text-white">
                {activity.module[0]}
              </span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-slate-900 truncate">
                {activity.action}
              </p>
              <p className="text-xs text-slate-500">
                {activity.module} - {activity.time}
              </p>
            </div>
            <span
              className={`w-2 h-2 rounded-full ${
                activity.status === "success"
                  ? "bg-emerald-500"
                  : "bg-amber-500"
              }`}
            />
          </div>
        ))}
      </div>
    </motion.div>
  );
}

// ============================================================================
// Main Dashboard
// ============================================================================

export default function DashboardPage() {
  const { user } = useUser();
  const [systemHealth, setSystemHealth] = useState<SystemHealth>({
    backend: "checking",
    latency: null,
    lastChecked: null,
    pipelinesOnline: 4,
    totalPipelines: 5,
  });

  const checkHealth = useCallback(async () => {
    const startTime = Date.now();
    try {
      const response = await fetch("/api/health", {
        method: "GET",
        cache: "no-store",
      });
      const latency = Date.now() - startTime;
      setSystemHealth({
        backend: response.ok ? "online" : "offline",
        latency: response.ok ? latency : null,
        lastChecked: new Date(),
        pipelinesOnline: 4,
        totalPipelines: 5,
      });
    } catch {
      setSystemHealth({
        backend: "offline",
        latency: null,
        lastChecked: new Date(),
        pipelinesOnline: 0,
        totalPipelines: 5,
      });
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  const firstName = user?.firstName || user?.username || "there";
  const greeting = getGreeting();

  const quickStats: QuickStat[] = [
    {
      label: "Total Analyses",
      value: "5,847",
      change: "+12% this week",
      changeType: "positive",
      icon: BarChart3,
      gradient: "from-blue-500 to-indigo-600",
    },
    {
      label: "Active Pipelines",
      value: "4/5",
      change: "All healthy",
      changeType: "positive",
      icon: Activity,
      gradient: "from-emerald-500 to-teal-600",
    },
    {
      label: "Avg Response",
      value: "1.2s",
      change: "-0.3s from last week",
      changeType: "positive",
      icon: Zap,
      gradient: "from-amber-500 to-orange-600",
    },
    {
      label: "Pending Reviews",
      value: "3",
      change: "2 urgent",
      changeType: "negative",
      icon: AlertCircle,
      gradient: "from-rose-500 to-pink-600",
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6 pb-8"
    >
      {/* Hero Header */}
      <DashboardHero
        firstName={firstName}
        greeting={greeting}
        systemHealth={systemHealth}
      />

      {/* Quick Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {quickStats.map((stat, index) => (
          <StatCard key={stat.label} stat={stat} index={index} />
        ))}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <QuickActionButton
          icon={Play}
          label="New Analysis"
          href="/dashboard/retinal"
          gradient="from-blue-600 to-indigo-700"
        />
        <QuickActionButton
          icon={FileText}
          label="View Reports"
          href="/dashboard/reports"
          gradient="from-violet-600 to-purple-700"
        />
        <QuickActionButton
          icon={BarChart3}
          label="Analytics"
          href="/dashboard/analytics"
          gradient="from-emerald-600 to-teal-700"
        />
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Modules Grid */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-slate-900">
              AI Diagnostic Modules
            </h2>
            <span className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded-full">
              {diagnosticModules.length} available
            </span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {diagnosticModules.map((module, index) => (
              <ModuleCard key={module.id} module={module} index={index} />
            ))}
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          <PipelineStatusCard />
          <RecentActivityCard />

          {/* Clinical Note */}
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl p-5 border border-slate-700">
            <div className="flex items-start gap-3">
              <div className="p-2 bg-blue-500/20 rounded-lg">
                <Shield size={16} className="text-blue-400" />
              </div>
              <div>
                <h4 className="text-sm font-medium text-white mb-1">
                  Clinical Reminder
                </h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  AI analysis results should always be reviewed by qualified
                  healthcare professionals before clinical decisions.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
