"use client";

/**
 * MediLens Dashboard - Clinical AI Platform
 *
 * Doctor/Clinician focused dashboard with:
 * - Quick access to diagnostic modules
 * - Recent patient analyses
 * - Clinical insights and alerts
 * - Pending reviews and priorities
 */

import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
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
  ChevronRight,
  CheckCircle2,
  AlertTriangle,
  FileText,
  Zap,
  Sparkles,
  AlertCircle,
  Play,
  BarChart3,
  RefreshCw,
  Users,
  Calendar,
  TrendingUp,
  Stethoscope,
  ClipboardList,
  UserCheck,
  AlertOctagon,
  Bookmark,
  Keyboard,
  BookOpen,
  Download,
  HelpCircle,
  Wifi,
  Database,
  ArrowUpRight,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface SystemHealth {
  backend: "online" | "offline" | "checking";
  latency: number | null;
  pipelinesOnline: number;
  totalPipelines: number;
}

interface DiagnosticModule {
  id: string;
  name: string;
  description: string;
  icon: React.ElementType;
  route: string;
  color: string;
  gradient: string;
}

interface RecentAnalysis {
  id: string;
  patientId: string;
  patientName: string;
  module: string;
  moduleId: string;
  result: "normal" | "abnormal" | "critical" | "pending";
  timestamp: Date;
  riskLevel?: "low" | "moderate" | "high";
}

interface ClinicalAlert {
  id: string;
  type: "urgent" | "followup" | "reminder";
  title: string;
  patient: string;
  timestamp: Date;
}

// ============================================================================
// Module Data
// ============================================================================

const diagnosticModules: DiagnosticModule[] = [
  {
    id: "retinal",
    name: "RetinaScan AI",
    description: "Diabetic retinopathy detection & grading",
    icon: Eye,
    route: "/dashboard/retinal",
    color: "#0891b2",
    gradient: "from-cyan-500 to-blue-500",
  },
  {
    id: "speech",
    name: "SpeechMD AI",
    description: "Voice biomarker & neurological analysis",
    icon: Mic,
    route: "/dashboard/speech",
    color: "#7c3aed",
    gradient: "from-violet-500 to-purple-500",
  },
  {
    id: "cardiology",
    name: "CardioPredict AI",
    description: "ECG analysis & arrhythmia detection",
    icon: Heart,
    route: "/dashboard/cardiology",
    color: "#dc2626",
    gradient: "from-rose-500 to-red-500",
  },
  {
    id: "radiology",
    name: "ChestXplorer AI",
    description: "Chest X-ray pathology detection",
    icon: Scan,
    route: "/dashboard/radiology",
    color: "#d97706",
    gradient: "from-amber-500 to-orange-500",
  },
  {
    id: "dermatology",
    name: "SkinSense AI",
    description: "Skin lesion classification & melanoma",
    icon: Sparkles,
    route: "/dashboard/dermatology",
    color: "#c026d3",
    gradient: "from-fuchsia-500 to-pink-500",
  },
  {
    id: "cognitive",
    name: "Cognitive Testing",
    description: "Memory & cognitive function assessment",
    icon: Brain,
    route: "/dashboard/cognitive",
    color: "#059669",
    gradient: "from-emerald-500 to-teal-500",
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

function formatTimeAgo(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);

  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  return `${Math.floor(diffHours / 24)}d ago`;
}

// ============================================================================
// Components
// ============================================================================

function WelcomeCard({
  firstName,
  greeting,
  systemOnline,
}: {
  firstName: string;
  greeting: string;
  systemOnline: boolean;
}) {
  return (
    <div className="relative overflow-hidden rounded-xl bg-zinc-900 border border-zinc-800 p-5">
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-20 -right-20 w-64 h-64 bg-gradient-to-bl from-blue-500/10 to-transparent rounded-full blur-3xl" />
      </div>
      <div className="relative z-10 flex items-center justify-between">
        <div>
          <p className="text-xs text-zinc-500 font-mono uppercase tracking-wider">
            {greeting}
          </p>
          <h1 className="text-lg font-semibold text-zinc-100 mt-1">
            Welcome back, <span className="text-blue-400">{firstName}</span>
          </h1>
          <p className="text-sm text-zinc-400 mt-0.5">
            AI-powered clinical diagnostics
          </p>
        </div>
        <div className="flex items-center gap-2 bg-zinc-800/60 border border-zinc-700/50 rounded-lg px-3 py-2">
          <div
            className={`w-2 h-2 rounded-full ${systemOnline ? "bg-emerald-500" : "bg-red-500"}`}
          />
          <span className="text-xs text-zinc-400">
            {systemOnline ? "System Ready" : "Offline"}
          </span>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  change,
  changeType,
  icon: Icon,
  color,
}: {
  label: string;
  value: string | number;
  change?: string;
  changeType?: "positive" | "negative" | "neutral";
  icon: React.ElementType;
  color: string;
}) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 hover:border-zinc-700 transition-colors">
      <div className="flex items-center justify-between mb-1.5">
        <p className="text-[10px] text-zinc-500 uppercase tracking-wide font-medium">
          {label}
        </p>
        <div className="p-1 rounded" style={{ backgroundColor: `${color}15` }}>
          <Icon size={12} style={{ color }} />
        </div>
      </div>
      <p className="text-lg font-semibold text-zinc-100">{value}</p>
      {change && (
        <p
          className={`text-[10px] mt-0.5 flex items-center gap-0.5 ${
            changeType === "positive"
              ? "text-emerald-500"
              : changeType === "negative"
                ? "text-red-500"
                : "text-zinc-500"
          }`}
        >
          {changeType === "positive" && <TrendingUp size={9} />}
          {change}
        </p>
      )}
    </div>
  );
}

function ModuleCard({ module }: { module: DiagnosticModule }) {
  const Icon = module.icon;
  return (
    <Link href={module.route} className="block group">
      <div className="relative overflow-hidden bg-zinc-900 border border-zinc-800 rounded-lg p-3 hover:border-zinc-700 transition-all">
        <div
          className={`absolute left-0 top-0 bottom-0 w-0.5 bg-gradient-to-b ${module.gradient}`}
        />
        <div
          className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
          style={{
            background: `linear-gradient(135deg, ${module.color}06 0%, transparent 50%)`,
          }}
        />
        <div className="relative z-10 flex items-center gap-2.5 pl-1.5">
          <div
            className="p-1.5 rounded"
            style={{ backgroundColor: `${module.color}15` }}
          >
            <Icon size={14} style={{ color: module.color }} />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-xs font-medium text-zinc-200 truncate">
              {module.name}
            </h3>
            <p className="text-[10px] text-zinc-500 truncate">
              {module.description}
            </p>
          </div>
          <ArrowRight
            size={12}
            className="text-zinc-600 group-hover:text-zinc-400 transition-all flex-shrink-0"
          />
        </div>
      </div>
    </Link>
  );
}

function QuickAction({
  icon: Icon,
  label,
  href,
  color,
  gradient,
}: {
  icon: React.ElementType;
  label: string;
  href: string;
  color: string;
  gradient: string;
}) {
  return (
    <Link href={href} className="group">
      <div className="relative overflow-hidden flex items-center gap-2 p-2.5 rounded-lg border border-zinc-800 bg-zinc-900 hover:border-zinc-700 transition-all">
        <div
          className={`absolute inset-0 bg-gradient-to-r ${gradient} opacity-0 group-hover:opacity-10 transition-opacity duration-300`}
        />
        <div
          className="relative z-10 p-1.5 rounded"
          style={{ backgroundColor: `${color}20` }}
        >
          <Icon size={14} style={{ color }} />
        </div>
        <span className="relative z-10 text-xs font-medium text-zinc-200">
          {label}
        </span>
        <ChevronRight
          size={12}
          className="relative z-10 text-zinc-500 ml-auto"
        />
      </div>
    </Link>
  );
}

function RecentPatientsCard({ analyses }: { analyses: RecentAnalysis[] }) {
  const getResultBadge = (result: string) =>
    ({
      normal: "text-emerald-500 bg-emerald-500/10",
      abnormal: "text-amber-500 bg-amber-500/10",
      critical: "text-red-500 bg-red-500/10",
      pending: "text-blue-500 bg-blue-500/10",
    })[result] || "text-zinc-500 bg-zinc-500/10";

  const getRiskBadge = (risk?: string) =>
    ({
      low: "text-emerald-500",
      moderate: "text-amber-500",
      high: "text-red-500",
    })[risk || ""] || "text-zinc-500";

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Users size={14} className="text-zinc-400" />
          <h3 className="text-sm font-medium text-zinc-200">Recent Patients</h3>
        </div>
        <Link
          href="/dashboard/reports"
          className="text-[10px] text-blue-400 hover:text-blue-300"
        >
          View all
        </Link>
      </div>
      <div className="space-y-2">
        {analyses.length === 0 ? (
          <p className="text-xs text-zinc-500 py-4 text-center">
            No recent analyses
          </p>
        ) : (
          analyses.map((analysis) => (
            <Link
              key={analysis.id}
              href={`/dashboard/${analysis.moduleId}`}
              className="flex items-center gap-3 p-2 rounded-lg bg-zinc-800/50 hover:bg-zinc-800 transition-colors"
            >
              <div className="w-8 h-8 rounded-full bg-zinc-700 flex items-center justify-center">
                <span className="text-xs text-zinc-300 font-medium">
                  {analysis.patientName
                    .split(" ")
                    .map((n) => n[0])
                    .join("")}
                </span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs text-zinc-200 font-medium truncate">
                  {analysis.patientName}
                </p>
                <p className="text-[10px] text-zinc-500">
                  {analysis.module} - {formatTimeAgo(analysis.timestamp)}
                </p>
              </div>
              <div className="text-right">
                <span
                  className={`text-[10px] px-1.5 py-0.5 rounded capitalize ${getResultBadge(analysis.result)}`}
                >
                  {analysis.result}
                </span>
                {analysis.riskLevel && (
                  <p
                    className={`text-[9px] mt-0.5 ${getRiskBadge(analysis.riskLevel)}`}
                  >
                    {analysis.riskLevel} risk
                  </p>
                )}
              </div>
            </Link>
          ))
        )}
      </div>
    </div>
  );
}

function ClinicalAlertsCard({ alerts }: { alerts: ClinicalAlert[] }) {
  const getAlertStyle = (type: string) =>
    ({
      urgent: {
        icon: AlertOctagon,
        color: "text-red-500",
        bg: "bg-red-500/10",
      },
      followup: {
        icon: Calendar,
        color: "text-amber-500",
        bg: "bg-amber-500/10",
      },
      reminder: {
        icon: Bookmark,
        color: "text-blue-500",
        bg: "bg-blue-500/10",
      },
    })[type] || {
      icon: AlertCircle,
      color: "text-zinc-500",
      bg: "bg-zinc-500/10",
    };

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <AlertTriangle size={14} className="text-zinc-400" />
          <h3 className="text-sm font-medium text-zinc-200">Clinical Alerts</h3>
          {alerts.filter((a) => a.type === "urgent").length > 0 && (
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-400">
              {alerts.filter((a) => a.type === "urgent").length} urgent
            </span>
          )}
        </div>
      </div>
      <div className="space-y-2">
        {alerts.length === 0 ? (
          <p className="text-xs text-zinc-500 py-4 text-center">No alerts</p>
        ) : (
          alerts.map((alert) => {
            const style = getAlertStyle(alert.type);
            const Icon = style.icon;
            return (
              <div
                key={alert.id}
                className="flex items-start gap-2 p-2 rounded-lg bg-zinc-800/50"
              >
                <div className={`p-1 rounded ${style.bg}`}>
                  <Icon size={12} className={style.color} />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-zinc-200 truncate">
                    {alert.title}
                  </p>
                  <p className="text-[10px] text-zinc-500">
                    {alert.patient} - {formatTimeAgo(alert.timestamp)}
                  </p>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

function PendingReviewsCard({ analyses }: { analyses: RecentAnalysis[] }) {
  const pending = analyses.filter(
    (a) =>
      a.result === "pending" ||
      a.result === "abnormal" ||
      a.result === "critical",
  );

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <ClipboardList size={14} className="text-zinc-400" />
          <h3 className="text-sm font-medium text-zinc-200">Pending Reviews</h3>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400">
            {pending.length}
          </span>
        </div>
      </div>
      <div className="space-y-2">
        {pending.length === 0 ? (
          <div className="flex items-center gap-2 py-4 justify-center">
            <CheckCircle2 size={14} className="text-emerald-500" />
            <p className="text-xs text-zinc-400">All caught up!</p>
          </div>
        ) : (
          pending.slice(0, 3).map((analysis) => (
            <Link
              key={analysis.id}
              href={`/dashboard/${analysis.moduleId}`}
              className="flex items-center justify-between p-2 rounded-lg bg-zinc-800/50 hover:bg-zinc-800 transition-colors"
            >
              <div className="flex items-center gap-2">
                <Stethoscope size={12} className="text-zinc-400" />
                <div>
                  <p className="text-xs text-zinc-200">
                    {analysis.patientName}
                  </p>
                  <p className="text-[10px] text-zinc-500">{analysis.module}</p>
                </div>
              </div>
              <span
                className={`text-[10px] px-1.5 py-0.5 rounded ${
                  analysis.result === "critical"
                    ? "bg-red-500/10 text-red-500"
                    : analysis.result === "abnormal"
                      ? "bg-amber-500/10 text-amber-500"
                      : "bg-blue-500/10 text-blue-500"
                }`}
              >
                {analysis.result === "pending" ? "Review" : "Verify"}
              </span>
            </Link>
          ))
        )}
      </div>
    </div>
  );
}

function TodaySummaryCard({
  stats,
}: {
  stats: { completed: number; pending: number; critical: number };
}) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-3">
        <Activity size={14} className="text-zinc-400" />
        <h3 className="text-sm font-medium text-zinc-200">Today's Summary</h3>
      </div>
      <div className="grid grid-cols-3 gap-3">
        <div className="text-center">
          <p className="text-lg font-semibold text-emerald-500">
            {stats.completed}
          </p>
          <p className="text-[10px] text-zinc-500">Completed</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-semibold text-amber-500">
            {stats.pending}
          </p>
          <p className="text-[10px] text-zinc-500">Pending</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-semibold text-red-500">{stats.critical}</p>
          <p className="text-[10px] text-zinc-500">Critical</p>
        </div>
      </div>
    </div>
  );
}

function SystemStatusCard({
  online,
  latency,
  pipelinesOnline = 0,
}: {
  online: boolean;
  latency?: number;
  pipelinesOnline?: number;
}) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-3">
        <Database size={14} className="text-zinc-400" />
        <h3 className="text-sm font-medium text-zinc-200">System Status</h3>
      </div>
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Wifi
              size={12}
              className={online ? "text-emerald-500" : "text-red-500"}
            />
            <span className="text-xs text-zinc-400">Backend API</span>
          </div>
          <span
            className={`text-xs ${online ? "text-emerald-500" : "text-red-500"}`}
          >
            {online ? "Connected" : "Offline"}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap size={12} className="text-amber-500" />
            <span className="text-xs text-zinc-400">Response</span>
          </div>
          <span className="text-xs text-zinc-300">{latency || "--"}ms</span>
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity size={12} className="text-blue-500" />
            <span className="text-xs text-zinc-400">Active Pipelines</span>
          </div>
          <span className="text-xs text-emerald-500">{pipelinesOnline}/5</span>
        </div>
      </div>
    </div>
  );
}

function KeyboardShortcutsCard() {
  const shortcuts = [
    { keys: ["Ctrl", "K"], action: "Search" },
    { keys: ["Ctrl", "/"], action: "Chat" },
  ];

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-2.5">
      <div className="flex items-center gap-1.5 mb-1.5">
        <Keyboard size={11} className="text-zinc-400" />
        <h3 className="text-[10px] font-medium text-zinc-300">Shortcuts</h3>
      </div>
      <div className="space-y-1">
        {shortcuts.map((s, i) => (
          <div key={i} className="flex items-center justify-between">
            <div className="flex items-center gap-0.5">
              {s.keys.map((key, j) => (
                <span
                  key={j}
                  className="px-1.5 py-0.5 text-[10px] font-mono font-medium text-zinc-300 bg-zinc-800 rounded border border-zinc-700"
                >
                  {key}
                </span>
              ))}
            </div>
            <span className="text-[10px] text-zinc-500">{s.action}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function QuickLinksCard() {
  const links = [
    { icon: BookOpen, label: "Docs", href: "#" },
    { icon: HelpCircle, label: "Help", href: "#" },
  ];

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-2.5">
      <div className="flex items-center gap-1.5 mb-1.5">
        <ArrowUpRight size={11} className="text-zinc-400" />
        <h3 className="text-[10px] font-medium text-zinc-300">Links</h3>
      </div>
      <div className="space-y-0.5">
        {links.map((link, i) => (
          <Link
            key={i}
            href={link.href}
            className="flex items-center gap-1.5 p-1.5 rounded hover:bg-zinc-800 transition-colors"
          >
            <link.icon size={11} className="text-zinc-400" />
            <span className="text-[10px] text-zinc-300">{link.label}</span>
          </Link>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Main Dashboard
// ============================================================================

export default function DashboardPage() {
  const { user } = useUser();
  const [systemOnline, setSystemOnline] = useState(false);
  const [pipelinesOnline, setPipelinesOnline] = useState(0);
  const [recentAnalyses, setRecentAnalyses] = useState<RecentAnalysis[]>([]);
  const [clinicalAlerts, setClinicalAlerts] = useState<ClinicalAlert[]>([]);

  const checkHealth = useCallback(async () => {
    const pipelineIds = [
      "retinal",
      "speech",
      "cardiology",
      "radiology",
      "dermatology",
    ];

    try {
      // Parallel health checks using Promise.allSettled for better performance
      const [mainHealthResult, ...pipelineResults] = await Promise.allSettled([
        fetch("/api/health", { method: "GET", cache: "no-store" }),
        ...pipelineIds.map((id) =>
          fetch(`/api/${id}/health`, { method: "GET", cache: "no-store" }),
        ),
      ]);

      // Check main backend health
      const isMainOnline =
        mainHealthResult.status === "fulfilled" && mainHealthResult.value.ok;
      setSystemOnline(isMainOnline);

      // Count online pipelines
      const onlineCount = pipelineResults.filter(
        (result) => result.status === "fulfilled" && result.value.ok,
      ).length;
      setPipelinesOnline(onlineCount);
    } catch {
      setSystemOnline(false);
      setPipelinesOnline(0);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);

    // Mock clinical data (in production, fetch from API)
    setRecentAnalyses([
      {
        id: "1",
        patientId: "P001",
        patientName: "Sarah Johnson",
        module: "Retinal Scan",
        moduleId: "retinal",
        result: "normal",
        timestamp: new Date(Date.now() - 1800000),
        riskLevel: "low",
      },
      {
        id: "2",
        patientId: "P002",
        patientName: "Michael Chen",
        module: "Speech Analysis",
        moduleId: "speech",
        result: "abnormal",
        timestamp: new Date(Date.now() - 3600000),
        riskLevel: "moderate",
      },
      {
        id: "3",
        patientId: "P003",
        patientName: "Emily Davis",
        module: "ECG Analysis",
        moduleId: "cardiology",
        result: "critical",
        timestamp: new Date(Date.now() - 7200000),
        riskLevel: "high",
      },
      {
        id: "4",
        patientId: "P004",
        patientName: "Robert Wilson",
        module: "Skin Analysis",
        moduleId: "dermatology",
        result: "pending",
        timestamp: new Date(Date.now() - 10800000),
      },
      {
        id: "5",
        patientId: "P005",
        patientName: "Lisa Anderson",
        module: "Chest X-ray",
        moduleId: "radiology",
        result: "normal",
        timestamp: new Date(Date.now() - 14400000),
        riskLevel: "low",
      },
    ]);

    setClinicalAlerts([
      {
        id: "1",
        type: "urgent",
        title: "Critical ECG findings require immediate review",
        patient: "Emily Davis",
        timestamp: new Date(Date.now() - 7200000),
      },
      {
        id: "2",
        type: "followup",
        title: "Speech assessment follow-up recommended",
        patient: "Michael Chen",
        timestamp: new Date(Date.now() - 86400000),
      },
      {
        id: "3",
        type: "reminder",
        title: "Quarterly retinal screening due",
        patient: "Sarah Johnson",
        timestamp: new Date(Date.now() - 172800000),
      },
    ]);

    return () => clearInterval(interval);
  }, [checkHealth]);

  const firstName = user?.firstName || user?.username || "Doctor";

  const todayStats = {
    completed: recentAnalyses.filter((a) => a.result === "normal").length,
    pending: recentAnalyses.filter((a) => a.result === "pending").length,
    critical: recentAnalyses.filter((a) => a.result === "critical").length,
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2 }}
      className="space-y-4 pb-6"
    >
      {/* Welcome */}
      <WelcomeCard
        firstName={firstName}
        greeting={getGreeting()}
        systemOnline={systemOnline}
      />

      {/* Stats Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
        <StatCard
          label="Today's Analyses"
          value="24"
          change="+8 from yesterday"
          changeType="positive"
          icon={BarChart3}
          color="#3b82f6"
        />
        <StatCard
          label="Patients Seen"
          value="18"
          change="+3 this week"
          changeType="positive"
          icon={Users}
          color="#059669"
        />
        <StatCard
          label="Pending Reviews"
          value="3"
          change="2 urgent"
          changeType="negative"
          icon={ClipboardList}
          color="#d97706"
        />
        <StatCard
          label="Accuracy Rate"
          value="98.2%"
          change="+0.5%"
          changeType="positive"
          icon={UserCheck}
          color="#7c3aed"
        />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-3 gap-2">
        <QuickAction
          icon={Play}
          label="New Analysis"
          href="/dashboard/retinal"
          color="#3b82f6"
          gradient="from-blue-500 to-indigo-500"
        />
        <QuickAction
          icon={FileText}
          label="Patient Reports"
          href="/dashboard/reports"
          color="#7c3aed"
          gradient="from-violet-500 to-purple-500"
        />
        <QuickAction
          icon={BarChart3}
          label="Analytics"
          href="/dashboard/analytics"
          color="#059669"
          gradient="from-emerald-500 to-teal-500"
        />
      </div>

      {/* Main 3-Column Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left: Modules + System */}
        <div className="space-y-3">
          <h2 className="text-xs font-medium text-zinc-400">
            Diagnostic Modules
          </h2>
          <div className="space-y-2">
            {diagnosticModules.map((module) => (
              <ModuleCard key={module.id} module={module} />
            ))}
          </div>
          <SystemStatusCard
            online={systemOnline}
            pipelinesOnline={pipelinesOnline}
          />
        </div>

        {/* Center: Clinical Overview */}
        <div className="space-y-3">
          <h2 className="text-xs font-medium text-zinc-400">
            Clinical Overview
          </h2>
          <TodaySummaryCard stats={todayStats} />
          <ClinicalAlertsCard alerts={clinicalAlerts} />
        </div>

        {/* Right: Patients & Reviews + Shortcuts + Links */}
        <div className="space-y-3">
          <h2 className="text-xs font-medium text-zinc-400">
            Patient Activity
          </h2>
          <RecentPatientsCard analyses={recentAnalyses} />
          <PendingReviewsCard analyses={recentAnalyses} />

          {/* Shortcuts & Quick Links - Compact */}
          <div className="grid grid-cols-2 gap-2">
            <KeyboardShortcutsCard />
            <QuickLinksCard />
          </div>

          {/* AI Disclaimer */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3">
            <div className="flex items-start gap-2">
              <Shield size={14} className="text-blue-500 mt-0.5" />
              <div>
                <p className="text-xs font-medium text-zinc-300">
                  AI-Assisted Diagnostics
                </p>
                <p className="text-[10px] text-zinc-500 mt-0.5 leading-relaxed">
                  All AI findings require clinical verification before
                  diagnosis.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
