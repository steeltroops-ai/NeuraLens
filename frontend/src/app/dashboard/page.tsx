"use client";

/**
 * MediLens Dashboard - Sleek Medical AI Platform
 *
 * Minimal, professional dashboard featuring:
 * - Personalized user greeting
 * - Sleek horizontal diagnostic cards
 * - Recent activity feed
 * - Clean, modern interface
 */

import { useState, useEffect, lazy, Suspense, useCallback } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { useUser } from "@clerk/nextjs";
import {
  Activity,
  Zap,
  Eye,
  Mic,
  Heart,
  Scan,
  ArrowRight,
  Clock,
  Shield,
  Sparkles,
  Brain,
  Hand,
  TrendingUp,
  FileText,
  Calendar,
  ChevronRight,
  BarChart3,
  CheckCircle2,
  Info,
} from "lucide-react";

// Lazy load diagnostic grid for optimal performance
const DiagnosticGrid = lazy(() => import("./_components/DiagnosticGrid"));

// Types
interface SystemHealth {
  backend: "online" | "offline" | "checking";
  latency: number | null;
  lastChecked: Date | null;
}

interface RecentAssessment {
  id: string;
  type: string;
  module: string;
  icon: React.ElementType;
  date: string;
  riskLevel: "low" | "moderate" | "high";
  score: number;
  gradient: string;
}

interface DiagnosticItem {
  id: string;
  name: string;
  description: string;
  icon: React.ElementType;
  route: string;
  gradient: string;
  accuracy: string;
  isNew?: boolean;
}

// All diagnostic modules in sleek format
const diagnosticItems: DiagnosticItem[] = [
  {
    id: "retinal",
    name: "RetinaScan AI",
    description: "Retinal fundus analysis",
    icon: Eye,
    route: "/dashboard/retinal",
    gradient: "from-cyan-500 to-teal-600",
    accuracy: "96.2%",
  },
  {
    id: "speech",
    name: "SpeechMD AI",
    description: "Voice biomarker analysis",
    icon: Mic,
    route: "/dashboard/speech",
    gradient: "from-blue-500 to-indigo-600",
    accuracy: "95.2%",
    isNew: true,
  },
  {
    id: "cardiology",
    name: "CardioPredict AI",
    description: "ECG signal analysis",
    icon: Heart,
    route: "/dashboard/cardiology",
    gradient: "from-red-500 to-rose-600",
    accuracy: "94.8%",
  },
  {
    id: "radiology",
    name: "ChestXplorer AI",
    description: "Chest X-ray analysis",
    icon: Scan,
    route: "/dashboard/radiology",
    gradient: "from-violet-500 to-purple-600",
    accuracy: "97.1%",
  },
  {
    id: "dermatology",
    name: "SkinSense AI",
    description: "Skin lesion detection",
    icon: Sparkles,
    route: "/dashboard/dermatology",
    gradient: "from-purple-400 to-pink-400",
    accuracy: "94.5%",
  },
  {
    id: "motor",
    name: "Motor Assessment",
    description: "Movement analysis",
    icon: Hand,
    route: "/dashboard/motor",
    gradient: "from-purple-400 to-indigo-400",
    accuracy: "93.5%",
  },
  {
    id: "cognitive",
    name: "Cognitive Testing",
    description: "Memory & cognition",
    icon: Brain,
    route: "/dashboard/cognitive",
    gradient: "from-orange-400 to-red-400",
    accuracy: "92.1%",
  },
  {
    id: "multimodal",
    name: "Multi-Modal",
    description: "Combined analysis",
    icon: Activity,
    route: "/dashboard/multimodal",
    gradient: "from-purple-500 to-violet-500",
    accuracy: "96.8%",
  },
  {
    id: "nri-fusion",
    name: "NRI Fusion",
    description: "Risk index engine",
    icon: Zap,
    route: "/dashboard/nri-fusion",
    gradient: "from-yellow-400 to-orange-400",
    accuracy: "97.2%",
  },
];

// Mock recent assessments
const recentAssessments: RecentAssessment[] = [
  {
    id: "1",
    type: "Speech Analysis",
    module: "SpeechMD AI",
    icon: Mic,
    date: "Today, 2:34 PM",
    riskLevel: "low",
    score: 23,
    gradient: "from-blue-500 to-indigo-600",
  },
  {
    id: "2",
    type: "Retinal Scan",
    module: "RetinaScan AI",
    icon: Eye,
    date: "Today, 11:15 AM",
    riskLevel: "moderate",
    score: 45,
    gradient: "from-cyan-500 to-teal-600",
  },
  {
    id: "3",
    type: "ECG Analysis",
    module: "CardioPredict AI",
    icon: Heart,
    date: "Yesterday",
    riskLevel: "low",
    score: 18,
    gradient: "from-red-500 to-rose-600",
  },
];

/**
 * Loading skeleton for diagnostic grid
 */
function DiagnosticGridSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: 6 }).map((_, i) => (
        <div
          key={i}
          className="h-20 animate-pulse rounded-xl bg-zinc-100 border border-zinc-200"
        />
      ))}
    </div>
  );
}

/**
 * Get greeting based on time of day
 */
function getGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
}

/**
 * System Health Badge - Minimal
 */
function SystemHealthBadge({ health }: { health: SystemHealth }) {
  const isOnline = health.backend === "online";
  return (
    <div className="flex items-center gap-2">
      <span
        className={`h-2 w-2 rounded-full ${isOnline ? "bg-emerald-500" : "bg-amber-500 animate-pulse"}`}
      />
      <span className="text-[12px] text-zinc-500">
        {isOnline ? "Systems Online" : "Checking..."}
      </span>
    </div>
  );
}

/**
 * Sleek Diagnostic Row Card
 */
function DiagnosticRowCard({
  item,
  index,
}: {
  item: DiagnosticItem;
  index: number;
}) {
  const Icon = item.icon;

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.2, delay: index * 0.03 }}
    >
      <Link href={item.route} className="group block">
        <div className="flex items-center gap-4 p-4 rounded-xl bg-white border border-zinc-200 hover:border-zinc-300 hover:shadow-md transition-all duration-200">
          {/* Icon */}
          <div
            className={`flex-shrink-0 p-2.5 rounded-xl bg-gradient-to-br ${item.gradient}`}
          >
            <Icon className="h-5 w-5 text-white" strokeWidth={2} />
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="text-[14px] font-semibold text-zinc-900 truncate">
                {item.name}
              </h3>
              {item.isNew && (
                <span className="px-1.5 py-0.5 bg-blue-100 text-blue-700 text-[9px] font-bold rounded uppercase">
                  New
                </span>
              )}
            </div>
            <p className="text-[12px] text-zinc-500 truncate">
              {item.description}
            </p>
          </div>

          {/* Accuracy */}
          <div className="hidden sm:block text-right">
            <div className="text-[13px] font-semibold text-zinc-900">
              {item.accuracy}
            </div>
            <div className="text-[10px] text-zinc-400">Accuracy</div>
          </div>

          {/* Arrow */}
          <ArrowRight className="h-4 w-4 text-zinc-300 group-hover:text-zinc-500 group-hover:translate-x-1 transition-all duration-200" />
        </div>
      </Link>
    </motion.div>
  );
}

/**
 * Recent Assessment Item - Compact
 */
function RecentAssessmentRow({ assessment }: { assessment: RecentAssessment }) {
  const Icon = assessment.icon;
  const getRiskColor = () => {
    switch (assessment.riskLevel) {
      case "low":
        return "text-emerald-600 bg-emerald-50";
      case "moderate":
        return "text-amber-600 bg-amber-50";
      case "high":
        return "text-red-600 bg-red-50";
    }
  };

  return (
    <div className="flex items-center gap-3 p-3 rounded-lg hover:bg-zinc-50 transition-colors cursor-pointer">
      <div
        className={`p-2 rounded-lg bg-gradient-to-br ${assessment.gradient}`}
      >
        <Icon className="h-4 w-4 text-white" strokeWidth={2} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-[13px] font-medium text-zinc-900 truncate">
          {assessment.type}
        </div>
        <div className="text-[11px] text-zinc-500">{assessment.date}</div>
      </div>
      <div
        className={`px-2 py-0.5 rounded text-[10px] font-medium ${getRiskColor()}`}
      >
        {assessment.score}%
      </div>
    </div>
  );
}

/**
 * Stats Card - Compact
 */
function StatCard({
  icon: Icon,
  label,
  value,
  color,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="flex items-center gap-3 p-3 rounded-lg bg-zinc-50 border border-zinc-100">
      <div className={`p-1.5 rounded-lg ${color}`}>
        <Icon className="h-3.5 w-3.5" strokeWidth={2} />
      </div>
      <div>
        <div className="text-[14px] font-semibold text-zinc-900">{value}</div>
        <div className="text-[10px] text-zinc-500">{label}</div>
      </div>
    </div>
  );
}

/**
 * Main Dashboard Page Component
 */
export default function DashboardPage() {
  const { user } = useUser();
  const [systemHealth, setSystemHealth] = useState<SystemHealth>({
    backend: "checking",
    latency: null,
    lastChecked: null,
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
      });
    } catch {
      setSystemHealth({
        backend: "offline",
        latency: null,
        lastChecked: new Date(),
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

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="space-y-6"
    >
      {/* Hero Header - Clean Greeting */}
      <header className="bg-white rounded-2xl border border-zinc-200 p-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-[24px] font-bold text-zinc-900 tracking-tight">
              {greeting}, {firstName}
            </h1>
            <p className="mt-1 text-[14px] text-zinc-500">
              Run clinical-grade AI analysis on medical imaging and biosignals
            </p>
          </div>
          <SystemHealthBadge health={systemHealth} />
        </div>
      </header>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Diagnostics */}
        <div className="lg:col-span-2 space-y-6">
          {/* Diagnostic Modules - Sleek Rows */}
          <section className="bg-white rounded-2xl border border-zinc-200 p-5">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-[16px] font-semibold text-zinc-900">
                  AI Diagnostics
                </h2>
                <p className="text-[12px] text-zinc-500">9 modules available</p>
              </div>
              <Link
                href="/dashboard/analytics"
                className="text-[12px] font-medium text-blue-600 hover:text-blue-700 flex items-center gap-1"
              >
                Analytics <ChevronRight className="h-3.5 w-3.5" />
              </Link>
            </div>
            <div className="space-y-2">
              {diagnosticItems.map((item, index) => (
                <DiagnosticRowCard key={item.id} item={item} index={index} />
              ))}
            </div>
          </section>
        </div>

        {/* Right Column - Activity & Stats */}
        <div className="space-y-6">
          {/* User Stats */}
          <section className="bg-white rounded-2xl border border-zinc-200 p-5">
            <h2 className="text-[14px] font-semibold text-zinc-900 mb-3">
              Your Overview
            </h2>
            <div className="grid grid-cols-2 gap-2">
              <StatCard
                icon={FileText}
                label="Assessments"
                value="12"
                color="bg-blue-50 text-blue-600"
              />
              <StatCard
                icon={CheckCircle2}
                label="Health Score"
                value="76%"
                color="bg-emerald-50 text-emerald-600"
              />
              <StatCard
                icon={Calendar}
                label="Last Test"
                value="Today"
                color="bg-violet-50 text-violet-600"
              />
              <StatCard
                icon={BarChart3}
                label="Risk Level"
                value="Low"
                color="bg-amber-50 text-amber-600"
              />
            </div>
          </section>

          {/* Recent Activity */}
          <section className="bg-white rounded-2xl border border-zinc-200 p-5">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-[14px] font-semibold text-zinc-900">
                Recent Activity
              </h2>
              <Link
                href="/dashboard/reports"
                className="text-[11px] text-blue-600 hover:text-blue-700"
              >
                View all
              </Link>
            </div>
            <div className="space-y-1">
              {recentAssessments.map((assessment) => (
                <RecentAssessmentRow
                  key={assessment.id}
                  assessment={assessment}
                />
              ))}
            </div>
          </section>

          {/* Info Card */}
          <div className="bg-blue-50 rounded-xl border border-blue-100 p-4">
            <div className="flex items-start gap-3">
              <Info className="h-4 w-4 text-blue-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-[12px] font-medium text-blue-900 mb-0.5">
                  Clinical Disclaimer
                </p>
                <p className="text-[11px] text-blue-700 leading-relaxed">
                  Results should be reviewed by a qualified healthcare
                  professional.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="flex flex-wrap items-center justify-center gap-6 py-3 text-[11px] text-zinc-400">
        <div className="flex items-center gap-1.5">
          <Sparkles className="h-3 w-3" />
          <span>MediLens v1.0</span>
        </div>
        <div className="flex items-center gap-1.5">
          <Shield className="h-3 w-3" />
          <span>HIPAA Compliant</span>
        </div>
        <div className="flex items-center gap-1.5">
          <Clock className="h-3 w-3" />
          <span>Real-time Processing</span>
        </div>
      </footer>
    </motion.div>
  );
}
