'use client';

/**
 * MediLens Dashboard - Neurologist Focus
 * 
 * Professional medical dashboard with:
 * - Patient case overview
 * - Recent assessments
 * - Neurological diagnostic tools
 * - Clinical insights
 * 
 * Designed for hackathon demo with realistic medical data
 */

import { useState, useEffect, memo, lazy, Suspense } from 'react';
import { motion } from 'framer-motion';
import {
  Users,
  AlertTriangle,
  CheckCircle,
  Clock
} from 'lucide-react';

// Lazy load dashboard components for optimal performance
const DiagnosticGrid = lazy(() => import('./_components/DiagnosticGrid'));

/**
 * Loading skeletons
 */
function StatsSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="h-24 animate-pulse rounded-lg bg-zinc-100 border border-zinc-200" />
      ))}
    </div>
  );
}

function RecentCasesSkeleton() {
  return (
    <div className="h-96 animate-pulse rounded-xl bg-zinc-100 border border-zinc-200" />
  );
}

function DiagnosticGridSkeleton() {
  return (
    <div className="space-y-6">
      <div className="h-6 w-32 animate-pulse rounded bg-zinc-200" />
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <div
            key={i}
            className="h-40 animate-pulse rounded-xl bg-zinc-100 border border-zinc-200"
          />
        ))}
      </div>
    </div>
  );
}

/**
 * Dashboard Overview Page Component
 */
export default function DashboardPage() {
  const [todayStats, setTodayStats] = useState({
    totalPatients: 0,
    completedAssessments: 0,
    pendingReviews: 0,
    criticalAlerts: 0
  });

  // Load today's statistics
  useEffect(() => {
    // Simulate realistic medical data
    setTodayStats({
      totalPatients: 23,
      completedAssessments: 18,
      pendingReviews: 5,
      criticalAlerts: 2
    });
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, ease: 'easeOut' }}
      className="space-y-6"
    >
      <Suspense fallback={<DashboardOverviewSkeleton />}>
        <DashboardOverview todayStats={todayStats} />
      </Suspense>
    </motion.div>
  );
}

/**
 * Full dashboard skeleton for initial load
 */
function DashboardOverviewSkeleton() {
  return (
    <div className="space-y-6">
      <StatsSkeleton />
      <RecentCasesSkeleton />
      <DiagnosticGridSkeleton />
    </div>
  );
}

/**
 * Dashboard Overview Component (Memoized for Performance)
 */
const DashboardOverview = memo(
  ({ todayStats }: { todayStats: any }) => {
    return (
      <div className="space-y-6">
        {/* Page Header */}
        <header className="bg-white rounded-xl border border-gray-300 shadow-md p-6">
          <h1 className="text-[20px] font-semibold text-gray-900 tracking-tight">
            Neurology Dashboard
          </h1>
          <p className="mt-1 text-[13px] text-gray-600">
            Today's patient assessments and diagnostic insights
          </p>
        </header>

        {/* Today's Statistics */}
        <section aria-labelledby="stats-heading">
          <h2 id="stats-heading" className="sr-only">Today's Statistics</h2>
          <Suspense fallback={<StatsSkeleton />}>
            <TodayStatsCards stats={todayStats} />
          </Suspense>
        </section>

        {/* Recent Patient Cases */}
        <section aria-labelledby="recent-cases-heading">
          <h2 id="recent-cases-heading" className="sr-only">Recent Patient Cases</h2>
          <Suspense fallback={<RecentCasesSkeleton />}>
            <RecentPatientCases />
          </Suspense>
        </section>

        {/* Diagnostic Modules Grid */}
        <section aria-labelledby="diagnostic-modules-heading" className="bg-white rounded-xl border border-zinc-300 shadow-sm p-6">
          <h2
            id="diagnostic-modules-heading"
            className="mb-4 text-[18px] font-semibold text-zinc-900"
          >
            Neurological Assessment Tools
          </h2>
          <Suspense fallback={<DiagnosticGridSkeleton />}>
            <DiagnosticGrid showSectionHeaders={true} />
          </Suspense>
        </section>
      </div>
    );
  }
);

DashboardOverview.displayName = 'DashboardOverview';

/**
 * Today's Statistics Cards
 */
function TodayStatsCards({ stats }: { stats: any }) {
  const statCards = [
    {
      title: 'Total Patients',
      value: stats.totalPatients,
      icon: Users,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      change: '+3 from yesterday'
    },
    {
      title: 'Completed Assessments',
      value: stats.completedAssessments,
      icon: CheckCircle,
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      change: '+12 from yesterday'
    },
    {
      title: 'Pending Reviews',
      value: stats.pendingReviews,
      icon: Clock,
      color: 'text-amber-600',
      bgColor: 'bg-amber-50',
      change: '2 urgent'
    },
    {
      title: 'Critical Alerts',
      value: stats.criticalAlerts,
      icon: AlertTriangle,
      color: 'text-red-600',
      bgColor: 'bg-red-50',
      change: 'Requires attention'
    }
  ];

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
      {statCards.map((stat, index) => (
        <motion.div
          key={stat.title}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: index * 0.1 }}
          className="bg-white rounded-lg border border-gray-300 shadow-md p-4"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[12px] font-medium text-zinc-600 uppercase tracking-wide">
                {stat.title}
              </p>
              <p className="text-[24px] font-semibold text-zinc-900 mt-1">
                {stat.value}
              </p>
              <p className="text-[11px] text-zinc-500 mt-1">
                {stat.change}
              </p>
            </div>
            <div className={`p-3 rounded-lg ${stat.bgColor}`}>
              <stat.icon className={`h-5 w-5 ${stat.color}`} />
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
}

/**
 * Recent Patient Cases Component
 */
function RecentPatientCases() {
  const recentCases = [
    {
      id: 'P-2024-001',
      patientInitials: 'J.D.',
      age: 67,
      condition: 'Suspected Parkinson\'s Disease',
      assessment: 'Speech Analysis',
      status: 'completed',
      riskScore: 78,
      timestamp: '2 hours ago',
      priority: 'high'
    },
    {
      id: 'P-2024-002',
      patientInitials: 'M.S.',
      age: 45,
      condition: 'Multiple Sclerosis Follow-up',
      assessment: 'Cognitive Testing',
      status: 'in-progress',
      riskScore: 45,
      timestamp: '4 hours ago',
      priority: 'medium'
    },
    {
      id: 'P-2024-003',
      patientInitials: 'R.K.',
      age: 72,
      condition: 'Diabetic Retinopathy',
      assessment: 'Retinal Scan',
      status: 'completed',
      riskScore: 89,
      timestamp: '6 hours ago',
      priority: 'critical'
    },
    {
      id: 'P-2024-004',
      patientInitials: 'A.L.',
      age: 58,
      condition: 'Stroke Risk Assessment',
      assessment: 'Multi-modal Analysis',
      status: 'pending-review',
      riskScore: 62,
      timestamp: '1 day ago',
      priority: 'medium'
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-50';
      case 'in-progress': return 'text-blue-600 bg-blue-50';
      case 'pending-review': return 'text-amber-600 bg-amber-50';
      default: return 'text-zinc-600 bg-zinc-50';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'border-l-red-500';
      case 'high': return 'border-l-orange-500';
      case 'medium': return 'border-l-yellow-500';
      default: return 'border-l-zinc-300';
    }
  };

  return (
    <div className="bg-white rounded-xl border border-gray-300 shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-[16px] font-semibold text-zinc-900">Recent Patient Cases</h3>
        <button className="text-[12px] text-blue-600 hover:text-blue-700 font-medium">
          View All Cases
        </button>
      </div>

      <div className="space-y-3">
        {recentCases.map((case_, index) => (
          <motion.div
            key={case_.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
            className={`border-l-4 ${getPriorityColor(case_.priority)} bg-zinc-50 rounded-r-lg p-4`}
          >
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-[13px] font-medium text-zinc-900">
                    {case_.patientInitials} ({case_.age}y)
                  </span>
                  <span className="text-[11px] text-zinc-500">
                    ID: {case_.id}
                  </span>
                  <span className={`px-2 py-1 rounded-full text-[10px] font-medium ${getStatusColor(case_.status)}`}>
                    {case_.status.replace('-', ' ')}
                  </span>
                </div>
                <p className="text-[12px] text-zinc-700 mb-1">{case_.condition}</p>
                <p className="text-[11px] text-zinc-500">{case_.assessment} • {case_.timestamp}</p>
              </div>
              <div className="text-right">
                <div className="text-[14px] font-semibold text-zinc-900">
                  Risk: {case_.riskScore}%
                </div>
                <div className={`text-[10px] font-medium ${case_.riskScore >= 80 ? 'text-red-600' :
                  case_.riskScore >= 60 ? 'text-orange-600' :
                    case_.riskScore >= 40 ? 'text-yellow-600' : 'text-green-600'
                  }`}>
                  {case_.riskScore >= 80 ? 'Critical' :
                    case_.riskScore >= 60 ? 'High' :
                      case_.riskScore >= 40 ? 'Moderate' : 'Low'}
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
