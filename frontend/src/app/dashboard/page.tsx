'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  Mic,
  Eye,
  Hand,
  Zap,
  Activity,
  Shield,
  Clock,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Info,
} from 'lucide-react';

import DashboardSidebar from '@/components/dashboard/DashboardSidebar';
import SpeechAssessment from '@/components/dashboard/SpeechAssessment';
import RetinalAssessment from '@/components/dashboard/RetinalAssessment';
import MotorAssessment from '@/components/dashboard/MotorAssessment';
import CognitiveAssessment from '@/components/dashboard/CognitiveAssessment';
import MultiModalAssessment from '@/components/dashboard/MultiModalAssessment';
import NRIFusionDashboard from '@/components/dashboard/NRIFusionDashboard';
import PerformanceMetrics from '@/components/dashboard/PerformanceMetrics';

export type AssessmentType =
  | 'overview'
  | 'speech'
  | 'retinal'
  | 'motor'
  | 'cognitive'
  | 'multimodal'
  | 'nri-fusion';

interface DashboardState {
  activeAssessment: AssessmentType;
  isProcessing: boolean;
  lastUpdate: Date | null;
  systemStatus: 'healthy' | 'warning' | 'error';
}

export default function Dashboard() {
  const [dashboardState, setDashboardState] = useState<DashboardState>({
    activeAssessment: 'overview',
    isProcessing: false,
    lastUpdate: null,
    systemStatus: 'healthy',
  });

  const [performanceMetrics, setPerformanceMetrics] = useState({
    speechLatency: 11.7,
    retinalLatency: 145.2,
    motorLatency: 42.3,
    cognitiveLatency: 38.1,
    nriLatency: 0.3,
    overallAccuracy: 95.2,
  });

  // System health check
  useEffect(() => {
    const checkSystemHealth = async () => {
      try {
        // Check backend health
        const response = await fetch('/api/health');
        if (response.ok) {
          setDashboardState((prev) => ({
            ...prev,
            systemStatus: 'healthy',
            lastUpdate: new Date(),
          }));
        } else {
          setDashboardState((prev) => ({ ...prev, systemStatus: 'warning' }));
        }
      } catch (error) {
        setDashboardState((prev) => ({ ...prev, systemStatus: 'error' }));
      }
    };

    checkSystemHealth();
    const interval = setInterval(checkSystemHealth, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const handleAssessmentChange = (assessment: AssessmentType) => {
    setDashboardState((prev) => ({ ...prev, activeAssessment: assessment }));
  };

  const handleProcessingStateChange = (isProcessing: boolean) => {
    setDashboardState((prev) => ({ ...prev, isProcessing }));
  };

  const renderMainContent = () => {
    switch (dashboardState.activeAssessment) {
      case 'speech':
        return (
          <SpeechAssessment onProcessingChange={handleProcessingStateChange} />
        );
      case 'retinal':
        return (
          <RetinalAssessment onProcessingChange={handleProcessingStateChange} />
        );
      case 'motor':
        return (
          <MotorAssessment onProcessingChange={handleProcessingStateChange} />
        );
      case 'cognitive':
        return (
          <CognitiveAssessment
            onProcessingChange={handleProcessingStateChange}
          />
        );
      case 'multimodal':
        return (
          <MultiModalAssessment
            onProcessingChange={handleProcessingStateChange}
          />
        );
      case 'nri-fusion':
        return (
          <NRIFusionDashboard
            onProcessingChange={handleProcessingStateChange}
          />
        );
      default:
        return (
          <DashboardOverview
            metrics={performanceMetrics}
            systemStatus={dashboardState.systemStatus}
          />
        );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur-sm">
        <div className="px-4 py-4 sm:px-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 p-2">
                <Brain className="h-5 w-5 text-white sm:h-6 sm:w-6" />
              </div>
              <div className="hidden sm:block">
                <h1 className="text-lg font-bold text-slate-900 sm:text-xl">
                  NeuroLens-X Dashboard
                </h1>
                <p className="text-xs text-slate-600 sm:text-sm">
                  Real-time Neurological Assessment Platform
                </p>
              </div>
              <div className="sm:hidden">
                <h1 className="text-lg font-bold text-slate-900">
                  NeuroLens-X
                </h1>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* System Status */}
              <div className="flex items-center space-x-2">
                {dashboardState.systemStatus === 'healthy' && (
                  <div className="flex items-center space-x-1 text-green-600">
                    <CheckCircle className="h-4 w-4" />
                    <span className="text-sm font-medium">System Healthy</span>
                  </div>
                )}
                {dashboardState.systemStatus === 'warning' && (
                  <div className="flex items-center space-x-1 text-yellow-600">
                    <AlertTriangle className="h-4 w-4" />
                    <span className="text-sm font-medium">System Warning</span>
                  </div>
                )}
                {dashboardState.systemStatus === 'error' && (
                  <div className="flex items-center space-x-1 text-red-600">
                    <AlertTriangle className="h-4 w-4" />
                    <span className="text-sm font-medium">System Error</span>
                  </div>
                )}
              </div>

              {/* Processing Indicator */}
              {dashboardState.isProcessing && (
                <div className="flex items-center space-x-2 text-blue-600">
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-blue-600 border-t-transparent"></div>
                  <span className="text-sm font-medium">Processing...</span>
                </div>
              )}

              {/* Performance Badge */}
              <div className="rounded-full bg-green-100 px-2 py-1 text-xs font-medium text-green-800 sm:px-3 sm:text-sm">
                <span className="hidden sm:inline">
                  {performanceMetrics.overallAccuracy}% Accuracy
                </span>
                <span className="sm:hidden">
                  {performanceMetrics.overallAccuracy}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="flex flex-col lg:flex-row">
        {/* Sidebar - Hidden on mobile, shown as overlay when needed */}
        <div className="hidden lg:block">
          <DashboardSidebar
            activeAssessment={dashboardState.activeAssessment}
            onAssessmentChange={handleAssessmentChange}
            systemStatus={dashboardState.systemStatus}
            performanceMetrics={performanceMetrics}
          />
        </div>

        {/* Mobile Navigation */}
        <div className="border-b border-slate-200 bg-white p-4 lg:hidden">
          <select
            value={dashboardState.activeAssessment}
            onChange={(e) =>
              handleAssessmentChange(e.target.value as AssessmentType)
            }
            className="w-full rounded-lg border border-slate-200 bg-white p-3 font-medium text-slate-900"
          >
            <option value="overview">Dashboard Overview</option>
            <option value="speech">Speech Analysis</option>
            <option value="retinal">Retinal Imaging</option>
            <option value="motor">Motor Function</option>
            <option value="cognitive">Cognitive Tests</option>
            <option value="multimodal">Multi-Modal Assessment</option>
            <option value="nri-fusion">NRI Fusion Engine</option>
          </select>
        </div>

        {/* Main Content */}
        <main className="flex-1 p-4 sm:p-6">
          <AnimatePresence mode="wait">
            <motion.div
              key={dashboardState.activeAssessment}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="h-full"
            >
              {renderMainContent()}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

// Dashboard Overview Component
function DashboardOverview({
  metrics,
  systemStatus,
}: {
  metrics: any;
  systemStatus: 'healthy' | 'warning' | 'error';
}) {
  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="mb-2 text-2xl font-bold text-slate-900">
              Welcome to NeuroLens-X
            </h2>
            <p className="mb-4 text-lg text-slate-600">
              Advanced neurological assessment platform with real-time AI
              analysis
            </p>
            <div className="flex items-center space-x-6 text-sm text-slate-500">
              <div className="flex items-center space-x-1">
                <Clock className="h-4 w-4" />
                <span>Real-time processing (&lt;100ms)</span>
              </div>
              <div className="flex items-center space-x-1">
                <Shield className="h-4 w-4" />
                <span>95%+ accuracy</span>
              </div>
              <div className="flex items-center space-x-1">
                <Activity className="h-4 w-4" />
                <span>Multi-modal fusion</span>
              </div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-blue-600">
              {metrics.overallAccuracy}%
            </div>
            <div className="text-sm text-slate-500">Overall Accuracy</div>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <PerformanceMetrics metrics={metrics} />

      {/* Quick Actions */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 sm:gap-6 lg:grid-cols-3 xl:grid-cols-3">
        <QuickActionCard
          icon={<Mic className="h-6 w-6" />}
          title="Speech Assessment"
          description="Analyze speech patterns for neurological indicators"
          latency={metrics.speechLatency}
          color="blue"
        />
        <QuickActionCard
          icon={<Eye className="h-6 w-6" />}
          title="Retinal Analysis"
          description="Examine retinal images for vascular changes"
          latency={metrics.retinalLatency}
          color="green"
        />
        <QuickActionCard
          icon={<Hand className="h-6 w-6" />}
          title="Motor Function"
          description="Assess movement patterns and coordination"
          latency={metrics.motorLatency}
          color="purple"
        />
        <QuickActionCard
          icon={<Brain className="h-6 w-6" />}
          title="Cognitive Tests"
          description="Evaluate memory, attention, and executive function"
          latency={metrics.cognitiveLatency}
          color="indigo"
        />
        <QuickActionCard
          icon={<Zap className="h-6 w-6" />}
          title="Multi-Modal"
          description="Combined assessment across all modalities"
          latency={metrics.nriLatency}
          color="yellow"
        />
        <QuickActionCard
          icon={<TrendingUp className="h-6 w-6" />}
          title="NRI Fusion"
          description="Advanced risk index calculation"
          latency={metrics.nriLatency}
          color="red"
        />
      </div>
    </div>
  );
}

// Quick Action Card Component
function QuickActionCard({
  icon,
  title,
  description,
  latency,
  color,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  latency: number;
  color: string;
}) {
  const colorClasses = {
    blue: 'from-blue-500 to-blue-600',
    green: 'from-green-500 to-green-600',
    purple: 'from-purple-500 to-purple-600',
    indigo: 'from-indigo-500 to-indigo-600',
    yellow: 'from-yellow-500 to-yellow-600',
    red: 'from-red-500 to-red-600',
  };

  return (
    <div className="cursor-pointer rounded-xl border border-slate-200 bg-white p-6 shadow-sm transition-shadow hover:shadow-md">
      <div
        className={`inline-flex rounded-lg bg-gradient-to-r p-3 ${colorClasses[color as keyof typeof colorClasses]} mb-4 text-white`}
      >
        {icon}
      </div>
      <h3 className="mb-2 text-lg font-semibold text-slate-900">{title}</h3>
      <p className="mb-3 text-sm text-slate-600">{description}</p>
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-500">Processing Time</span>
        <span className="font-medium text-green-600">{latency}ms</span>
      </div>
    </div>
  );
}
