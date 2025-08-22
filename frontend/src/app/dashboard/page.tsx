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
  Home,
  Settings,
  BarChart3,
  Users,
  FileText,
  HelpCircle,
} from 'lucide-react';

import DashboardSidebar from '@/components/dashboard/DashboardSidebar';
import SpeechAssessment from '@/components/dashboard/SpeechAssessment';
import RetinalAssessment from '@/components/dashboard/RetinalAssessment';
import MotorAssessment from '@/components/dashboard/MotorAssessment';
import CognitiveAssessment from '@/components/dashboard/CognitiveAssessment';
import MultiModalAssessment from '@/components/dashboard/MultiModalAssessment';
import NRIFusionDashboard from '@/components/dashboard/NRIFusionDashboard';
import PerformanceMetrics from '@/components/dashboard/PerformanceMetrics';
import type { AssessmentType, DashboardState } from '@/types/dashboard';

const sidebarItems = [
  { id: 'overview', label: 'Overview', icon: Home },
  { id: 'speech', label: 'Speech Analysis', icon: Mic },
  { id: 'retinal', label: 'Retinal Imaging', icon: Eye },
  { id: 'motor', label: 'Motor Function', icon: Hand },
  { id: 'cognitive', label: 'Cognitive Tests', icon: Brain },
  { id: 'multimodal', label: 'Multi-Modal', icon: Activity },
  { id: 'nri-fusion', label: 'NRI Fusion', icon: Zap },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'patients', label: 'Patients', icon: Users },
  { id: 'reports', label: 'Reports', icon: FileText },
  { id: 'settings', label: 'Settings', icon: Settings },
  { id: 'help', label: 'Help', icon: HelpCircle },
];

export default function Dashboard() {
  const [dashboardState, setDashboardState] = useState<DashboardState>({
    activeAssessment: 'overview',
    isProcessing: false,
    lastUpdate: null,
    systemStatus: 'healthy',
  });

  const [performanceMetrics] = useState({
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
    const interval = setInterval(checkSystemHealth, 30000);
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
    <div className="flex min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Fixed Sidebar */}
      <div className="fixed inset-y-0 left-0 z-50 w-64 border-r border-slate-200 bg-white shadow-lg">
        {/* Sidebar Header */}
        <div className="flex items-center space-x-3 border-b border-slate-200 p-6">
          <div className="rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 p-2">
            <Brain className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-900">NeuroLens-X</h1>
            <p className="text-xs text-slate-600">Dashboard</p>
          </div>
        </div>

        {/* System Status */}
        <div className="border-b border-slate-200 p-4">
          <div className="flex items-center space-x-2">
            {dashboardState.systemStatus === 'healthy' && (
              <div className="flex items-center space-x-2 text-green-600">
                <CheckCircle className="h-4 w-4" />
                <span className="text-sm font-medium">System Healthy</span>
              </div>
            )}
            {dashboardState.systemStatus === 'warning' && (
              <div className="flex items-center space-x-2 text-yellow-600">
                <AlertTriangle className="h-4 w-4" />
                <span className="text-sm font-medium">System Warning</span>
              </div>
            )}
            {dashboardState.systemStatus === 'error' && (
              <div className="flex items-center space-x-2 text-red-600">
                <AlertTriangle className="h-4 w-4" />
                <span className="text-sm font-medium">System Error</span>
              </div>
            )}
          </div>

          <div className="mt-2 w-fit rounded-full bg-green-100 px-3 py-1 text-xs font-medium text-green-800">
            {performanceMetrics.overallAccuracy}% Accuracy
          </div>

          {dashboardState.isProcessing && (
            <div className="mt-2 flex items-center space-x-2 text-blue-600">
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-blue-600 border-t-transparent"></div>
              <span className="text-sm font-medium">Processing...</span>
            </div>
          )}
        </div>

        {/* Navigation Menu */}
        <nav className="flex-1 overflow-y-auto p-4">
          <div className="space-y-2">
            {sidebarItems.map((item) => {
              const Icon = item.icon;
              const isActive = dashboardState.activeAssessment === item.id;

              return (
                <button
                  key={item.id}
                  onClick={() =>
                    handleAssessmentChange(item.id as AssessmentType)
                  }
                  className={`flex w-full items-center space-x-3 rounded-lg px-3 py-2 text-left transition-colors ${
                    isActive
                      ? 'border border-blue-200 bg-blue-100 text-blue-700'
                      : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                  }`}
                >
                  <Icon className="h-5 w-5" />
                  <span className="text-sm font-medium">{item.label}</span>
                </button>
              );
            })}
          </div>
        </nav>

        {/* Footer */}
        <div className="border-t border-slate-200 p-4">
          <div className="text-center text-xs text-slate-500">
            © 2024 NeuroLens-X
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="ml-64 flex-1">
        <main className="p-6">
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
              Welcome to NeuroLens-X Dashboard
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
                <span>90%+ clinical accuracy</span>
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
            <div className="mt-2 flex items-center justify-end space-x-1">
              {systemStatus === 'healthy' && (
                <>
                  <CheckCircle className="h-4 w-4 text-green-600" />
                  <span className="text-sm text-green-600">System Healthy</span>
                </>
              )}
              {systemStatus === 'warning' && (
                <>
                  <AlertTriangle className="h-4 w-4 text-yellow-600" />
                  <span className="text-sm text-yellow-600">
                    System Warning
                  </span>
                </>
              )}
              {systemStatus === 'error' && (
                <>
                  <AlertTriangle className="h-4 w-4 text-red-600" />
                  <span className="text-sm text-red-600">System Error</span>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <PerformanceMetrics metrics={metrics} />
    </div>
  );
}
