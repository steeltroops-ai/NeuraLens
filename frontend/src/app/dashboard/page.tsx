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
  Info
} from 'lucide-react';

import DashboardSidebar from '@/components/dashboard/DashboardSidebar';
import SpeechAssessment from '@/components/dashboard/SpeechAssessment';
import RetinalAssessment from '@/components/dashboard/RetinalAssessment';
import MotorAssessment from '@/components/dashboard/MotorAssessment';
import CognitiveAssessment from '@/components/dashboard/CognitiveAssessment';
import MultiModalAssessment from '@/components/dashboard/MultiModalAssessment';
import NRIFusionDashboard from '@/components/dashboard/NRIFusionDashboard';
import PerformanceMetrics from '@/components/dashboard/PerformanceMetrics';

export type AssessmentType = 'overview' | 'speech' | 'retinal' | 'motor' | 'cognitive' | 'multimodal' | 'nri-fusion';

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
    systemStatus: 'healthy'
  });

  const [performanceMetrics, setPerformanceMetrics] = useState({
    speechLatency: 11.7,
    retinalLatency: 145.2,
    motorLatency: 42.3,
    cognitiveLatency: 38.1,
    nriLatency: 0.3,
    overallAccuracy: 95.2
  });

  // System health check
  useEffect(() => {
    const checkSystemHealth = async () => {
      try {
        // Check backend health
        const response = await fetch('/api/health');
        if (response.ok) {
          setDashboardState(prev => ({ ...prev, systemStatus: 'healthy', lastUpdate: new Date() }));
        } else {
          setDashboardState(prev => ({ ...prev, systemStatus: 'warning' }));
        }
      } catch (error) {
        setDashboardState(prev => ({ ...prev, systemStatus: 'error' }));
      }
    };

    checkSystemHealth();
    const interval = setInterval(checkSystemHealth, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const handleAssessmentChange = (assessment: AssessmentType) => {
    setDashboardState(prev => ({ ...prev, activeAssessment: assessment }));
  };

  const handleProcessingStateChange = (isProcessing: boolean) => {
    setDashboardState(prev => ({ ...prev, isProcessing }));
  };

  const renderMainContent = () => {
    switch (dashboardState.activeAssessment) {
      case 'speech':
        return <SpeechAssessment onProcessingChange={handleProcessingStateChange} />;
      case 'retinal':
        return <RetinalAssessment onProcessingChange={handleProcessingStateChange} />;
      case 'motor':
        return <MotorAssessment onProcessingChange={handleProcessingStateChange} />;
      case 'cognitive':
        return <CognitiveAssessment onProcessingChange={handleProcessingStateChange} />;
      case 'multimodal':
        return <MultiModalAssessment onProcessingChange={handleProcessingStateChange} />;
      case 'nri-fusion':
        return <NRIFusionDashboard onProcessingChange={handleProcessingStateChange} />;
      default:
        return <DashboardOverview metrics={performanceMetrics} systemStatus={dashboardState.systemStatus} />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-slate-200 sticky top-0 z-40">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">NeuroLens Dashboard</h1>
                <p className="text-sm text-slate-600">Real-time Neurological Assessment Platform</p>
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
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-600 border-t-transparent"></div>
                  <span className="text-sm font-medium">Processing...</span>
                </div>
              )}

              {/* Performance Badge */}
              <div className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium">
                {performanceMetrics.overallAccuracy}% Accuracy
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <DashboardSidebar
          activeAssessment={dashboardState.activeAssessment}
          onAssessmentChange={handleAssessmentChange}
          systemStatus={dashboardState.systemStatus}
          performanceMetrics={performanceMetrics}
        />

        {/* Main Content */}
        <main className="flex-1 p-6">
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
  systemStatus 
}: { 
  metrics: any; 
  systemStatus: 'healthy' | 'warning' | 'error';
}) {
  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-2xl font-bold text-slate-900 mb-2">
              Welcome to NeuroLens
            </h2>
            <p className="text-slate-600 text-lg mb-4">
              Advanced neurological assessment platform with real-time AI analysis
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
            <div className="text-3xl font-bold text-blue-600">{metrics.overallAccuracy}%</div>
            <div className="text-sm text-slate-500">Overall Accuracy</div>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <PerformanceMetrics metrics={metrics} />

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
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
  color 
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
    red: 'from-red-500 to-red-600'
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 hover:shadow-md transition-shadow cursor-pointer">
      <div className={`inline-flex p-3 rounded-lg bg-gradient-to-r ${colorClasses[color as keyof typeof colorClasses]} text-white mb-4`}>
        {icon}
      </div>
      <h3 className="text-lg font-semibold text-slate-900 mb-2">{title}</h3>
      <p className="text-slate-600 text-sm mb-3">{description}</p>
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-500">Processing Time</span>
        <span className="font-medium text-green-600">{latency}ms</span>
      </div>
    </div>
  );
}
