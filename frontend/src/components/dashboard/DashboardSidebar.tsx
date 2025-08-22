'use client';

import React from 'react';
import { motion } from 'framer-motion';
import {
  Home,
  Mic,
  Eye,
  Hand,
  Brain,
  Zap,
  TrendingUp,
  Activity,
  Clock,
  CheckCircle,
  AlertTriangle,
  Info,
} from 'lucide-react';

import type { AssessmentType } from '@/types/dashboard';

interface DashboardSidebarProps {
  activeAssessment: AssessmentType;
  onAssessmentChange: (assessment: AssessmentType) => void;
  systemStatus: 'healthy' | 'warning' | 'error';
  performanceMetrics: {
    speechLatency: number;
    retinalLatency: number;
    motorLatency: number;
    cognitiveLatency: number;
    nriLatency: number;
    overallAccuracy: number;
  };
}

interface NavigationItem {
  id: AssessmentType;
  label: string;
  description: string;
  icon: React.ReactNode;
  latency?: number;
  accuracy?: number;
  category: 'overview' | 'individual' | 'advanced';
  conditions?: string[];
}

export default function DashboardSidebar({
  activeAssessment,
  onAssessmentChange,
  systemStatus,
  performanceMetrics,
}: DashboardSidebarProps) {
  const navigationItems: NavigationItem[] = [
    // Overview
    {
      id: 'overview',
      label: 'Dashboard',
      description: 'System overview',
      icon: <Home className="h-5 w-5" />,
      category: 'overview',
    },

    // Individual Assessments - CLEAN BUTTONS ONLY
    {
      id: 'speech',
      label: 'Speech Test',
      description: 'Voice analysis',
      icon: <Mic className="h-5 w-5" />,
      category: 'individual',
    },
    {
      id: 'retinal',
      label: 'Eye Test',
      description: 'Retinal imaging',
      icon: <Eye className="h-5 w-5" />,
      category: 'individual',
    },
    {
      id: 'motor',
      label: 'Motor Test',
      description: 'Movement analysis',
      icon: <Hand className="h-5 w-5" />,
      category: 'individual',
    },
    {
      id: 'cognitive',
      label: 'Cognitive Test',
      description: 'Memory & attention',
      icon: <Brain className="h-5 w-5" />,
      category: 'individual',
    },

    // Advanced Assessments - SIMPLIFIED
    {
      id: 'multimodal',
      label: 'Full Assessment',
      description: 'All tests combined',
      icon: <Activity className="h-5 w-5" />,
      category: 'advanced',
    },
    {
      id: 'nri-fusion',
      label: 'Risk Analysis',
      description: 'Advanced analytics',
      icon: <TrendingUp className="h-5 w-5" />,
      category: 'advanced',
    },
  ];

  const getCategoryTitle = (category: string) => {
    switch (category) {
      case 'overview':
        return 'Overview';
      case 'individual':
        return 'Tests';
      case 'advanced':
        return 'Advanced';
      default:
        return '';
    }
  };

  const getLatencyColor = (latency?: number) => {
    if (!latency) return 'text-slate-500';
    if (latency < 50) return 'text-green-600';
    if (latency < 100) return 'text-yellow-600';
    return 'text-red-600';
  };

  const renderNavigationItem = (item: NavigationItem) => {
    const isActive = activeAssessment === item.id;

    return (
      <motion.button
        key={item.id}
        onClick={() => onAssessmentChange(item.id)}
        className={`w-full rounded-lg p-4 text-left transition-all duration-200 ${
          isActive
            ? 'border-blue-200 bg-blue-50 shadow-sm'
            : 'border-transparent hover:bg-slate-50'
        } border`}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <div className="flex items-start space-x-3">
          <div
            className={`rounded-lg p-2 ${
              isActive
                ? 'bg-blue-100 text-blue-600'
                : 'bg-slate-100 text-slate-600'
            }`}
          >
            {item.icon}
          </div>

          <div className="min-w-0 flex-1">
            <h3
              className={`text-sm font-medium ${
                isActive ? 'text-blue-900' : 'text-slate-900'
              }`}
            >
              {item.label}
            </h3>
            <p className="text-xs text-slate-600">{item.description}</p>
          </div>
        </div>
      </motion.button>
    );
  };

  return (
    <aside className="h-screen w-64 overflow-y-auto border-r border-slate-200 bg-white">
      <div className="p-6">
        {/* Clean Header - Updated */}
        <div className="mb-8">
          <h2 className="text-lg font-semibold text-slate-900">NeuroLens</h2>
          <p className="text-sm text-slate-600">Assessment Platform</p>
        </div>

        {/* Navigation */}
        <nav className="space-y-6">
          {['overview', 'individual', 'advanced'].map((category) => (
            <div key={category}>
              <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
                {getCategoryTitle(category)}
              </h2>
              <div className="space-y-2">
                {navigationItems
                  .filter((item) => item.category === category)
                  .map(renderNavigationItem)}
              </div>
            </div>
          ))}
        </nav>
      </div>
    </aside>
  );
}
